// 岁月织机 — L2 经线（拉弦 / 二胡竖向丝线）v5
//
// 重构要点：
//   - 几何体改为矩形截面 strip（PlaneGeometry），模拟粗麻绳/丝线质感
//   - 每根线有随机宽度微扰，避免过于整齐
//   - 高光颜色改为地区调色板的 bowed 色（不再固定白色）
//   - 高光横向扩散：用屏幕空间 UV 在 X 方向叠加光晕，比原来宽 3 倍
//   - 线体基础纹理：用噪声模拟麻布纤维感（沿 vT 方向随机亮暗变化）

import { useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { bridge, colors } from './AudioBridge';

const NUM_THREADS      = 20;
const SEGS_PER_THREAD  = 64;   // 增加分段，让纤维噪声更细腻
const WIDTH            = 12;
const HEIGHT           = 8;
const THREAD_W         = 0.045; // 每根线的基础宽度（世界单位），比 Line 粗很多

// ── Shader ──────────────────────────────────────────────────────────────────

const vertexShader = /* glsl */`
attribute float aT;       // 沿线归一化位置 [0,1]
attribute float aXSide;   // -1 = 左边缘，+1 = 右边缘
attribute float aWidth;   // 该线的实际宽度（含随机扰动）

uniform float uAmp;
uniform float uTime;
uniform float uSeed;

varying float vT;
varying float vXSide;     // 传给 frag 用于横向光晕

void main() {
  vT     = aT;
  vXSide = aXSide;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const fragmentShader = /* glsl */`
uniform vec3  uColor;
uniform vec3  uGlowColor;   // 地区高光色（不再固定白色）
uniform float uAmp;
uniform float uTime;
uniform float uSeed;
uniform float uBaseOpacity;
uniform float uThreshold;

varying float vT;
varying float vXSide;

// 简单 hash 噪声，模拟麻布纤维疏密
float hash(float n) { return fract(sin(n) * 43758.5453); }
float fbr(float t, float seed) {
  float f = 0.0;
  f += hash(t * 31.7 + seed) * 0.5;
  f += hash(t * 67.3 + seed + 1.1) * 0.3;
  f += hash(t * 127.1 + seed + 2.3) * 0.2;
  return f;
}

void main() {
  // 超阈余量
  float excess = clamp((uAmp - uThreshold) / max(0.2, 1.0 - uThreshold), 0.0, 1.0);

  // 流光中心沿线移动
  float speed  = 0.6 + uAmp * 2.5;
  float center = 0.5 + sin(uTime * speed + uSeed) * 0.42;

  // 纵向高光宽度（比原来更宽：0.18~0.38）
  float vWidth = 0.18 + uAmp * 0.20;
  float vDist  = abs(vT - center);
  float vMask  = max(0.0, 1.0 - vDist / max(vWidth, 0.001));
  vMask = vMask * vMask;

  // 横向光晕：用 vXSide 在线宽方向叠加柔边（-1~+1 范围内的余弦晕）
  // 中心最亮，边缘渐暗；让高光感觉"溢出"线宽两侧
  float hGlow = cos(vXSide * 1.2);          // 中心 cos(0)=1，边缘 cos(1.2)≈0.36
  hGlow = hGlow * hGlow;                     // 平方让中心更集中
  float bright = excess * vMask * hGlow;

  // 麻布纤维噪声：让线体亮暗不均匀（基础层）
  float fiber = 0.6 + fbr(vT, uSeed) * 0.4; // [0.6, 1.0]

  // 基础层：线体本色 × 纤维噪声
  float baseAlpha = uBaseOpacity * fiber;

  // 高光层：地区色光晕
  float glowAlpha = bright * 0.9;

  // 混合：基础 + 高光（高光色叠加在线体色上）
  vec3 finalColor = mix(uColor, uGlowColor, clamp(bright * 1.5, 0.0, 1.0));
  float alpha = clamp(baseAlpha + glowAlpha, 0.0, 1.0);

  gl_FragColor = vec4(finalColor, alpha);
}
`;

// ── 组件 ─────────────────────────────────────────────────────────────────────

export function WarpThreads() {
  const { mats, meshes } = useMemo(() => {
    const allMats:   THREE.ShaderMaterial[] = [];
    const allMeshes: THREE.Mesh[]           = [];

    for (let i = 0; i < NUM_THREADS; i++) {
      const x0    = ((i / (NUM_THREADS - 1)) - 0.5) * WIDTH;
      const seed  = i * 7.13 + 2.7;
      const phi   = (i * 0.618033988) % 1.0;
      const threshold = phi * 0.88;
      // 随机宽度扰动 ±20%，让每根线粗细略有不同
      const threadW = THREAD_W * (0.8 + (hash11(i * 13.7)) * 0.4);

      // ── 几何：矩形 strip（左右各一列顶点，共 SEGS+1 行）────────────────
      const rows  = SEGS_PER_THREAD + 1;
      const verts = rows * 2;               // 左列 + 右列

      const positions = new Float32Array(verts * 3);
      const aT        = new Float32Array(verts);
      const aXSide    = new Float32Array(verts);
      const aWidth    = new Float32Array(verts);
      const indices   = [];

      for (let r = 0; r < rows; r++) {
        const t  = r / SEGS_PER_THREAD;
        const y  = (t - 0.5) * HEIGHT;
        const li = r * 2;       // 左顶点索引
        const ri = r * 2 + 1;  // 右顶点索引

        positions[li * 3]     = x0 - threadW * 0.5;
        positions[li * 3 + 1] = y;
        positions[li * 3 + 2] = 0;

        positions[ri * 3]     = x0 + threadW * 0.5;
        positions[ri * 3 + 1] = y;
        positions[ri * 3 + 2] = 0;

        aT[li]     = t;  aT[ri]     = t;
        aXSide[li] = -1; aXSide[ri] = 1;
        aWidth[li] = threadW; aWidth[ri] = threadW;

        if (r < SEGS_PER_THREAD) {
          // 两个三角形组成一个矩形格
          indices.push(li, ri, li + 2, ri, ri + 2, li + 2);
        }
      }

      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geo.setAttribute('aT',       new THREE.Float32BufferAttribute(aT, 1));
      geo.setAttribute('aXSide',   new THREE.Float32BufferAttribute(aXSide, 1));
      geo.setAttribute('aWidth',   new THREE.Float32BufferAttribute(aWidth, 1));
      geo.setIndex(indices);

      const mat = new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms: {
          uColor:       { value: new THREE.Color() },
          uGlowColor:   { value: new THREE.Color(1, 1, 1) },
          uAmp:         { value: 0 },
          uTime:        { value: 0 },
          uSeed:        { value: seed },
          uBaseOpacity: { value: 0.06 },   // 比原来的 0.025 明显，让线体可见
          uThreshold:   { value: threshold },
        },
        transparent: true,
        depthWrite: false,
        blending: THREE.NormalBlending,
        side: THREE.DoubleSide,
      });

      allMats.push(mat);
      allMeshes.push(new THREE.Mesh(geo, mat));
    }

    return { mats: allMats, meshes: allMeshes };
  }, []);

  useFrame(({ clock }) => {
    const t   = clock.elapsedTime;
    const amp = bridge.amp.erhu;

    for (let i = 0; i < NUM_THREADS; i++) {
      const mat = mats[i];
      mat.uniforms.uColor.value.copy(colors.bowed);
      // 高光色：用地区调色板的 bowed 色加亮（饱和度提高），而非固定白色
      // 取 bowed 色 + vocal.aux 的混合，让高光带地区感
      const gc = mat.uniforms.uGlowColor.value as THREE.Color;
      gc.copy(colors.bowed).lerp(colors.vocal.aux, 0.5).multiplyScalar(2.2);
      mat.uniforms.uAmp.value  = amp * (0.4 + bridge.introProgress * 0.6);
      mat.uniforms.uTime.value = t;
    }
  });

  return (
    <group position={[0, 0, -0.1]}>
      {meshes.map((m, i) => <primitive key={i} object={m} />)}
    </group>
  );
}

// 简单伪随机 [0,1]
function hash11(n: number): number {
  return ((Math.sin(n) * 43758.5453) % 1 + 1) % 1;
}
