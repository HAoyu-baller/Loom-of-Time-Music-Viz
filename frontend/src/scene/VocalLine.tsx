// 岁月织机 — L5 人声水袖飘带（v5：多层ShaderMaterial纯代码实现）
//
// 架构：4 层独立 ShaderMaterial 带，AdditiveBlending 叠加
//   每层相位偏移错开 → 折叠绸缎感
//   宽度随曲率变化 → 弯曲处收窄，舒展处宽开
//   Fragment shader 实现边缘渐消 + 冷暖色轴渐变
//   无任何贴图素材依赖

import { useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { bridge, colors } from './AudioBridge';

const POINTS = 128;

// yungui 粒子颜色写入临时缓存（避免每帧分配）
const _tmpYg  = new THREE.Color();
const _tmpYg2 = new THREE.Color();

// ── 云贵粒子 ShaderMaterial（支持逐点 gl_PointSize + 软圆）───────────
const ygVertFull = /* glsl */`
attribute float aSize;
varying vec4 vColor;
void main() {
  vColor = vec4(color, 1.0);
  vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = aSize * (300.0 / -mvPos.z);
  gl_Position  = projectionMatrix * mvPos;
}
`;
const ygFragFull = /* glsl */`
varying vec4 vColor;
void main() {
  float d = length(gl_PointCoord - vec2(0.5));
  if (d > 0.5) discard;
  float alpha = 1.0 - smoothstep(0.28, 0.5, d);
  gl_FragColor = vec4(vColor.rgb * vColor.a * alpha, 1.0);
}
`;

// ── 云贵粒子常量（模块级，避免 useMemo dep 问题）────────────────────
const YG_COUNT  = 600;
const YG_TRAIL  = 14;
const YG_HALF_W = 0.55;
const YG_TOTAL  = YG_COUNT * (1 + YG_TRAIL);

// ── 4 层参数 ─────────────────────────────────────────────────────────
// phaseT: 时间相位偏移（秒），让各层波形不完全同步
// ampScale: 振幅系数
// baseHalfW: 基础半宽（世界单位）
// yOffset: 额外垂直偏移，轻微错层
// opacity: 基础不透明度（AdditiveBlending 下近似峰值亮度）
const LAYERS = [
  { phaseT: 0.0,  ampScale: 1.00, baseHalfW: 0.55, yOffset:  0.00, opacity: 0.30 },
  { phaseT: 0.38, ampScale: 0.88, baseHalfW: 0.42, yOffset:  0.06, opacity: 0.25 },
  { phaseT: 0.76, ampScale: 0.72, baseHalfW: 0.32, yOffset: -0.04, opacity: 0.20 },
  { phaseT: 1.20, ampScale: 0.56, baseHalfW: 0.22, yOffset:  0.10, opacity: 0.15 },
] as const;

// ── fbm：慢波参数，减少周期数，降低时间速率 ─────────────────────────
// 约 2.2 个周期横跨全屏，时间系数比原来减半
function fbm(x: number, t: number): number {
  return (
    Math.sin(x * 0.85 + t * 0.08) * 0.38 +
    Math.sin(x * 1.70 + t * 0.12 + 1.3) * 0.28 +
    Math.sin(x * 0.45 + t * 0.05 - 0.7) * 0.34
  );
}

// ── ShaderMaterial：边缘渐消 + 冷暖色轴渐变 ─────────────────────────
const ribbonVert = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// uv.x = u (0→1, 从左到右)
// uv.y = v (0→1, 从上边到下边)  中轴 v=0.5
const ribbonFrag = /* glsl */`
uniform vec3  uColorA;   // 活跃端颜色（右侧，音量高处）
uniform vec3  uColorB;   // 尾端颜色（左侧，余韵）
uniform float uOpacity;
uniform float uBrightness;

varying vec2 vUv;

void main() {
  // 距中轴的归一化距离 [0,1]
  float dist = abs(vUv.y - 0.5) * 2.0;

  // 边缘渐消：幂函数，中心最亮，边缘彻底透明
  float edgeFade = pow(1.0 - dist, 2.4);

  // 沿 u 轴冷暖渐变
  vec3 rgb = mix(uColorB, uColorA, vUv.x);

  // 额外辉光：中心处轻微加亮
  rgb += vec3(edgeFade * 0.12);

  float alpha = edgeFade * uOpacity * uBrightness;

  gl_FragColor = vec4(rgb, alpha);
}
`;

interface LayerUniforms {
  [key: string]: THREE.IUniform;
  uColorA:    { value: THREE.Color };
  uColorB:    { value: THREE.Color };
  uOpacity:   { value: number };
  uBrightness:{ value: number };
}

// ── 陕西专用常量 ──────────────────────────────────────────────────────
const SX_HALF_W     = 0.08;   // 细而紧绷的管道半宽
const SX_NUM_SEG    = 6;      // 综框段数
const SX_STEP_H     = 0.55;   // 方波基础幅度（世界单位）
const SX_TRANSITION = 3;      // 段间竖截面过渡点数

// fragment shader：硬边管道，仅最外缘 20% 柔化
const shanxiVert = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;
const shanxiFrag = /* glsl */`
uniform vec3  uColor;
uniform float uOpacity;
varying vec2  vUv;
void main() {
  float dist = abs(vUv.y - 0.5) * 2.0;
  float edge = smoothstep(1.0, 0.6, dist);
  gl_FragColor = vec4(uColor, edge * uOpacity);
}
`;
interface SxUniforms { [k: string]: THREE.IUniform; uColor: { value: THREE.Color }; uOpacity: { value: number }; }

// ── 华南专用：盘金绣绞线 ──────────────────────────────────────────────
const hnVert = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// uv.x = 横向 [0,1]   uv.y = 纵向 [0,1]  中轴 v=0.5
const hnFrag = /* glsl */`
uniform vec3  uColorA;    // vocalCore 赤金
uniform vec3  uColorB;    // vocalGlow 暖黄
uniform float uOpacity;
uniform float uAmp;
uniform float uPulsePos;  // 光子中心 [0,1]
uniform float uHeat;      // 余热 [0,1]

varying vec2 vUv;

void main() {
  // 1. 梭形端部收窄遮罩：两端 u→0/1 时透明，中间最宽
  float spindle  = sin(vUv.x * 3.14159);           // 0→1→0
  float spindlePow = pow(spindle, 0.5);             // 收窄速率（<1 让中段更宽）

  // 2. 纵向边缘渐消（距中轴越远越透明）
  float dist     = abs(vUv.y - 0.5) * 2.0;
  float edgeFade = pow(1.0 - dist, 2.0);

  float alpha    = edgeFade * spindlePow;

  // 3. 螺旋绞线纹理（麻花条纹）
  float spiral   = sin(vUv.x * 20.0 + vUv.y * 6.0) * 0.5 + 0.5;
  float cordMask = 0.55 + spiral * 0.45;

  // 4. 基底金色（左暖黄 → 右赤金）
  vec3 baseColor = mix(uColorB, uColorA, clamp(vUv.x * 0.4 + 0.3, 0.0, 1.0));
  baseColor *= cordMask;

  // 5. 流光脉冲（Gaussian，可 wrap）
  float pulseW  = 0.05 + uAmp * 0.08;
  float pulseDx = abs(vUv.x - uPulsePos);
  pulseDx       = min(pulseDx, 1.0 - pulseDx);
  float pulse   = exp(-pulseDx * pulseDx / (pulseW * pulseW + 0.0001));
  pulse        *= uAmp * 2.0;

  // 6. 余热（暖黄泛光）
  vec3 heatColor = uColorB * uHeat * 2.0;

  vec3 finalRgb = baseColor + vec3(pulse) + heatColor;
  gl_FragColor  = vec4(finalRgb, alpha * uOpacity);
}
`;

interface HnUniforms {
  [k: string]: THREE.IUniform;
  uColorA:   { value: THREE.Color };
  uColorB:   { value: THREE.Color };
  uOpacity:  { value: number };
  uAmp:      { value: number };
  uPulsePos: { value: number };
  uHeat:     { value: number };
}

// ── 东北专用：双螺旋宽面带 ────────────────────────────────────────────
const DB_STRANDS = 2;
const DB_PHASE_OFFSETS = [0, Math.PI];
const DB_COLORS = [
  { inner: new THREE.Color(0.0, 0.64, 0.51), outer: new THREE.Color(1.0, 0.75, 0.0) },
  { inner: new THREE.Color(1.0, 0.06, 0.32), outer: new THREE.Color(0.10, 0.03, 0.03) },
];

const dbStrandVert = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;
const dbStrandFrag = /* glsl */`
uniform vec3  uInner;
uniform vec3  uOuter;
uniform float uOpacity;
uniform float uAmp;
uniform float uFacet;
varying vec2  vUv;
void main() {
  float dist     = abs(vUv.y - 0.5) * 2.0;
  float edgeFade = pow(1.0 - dist, 1.6);

  vec3 rgb = mix(uInner, uOuter, pow(dist, 0.5));

  // 几何切面折痕：沿 u 轴高频正弦，模拟冰凌折痕
  float facet     = sin(vUv.x * 18.0 + uFacet) * 0.5 + 0.5;
  float facetMask = 0.6 + facet * 0.4;
  rgb *= facetMask;

  // 中心辉光 + 音量驱动过曝
  float glow = edgeFade * (1.0 + uAmp * 3.5);
  rgb += uInner * glow * 0.35;

  // 高音时自发光过曝，触发 Bloom 爆炸
  rgb *= (1.0 + uAmp * 2.0);

  gl_FragColor = vec4(rgb, edgeFade * uOpacity);
}
`;
interface DbStrandUniforms { [k: string]: THREE.IUniform; uInner: { value: THREE.Color }; uOuter: { value: THREE.Color }; uOpacity: { value: number }; uAmp: { value: number }; uFacet: { value: number }; }

export function VocalLine() {
  const { viewport } = useThree();

  // ── 内部低频弹簧（过阻尼：慢跟随 + 自然衰落，无回弹）────────────
  const springRef = useRef({ vel: 0, cur: 0 });

  // ── 东北：双螺旋物理状态 ─────────────────────────────────────────
  const dbPhysics = useRef({ spinPhase: 0, twist: 2.0, radius: 0.05 });

  const { dbMeshes, dbUniforms, dbPosArrays } = useMemo(() => {
    const meshes: THREE.Mesh[] = [];
    const unis: DbStrandUniforms[] = [];
    const posArrs: Float32Array[] = [];

    for (let s = 0; s < DB_STRANDS; s++) {
      const geo = new THREE.BufferGeometry();
      const posAttr = new THREE.Float32BufferAttribute(new Float32Array(POINTS * 3 * 3), 3);
      const uvs = new Float32Array(POINTS * 3 * 2);
      for (let i = 0; i < POINTS; i++) {
        const u = i / (POINTS - 1);
        uvs[i * 6]     = u; uvs[i * 6 + 1] = 0;
        uvs[i * 6 + 2] = u; uvs[i * 6 + 3] = 0.5;
        uvs[i * 6 + 4] = u; uvs[i * 6 + 5] = 1;
      }
      const idx = new Uint16Array((POINTS - 1) * 12);
      for (let i = 0; i < POINTS - 1; i++) {
        const b = i * 3;
        idx[i*12]   = b;   idx[i*12+1] = b+1; idx[i*12+2]  = b+3;
        idx[i*12+3] = b+1; idx[i*12+4] = b+4; idx[i*12+5]  = b+3;
        idx[i*12+6] = b+1; idx[i*12+7] = b+2; idx[i*12+8]  = b+4;
        idx[i*12+9] = b+2; idx[i*12+10]= b+5; idx[i*12+11] = b+4;
      }
      geo.setAttribute('position', posAttr);
      geo.setAttribute('uv',       new THREE.Float32BufferAttribute(uvs, 2));
      geo.setIndex(new THREE.BufferAttribute(idx, 1));

      const u: DbStrandUniforms = {
        uInner:   { value: DB_COLORS[s].inner.clone() },
        uOuter:   { value: DB_COLORS[s].outer.clone() },
        uOpacity: { value: 0 },
        uAmp:     { value: 0 },
        uFacet:   { value: 0 },
      };
      const mat = new THREE.ShaderMaterial({
        vertexShader: dbStrandVert, fragmentShader: dbStrandFrag,
        uniforms: u, transparent: true, depthWrite: false,
        blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      });
      meshes.push(new THREE.Mesh(geo, mat));
      unis.push(u);
      posArrs.push(posAttr.array as Float32Array);
    }
    return { dbMeshes: meshes, dbUniforms: unis, dbPosArrays: posArrs };
  }, []);

  // ── 华南：余热 & 光子位置状态 ─────────────────────────────────────
  const heatRef  = useRef(0);
  const pulseRef = useRef(0);

  // ── 华南绞线 Mesh ─────────────────────────────────────────────────
  const { hnMesh, hnUniforms } = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(POINTS * 3 * 3);
    const uvs = new Float32Array(POINTS * 3 * 2);
    for (let i = 0; i < POINTS; i++) {
      const u = i / (POINTS - 1);
      uvs[i * 6]     = u; uvs[i * 6 + 1] = 0;
      uvs[i * 6 + 2] = u; uvs[i * 6 + 3] = 0.5;
      uvs[i * 6 + 4] = u; uvs[i * 6 + 5] = 1;
    }
    const idx = new Uint16Array((POINTS - 1) * 12);
    for (let i = 0; i < POINTS - 1; i++) {
      const b = i * 3;
      idx[i*12]    = b;   idx[i*12+1]  = b+1; idx[i*12+2]  = b+3;
      idx[i*12+3]  = b+1; idx[i*12+4]  = b+4; idx[i*12+5]  = b+3;
      idx[i*12+6]  = b+1; idx[i*12+7]  = b+2; idx[i*12+8]  = b+4;
      idx[i*12+9]  = b+2; idx[i*12+10] = b+5; idx[i*12+11] = b+4;
    }
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('uv',       new THREE.Float32BufferAttribute(uvs, 2));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));

    const u: HnUniforms = {
      uColorA:   { value: new THREE.Color(1, 0.5, 0.1) },
      uColorB:   { value: new THREE.Color(1, 0.8, 0.2) },
      uOpacity:  { value: 0 },
      uAmp:      { value: 0 },
      uPulsePos: { value: 0 },
      uHeat:     { value: 0 },
    };
    const mat = new THREE.ShaderMaterial({
      vertexShader:   hnVert,
      fragmentShader: hnFrag,
      uniforms: u,
      transparent: true,
      depthWrite:  false,
      blending:    THREE.AdditiveBlending,
      side:        THREE.DoubleSide,
    });
    return { hnMesh: new THREE.Mesh(geo, mat), hnUniforms: u };
  }, []);

  // ── 构建 4 层 Mesh ────────────────────────────────────────────────
  const layers = useMemo(() => {
    return LAYERS.map(({ opacity }) => {
      // 3行 × POINTS 列的网格（上边 / 中轴 / 下边）
      const geo = new THREE.BufferGeometry();
      const pos = new Float32Array(POINTS * 3 * 3);
      const uvs = new Float32Array(POINTS * 3 * 2);

      for (let i = 0; i < POINTS; i++) {
        const u = i / (POINTS - 1);
        // 上边 v=0, 中轴 v=0.5, 下边 v=1
        uvs[i * 6]     = u; uvs[i * 6 + 1] = 0;
        uvs[i * 6 + 2] = u; uvs[i * 6 + 3] = 0.5;
        uvs[i * 6 + 4] = u; uvs[i * 6 + 5] = 1;
      }

      const idx = new Uint16Array((POINTS - 1) * 12);
      for (let i = 0; i < POINTS - 1; i++) {
        const b = i * 3;
        idx[i * 12]      = b;     idx[i * 12 + 1]  = b + 1; idx[i * 12 + 2]  = b + 3;
        idx[i * 12 + 3]  = b + 1; idx[i * 12 + 4]  = b + 4; idx[i * 12 + 5]  = b + 3;
        idx[i * 12 + 6]  = b + 1; idx[i * 12 + 7]  = b + 2; idx[i * 12 + 8]  = b + 4;
        idx[i * 12 + 9]  = b + 2; idx[i * 12 + 10] = b + 5; idx[i * 12 + 11] = b + 4;
      }

      geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      geo.setAttribute('uv',       new THREE.Float32BufferAttribute(uvs, 2));
      geo.setIndex(new THREE.BufferAttribute(idx, 1));

      const uniforms: LayerUniforms = {
        uColorA:     { value: new THREE.Color(1, 1, 1) },
        uColorB:     { value: new THREE.Color(0.6, 0.8, 0.9) },
        uOpacity:    { value: opacity },
        uBrightness: { value: 1 },
      };

      const mat = new THREE.ShaderMaterial({
        vertexShader:   ribbonVert,
        fragmentShader: ribbonFrag,
        uniforms,
        transparent:  true,
        depthWrite:   false,
        blending:     THREE.AdditiveBlending,
        side:         THREE.DoubleSide,
      });

      return { mesh: new THREE.Mesh(geo, mat), uniforms };
    });
  }, []);

  // ── coreLine（白芯线，AdditiveBlending，保留）──────────────────────
  const coreGeo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(POINTS * 3), 3));
    return g;
  }, []);
  const coreMat = useMemo(() => new THREE.LineBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.6,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  }), []);
  const coreLine = useMemo(() => new THREE.Line(coreGeo, coreMat), [coreGeo, coreMat]);

  // ── 陕西管道 Mesh ─────────────────────────────────────────────────
  // POINTS 个竖截面，每截面 3 行（上/中/下），另加 SX_NUM_SEG-1 个段间过渡截面
  // 简化：直接用 POINTS 行，段间过渡由 cy 计算控制
  const { sxMesh, sxUniforms } = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(POINTS * 3 * 3);
    const uvs = new Float32Array(POINTS * 3 * 2);
    for (let i = 0; i < POINTS; i++) {
      const u = i / (POINTS - 1);
      uvs[i * 6]     = u; uvs[i * 6 + 1] = 0;
      uvs[i * 6 + 2] = u; uvs[i * 6 + 3] = 0.5;
      uvs[i * 6 + 4] = u; uvs[i * 6 + 5] = 1;
    }
    const idx = new Uint16Array((POINTS - 1) * 12);
    for (let i = 0; i < POINTS - 1; i++) {
      const b = i * 3;
      idx[i*12]    = b;   idx[i*12+1]  = b+1; idx[i*12+2]  = b+3;
      idx[i*12+3]  = b+1; idx[i*12+4]  = b+4; idx[i*12+5]  = b+3;
      idx[i*12+6]  = b+1; idx[i*12+7]  = b+2; idx[i*12+8]  = b+4;
      idx[i*12+9]  = b+2; idx[i*12+10] = b+5; idx[i*12+11] = b+4;
    }
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('uv',       new THREE.Float32BufferAttribute(uvs, 2));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));

    const u: SxUniforms = {
      uColor:   { value: new THREE.Color(1, 0.2, 0.1) },
      uOpacity: { value: 0 },
    };
    const mat = new THREE.ShaderMaterial({
      vertexShader: shanxiVert, fragmentShader: shanxiFrag,
      uniforms: u, transparent: true, depthWrite: false,
      blending: THREE.NormalBlending, side: THREE.DoubleSide,
    });
    return { sxMesh: new THREE.Mesh(geo, mat), sxUniforms: u };
  }, []);

  // 陕西：震颤弹簧 + 前帧方波符号缓存
  const snapRef        = useRef({ vel: 0, pos: 0 });
  const prevSegSignRef = useRef(new Int8Array(SX_NUM_SEG));

  // ── 云贵苗银丝带粒子（带拖尾） ────────────────────────────────────
  const { ygPoints, ygPosAttr, ygColAttr, ygSizeAttr } = useMemo(() => {
    const geo     = new THREE.BufferGeometry();
    const pos     = new Float32Array(YG_TOTAL * 3);
    const col     = new Float32Array(YG_TOTAL * 3);
    const sizes   = new Float32Array(YG_TOTAL);
    const posAttr  = new THREE.Float32BufferAttribute(pos,   3);
    const colAttr  = new THREE.Float32BufferAttribute(col,   3);
    const sizeAttr = new THREE.Float32BufferAttribute(sizes, 1);
    sizeAttr.usage = THREE.DynamicDrawUsage;
    geo.setAttribute('position', posAttr);
    geo.setAttribute('color',    colAttr);
    geo.setAttribute('aSize',    sizeAttr);
    const mat = new THREE.ShaderMaterial({
      vertexShader:   ygVertFull,
      fragmentShader: ygFragFull,
      vertexColors:   true,
      transparent:    true,
      blending:       THREE.AdditiveBlending,
      depthWrite:     false,
    });
    const pts = new THREE.Points(geo, mat);
    pts.frustumCulled = false;
    return { ygPoints: pts, ygPosAttr: posAttr, ygColAttr: colAttr, ygSizeAttr: sizeAttr };
  }, []);

  // 每粒子固定参数 + 拖尾历史位置环形缓冲
  const ygParamRef = useRef(
    Array.from({ length: YG_COUNT }, () => ({
      u:         Math.random(),
      vOffset:   (Math.random() - 0.5) * 2 * YG_HALF_W,
      speed:     0.003 + Math.random() * 0.006,
      trailX:    new Float32Array(YG_TRAIL),
      trailY:    new Float32Array(YG_TRAIL),
      trailHead: 0,
    }))
  );

  // ── 每帧更新 ──────────────────────────────────────────────────────
  useFrame(({ clock }, dt) => {
    const t      = clock.elapsedTime;
    const safeDt = Math.min(dt, 0.05);

    // 过阻尼弹簧（stiffness=2.5, damping=3.5）
    const rawAmp = bridge.amp.vocal;
    const sp = springRef.current;
    const force = (rawAmp - sp.cur) * 2.5 - sp.vel * 3.5;
    sp.vel += force * safeDt;
    sp.cur = Math.max(0, sp.cur + sp.vel * safeDt);
    const amp = sp.cur;

    const introFade  = Math.min(1, bridge.introProgress / 0.4);
    const brightness = (0.5 + amp * 0.5) * introFade;
    const sceneW     = viewport.width;
    const ampFactor  = 0.625 + amp * 1.875;  // 振幅包络，range 0.625→2.5（+25%）

    // idle 呼吸（仅静音时）
    const breatheScale = bridge.isPlaying ? 0 : 0.04;

    // 写入中心 Y 坐标供其他层使用
    bridge.vocalCenterY = fbm(1.1, t) * ampFactor;

    // ── 更新每一层（仅江南显示水袖，其他地区隐藏）──────────────────
    const isJiangnan = bridge.targetRegion === 'jiangnan';
    const ribbonBrightness = isJiangnan ? brightness : 0;

    for (let li = 0; li < LAYERS.length; li++) {
      const { phaseT, ampScale, baseHalfW, yOffset } = LAYERS[li];
      const { mesh, uniforms } = layers[li];
      const posArr = (mesh.geometry.attributes.position as THREE.BufferAttribute).array as Float32Array;

      // 颜色跟随地区调色板
      uniforms.uColorA.value.copy(colors.vocal.main);
      uniforms.uColorB.value.copy(colors.vocal.aux);
      uniforms.uBrightness.value = ribbonBrightness;

      const layerT = t + phaseT;

      for (let i = 0; i < POINTS; i++) {
        const u  = i / (POINTS - 1);
        const x  = (u - 0.5) * sceneW;

        const breathe = bridge.isPlaying ? 0 : Math.sin(layerT * 0.4 + u * 1.5) * breatheScale;
        const cy = fbm(u * 2.2, layerT) * ampFactor * ampScale + yOffset + breathe;

        // 曲率计算（二阶差分近似），用于宽度收窄
        let curvature = 0;
        if (i > 0 && i < POINTS - 1) {
          const uPrev = (i - 1) / (POINTS - 1);
          const uNext = (i + 1) / (POINTS - 1);
          const cyPrev = fbm(uPrev * 2.2, layerT) * ampFactor * ampScale + yOffset;
          const cyNext = fbm(uNext * 2.2, layerT) * ampFactor * ampScale + yOffset;
          const d2y = cyNext - 2 * cy + cyPrev;
          curvature = Math.abs(d2y) * 40; // 放大到可感知范围
        }
        const halfW = baseHalfW / (1 + curvature * 2.5);

        // 法向量（垂直于切线）
        let nx = 0, ny = 1;
        if (i < POINTS - 1) {
          const u2   = (i + 1) / (POINTS - 1);
          const dx   = (u2 - u) * sceneW;
          const layerT2 = layerT; // 同一层时间
          const dy   = (fbm(u2 * 2.2, layerT2) * ampFactor * ampScale + yOffset
                      - fbm(u  * 2.2, layerT2) * ampFactor * ampScale - yOffset);
          const len  = Math.sqrt(dx * dx + dy * dy) || 1;
          nx = -dy / len;
          ny =  dx / len;
        }

        const base = i * 9;
        // 上边
        posArr[base]     = x + nx * halfW;
        posArr[base + 1] = cy + ny * halfW;
        posArr[base + 2] = 0;
        // 中轴
        posArr[base + 3] = x;
        posArr[base + 4] = cy;
        posArr[base + 5] = 0;
        // 下边
        posArr[base + 6] = x - nx * halfW;
        posArr[base + 7] = cy - ny * halfW;
        posArr[base + 8] = 0;
      }

      (mesh.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    }

    // ── 更新白芯线（跟随第 0 层中轴）────────────────────────────────
    const coreArr = (coreGeo.attributes.position as THREE.BufferAttribute).array as Float32Array;
    for (let i = 0; i < POINTS; i++) {
      const u  = i / (POINTS - 1);
      const x  = (u - 0.5) * sceneW;
      const cy = fbm(u * 2.2, t) * ampFactor + LAYERS[0].yOffset;
      coreArr[i * 3]     = x;
      coreArr[i * 3 + 1] = cy;
      coreArr[i * 3 + 2] = 0;
    }
    (coreGeo.attributes.position as THREE.BufferAttribute).needsUpdate = true;

    // 陕西时：coreLine 用铁锈橙；云贵/华南时隐藏；其他地区白色
    const isShanxi     = bridge.targetRegion === 'shanxi';
    const isYunGuiLine = bridge.targetRegion === 'yungui';
    const isHuanan     = bridge.targetRegion === 'huanan';
    const isDongbei    = bridge.targetRegion === 'dongbei';
    coreMat.color.set(isShanxi ? colors.vocal.aux : new THREE.Color(0xffffff));
    coreMat.opacity = (isYunGuiLine || isHuanan || isDongbei) ? 0 : isShanxi
      ? (0.3 + amp * 0.3) * introFade
      : (0.4 + amp * 0.45) * introFade;

    // ── 陕西管道更新 ──────────────────────────────────────────────────
    if (isShanxi) {
      // 震颤弹簧（极硬，stiffness=20, damping=8）
      const snap = snapRef.current;
      const snapForce = (0 - snap.pos) * 20 - snap.vel * 8;
      snap.vel += snapForce * safeDt;
      snap.pos += snap.vel * safeDt;

      const sxPosArr = (sxMesh.geometry.attributes.position as THREE.BufferAttribute).array as Float32Array;
      const segLen = POINTS / SX_NUM_SEG;

      // 计算每段的方波 Y 值，并检测符号翻转触发震颤
      const segY = new Float32Array(SX_NUM_SEG);
      for (let s = 0; s < SX_NUM_SEG; s++) {
        const segPhase = (s / SX_NUM_SEG) * Math.PI * 2;
        const rawSign  = Math.sign(Math.sin(segPhase + t * 0.18));
        segY[s] = rawSign * SX_STEP_H * (0.4 + amp * 1.2);

        // 检测符号翻转 → 触发震颤冲量
        const curSign = rawSign >= 0 ? 1 : -1;
        if (prevSegSignRef.current[s] !== 0 && prevSegSignRef.current[s] !== curSign) {
          snap.vel += 0.35;
        }
        prevSegSignRef.current[s] = curSign as 1 | -1;
      }

      for (let i = 0; i < POINTS; i++) {
        const u   = i / (POINTS - 1);
        const x   = (u - 0.5) * sceneW;
        const seg = Math.min(SX_NUM_SEG - 1, Math.floor(i / segLen));

        // 段间过渡：段尾的 SX_TRANSITION 个点做竖截面插值
        let cy: number;
        const posInSeg = i - seg * segLen;
        if (seg < SX_NUM_SEG - 1 && posInSeg >= segLen - SX_TRANSITION) {
          // 过渡区：从当前段 Y 线性到下一段 Y
          const frac = (posInSeg - (segLen - SX_TRANSITION)) / SX_TRANSITION;
          cy = segY[seg] + (segY[seg + 1] - segY[seg]) * frac;
        } else {
          cy = segY[seg];
        }

        // 棉结疙瘩：极小高频扰动
        const slub = Math.sin(i * 7.3 + t * 0.03) * 0.012 + Math.sin(i * 13.1 + t * 0.02) * 0.008;
        cy += slub + snap.pos;

        const base = i * 9;
        // 上边
        sxPosArr[base]     = x;
        sxPosArr[base + 1] = cy + SX_HALF_W;
        sxPosArr[base + 2] = 0;
        // 中轴
        sxPosArr[base + 3] = x;
        sxPosArr[base + 4] = cy;
        sxPosArr[base + 5] = 0;
        // 下边
        sxPosArr[base + 6] = x;
        sxPosArr[base + 7] = cy - SX_HALF_W;
        sxPosArr[base + 8] = 0;
      }
      (sxMesh.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;

      sxUniforms.uColor.value.copy(colors.vocal.main);
      sxUniforms.uOpacity.value = (0.88 + amp * 0.04) * introFade;
    } else {
      // 非陕西时隐藏管道
      sxUniforms.uOpacity.value = 0;
    }

    // ── 云贵苗银丝带粒子更新 ─────────────────────────────────────────
    const isYunGui = bridge.targetRegion === 'yungui' || bridge.targetRegion === 'default';
    if (isYunGui) {
      const ygPos  = ygPosAttr.array as Float32Array;
      const ygCol  = ygColAttr.array as Float32Array;
      const ygSize = ygSizeAttr.array as Float32Array;
      const ps     = ygParamRef.current;
      const speedMult  = 1.0 + amp * 3.0;
      const bright     = 0.4 + amp * 0.55;
      // 主粒子大小：静音 0.14，高音最大 0.26
      const headSize   = 0.14 + amp * 0.12;

      for (let i = 0; i < YG_COUNT; i++) {
        const p = ps[i];
        p.u = (p.u + p.speed * speedMult * safeDt) % 1.0;

        const x  = (p.u - 0.5) * sceneW;
        const cy = fbm(p.u * 2.2, t) * ampFactor + p.vOffset;

        p.trailX[p.trailHead] = x;
        p.trailY[p.trailHead] = cy;
        p.trailHead = (p.trailHead + 1) % YG_TRAIL;

        const base = i * (1 + YG_TRAIL);

        // 主粒子位置 + 大小
        ygPos[base * 3]     = x;
        ygPos[base * 3 + 1] = cy;
        ygPos[base * 3 + 2] = 0;
        ygSize[base] = headSize;

        // 主粒子颜色：vocal.main（冷银蓝）
        _tmpYg.copy(colors.vocal.main);
        ygCol[base * 3]     = _tmpYg.r * bright;
        ygCol[base * 3 + 1] = _tmpYg.g * bright;
        ygCol[base * 3 + 2] = _tmpYg.b * bright;

        // 拖尾点：从 vocal.main 渐变到 vocal.aux（暖色），同时缩小 + 变暗
        for (let tr = 0; tr < YG_TRAIL; tr++) {
          // tr=0 是最新历史帧（最接近头部），tr=YG_TRAIL-1 是最老
          const bufIdx = (p.trailHead + tr) % YG_TRAIL;
          // fade: 最新帧 → 1（最亮最大），最老帧 → 0
          const fade = (tr + 1) / (YG_TRAIL + 1);
          const tb   = (base + 1 + tr) * 3;

          ygPos[tb]     = p.trailX[bufIdx];
          ygPos[tb + 1] = p.trailY[bufIdx];
          ygPos[tb + 2] = 0;

          // 大小：头部 headSize，尾端收到 20%
          ygSize[base + 1 + tr] = headSize * (0.2 + fade * 0.8);

          // 颜色：fade=1(新) → vocal.main；fade=0(旧) → vocal.aux（暖）
          _tmpYg.copy(colors.vocal.main);
          _tmpYg2.copy(colors.vocal.aux);
          _tmpYg.lerp(_tmpYg2, 1.0 - fade);          // 新→冷，旧→暖
          const trB = bright * fade * 0.75;
          ygCol[tb]     = _tmpYg.r * trB;
          ygCol[tb + 1] = _tmpYg.g * trB;
          ygCol[tb + 2] = _tmpYg.b * trB;
        }
      }

      ygPosAttr.needsUpdate  = true;
      ygColAttr.needsUpdate  = true;
      ygSizeAttr.needsUpdate = true;
    } else {
      // 非云贵：所有点推到屏幕外，重置拖尾
      const ygPos = ygPosAttr.array as Float32Array;
      for (let i = 0; i < YG_TOTAL; i++) {
        ygPos[i * 3 + 2] = -100;
      }
      for (const p of ygParamRef.current) {
        p.trailX.fill(0); p.trailY.fill(0); p.trailHead = 0;
      }
      ygPosAttr.needsUpdate = true;
    }

    // ── 华南盘金绣绞线更新 ───────────────────────────────────────────
    if (isHuanan) {
      // 光子位移：静止时慢速，高音时极速
      pulseRef.current = (pulseRef.current + (0.012 + amp * 0.1) * safeDt) % 1.0;

      // 余热充能（高音触发）与冷却
      heatRef.current += Math.max(0, amp - 0.2) * 3.0 * safeDt;
      heatRef.current -= 1.5 * safeDt;
      heatRef.current  = Math.max(0, Math.min(1, heatRef.current));

      // 更新 uniforms
      hnUniforms.uColorA.value.copy(colors.vocal.main);
      hnUniforms.uColorB.value.copy(colors.vocal.aux);
      hnUniforms.uOpacity.value  = (0.85 + amp * 0.1) * introFade;
      hnUniforms.uAmp.value      = amp;
      hnUniforms.uPulsePos.value = pulseRef.current;
      hnUniforms.uHeat.value     = heatRef.current;

      // 更新管道几何（极小 Y 惰性，体现"重型金属"感）
      const hnPosArr = (hnMesh.geometry.attributes.position as THREE.BufferAttribute).array as Float32Array;
      const HN_MAX_HALF_W = 0.7;
      for (let i = 0; i < POINTS; i++) {
        const u  = i / (POINTS - 1);
        const x  = (u - 0.5) * sceneW;
        const cy = fbm(u * 2.2, t) * ampFactor * 0.45;

        // 梭形宽度：两端收窄，中间最宽
        const spindle  = Math.sin(u * Math.PI);
        const halfW    = HN_MAX_HALF_W * Math.pow(spindle, 0.6);

        // 法向量（切线垂直方向）
        let nx = 0, ny = 1;
        if (i < POINTS - 1) {
          const u2  = (i + 1) / (POINTS - 1);
          const dx  = (u2 - u) * sceneW;
          const dy  = fbm(u2 * 2.2, t) * ampFactor * 0.45 - cy;
          const len = Math.sqrt(dx * dx + dy * dy) || 1;
          nx = -dy / len; ny = dx / len;
        }

        const base = i * 9;
        hnPosArr[base]     = x + nx * halfW;
        hnPosArr[base + 1] = cy + ny * halfW;
        hnPosArr[base + 2] = 0;
        hnPosArr[base + 3] = x;
        hnPosArr[base + 4] = cy;
        hnPosArr[base + 5] = 0;
        hnPosArr[base + 6] = x - nx * halfW;
        hnPosArr[base + 7] = cy - ny * halfW;
        hnPosArr[base + 8] = 0;
      }
      (hnMesh.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    } else {
      hnUniforms.uOpacity.value = 0;
    }

    // ── 东北双螺旋更新 ────────────────────────────────────────────────
    if (isDongbei) {
      const db = dbPhysics.current;

      // 速度降至 10%：自转 0.08 rad/s 基础，高音最多 +0.8
      db.spinPhase += (0.08 + amp * 0.8) * safeDt;
      db.radius = THREE.MathUtils.lerp(db.radius, 0.06 + amp * 0.7, 0.08);
      db.twist  = THREE.MathUtils.lerp(db.twist,  1.5  + amp * 6.0, 0.06);

      const bright   = 0.75 + amp * 0.5;

      for (let s = 0; s < DB_STRANDS; s++) {
        const phaseOff = DB_PHASE_OFFSETS[s];
        const posArr   = dbPosArrays[s];

        // 宽度升级：静音 0.35，高音最宽 0.85
        const maxHalfW = 0.35 + amp * 0.5;

        for (let i = 0; i < POINTS; i++) {
          const u      = i / (POINTS - 1);
          const x      = (u - 0.5) * sceneW;
          const anchor = Math.sin(u * Math.PI);
          const r      = db.radius * anchor;
          const halfW  = maxHalfW * anchor;
          const angle  = db.spinPhase + u * db.twist * Math.PI * 2 + phaseOff;

          // 螺旋中轴
          const cy = Math.sin(angle) * r;
          const cz = Math.cos(angle) * r;

          // 法向量（绕 X 轴，垂直于螺旋切线，简化为径向方向）
          const ny = Math.sin(angle);
          const nz = Math.cos(angle);

          // 上边
          posArr[i * 9]     = x;
          posArr[i * 9 + 1] = cy + ny * halfW;
          posArr[i * 9 + 2] = cz + nz * halfW;
          // 中轴
          posArr[i * 9 + 3] = x;
          posArr[i * 9 + 4] = cy;
          posArr[i * 9 + 5] = cz;
          // 下边
          posArr[i * 9 + 6] = x;
          posArr[i * 9 + 7] = cy - ny * halfW;
          posArr[i * 9 + 8] = cz - nz * halfW;
        }

        (dbMeshes[s].geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
        dbUniforms[s].uOpacity.value = bright;
        dbUniforms[s].uAmp.value     = amp;
        dbUniforms[s].uFacet.value   = db.spinPhase * 2.0 + s * Math.PI;
      }
    } else {
      for (let s = 0; s < DB_STRANDS; s++) dbUniforms[s].uOpacity.value = 0;
      dbPhysics.current.spinPhase = 0;
      dbPhysics.current.radius    = 0.06;
      dbPhysics.current.twist     = 1.5;
    }
  });

  return (
    <group position={[0, 0, 0.2]}>
      {layers.map(({ mesh }, i) => <primitive key={i} object={mesh} />)}
      <primitive object={sxMesh} />
      <primitive object={hnMesh} />
      {dbMeshes.map((m, i) => <primitive key={`db${i}`} object={m} />)}
      <primitive object={ygPoints} />
      <primitive object={coreLine} />
    </group>
  );
}
