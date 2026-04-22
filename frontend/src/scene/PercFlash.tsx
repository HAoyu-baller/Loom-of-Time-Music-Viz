// 岁月织机 — L6 打纬重击（PercFlash v6）
//
// 优化点（v6）：
//   A. 工作区限制：飞梭 targetY 在屏幕中间 70% 高度内移动，progress 0→1 从 -35%vh 爬到 +35%vh
//   B. 2 秒冷却：触发之间至少间隔 2.0s，避免快节奏音乐连续叠出
//   C. 新下滑逻辑：远起点 → 砸 → 停 → 向左侧飞出并微微上弹（织机梭子横向退场）
//   D. 起点拉远：从屏幕顶上方 2 倍 shuttleH 处冲入画面
//   E. 单 slot（MAX_SCANS=1）：冷却后不需要多 slot
//
//   透明底修复：ShaderMaterial，fragment shader 只用贴图自身 alpha × uOpacity
//   UV 裁切：图片两侧各有约 12% 纯透明空白，用 uUvOffset/uUvScale 裁掉
//   模糊修复：贴图设置 LinearFilter，不生成 mipmap

import { useMemo } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { bridge } from './AudioBridge';

const MAX_SCANS     = 1;            // 单 slot：冷却后不需要多 slot
const HOLD_DURATION = 0.10;         // 打纬压住时长
const EXIT_DURATION = 0.45;         // 侧飞退场时长
const MIN_INTERVAL  = 2.0;          // 两次触发最小间隔（秒）

// 工作区：飞梭 targetY 只在屏幕中间 70% 内移动
const WORK_Y_RATIO  = 0.35;         // 工作区上下各占视野 35%

// shuttle.png 实测：图片 2400×288，有效内容区 x:286~2114, y:8~260
const UV_X0 = 286  / 2400;
const UV_X1 = 2114 / 2400;
const UV_Y0 = 8    / 288;
const UV_Y1 = 260  / 288;
const CONTENT_ASPECT = 1928 / 252;  // ≈ 7.65

// 下砸时长：BPM 越快落得越快，范围 0.45 ~ 0.85s
function slamDuration(bpm: number): number {
  const clamped = Math.max(40, Math.min(240, bpm));
  return 0.85 - (clamped - 40) / 200 * 0.40;
}

// ── ShaderMaterial：UV 裁切 + 原色 alpha × uOpacity ─────────────────────
const shuttleVert = /* glsl */`
varying vec2 vUv;
uniform vec2 uUvOffset;
uniform vec2 uUvScale;
void main() {
  vUv = uUvOffset + uv * uUvScale;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const shuttleFrag = /* glsl */`
uniform sampler2D uTex;
uniform float     uOpacity;
uniform float     uHasTex;
varying vec2 vUv;

void main() {
  if (uHasTex < 0.5) {
    gl_FragColor = vec4(0.0);
    return;
  }
  vec4 c = texture2D(uTex, vUv);
  vec3 rgb = pow(c.rgb, vec3(1.5));
  rgb *= vec3(1.08, 0.88, 0.72);
  rgb = clamp(rgb, 0.0, 1.0);
  gl_FragColor = vec4(rgb, c.a * uOpacity);
}
`;

interface ScanState {
  active:  boolean;
  age:     number;
  startY:  number;
  targetY: number;
  slamDur: number;
  exitDirX: number;     // +1 = 向右飞出，-1 = 向左飞出（随机交替更自然）
}

function easeOutCubic(x: number): number { return 1 - Math.pow(1 - x, 3); }
function easeInQuad(x: number):  number { return x * x; }
function lerp(a: number, b: number, t: number): number { return a + (b - a) * t; }

export function PercFlash() {
  const { viewport } = useThree();

  // ── 飞梭 ShaderMaterial ───────────────────────────────────────────────
  const { scanMeshes, scanUniforms } = useMemo(() => {
    const loader = new THREE.TextureLoader();

    const uniforms = Array.from({ length: MAX_SCANS }, () => ({
      uTex:      { value: null as THREE.Texture | null },
      uOpacity:  { value: 0 },
      uHasTex:   { value: 0 },
      uUvOffset: { value: new THREE.Vector2(UV_X0, UV_Y0) },
      uUvScale:  { value: new THREE.Vector2(UV_X1 - UV_X0, UV_Y1 - UV_Y0) },
    }));

    const meshes = uniforms.map((u) => {
      const geo = new THREE.PlaneGeometry(1, 1);
      const mat = new THREE.ShaderMaterial({
        vertexShader:   shuttleVert,
        fragmentShader: shuttleFrag,
        uniforms: u,
        transparent: true,
        depthWrite: false,
        blending: THREE.NormalBlending,
      });
      return new THREE.Mesh(geo, mat);
    });

    loader.load('/shuttle.png', (tex) => {
      tex.wrapS    = THREE.ClampToEdgeWrapping;
      tex.wrapT    = THREE.ClampToEdgeWrapping;
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      for (const u of uniforms) {
        u.uTex.value    = tex;
        u.uHasTex.value = 1;
      }
    });

    return { scanMeshes: meshes, scanUniforms: uniforms };
  }, []);

  const scanPool = useMemo<ScanState[]>(() =>
    Array.from({ length: MAX_SCANS }, () => ({
      active: false, age: 0, startY: 0, targetY: 0, slamDur: 0.6, exitDirX: -1,
    })), []);

  // 冷却状态：上次成功触发的 elapsed 时刻（-∞ 表示首次可触发）
  const gate = useMemo(() => ({
    prevPulse: 0,
    lastFireTime: -Infinity,
  }), []);

  useFrame(({ clock }, dt) => {
    const safeDt  = Math.min(dt, 0.05);
    const elapsed = clock.elapsedTime;
    const vw = viewport.width;
    const vh = viewport.height;
    const shuttleH = vw / CONTENT_ASPECT;

    // ── 触发（带 2s 冷却）──────────────────────────────────────────────
    const triggered = bridge.pulse.perc > 0.8 && gate.prevPulse <= 0.8;
    const offCooldown = elapsed - gate.lastFireTime >= MIN_INTERVAL;
    if (triggered && offCooldown) {
      bridge.filteredPercHitCount++;
      const scan = scanPool.find(s => !s.active);
      if (scan) {
        scan.active  = true;
        scan.age     = 0;
        // 远起点：屏幕顶上方 2 倍 shuttleH，拉出下砸势能
        scan.startY  = vh / 2 + shuttleH * 2.0;
        // 工作区：Y 限制在 [-vh*0.35, +vh*0.35] 间，随 weaveProgress 爬升
        //         再补偿 -shuttleH/2，让飞梭底边对齐已织区顶沿
        const workTop    =  vh * WORK_Y_RATIO;
        const workBottom = -vh * WORK_Y_RATIO;
        scan.targetY = workBottom + (workTop - workBottom) * bridge.weaveProgress
                       - shuttleH * 0.25;
        scan.slamDur = slamDuration(bridge.estimatedBpm);
        // 随机决定退场方向（左右交替更自然）
        scan.exitDirX = Math.random() > 0.5 ? 1 : -1;
        gate.lastFireTime = elapsed;
      }
    }
    gate.prevPulse = bridge.pulse.perc;

    for (let si = 0; si < MAX_SCANS; si++) {
      const scan = scanPool[si];
      const u    = scanUniforms[si];
      const mesh = scanMeshes[si];

      if (!scan.active) {
        u.uOpacity.value = 0;
        continue;
      }

      scan.age += safeDt;
      if (scan.age >= scan.slamDur + HOLD_DURATION + EXIT_DURATION) {
        scan.active      = false;
        u.uOpacity.value = 0;
        continue;
      }

      mesh.scale.set(vw, shuttleH * 0.5, 1);

      const { startY, targetY, slamDur, exitDirX } = scan;
      // 退场终点：纯水平飞出屏幕（无上弹、无淡出）
      const exitX = exitDirX * (vw * 0.5 + vw * 0.15);

      if (scan.age < slamDur) {
        // 阶段 1：下砸（easeOutCubic 加速感）
        const p = easeOutCubic(scan.age / slamDur);
        mesh.position.set(0, lerp(startY, targetY, p), 0.1);
        u.uOpacity.value = lerp(0.0, 1.0, p);
      } else if (scan.age < slamDur + HOLD_DURATION) {
        // 阶段 2：打纬压住
        mesh.position.set(0, targetY, 0.1);
        u.uOpacity.value = 1.0;
      } else {
        // 阶段 3：直来直去侧飞（保持 targetY，加速离场，全程全不透明）
        const exitT = (scan.age - slamDur - HOLD_DURATION) / EXIT_DURATION;
        const exitP = easeInQuad(exitT);
        mesh.position.set(lerp(0, exitX, exitP), targetY, 0.1);
        u.uOpacity.value = 1.0;
      }
    }
  });

  return (
    <group>
      {scanMeshes.map((m, i) => <primitive key={i} object={m} />)}
    </group>
  );
}
