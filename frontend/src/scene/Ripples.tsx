// 岁月织机 — L4 弹拨精灵动画（Pluck Sprite v3）
//
// 地区素材路径（抽帧序列，带下划线文件夹）：
//   jiangnan → /jiang_nan/peony_001~026.png  白底黑底，AdditiveBlending + 调色板染色
//   shanxi   → /shan_xi/peony_001~030.png    黑底橙红，ShaderMaterial 亮度键去背 + 原色保留
//   yungui   → /yun_gui/peony_001~030.png    黑底彩色，ShaderMaterial 亮度键去背 + 原色保留
//   huanan   → /hua_nan/peony_001~025.png    黑底彩色，ShaderMaterial 亮度键去背 + 原色保留
//   dongbei  → /dong_bei/peony_001~024.png   深蓝背景，ShaderMaterial 亮度键去背 + 原色保留
//
// 渲染策略分两类：
//   WHITE_BLEND  — 纯白图案黑底（jiangnan）：MeshBasicMaterial AdditiveBlending，调色板染色
//   LUMA_KEYED   — 彩色图有背景（其余四区）：ShaderMaterial，亮度阈值切除深色背景，保留原色

import { useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { bridge, colors } from './AudioBridge';
import { Textures } from './TextureGen';

const MAX_KNOTS   = 10;
const MAX_SCARS   = 30;
const SPRITE_SIZE = 0.7;   // 比原来减半（原 1.4）
const NUM_THREADS = 20;
const WARP_WIDTH  = 12;
const REGIONS     = ['huanan', 'shanxi', 'yungui', 'jiangnan', 'dongbei'] as const;

// 各地区帧数（与 public 目录实际文件数对应）
const REGION_FRAME_COUNT: Record<string, number> = {
  huanan:   25,
  shanxi:   30,
  yungui:   30,
  jiangnan: 26,
  dongbei:  24,
};
type RegionId = typeof REGIONS[number];

const WARP_X_POSITIONS = Array.from({ length: NUM_THREADS },
  (_, i) => ((i / (NUM_THREADS - 1)) - 0.5) * WARP_WIDTH,
);

// ── 地区渲染配置 ──────────────────────────────────────────────────────────────
type BlendMode = 'WHITE_BLEND' | 'LUMA_KEYED';
interface RegionConfig {
  folder:    string;   // public 目录下的文件夹名（带下划线）
  mode:      BlendMode;
  lumaThreshold: number; // LUMA_KEYED 专用：[0,1] 低于此亮度的像素被视为背景
  lumaEdge:  number;    // 边缘柔化范围
  tint:      boolean;   // 是否叠加调色板颜色（LUMA_KEYED 下通常 false 保留原色）
  sizeScale: number;    // 相对于 SPRITE_SIZE 的缩放系数
}

const REGION_CONFIG: Record<RegionId, RegionConfig> = {
  jiangnan: { folder: 'jiang_nan', mode: 'WHITE_BLEND', lumaThreshold: 0,    lumaEdge: 0,    tint: false, sizeScale: 1.0 },
  shanxi:   { folder: 'shan_xi',   mode: 'LUMA_KEYED',  lumaThreshold: 0.08, lumaEdge: 0.10, tint: false, sizeScale: 1.1 },
  yungui:   { folder: 'yun_gui',   mode: 'LUMA_KEYED',  lumaThreshold: 0.06, lumaEdge: 0.08, tint: false, sizeScale: 1.1 },
  huanan:   { folder: 'hua_nan',   mode: 'LUMA_KEYED',  lumaThreshold: 0.07, lumaEdge: 0.10, tint: false, sizeScale: 1.2 },
  dongbei:  { folder: 'dong_bei',  mode: 'LUMA_KEYED',  lumaThreshold: 0.12, lumaEdge: 0.14, tint: false, sizeScale: 1.1 },
};

// ── LUMA_KEYED ShaderMaterial ─────────────────────────────────────────────────
// 根据像素亮度（luminance）决定透明度：
//   亮度 < lumaThreshold → alpha = 0（背景剔除）
//   亮度 > lumaThreshold + lumaEdge → alpha = 原始 alpha
//   中间 → smoothstep 柔化边缘
const lumaVertShader = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const lumaFragShader = /* glsl */`
uniform sampler2D uTex;
uniform float     uOpacity;
uniform float     uLumaThreshold;
uniform float     uLumaEdge;

varying vec2 vUv;

void main() {
  vec4 c = texture2D(uTex, vUv);
  float luma = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
  float keyAlpha = smoothstep(uLumaThreshold, uLumaThreshold + uLumaEdge, luma);
  gl_FragColor = vec4(c.rgb, c.a * keyAlpha * uOpacity);
}
`;

interface KnotState {
  active:      boolean;
  age:         number;
  maxAge:      number;
  cx:          number;
  cy:          number;
  region:      RegionId;
  scarSpawned: boolean;
}

interface ScarState {
  active: boolean;
  age:    number;
  maxAge: number;
  x:      number;
  y:      number;
}

export function Ripples() {
  // ── 加载纹理（按地区分帧）────────────────────────────────────────
  const regionFrames = useMemo(() => {
    const loader = new THREE.TextureLoader();
    const map: Record<string, THREE.Texture[]> = {};
    for (const region of REGIONS) {
      const cfg = REGION_CONFIG[region];
      const count = REGION_FRAME_COUNT[region] ?? 4;
      map[region] = Array.from({ length: count }, (_, fi) => {
        const num = String(fi + 1).padStart(3, '0');
        const tex = loader.load(`/${cfg.folder}/peony_${num}.png`);
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        return tex;
      });
    }
    map['default'] = map['jiangnan'];
    return map;
  }, []);

  // ── 每个 knot mesh 持有自己的材质（避免共享 needsUpdate 冲突）────
  const sharedGeo = useMemo(() => new THREE.PlaneGeometry(1, 1), []);

  // WHITE_BLEND mesh 池（MeshBasicMaterial，10 个）
  const whiteMeshes = useMemo(() =>
    Array.from({ length: MAX_KNOTS }, () =>
      new THREE.Mesh(sharedGeo, new THREE.MeshBasicMaterial({
        transparent: true, opacity: 0, depthWrite: false,
        blending: THREE.AdditiveBlending,
      }))
    ), [sharedGeo]);

  // LUMA_KEYED mesh 池（ShaderMaterial，10 个，uniforms 独立）
  const { lumaMeshes, lumaUniforms } = useMemo(() => {
    const meshes: THREE.Mesh[] = [];
    const uniforms: Array<{ uTex: { value: THREE.Texture | null }; uOpacity: { value: number }; uLumaThreshold: { value: number }; uLumaEdge: { value: number } }> = [];
    for (let i = 0; i < MAX_KNOTS; i++) {
      const u = {
        uTex:           { value: null as THREE.Texture | null },
        uOpacity:       { value: 0 },
        uLumaThreshold: { value: 0.08 },
        uLumaEdge:      { value: 0.10 },
      };
      uniforms.push(u);
      meshes.push(new THREE.Mesh(sharedGeo, new THREE.ShaderMaterial({
        vertexShader:   lumaVertShader,
        fragmentShader: lumaFragShader,
        uniforms: u,
        transparent: true,
        depthWrite: false,
        blending: THREE.NormalBlending,
      })));
    }
    return { lumaMeshes: meshes, lumaUniforms: uniforms };
  }, [sharedGeo]);

  // ── knot 状态池 ────────────────────────────────────────────────────
  const knotPool = useMemo<KnotState[]>(() =>
    Array.from({ length: MAX_KNOTS }, () => ({
      active: false, age: 0, maxAge: 1.4,
      cx: 0, cy: 0, region: 'jiangnan' as RegionId, scarSpawned: false,
    })), []);

  // 每个 knot 绑定到哪个 mesh 池（按触发时地区决定）
  // 实际上我们用一个统一的 pool，根据 region 动态决定用哪个 mesh
  // 简化：每个 slot 同时存在 white + luma 两个 mesh，显示哪个由 region 决定
  const knotSlots = useMemo(() =>
    Array.from({ length: MAX_KNOTS }, (_, i) => ({
      white: whiteMeshes[i],
      luma:  lumaMeshes[i],
      lumaU: lumaUniforms[i],
    })), [whiteMeshes, lumaMeshes, lumaUniforms]);

  // ── 绣迹（Scar）────────────────────────────────────────────────────
  const scarPool = useMemo<ScarState[]>(() =>
    Array.from({ length: MAX_SCARS }, () => ({
      active: false, age: 0, maxAge: 20, x: 0, y: 0,
    })), []);

  const scarGeo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(MAX_SCARS * 3), 3));
    g.setAttribute('color',    new THREE.Float32BufferAttribute(new Float32Array(MAX_SCARS * 3), 3));
    return g;
  }, []);

  const scarPoints = useMemo(() => new THREE.Points(scarGeo,
    new THREE.PointsMaterial({
      map: Textures.stitchKnot, size: 0.20, vertexColors: true,
      transparent: true, opacity: 1, blending: THREE.AdditiveBlending,
      depthWrite: false, sizeAttenuation: true,
    }),
  ), [scarGeo]);

  const prevPulse = useMemo(() => ({ plucked: 0 }), []);
  let scarWriteIdx = 0;

  useFrame((_, dt) => {
    const safeDt = Math.min(dt, 0.05);
    const rawRegion = bridge.targetRegion as string;
    const region: RegionId = (REGIONS as readonly string[]).includes(rawRegion)
      ? rawRegion as RegionId
      : 'jiangnan';

    // ── 触发新织点（降低阈值 + idle 时随机触发，提高出现频率）─────────
    if (bridge.pulse.plucked > 0.35 && prevPulse.plucked <= 0.35) {
      const spawn = (extraX?: number) => {
        const idx = knotPool.findIndex(k => !k.active);
        if (idx === -1) return;
        const k = knotPool[idx];
        k.active      = true;
        k.age         = 0;
        k.maxAge      = 1.0 + Math.random() * 0.6;  // 总时长 1.0~1.6s
        k.cy          = bridge.vocalCenterY;
        k.cx          = extraX ?? WARP_X_POSITIONS[Math.floor(Math.random() * NUM_THREADS)];
        k.region      = region;
        k.scarSpawned = false;
      };
      spawn();
      // 强音时同时触发第二个（增加密度）
      if (bridge.pulse.plucked > 0.70) spawn();
    }
    // idle 时也保持低频出现
    if (!bridge.isPlaying && Math.random() < 0.025) {
      const idx = knotPool.findIndex(k => !k.active);
      if (idx !== -1) {
        const k = knotPool[idx];
        k.active      = true;
        k.age         = 0;
        k.maxAge      = 1.0 + Math.random() * 0.6;
        k.cy          = (Math.random() - 0.5) * 3;
        k.cx          = WARP_X_POSITIONS[Math.floor(Math.random() * NUM_THREADS)];
        k.region      = region;
        k.scarSpawned = false;
      }
    }
    prevPulse.plucked = bridge.pulse.plucked;

    // ── 更新每个织点 ────────────────────────────────────────────────
    for (let ki = 0; ki < MAX_KNOTS; ki++) {
      const k    = knotPool[ki];
      const slot = knotSlots[ki];
      const cfg  = REGION_CONFIG[k.region] ?? REGION_CONFIG['jiangnan'];

      // 默认隐藏两个 mesh
      if (!k.active) {
        (slot.white.material as THREE.MeshBasicMaterial).opacity = 0;
        slot.white.scale.setScalar(0.01);
        slot.lumaU.uOpacity.value = 0;
        slot.luma.scale.setScalar(0.01);
        continue;
      }

      k.age += safeDt;
      if (k.age >= k.maxAge) {
        k.active = false;
        (slot.white.material as THREE.MeshBasicMaterial).opacity = 0;
        slot.white.scale.setScalar(0.01);
        slot.lumaU.uOpacity.value = 0;
        slot.luma.scale.setScalar(0.01);
        continue;
      }

      const progress = k.age / k.maxAge;
      const FADE_IN_END    = 0.25;
      const FADE_OUT_START = 0.65;

      let opacity = 1;
      let scaleMultiplier = 1;

      if (progress < FADE_IN_END) {
        const t = progress / FADE_IN_END;
        opacity = t;
        scaleMultiplier = 0.5 + t * 0.5;
      } else if (progress > FADE_OUT_START) {
        const t = (progress - FADE_OUT_START) / (1 - FADE_OUT_START);
        opacity = 1 - t;
        scaleMultiplier = 1 - t * 0.2;
      }

      const bloomRange = FADE_OUT_START - FADE_IN_END;
      const bloomT = Math.max(0, Math.min(1, (progress - FADE_IN_END) / bloomRange));
      const frameCount = (regionFrames[k.region] ?? regionFrames['jiangnan']).length;
      const frameIdx = Math.min(frameCount - 1, Math.floor(bloomT * frameCount));
      const frames = regionFrames[k.region] ?? regionFrames['jiangnan'];
      const finalOpacity = opacity * 0.88;
      const finalScale   = SPRITE_SIZE * cfg.sizeScale * scaleMultiplier;

      if (cfg.mode === 'WHITE_BLEND') {
        // 隐藏 luma mesh
        slot.lumaU.uOpacity.value = 0;
        slot.luma.scale.setScalar(0.01);

        // 更新 white mesh
        const wmat = slot.white.material as THREE.MeshBasicMaterial;
        wmat.map     = frames[frameIdx];
        wmat.opacity = finalOpacity;
        if (cfg.tint) {
          wmat.color.copy(colors.plucked0).lerp(colors.plucked1, bloomT);
        } else {
          wmat.color.setScalar(1);
        }
        wmat.needsUpdate = true;
        slot.white.position.set(k.cx, k.cy, 0.05);
        slot.white.scale.setScalar(finalScale);
      } else {
        // 隐藏 white mesh
        const wmat = slot.white.material as THREE.MeshBasicMaterial;
        wmat.opacity = 0;
        slot.white.scale.setScalar(0.01);

        // 更新 luma mesh
        slot.lumaU.uTex.value           = frames[frameIdx];
        slot.lumaU.uOpacity.value       = finalOpacity;
        slot.lumaU.uLumaThreshold.value = cfg.lumaThreshold;
        slot.lumaU.uLumaEdge.value      = cfg.lumaEdge;
        slot.luma.position.set(k.cx, k.cy, 0.05);
        slot.luma.scale.setScalar(finalScale);
      }

      // 在 FADE_OUT_START 时生成绣迹
      if (progress >= FADE_OUT_START && !k.scarSpawned) {
        k.scarSpawned = true;
        const scar = scarPool[scarWriteIdx % MAX_SCARS];
        scarWriteIdx++;
        scar.active = true; scar.age = 0;
        scar.maxAge = 15 + Math.random() * 10;
        scar.x = k.cx; scar.y = k.cy;
      }
    }

    // ── 更新绣迹 ──────────────────────────────────────────────────
    const scarPos = (scarGeo.attributes.position as THREE.BufferAttribute).array as Float32Array;
    const scarCol = (scarGeo.attributes.color    as THREE.BufferAttribute).array as Float32Array;

    for (let si = 0; si < MAX_SCARS; si++) {
      const s = scarPool[si];
      if (!s.active) {
        scarPos[si * 3 + 2] = -100;
        scarCol[si * 3] = scarCol[si * 3 + 1] = scarCol[si * 3 + 2] = 0;
        continue;
      }
      s.age += safeDt;
      if (s.age >= s.maxAge) {
        s.active = false;
        scarPos[si * 3 + 2] = -100;
        scarCol[si * 3] = scarCol[si * 3 + 1] = scarCol[si * 3 + 2] = 0;
        continue;
      }
      const a = 0.022 * (1 - Math.pow(s.age / s.maxAge, 3));
      scarPos[si * 3]     = s.x;
      scarPos[si * 3 + 1] = s.y;
      scarPos[si * 3 + 2] = 0.05;
      scarCol[si * 3]     = colors.plucked0.r * a;
      scarCol[si * 3 + 1] = colors.plucked0.g * a;
      scarCol[si * 3 + 2] = colors.plucked0.b * a;
    }

    (scarGeo.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (scarGeo.attributes.color    as THREE.BufferAttribute).needsUpdate = true;
  });

  return (
    <group>
      {knotSlots.map((slot, i) => (
        <group key={i}>
          <primitive object={slot.white} />
          <primitive object={slot.luma} />
        </group>
      ))}
      <primitive object={scarPoints} />
    </group>
  );
}
