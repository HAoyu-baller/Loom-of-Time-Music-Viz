// 岁月织机 — 主场景组合
// 挂载所有可视化层 + UnrealBloom + Vignette 后处理

import { useRef, useMemo } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing';
import { BlendFunction } from 'postprocessing';
import * as THREE from 'three';
import { tickBridge, bridge, colors } from './AudioBridge';
import type { RegionKey } from '../palettes';
import { VocalLine } from './VocalLine';
import { WarpThreads } from './WarpThreads';
import { Particles } from './Particles';
import { Ripples } from './Ripples';
import { PercFlash } from './PercFlash';

// ── 地区文化底纹路径 ─────────────────────────────────────────────────
const BG_TEXTURE_PATHS: Partial<Record<RegionKey, string>> = {
  huanan:   '/bg_huanan.png',
  shanxi:   '/bg_shanxi.png',
  yungui:   '/bg_yungui.png',
  jiangnan: '/bg_jiangnan.png',
  dongbei:  '/bg_dongbei.png',
};

// ── AmbientPattern Shader ────────────────────────────────────────────
// fragment shader 根据 vUv.y 和 uWeaveProgress 决定各像素透明度：
//   工作区：vUv.y ∈ [uWorkBottom, uWorkTop]（对应屏幕中间 70% 高度）
//   进度条：woven 区 = vUv.y < uWorkBottom + (uWorkTop - uWorkBottom) * progress
//   已织区 12% 透明度 / 未织区 3% / 当前织入行（一条细亮线）60%
const ambientVertShader = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const ambientFragShader = /* glsl */`
uniform sampler2D uTex;
uniform float     uWeaveProgress;
uniform float     uHasTex;
uniform float     uWorkBottom;   // 工作区下沿 UV（≈0.15）
uniform float     uWorkTop;      // 工作区上沿 UV（≈0.85）
uniform float     uRegionBoost;  // 按地区调整整体背景花纹强度（江南浅色需加强）

varying vec2 vUv;

void main() {
  vec4 texColor = uHasTex > 0.5
    ? texture2D(uTex, vUv)
    : vec4(1.0, 1.0, 1.0, 1.0);

  // 当前织入行的世界 UV 位置（与飞梭 targetY 对齐）
  float weavingY = uWorkBottom + (uWorkTop - uWorkBottom) * uWeaveProgress;

  // 基础分层：工作区内已织 12% / 未织 3%；工作区外 1%
  float inWorkArea = step(uWorkBottom, vUv.y) * step(vUv.y, uWorkTop);
  float woven     = step(vUv.y, weavingY) * inWorkArea;
  float baseAlpha = mix(0.01, mix(0.03, 0.12, woven), inWorkArea);

  // 当前织入行高亮：一条 0.006 uv 宽的亮带，最强 0.60
  float lineDist  = abs(vUv.y - weavingY);
  float lineGlow  = smoothstep(0.006, 0.0, lineDist) * inWorkArea;
  float alpha     = max(baseAlpha, lineGlow * 0.60) * uRegionBoost;

  gl_FragColor = vec4(texColor.rgb, alpha * texColor.a);
}
`;

// ── 织物布匹贴图路径（shuttle 落下后逐行 reveal）────────────────────
const FABRIC_TEXTURE_PATHS: Partial<Record<RegionKey, string>> = {
  huanan:   '/huanan_bg.png',
  shanxi:   '/shanbei_bg.png',
  yungui:   '/yungui_bg.png',
  jiangnan: '/jiangnan_bg.png',
  dongbei:  '/dongbei_bg.png',
};

// ── FabricReveal Shader ──────────────────────────────────────────────
// 随 uWeaveProgress 从下往上逐行显现布匹图，已织区最终透明度 0.72
const fabricVertShader = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const fabricFragShader = /* glsl */`
uniform sampler2D uTex;
uniform float     uWeaveProgress;
uniform float     uWeaveGlow;
uniform float     uHasTex;
uniform float     uWorkBottom;
uniform float     uWorkTop;

varying vec2 vUv;

void main() {
  if (uHasTex < 0.5) { gl_FragColor = vec4(0.0); return; }
  vec4 texColor = texture2D(uTex, vUv);

  float weavingY = uWorkBottom + (uWorkTop - uWorkBottom) * uWeaveProgress;

  // 已织区显示，未织区隐藏；当前织入行加一条柔和亮边
  float woven    = step(vUv.y, weavingY);
  float lineDist = abs(vUv.y - weavingY);
  float lineGlow = smoothstep(0.012, 0.0, lineDist) * step(uWorkBottom, vUv.y) * step(vUv.y, uWorkTop);

  // 最后 3 次穿梭后透明度从 0.72 升到 0.95
  float wovenOpacity = mix(0.72, 0.95, uWeaveGlow);
  float alpha = woven * wovenOpacity + lineGlow * 0.25;
  float inWork = step(uWorkBottom, vUv.y) * step(vUv.y, uWorkTop);
  alpha *= inWork;

  gl_FragColor = vec4(texColor.rgb, alpha * texColor.a);
}
`;

// ── FabricReveal 组件 ─────────────────────────────────────────────────
function FabricReveal() {
  const lastRegion = useRef<RegionKey>('default');
  const { viewport } = useThree();

  const textures = useMemo(() => {
    const loader = new THREE.TextureLoader();
    const result: Partial<Record<RegionKey, THREE.Texture>> = {};
    for (const [key, path] of Object.entries(FABRIC_TEXTURE_PATHS)) {
      const tex = loader.load(path);
      tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
      tex.repeat.set(1, 1);
      result[key as RegionKey] = tex;
    }
    return result;
  }, []);

  const { mesh, uniforms } = useMemo(() => {
    const geo = new THREE.PlaneGeometry(1, 1);
    const u = {
      uTex:           { value: null as THREE.Texture | null },
      uWeaveProgress: { value: 0 },
      uWeaveGlow:     { value: 0 },
      uHasTex:        { value: 0 },
      uWorkBottom:    { value: 0.15 },
      uWorkTop:       { value: 0.85 },
    };
    const mat = new THREE.ShaderMaterial({
      vertexShader:   fabricVertShader,
      fragmentShader: fabricFragShader,
      uniforms: u,
      transparent: true,
      depthWrite: false,
    });
    return { mesh: new THREE.Mesh(geo, mat), uniforms: u };
  }, []);

  useFrame(() => {
    mesh.scale.set(viewport.width, viewport.height, 1);
    const region = bridge.targetRegion;
    if (region !== lastRegion.current) {
      lastRegion.current = region;
      const tex = FABRIC_TEXTURE_PATHS[region] ? (textures[region] ?? null) : null;
      uniforms.uTex.value   = tex;
      uniforms.uHasTex.value = tex ? 1.0 : 0.0;
      (mesh.material as THREE.ShaderMaterial).needsUpdate = true;
    }
    uniforms.uWeaveProgress.value = bridge.weaveProgress;
    uniforms.uWeaveGlow.value     = bridge.weaveGlow;
  });

  return <primitive object={mesh} position={[0, 0, -0.3]} />;
}

const REGION_BG_BOOST: Partial<Record<RegionKey, number>> = {
  jiangnan: 2.2,   // 月白+黛蓝的水墨底图本身极淡，明显加强
  yungui:   1.0,
  shanxi:   1.0,
  huanan:   1.0,
  dongbei:  1.0,
  default:  1.0,
};

// ── 背景环境纹理层（逐行堆叠织入）──────────────────────────────────────
function AmbientPattern() {
  const lastRegion = useRef<RegionKey>('default');
  const { viewport } = useThree();

  // 预加载所有背景纹理
  const textures = useMemo(() => {
    const loader = new THREE.TextureLoader();
    const result: Partial<Record<RegionKey, THREE.Texture>> = {};
    for (const [key, path] of Object.entries(BG_TEXTURE_PATHS)) {
      const tex = loader.load(path);
      tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
      // Plane 现在是 viewport 尺寸（约 15×10），repeat(1.5, 1) 让图案每 10×10 世界单位重复一次
      tex.repeat.set(1.5, 1);
      result[key as RegionKey] = tex;
    }
    return result;
  }, []);

  const { mesh, uniforms } = useMemo(() => {
    // Plane 单位尺寸，每帧按 viewport scale，让 vUv.y 直接对应屏幕底到屏幕顶
    const geo = new THREE.PlaneGeometry(1, 1);
    const u = {
      uTex:           { value: null as THREE.Texture | null },
      uWeaveProgress: { value: 0 },
      uHasTex:        { value: 0 },
      uWorkBottom:    { value: 0.15 },   // 屏幕中间 70% 下沿
      uWorkTop:       { value: 0.85 },   // 屏幕中间 70% 上沿
      uRegionBoost:   { value: 1.0 },    // 按地区调整花纹强度
    };
    const mat = new THREE.ShaderMaterial({
      vertexShader:   ambientVertShader,
      fragmentShader: ambientFragShader,
      uniforms: u,
      transparent: true,
      depthWrite: false,
    });
    return { mesh: new THREE.Mesh(geo, mat), uniforms: u };
  }, []);

  useFrame(() => {
    const region = bridge.targetRegion;

    // 每帧让 Plane 尺寸匹配 viewport，使 vUv.y 对应屏幕空间
    mesh.scale.set(viewport.width, viewport.height, 1);

    // 地区切换时替换贴图 + 对应 BG boost
    if (region !== lastRegion.current) {
      lastRegion.current = region;
      const tex = textures[region] ?? null;
      uniforms.uTex.value        = tex;
      uniforms.uHasTex.value     = tex ? 1.0 : 0.0;
      uniforms.uRegionBoost.value = REGION_BG_BOOST[region] ?? 1.0;
      (mesh.material as THREE.ShaderMaterial).needsUpdate = true;
    }

    uniforms.uWeaveProgress.value = bridge.weaveProgress;
  });

  return <primitive object={mesh} position={[0, 0, -0.5]} />;
}

function BackgroundColor() {
  const { scene } = useThree();
  useFrame(() => {
    scene.background = colors.bg;
  });
  return null;
}

function Ticker() {
  useFrame(({ clock }, dt) => {
    tickBridge(dt, clock.elapsedTime);
  });
  return null;
}

export function LoomScene() {
  return (
    <>
      <Ticker />
      <BackgroundColor />

      <AmbientPattern />
      <FabricReveal />
      <WarpThreads />
      <Particles />
      <Ripples />
      <VocalLine />
      <PercFlash />

      {/* ── 后处理管线 ─────────────────────────────────────── */}
      <EffectComposer>
        <Bloom
          blendFunction={BlendFunction.SCREEN}
          intensity={1.6}
          luminanceThreshold={0.1}
          luminanceSmoothing={0.85}
          mipmapBlur
        />
<Vignette
          blendFunction={BlendFunction.NORMAL}
          offset={0.18}
          darkness={0.95}
        />
      </EffectComposer>
    </>
  );
}
