// 岁月织机 — L3 风粒子（地域行为分化）
// 挂载方式学 VocalLine 云贵：useMemo 一次性创建 THREE.Points + primitive。
// 关键：粒子位置基于 viewport.width/height（实际视野尺寸），而非硬编码。

import { useRef, useMemo } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { bridge, colors } from './AudioBridge';

const MAX_PARTICLES = 1000;

const SPRITE_SIZE: Record<string, number> = {
  default:  1.76,
  huanan:   2.74,
  shanxi:   3.28,
  yungui:   3.10,
  jiangnan: 2.56,
  dongbei:  3.92,
};

interface Particle {
  x: number; y: number;
  vx: number; vy: number;
  life: number; maxLife: number;
  seed: number;
  phase: number;
  active: boolean;
}

function createPool(): Particle[] {
  return Array.from({ length: MAX_PARTICLES }, () => ({
    x: 0, y: 0, vx: 0, vy: 0, life: 0, maxLife: 1, seed: 0, phase: 0, active: false,
  }));
}

function spawnParticle(pool: Particle[], region: string, W: number, H: number) {
  const p = pool.find(pt => !pt.active);
  if (!p) return;
  p.active = true;
  p.seed = Math.random() * 1000;
  p.life = 1;
  p.phase = 0;

  switch (region) {
    case 'shanxi':
      p.x = (Math.random() - 0.5) * W * 0.9;
      p.y = H * 0.5 + Math.random() * 0.3;
      p.vx = (Math.random() - 0.5) * 0.1;
      p.vy = -0.9 - Math.random() * 0.5;
      p.maxLife = 4 + Math.random() * 2;
      break;
    case 'yungui':
      p.x = (Math.random() - 0.5) * W * 0.9;
      p.y = (Math.random() - 0.5) * H * 0.9;
      p.vx = 0; p.vy = 0;
      p.maxLife = 1.5 + Math.random() * 2;
      break;
    case 'jiangnan': {
      p.phase = Math.random() * Math.PI * 2;
      // 8 字幅度随视野大小缩放：横向用 viewport 宽度的 35%，纵向 25%
      const ax = W * 0.35;
      const ay = H * 0.25;
      p.vx = (Math.random() - 0.5) * ax * 0.2;   // 静态偏移：烟带厚度
      p.vy = (Math.random() - 0.5) * ay * 0.2;
      p.x = Math.cos(p.phase) * ax + p.vx;
      p.y = Math.sin(p.phase * 2) * ay + p.vy;
      p.maxLife = 8 + Math.random() * 4;
      break;
    }
    case 'dongbei':
      p.x = (Math.random() - 0.5) * W * 0.7;
      p.y = -H * 0.45;
      p.vx = (Math.random() - 0.5) * 0.15;
      p.vy = 1.8 + Math.random() * 0.6;
      p.maxLife = 3 + Math.random() * 1.5;
      break;
    default: { // huanan + default
      p.x = (Math.random() - 0.5) * W * 0.2;   // 底部中央聚集
      p.y = -H * 0.45;
      const angle = (Math.random() - 0.5) * Math.PI * 0.7;
      p.vx = Math.sin(angle) * 0.4;
      p.vy = Math.cos(angle) * 0.5 + 0.2;
      p.maxLife = 4 + Math.random() * 3;
      break;
    }
  }
}

export function Particles() {
  const lastRegionRef = useRef<string>('');
  const pool = useMemo(createPool, []);
  const { viewport } = useThree();

  // 学 VocalLine 云贵：一次性创建 Points 实例
  const { points, material, posAttr, colAttr } = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(MAX_PARTICLES * 3);
    const col = new Float32Array(MAX_PARTICLES * 3);
    const pa = new THREE.Float32BufferAttribute(pos, 3);
    const ca = new THREE.Float32BufferAttribute(col, 3);
    g.setAttribute('position', pa);
    g.setAttribute('color', ca);
    const m = new THREE.PointsMaterial({
      size: SPRITE_SIZE.default,
      vertexColors: true,
      transparent: true,
      opacity: 1,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    });
    const p = new THREE.Points(g, m);
    p.frustumCulled = false;
    p.position.set(0, 0, 0.05);
    return { points: p, material: m, posAttr: pa, colAttr: ca };
  }, []);

  useFrame(({ clock }, dt) => {
    const region = bridge.targetRegion;
    const windAmp = bridge.amp.wind;
    const W = viewport.width;
    const H = viewport.height;

    // 地区切换：更新点尺寸
    if (region !== lastRegionRef.current) {
      lastRegionRef.current = region;
      material.size = SPRITE_SIZE[region] ?? SPRITE_SIZE.default;
      material.needsUpdate = true;
    }

    // 生成粒子：idle 保底 3 个/帧
    const baseSpawn = bridge.isPlaying ? 0 : 3;
    const spawnRate = region === 'yungui'
      ? Math.floor(windAmp * 24) + baseSpawn
      : Math.floor(windAmp * 12) + baseSpawn;
    for (let i = 0; i < spawnRate; i++) spawnParticle(pool, region, W, H);

    const elapsed = clock.elapsedTime;
    const posArr = posAttr.array as Float32Array;
    const colArr = colAttr.array as Float32Array;

    // 亮度：idle 时 0.9，播放时最高 1.5
    const brightness = 0.9 + windAmp * 0.6;

    for (let i = 0; i < MAX_PARTICLES; i++) {
      const p = pool[i];

      if (!p.active) {
        posArr[i * 3 + 2] = -100;
        colArr[i * 3] = colArr[i * 3 + 1] = colArr[i * 3 + 2] = 0;
        continue;
      }

      p.life -= dt / p.maxLife;
      if (p.life <= 0) {
        p.active = false;
        posArr[i * 3 + 2] = -100;
        colArr[i * 3] = colArr[i * 3 + 1] = colArr[i * 3 + 2] = 0;
        continue;
      }

      // ── 按 region 分派运动逻辑 ──────────────────────────────────
      const currentFrame = Math.floor(elapsed * 60);
      let flashBoost = 1;
      let apexBoost  = 1;

      if (region === 'jiangnan') {
        // 江南圆场步 Lissajous：幅度随 viewport 缩放
        const ax = W * 0.35;
        const ay = H * 0.25;
        const theta = p.phase + elapsed * 0.4;
        p.x = Math.cos(theta) * ax + p.vx;
        p.y = Math.sin(theta * 2) * ay + p.vy;
      } else if (region === 'yungui') {
        // 云贵傩戏：非连续瞬移
        const seedI = (p.seed | 0);
        if ((currentFrame + seedI) % 15 === 0) {
          p.x += (Math.random() - 0.5) * 0.5;
          p.y += (Math.random() - 0.5) * 0.5;
        }
      } else if (region === 'shanxi') {
        // 陕西秦腔：砸夯 + 亮相亮闪
        const damping = Math.max(0, 1.0 - (1.0 - 0.82) * dt * 60);
        if (p.phase === 0) {
          if (p.y < 0.5) {
            p.vx *= damping;
            p.vy *= damping;
          }
          if (Math.abs(p.vy) < 0.03 && p.y < 0.5) {
            p.phase = elapsed;
          }
        }
        if (p.phase > 0) {
          flashBoost = 1 + 2 * Math.exp(-(elapsed - p.phase) * 8);
          if (windAmp > 0.05) {
            const toX = 0 - p.x;
            const toY = bridge.vocalCenterY - p.y;
            const dist = Math.sqrt(toX * toX + toY * toY) + 0.1;
            p.vx += (toX / dist) * windAmp * 0.008;
            p.vy += (toY / dist) * windAmp * 0.008;
            p.vx += (-toY / dist) * windAmp * 0.005;
            p.vy += ( toX / dist) * windAmp * 0.005;
            if (dist < 0.15 && windAmp > 0.3) p.life = 0;
          }
        }
        p.x += p.vx * dt * 60;
        p.y += p.vy * dt * 60;
      } else if (region === 'dongbei') {
        // 东北抛手绢：重力抛物线
        const GRAVITY = 0.035;
        p.vy -= GRAVITY * dt * 60;
        p.x += p.vx * dt * 60;
        p.y += p.vy * dt * 60;
        apexBoost = 1 + 0.3 * Math.exp(-p.vy * p.vy * 10);
        if (p.life < 0.3 && windAmp > 0.05) {
          const toX = 0 - p.x;
          const toY = bridge.vocalCenterY - p.y;
          const dist = Math.sqrt(toX * toX + toY * toY) + 0.1;
          p.vx += (toX / dist) * windAmp * 0.008;
          p.vy += (toY / dist) * windAmp * 0.008;
          p.vx += (-toY / dist) * windAmp * 0.005;
          p.vy += ( toX / dist) * windAmp * 0.005;
          if (dist < 0.15 && windAmp > 0.3) p.life = 0;
        }
      } else {
        // 华南 / default：粤剧开扇
        p.vx += Math.sin(p.seed + elapsed * 0.5) * 0.002;
        if (windAmp > 0.05) {
          const toX = 0 - p.x;
          const toY = bridge.vocalCenterY - p.y;
          const dist = Math.sqrt(toX * toX + toY * toY) + 0.1;
          p.vx += (toX / dist) * windAmp * 0.008;
          p.vy += (toY / dist) * windAmp * 0.008;
          p.vx += (-toY / dist) * windAmp * 0.005;
          p.vy += ( toX / dist) * windAmp * 0.005;
          if (dist < 0.15 && windAmp > 0.3) p.life = 0;
        }
        p.x += p.vx * dt * 60;
        p.y += p.vy * dt * 60;
      }

      posArr[i * 3]     = p.x;
      posArr[i * 3 + 1] = p.y;
      posArr[i * 3 + 2] = 0;

      let alpha = p.life;
      if (p.life > 0.85) alpha = (1 - p.life) / 0.15;
      if (region === 'yungui') alpha *= 0.5 + 0.5 * Math.sin(elapsed * 3 + p.seed);

      // 顶点颜色：colors.wind.main（全局 lerp 平滑过渡）* alpha * brightness * flashBoost * apexBoost
      const a = Math.max(0, alpha) * brightness * flashBoost * apexBoost;
      const w = colors.wind.main;
      colArr[i * 3]     = w.r * a;
      colArr[i * 3 + 1] = w.g * a;
      colArr[i * 3 + 2] = w.b * a;
    }

    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
  });

  return <primitive object={points} />;
}
