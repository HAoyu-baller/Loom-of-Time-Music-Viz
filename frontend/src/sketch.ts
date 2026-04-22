// 岁月织机 — p5.js 可视化核心
// 六层渲染架构：背景淡化 / 经线 / 风粒子 / 弹拨涟漪 / 人声主线 / 打击闪光

import p5 from 'p5';
import { PALETTES, lerpPalette, deepCopyPalette } from './palettes';
import type { Palette, RegionKey } from './palettes';

// ── 与 React 通信的桥接对象（mutable ref，p5 每帧直接读取）──────────────
export interface SketchBridge {
  analysers: Partial<Record<string, AnalyserNode>>;
  targetRegion: RegionKey;
  isPlaying: boolean;
}

// ── 内部类型 ──────────────────────────────────────────────────────────────
interface Particle {
  x: number; y: number;
  vx: number; vy: number;
  life: number; maxLife: number;
  size: number;
}

interface Ripple {
  cx: number; cy: number;
  radius: number; maxRadius: number;
  life: number; maxLife: number;
}

interface PercFlash {
  life: number; maxLife: number;
  offsetAngle: number;
  spokes: number;
}

// ── 工具函数 ──────────────────────────────────────────────────────────────
function getRMS(analyser: AnalyserNode): number {
  const data = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(data);
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
  return Math.sqrt(sum / data.length);
}

const NUM_THREADS = 22;

// ── 主 sketch 工厂 ────────────────────────────────────────────────────────
export function createSketch(bridge: SketchBridge) {
  return (p: p5) => {
    // @types/p5 缺少 curveVertex — 使用局部包装
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const cv = (x: number, y: number) => (p as any).curveVertex(x, y);

    // 双缓冲调色板
    let activePalette: Palette = deepCopyPalette(PALETTES.default);

    // 运行时集合
    const particles: Particle[] = [];
    const ripples: Ripple[] = [];
    const flashes: PercFlash[] = [];

    // 每根经线的 noise seed
    const threadSeeds: number[] = [];

    // 峰值检测上一帧值
    let prevPlucked = 0;
    let prevPerc = 0;

    // ── setup ────────────────────────────────────────────────────────────
    p.setup = () => {
      const cnv = p.createCanvas(p.windowWidth, p.windowHeight);
      cnv.style('display', 'block');
      p.colorMode(p.HSB, 360, 100, 100, 100);
      p.noFill();
      for (let i = 0; i < NUM_THREADS; i++) {
        threadSeeds.push(p.random(0, 1000));
      }
    };

    p.windowResized = () => {
      p.resizeCanvas(p.windowWidth, p.windowHeight);
    };

    // ── draw（每帧）──────────────────────────────────────────────────────
    p.draw = () => {
      // 1. 调色板渐变（约 8 秒 @ 60fps）
      const target = PALETTES[bridge.targetRegion] ?? PALETTES.default;
      activePalette = lerpPalette(activePalette, target, 0.003);

      // 2. 读取音频振幅
      const raw = {
        vocal:   bridge.analysers['vocal']   ? getRMS(bridge.analysers['vocal']!)   : 0,
        erhu:    bridge.analysers['erhu']    ? getRMS(bridge.analysers['erhu']!)    : 0,
        wind:    bridge.analysers['wind']    ? getRMS(bridge.analysers['wind']!)    : 0,
        plucked: bridge.analysers['plucked'] ? getRMS(bridge.analysers['plucked']!) : 0,
        perc:    bridge.analysers['perc']    ? getRMS(bridge.analysers['perc']!)    : 0,
      };

      // 3. 缩放 + 空闲时模拟环境振幅
      const t = p.frameCount;
      const idle = !bridge.isPlaying;
      const amps = {
        vocal:   idle ? 0.12 + Math.sin(t * 0.018) * 0.05 : p.constrain(raw.vocal   * 9,  0, 1),
        erhu:    idle ? 0.06 + Math.sin(t * 0.011) * 0.03 : p.constrain(raw.erhu    * 11, 0, 1),
        wind:    idle ? 0.04                               : p.constrain(raw.wind    * 11, 0, 1),
        plucked: idle ? 0                                  : p.constrain(raw.plucked * 13, 0, 1),
        perc:    idle ? 0                                  : p.constrain(raw.perc    * 16, 0, 1),
      };

      // 4. 峰值检测 → 触发涟漪 / 闪光
      const PLUCKED_THR = 0.18;
      const PERC_THR    = 0.22;
      if (amps.plucked > PLUCKED_THR && prevPlucked <= PLUCKED_THR) spawnRipple();
      if (amps.perc    > PERC_THR    && prevPerc    <= PERC_THR)    spawnFlash();
      prevPlucked = amps.plucked;
      prevPerc    = amps.perc;

      // 5. 风粒子生成
      const spawnN = Math.floor(amps.wind * 5);
      for (let i = 0; i < spawnN; i++) spawnParticle();
      // 空闲时偶发几粒子保持画面有生气
      if (idle && t % 12 === 0) spawnParticle();

      // ── L1: 背景淡化（运动拖尾）──────────────────────────────────────
      p.blendMode(p.BLEND);
      const bg = activePalette.bg;
      p.background(bg.h, bg.s, bg.b, 16);

      // ── L2: 经线（拉弦 / 竖向线）────────────────────────────────────
      drawWarpThreads(amps.erhu);

      // ── L3: 风粒子（加法混合发光）───────────────────────────────────
      p.blendMode(p.ADD);
      updateAndDrawParticles();
      p.blendMode(p.BLEND);

      // ── L4: 弹拨涟漪（加法混合）─────────────────────────────────────
      p.blendMode(p.ADD);
      updateAndDrawRipples();
      p.blendMode(p.BLEND);

      // ── L5: 人声主线 ─────────────────────────────────────────────────
      drawVocalLine(amps.vocal);

      // ── L6: 打击闪光（加法混合）─────────────────────────────────────
      p.blendMode(p.ADD);
      updateAndDrawFlashes();
      p.blendMode(p.BLEND);
    };

    // ════════════════════════════════════════════════════════════════════
    //  Layer 2 — 经线（竖向丝绡，随二胡振幅颤动）
    // ════════════════════════════════════════════════════════════════════
    function drawWarpThreads(amp: number) {
      const c = activePalette.bowed;
      const baseAlpha = 28 + amp * 44;
      p.strokeWeight(0.9);
      p.noFill();

      for (let i = 0; i < NUM_THREADS; i++) {
        const x0 = p.map(i, 0, NUM_THREADS - 1, p.width * 0.04, p.width * 0.96);
        const seed = threadSeeds[i];
        p.stroke(c.h, c.s, c.b, baseAlpha);
        p.beginShape();
        const STEPS = 36;
        for (let j = 0; j <= STEPS; j++) {
          const y  = p.map(j, 0, STEPS, 0, p.height);
          const nt = p.frameCount * 0.004 + j * 0.07;
          const dx = (p.noise(seed, nt) - 0.5) * 2 * (8 + amp * 28);
          cv(x0 + dx, y);
        }
        p.endShape();
      }
    }

    // ════════════════════════════════════════════════════════════════════
    //  Layer 5 — 人声主线（横贯画布的蜿蜒曲线，双通道：光晕 + 核心）
    // ════════════════════════════════════════════════════════════════════
    function drawVocalLine(amp: number) {
      const STEPS = 64;
      const yBase = p.height * 0.5;
      const ft = p.frameCount * 0.0028;

      const buildPoints = () => {
        const pts: [number, number][] = [];
        for (let i = 0; i <= STEPS; i++) {
          const x = p.map(i, 0, STEPS, -p.width * 0.02, p.width * 1.02);
          const n = p.noise(i * 0.07 + ft, ft * 0.6);
          const y = yBase + (n - 0.5) * p.height * (0.10 + amp * 0.38);
          pts.push([x, y]);
        }
        return pts;
      };

      const pts = buildPoints();

      // 光晕通道（宽 + 低透明度）
      const gc = activePalette.vocal.glow;
      p.blendMode(p.ADD);
      p.strokeWeight(14 + amp * 22);
      p.stroke(gc.h, gc.s, gc.b, 10 + amp * 16);
      p.beginShape();
      for (const [x, y] of pts) cv(x, y);
      p.endShape();
      p.blendMode(p.BLEND);

      // 核心通道（细 + 高饱和）
      const cc = activePalette.vocal.core;
      p.strokeWeight(1.8 + amp * 3.5);
      p.stroke(cc.h, cc.s, cc.b, 60 + amp * 35);
      p.beginShape();
      for (const [x, y] of pts) cv(x, y);
      p.endShape();
    }

    // ════════════════════════════════════════════════════════════════════
    //  Layer 3 — 风粒子
    // ════════════════════════════════════════════════════════════════════
    function spawnParticle() {
      if (particles.length > 380) return;
      particles.push({
        x:       p.random(0, p.width),
        y:       p.height + p.random(10, 30),
        vx:      p.random(-0.6, 0.6),
        vy:      -p.random(0.9, 2.8),
        life:    p.random(100, 240),
        maxLife: 0,
        size:    p.random(2, 7),
      });
      const last = particles[particles.length - 1];
      last.maxLife = last.life;
    }

    function updateAndDrawParticles() {
      const c = activePalette.wind.main;
      p.noStroke();
      for (let i = particles.length - 1; i >= 0; i--) {
        const pt = particles[i];
        pt.life--;
        if (pt.life <= 0 || pt.y < -30) { particles.splice(i, 1); continue; }

        const nf = (p.noise(pt.x * 0.004, pt.y * 0.004, p.frameCount * 0.006) - 0.5) * 0.35;
        pt.vx = pt.vx * 0.97 + nf;
        pt.x += pt.vx;
        pt.y += pt.vy;

        const lr = pt.life / pt.maxLife;
        const alpha = lr * (25 + lr * 45);
        p.fill(c.h, c.s, c.b, alpha);
        p.circle(pt.x, pt.y, pt.size * (0.4 + lr * 0.6) + 0.5);
      }
    }

    // ════════════════════════════════════════════════════════════════════
    //  Layer 4 — 弹拨涟漪（形态随地域变化）
    // ════════════════════════════════════════════════════════════════════
    function spawnRipple() {
      const cx = p.random(p.width * 0.12, p.width * 0.88);
      const cy = p.random(p.height * 0.18, p.height * 0.82);
      const ml = p.random(65, 200);
      ripples.push({ cx, cy, radius: 0, maxRadius: ml, life: 75, maxLife: 75 });
    }

    function updateAndDrawRipples() {
      for (let i = ripples.length - 1; i >= 0; i--) {
        const r = ripples[i];
        r.life--;
        if (r.life <= 0) { ripples.splice(i, 1); continue; }
        const lr = r.life / r.maxLife;
        r.radius = (1 - lr) * r.maxRadius;
        drawRippleShape(r.cx, r.cy, r.radius, lr);
      }
    }

    function drawRippleShape(cx: number, cy: number, radius: number, alpha01: number) {
      const p1 = activePalette.plucked[0];
      const p2 = activePalette.plucked[1];
      const a  = alpha01 * 62;
      p.noFill();

      switch (bridge.targetRegion) {

        // 华南：铜钱纹 — 同心双圆 + 内嵌六边形
        case 'huanan': {
          p.stroke(p1.h, p1.s, p1.b, a);
          p.strokeWeight(1.6);
          p.circle(cx, cy, radius * 2);
          p.circle(cx, cy, radius * 0.52 * 2);
          p.stroke(p2.h, p2.s, p2.b, a * 0.85);
          p.strokeWeight(1);
          p.beginShape();
          for (let k = 0; k <= 6; k++) {
            const ang = k * p.TWO_PI / 6 - p.PI / 6;
            p.vertex(cx + Math.cos(ang) * radius * 0.52, cy + Math.sin(ang) * radius * 0.52);
          }
          p.endShape(p.CLOSE);
          break;
        }

        // 陕西：剪纸展开 — 8 辐射线 + 菱形
        case 'shanxi': {
          p.stroke(p1.h, p1.s, p1.b, a);
          p.strokeWeight(1.3);
          for (let k = 0; k < 8; k++) {
            const ang = k * p.TWO_PI / 8;
            p.line(cx, cy, cx + Math.cos(ang) * radius, cy + Math.sin(ang) * radius);
          }
          p.stroke(p2.h, p2.s, p2.b, a);
          p.beginShape();
          p.vertex(cx,               cy - radius * 0.52);
          p.vertex(cx + radius * 0.32, cy);
          p.vertex(cx,               cy + radius * 0.52);
          p.vertex(cx - radius * 0.32, cy);
          p.endShape(p.CLOSE);
          break;
        }

        // 云贵：银铃震动 — 圆圈 + 对数螺旋
        case 'yungui': {
          p.stroke(p1.h, p1.s, p1.b, a);
          p.strokeWeight(1.6);
          p.circle(cx, cy, radius * 2);
          p.stroke(p2.h, p2.s, p2.b, a * 0.72);
          p.strokeWeight(0.8);
          p.beginShape();
          const totalAng = p.TWO_PI * 1.8;
          for (let ang = 0.01; ang < totalAng; ang += 0.08) {
            const r = (radius * 0.42) * (ang / totalAng);
            cv(cx + Math.cos(ang) * r, cy + Math.sin(ang) * r);
          }
          p.endShape();
          break;
        }

        // 江南：水墨扩散 — 噪声偏移的不规则单环
        case 'jiangnan': {
          p.stroke(p1.h, p1.s, p1.b, a * 0.55);
          p.strokeWeight(0.9);
          p.beginShape();
          const STEPS = 44;
          for (let k = 0; k <= STEPS + 3; k++) {
            const ang = (k / STEPS) * p.TWO_PI;
            const nr  = radius + (p.noise(
              cx * 0.008 + Math.cos(ang) * 6,
              cy * 0.008 + Math.sin(ang) * 6,
              p.frameCount * 0.018,
            ) - 0.5) * radius * 0.28;
            cv(cx + Math.cos(ang) * nr, cy + Math.sin(ang) * nr);
          }
          p.endShape(p.CLOSE);
          break;
        }

        // 东北：六角雪花 — 6 主射线 + 侧枝
        case 'dongbei': {
          p.stroke(p1.h, p1.s, p1.b, a);
          p.strokeWeight(1.3);
          for (let k = 0; k < 6; k++) {
            const ang = k * p.TWO_PI / 6;
            p.line(cx, cy, cx + Math.cos(ang) * radius, cy + Math.sin(ang) * radius);
            const mx  = cx + Math.cos(ang) * radius * 0.5;
            const my  = cy + Math.sin(ang) * radius * 0.5;
            const bl  = radius * 0.32;
            p.line(mx, my, mx + Math.cos(ang + p.HALF_PI / 2) * bl, my + Math.sin(ang + p.HALF_PI / 2) * bl);
            p.line(mx, my, mx + Math.cos(ang - p.HALF_PI / 2) * bl, my + Math.sin(ang - p.HALF_PI / 2) * bl);
          }
          break;
        }

        // 默认 / default
        default: {
          p.stroke(p1.h, p1.s, p1.b, a);
          p.strokeWeight(1.6);
          p.circle(cx, cy, radius * 2);
          break;
        }
      }
    }

    // ════════════════════════════════════════════════════════════════════
    //  Layer 6 — 打击闪光（从画布中心向外辐射，快速衰减）
    // ════════════════════════════════════════════════════════════════════
    function spawnFlash() {
      flashes.push({
        life:        18,
        maxLife:     18,
        offsetAngle: p.random(p.TWO_PI),
        spokes:      Math.floor(p.random(6, 14)),
      });
    }

    function updateAndDrawFlashes() {
      const c   = activePalette.perc;
      const cx  = p.width  * 0.5;
      const cy  = p.height * 0.5;
      const maxLen = Math.min(p.width, p.height) * 0.5;

      for (let i = flashes.length - 1; i >= 0; i--) {
        const f = flashes[i];
        f.life--;
        if (f.life <= 0) { flashes.splice(i, 1); continue; }
        const lr    = f.life / f.maxLife;
        const alpha = lr * 72;
        const len   = maxLen * lr;
        p.stroke(c.h, c.s, c.b, alpha);
        p.strokeWeight(lr * 2.2);
        for (let k = 0; k < f.spokes; k++) {
          const ang = f.offsetAngle + k * p.TWO_PI / f.spokes;
          p.line(cx, cy, cx + Math.cos(ang) * len, cy + Math.sin(ang) * len);
        }
      }
    }

  }; // end sketch function
}
