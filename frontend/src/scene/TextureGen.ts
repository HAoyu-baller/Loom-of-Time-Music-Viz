// 岁月织机 — 运行时纹理生成器
// 用 Canvas 2D API 程序化生成所有粒子精灵，无需外部图片文件
//
// 每种精灵对应一个地区的视觉气质：
//   glow_soft    → 通用软辉光点（fallback + PercFlash）
//   glow_ring    → 辉光环（PercFlash 扩散环）
//   spark        → 华南：金箔光斑（不规则椭圆 + 暖散射）
//   grain        → 陕西：黄土尘埃（不规则硬边颗粒）
//   star         → 云贵：银光星爆（4 射线 + 核心点）
//   ink          → 江南：水墨渐散（软边墨晕 + 飞白）
//   snow         → 东北：六瓣雪花（精确几何晶体）

import * as THREE from 'three';

function makeCanvas(size = 128): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  return [c, c.getContext('2d')!];
}

// ── 通用软辉光点 ──────────────────────────────────────────────────────
function genGlowSoft(size = 128): THREE.CanvasTexture {
  const [c, ctx] = makeCanvas(size);
  const r = size / 2;
  const g = ctx.createRadialGradient(r, r, 0, r, r, r);
  g.addColorStop(0.00, 'rgba(255,255,255,1.0)');
  g.addColorStop(0.25, 'rgba(255,255,255,0.85)');
  g.addColorStop(0.55, 'rgba(255,255,255,0.3)');
  g.addColorStop(1.00, 'rgba(255,255,255,0.0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(c);
}

// ── 辉光环（PercFlash 扩散环叠加用）──────────────────────────────────
function genGlowRing(size = 128): THREE.CanvasTexture {
  const [c, ctx] = makeCanvas(size);
  const r = size / 2;
  const g = ctx.createRadialGradient(r, r, r * 0.58, r, r, r * 0.98);
  g.addColorStop(0.0, 'rgba(255,255,255,0.0)');
  g.addColorStop(0.4, 'rgba(255,255,255,0.85)');
  g.addColorStop(0.7, 'rgba(255,255,255,0.5)');
  g.addColorStop(1.0, 'rgba(255,255,255,0.0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(c);
}

// ── 华南：金箔光斑 ────────────────────────────────────────────────────
// 不规则椭圆，暖金边缘散射，模拟金箔在光下的漫反射
function genSpark(size = 128): THREE.CanvasTexture {
  const [c, ctx] = makeCanvas(size);
  const cx = size / 2, cy = size / 2;

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(1.15, 0.78);  // 扁椭圆
  ctx.rotate(0.6);

  const g = ctx.createRadialGradient(0, 0, 0, 0, 0, cx);
  g.addColorStop(0.00, 'rgba(255,255,255,1.0)');
  g.addColorStop(0.30, 'rgba(255,240,180,0.8)');
  g.addColorStop(0.65, 'rgba(255,200,80,0.25)');
  g.addColorStop(1.00, 'rgba(255,160,40,0.0)');
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(0, 0, cx, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // 高光核心
  const gc = ctx.createRadialGradient(cx - 4, cy - 4, 0, cx, cy, cx * 0.18);
  gc.addColorStop(0, 'rgba(255,255,255,0.9)');
  gc.addColorStop(1, 'rgba(255,255,255,0.0)');
  ctx.fillStyle = gc;
  ctx.fillRect(0, 0, size, size);

  return new THREE.CanvasTexture(c);
}

// ── 陕西：黄土尘埃颗粒 ───────────────────────────────────────────────
// 不规则七边形 + 硬边，颗粒感，不发光
function genGrain(size = 128): THREE.CanvasTexture {
  const [c, ctx] = makeCanvas(size);
  const cx = size / 2, cy = size / 2;

  // 不规则多边形轮廓（固定种子，保证每次相同）
  const pts = 8;
  const radii = [0.42, 0.36, 0.44, 0.30, 0.40, 0.34, 0.46, 0.32];
  ctx.beginPath();
  for (let i = 0; i < pts; i++) {
    const a = (i / pts) * Math.PI * 2 - 0.3;
    const r = cx * radii[i];
    const x = cx + Math.cos(a) * r;
    const y = cy + Math.sin(a) * r;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.closePath();

  const g = ctx.createRadialGradient(cx - 3, cy - 3, 0, cx, cy, cx * 0.45);
  g.addColorStop(0.0, 'rgba(255,255,255,1.0)');
  g.addColorStop(0.6, 'rgba(255,255,255,0.7)');
  g.addColorStop(1.0, 'rgba(255,255,255,0.0)');
  ctx.fillStyle = g;
  ctx.fill();

  return new THREE.CanvasTexture(c);
}

// ── 云贵：4 射线银光星爆 ──────────────────────────────────────────────
// 细长十字形射线 + 45° 斜线（共 4 轴 8 射线）+ 核心辉光
function genStar(size = 128): THREE.CanvasTexture {
  const [c, ctx] = makeCanvas(size);
  const cx = size / 2, cy = size / 2;

  ctx.lineCap = 'round';

  // 4 条主射线（水平 + 垂直）
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI;
    const len = cx * 0.92;
    const ex1 = cx + Math.cos(a) * len;
    const ey1 = cy + Math.sin(a) * len;
    const ex2 = cx - Math.cos(a) * len;
    const ey2 = cy - Math.sin(a) * len;

    const g = ctx.createLinearGradient(ex2, ey2, ex1, ey1);
    g.addColorStop(0.00, 'rgba(255,255,255,0.0)');
    g.addColorStop(0.40, 'rgba(255,255,255,0.75)');
    g.addColorStop(0.50, 'rgba(255,255,255,1.0)');
    g.addColorStop(0.60, 'rgba(255,255,255,0.75)');
    g.addColorStop(1.00, 'rgba(255,255,255,0.0)');

    ctx.save();
    ctx.lineWidth = i % 2 === 0 ? 2.8 : 1.8; // 主轴粗，次轴细
    ctx.strokeStyle = g;
    ctx.beginPath();
    ctx.moveTo(ex2, ey2);
    ctx.lineTo(ex1, ey1);
    ctx.stroke();
    ctx.restore();
  }

  // 核心辉光圆
  const gc = ctx.createRadialGradient(cx, cy, 0, cx, cy, cx * 0.22);
  gc.addColorStop(0.0, 'rgba(255,255,255,1.0)');
  gc.addColorStop(0.5, 'rgba(255,255,255,0.6)');
  gc.addColorStop(1.0, 'rgba(255,255,255,0.0)');
  ctx.fillStyle = gc;
  ctx.beginPath();
  ctx.arc(cx, cy, cx * 0.22, 0, Math.PI * 2);
  ctx.fill();

  return new THREE.CanvasTexture(c);
}

// ── 江南：水墨渐散 ───────────────────────────────────────────────────
// 中心软墨晕 + 6 个外围飞白晕，边缘参差不齐，宣纸渗透感
function genInk(size = 128): THREE.CanvasTexture {
  const [c, ctx] = makeCanvas(size);
  const cx = size / 2, cy = size / 2;

  // 主墨晕
  const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, cx * 0.95);
  g.addColorStop(0.00, 'rgba(255,255,255,0.88)');
  g.addColorStop(0.35, 'rgba(255,255,255,0.5)');
  g.addColorStop(0.70, 'rgba(255,255,255,0.15)');
  g.addColorStop(1.00, 'rgba(255,255,255,0.0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);

  // 外围 6 个不均匀飞白（固定偏移，保证确定性）
  const offsets = [
    [0.28, 0.0], [0.22, 1.05], [0.30, 2.1],
    [0.25, 3.14], [0.20, 4.2], [0.27, 5.25],
  ];
  for (const [dr, angle] of offsets) {
    const bx = cx + Math.cos(angle) * cx * dr;
    const by = cy + Math.sin(angle) * cy * dr;
    const gb = ctx.createRadialGradient(bx, by, 0, bx, by, cx * 0.32);
    gb.addColorStop(0.0, 'rgba(255,255,255,0.35)');
    gb.addColorStop(1.0, 'rgba(255,255,255,0.0)');
    ctx.fillStyle = gb;
    ctx.fillRect(0, 0, size, size);
  }

  return new THREE.CanvasTexture(c);
}

// ── 东北：六瓣雪花晶体 ───────────────────────────────────────────────
// 6 主轴 + 每轴 2 对分叉 + 尖端小分叉，几何精确
function genSnow(size = 128): THREE.CanvasTexture {
  const [c, ctx] = makeCanvas(size);
  const cx = size / 2, cy = size / 2;

  ctx.strokeStyle = 'rgba(255,255,255,0.95)';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    const arm = cx * 0.86;

    // 主轴
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(a) * arm, cy + Math.sin(a) * arm);
    ctx.stroke();

    // 两层分叉（0.38 和 0.65 处）
    for (const frac of [0.38, 0.65]) {
      const bx = cx + Math.cos(a) * arm * frac;
      const by = cy + Math.sin(a) * arm * frac;
      const bl = arm * (frac < 0.5 ? 0.26 : 0.20);
      ctx.lineWidth = 1.4;
      for (const sign of [-1, 1]) {
        const ba = a + sign * (Math.PI / 3);
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx + Math.cos(ba) * bl, by + Math.sin(ba) * bl);
        ctx.stroke();
      }
    }

    // 尖端小分叉
    const tx = cx + Math.cos(a) * arm * 0.88;
    const ty = cy + Math.sin(a) * arm * 0.88;
    ctx.lineWidth = 0.9;
    for (const sign of [-1, 1]) {
      const ba = a + sign * (Math.PI / 4);
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx + Math.cos(ba) * arm * 0.10, ty + Math.sin(ba) * arm * 0.10);
      ctx.stroke();
    }
  }

  // 中心六边形核心
  ctx.fillStyle = 'rgba(255,255,255,1.0)';
  ctx.beginPath();
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    const x = cx + Math.cos(a) * 3.5;
    const y = cy + Math.sin(a) * 3.5;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();

  return new THREE.CanvasTexture(c);
}

// ── 打纬光刃：水平高斯软边条（BattenStrike 扫描线用）──────────────────
// w=512 宽，h=32 高；中心行纯白，向上下高斯衰减到透明，横向均匀
function genLightStreak(w = 512, h = 32): THREE.CanvasTexture {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d')!;

  const cy = h / 2;
  const sigma = h * 0.38; // 高斯标准差（约占高度 38%）

  const imgData = ctx.createImageData(w, h);
  const d = imgData.data;

  for (let y = 0; y < h; y++) {
    const dy = y - cy;
    const gauss = Math.exp(-(dy * dy) / (2 * sigma * sigma));
    const alpha = Math.round(gauss * 255);
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      d[idx]     = 255;   // R
      d[idx + 1] = 255;   // G
      d[idx + 2] = 255;   // B
      d[idx + 3] = alpha; // A
    }
  }
  ctx.putImageData(imgData, 0, 0);
  return new THREE.CanvasTexture(c);
}

// ── 刺绣结印记：细十字 + 中心亮点（JacquardKnots 消散后留痕用）──────────
// 4 条极细白线延伸到边缘 70%，中心 4×4 亮点，整体约 60% 最大亮度
function genStitchKnot(size = 64): THREE.CanvasTexture {
  const [c, ctx] = makeCanvas(size);
  const cx = size / 2, cy = size / 2;
  const reach = size * 0.35; // 延伸距离（边长 70% 的一半）

  ctx.strokeStyle = 'rgba(255,255,255,0.62)';
  ctx.lineCap = 'round';

  // 水平线
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(cx - reach, cy);
  ctx.lineTo(cx + reach, cy);
  ctx.stroke();

  // 垂直线
  ctx.beginPath();
  ctx.moveTo(cx, cy - reach);
  ctx.lineTo(cx, cy + reach);
  ctx.stroke();

  // 中心亮点（4×4 区域辉光）
  const gc = ctx.createRadialGradient(cx, cy, 0, cx, cy, 4);
  gc.addColorStop(0.0, 'rgba(255,255,255,0.60)');
  gc.addColorStop(1.0, 'rgba(255,255,255,0.0)');
  ctx.fillStyle = gc;
  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, Math.PI * 2);
  ctx.fill();

  return new THREE.CanvasTexture(c);
}

// ── 单例导出（模块加载时生成一次，后续复用）────────────────────────────
export const Textures = {
  glowSoft:    genGlowSoft(),
  glowRing:    genGlowRing(),
  spark:       genSpark(),
  grain:       genGrain(),
  star:        genStar(),
  ink:         genInk(),
  snow:        genSnow(),
  lightStreak: genLightStreak(),
  stitchKnot:  genStitchKnot(),
} as const;

// 按地区 key 取对应精灵
export function spriteForRegion(region: string): THREE.CanvasTexture {
  switch (region) {
    case 'huanan':   return Textures.spark;
    case 'shanxi':   return Textures.grain;
    case 'yungui':   return Textures.star;
    case 'jiangnan': return Textures.ink;
    case 'dongbei':  return Textures.snow;
    default:         return Textures.glowSoft;
  }
}
