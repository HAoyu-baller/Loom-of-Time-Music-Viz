// 岁月织机 — 地域色彩调色板
// 真源是 REGION_COLORS (hex)，运行时用 HSB（带色相环绕的 lerp 避免极端色间灰泥过渡）

export interface HSBColor {
  h: number; // 0-360
  s: number; // 0-100
  b: number; // 0-100
}

export interface Palette {
  bg: HSBColor;
  vocal: { core: HSBColor; glow: HSBColor; accent: HSBColor };
  bowed: HSBColor;
  wind: { main: HSBColor; base: HSBColor };
  plucked: [HSBColor, HSBColor];
  perc: HSBColor;
}

export type RegionKey = 'default' | 'huanan' | 'shanxi' | 'yungui' | 'jiangnan' | 'dongbei';

// ── 用户可见的 hex 配色真源 ─────────────────────────────────────────
// vocal: main(主色) / aux(辅助色) / accent(点缀色)
// wind:  main(主色) / base(底色，用作场景背景)
// '#__KEEP__' 占位符 = 沿用该字段的现有 HSB 值（用于 dongbei vocal 保留不变）
export const REGION_COLORS: Record<RegionKey, {
  vocal: { main: string; aux: string; accent: string };
  wind:  { main: string; base: string };
}> = {
  jiangnan: {
    vocal: { main: '#F4F6F9', aux: '#89C5D8', accent: '#E8D3C7' },
    wind:  { main: '#C8D6C7', base: '#2C3539' },
  },
  shanxi: {
    vocal: { main: '#8C2111', aux: '#D28C4B', accent: '#D9381E' },
    wind:  { main: '#BFA896', base: '#382418' },
  },
  yungui: {
    vocal: { main: '#E0FFFF', aux: '#00E5FF', accent: '#FFFFFF' },
    wind:  { main: '#8A2BE2', base: '#081820' },
  },
  huanan: {
    vocal: { main: '#FFD700', aux: '#FF8C00', accent: '#FF4500' },
    wind:  { main: '#FFFFFF', base: '#0A4A3C' },
  },
  dongbei: {
    // vocal 保持现有 DB_COLORS 冰绿/猩红双螺旋不变
    vocal: { main: '#__KEEP__', aux: '#__KEEP__', accent: '#__KEEP__' },
    wind:  { main: '#E6F4F1', base: '#0A0E12' },
  },
  default: {
    vocal: { main: '#D4B996', aux: '#F2E4CE', accent: '#E0C9A8' },
    wind:  { main: '#E8D3A8', base: '#0F0E08' },
  },
};

// ── hex → HSB 工具 ──────────────────────────────────────────────────
export function hexToHsb(hex: string): HSBColor {
  let s = hex.replace('#', '').trim();
  if (s.length === 3) s = s.split('').map(c => c + c).join('');
  const r = parseInt(s.slice(0, 2), 16) / 255;
  const g = parseInt(s.slice(2, 4), 16) / 255;
  const bl = parseInt(s.slice(4, 6), 16) / 255;
  const max = Math.max(r, g, bl), min = Math.min(r, g, bl);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === r)      h = ((g - bl) / d) % 6;
    else if (max === g) h = (bl - r) / d + 2;
    else                h = (r - g)  / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  const sat = max === 0 ? 0 : (d / max) * 100;
  const bri = max * 100;
  return { h, s: sat, b: bri };
}

// 历史 HSB 兜底值（仅 dongbei vocal 会用到：用户明确要求保持）
const DONGBEI_VOCAL_KEEP = {
  core:   { h: 208, s: 72, b: 95 },
  glow:   { h: 205, s: 30, b: 100 },
  accent: { h: 205, s: 30, b: 100 },  // 用 glow 占位，因 dongbei 暂不使用 accent
};

// 其他声部（bowed / plucked / perc）的每地区 HSB 保持历史值，
// 本轮仅 vocal 和 wind 接入 REGION_COLORS
const LEGACY_OTHER: Record<RegionKey, { bowed: HSBColor; plucked: [HSBColor, HSBColor]; perc: HSBColor }> = {
  default:  { bowed: { h:  40, s: 20, b: 18 }, plucked: [{ h: 38, s: 55, b: 85 }, { h: 45, s: 45, b: 90 }], perc: { h:  35, s: 65, b: 80 } },
  huanan:   { bowed: { h: 162, s: 35, b: 16 }, plucked: [{ h:356, s: 75, b: 88 }, { h: 42, s: 65, b: 92 }], perc: { h: 355, s: 80, b: 85 } },
  shanxi:   { bowed: { h:  38, s: 48, b: 18 }, plucked: [{ h:  4, s: 88, b: 90 }, { h: 38, s: 75, b: 92 }], perc: { h:   0, s: 92, b: 90 } },
  yungui:   { bowed: { h: 232, s: 42, b: 17 }, plucked: [{ h:190, s: 30, b: 95 }, { h:195, s: 65, b: 90 }], perc: { h: 262, s: 68, b: 78 } },
  jiangnan: { bowed: { h: 200, s: 15, b: 20 }, plucked: [{ h:210, s: 20, b: 75 }, { h:200, s:  8, b: 92 }], perc: { h: 208, s: 28, b: 72 } },
  dongbei:  { bowed: { h:  25, s: 10, b: 22 }, plucked: [{ h:210, s: 65, b: 92 }, { h:200, s: 10, b: 98 }], perc: { h: 198, s: 52, b: 92 } },
};

function buildPalette(region: RegionKey): Palette {
  const rc = REGION_COLORS[region];
  const other = LEGACY_OTHER[region];
  // vocal：hex 真源，遇 '#__KEEP__' 用 DONGBEI_VOCAL_KEEP
  const vMain   = rc.vocal.main   === '#__KEEP__' ? DONGBEI_VOCAL_KEEP.core   : hexToHsb(rc.vocal.main);
  const vAux    = rc.vocal.aux    === '#__KEEP__' ? DONGBEI_VOCAL_KEEP.glow   : hexToHsb(rc.vocal.aux);
  const vAccent = rc.vocal.accent === '#__KEEP__' ? DONGBEI_VOCAL_KEEP.accent : hexToHsb(rc.vocal.accent);
  return {
    bg:      hexToHsb(rc.wind.base),   // bg 历史字段保留（与 wind.base 同源）
    vocal:   { core: vMain, glow: vAux, accent: vAccent },
    wind:    { main: hexToHsb(rc.wind.main), base: hexToHsb(rc.wind.base) },
    bowed:   other.bowed,
    plucked: other.plucked,
    perc:    other.perc,
  };
}

export const PALETTES: Record<RegionKey, Palette> = {
  default:  buildPalette('default'),
  huanan:   buildPalette('huanan'),
  shanxi:   buildPalette('shanxi'),
  yungui:   buildPalette('yungui'),
  jiangnan: buildPalette('jiangnan'),
  dongbei:  buildPalette('dongbei'),
};

// ── HSB lerp（带色相环绕，避免极端色间灰泥过渡）────────────────────
function lerpHue(a: number, b: number, t: number): number {
  let d = b - a;
  if (d > 180) d -= 360;
  if (d < -180) d += 360;
  return (a + d * t + 360) % 360;
}

function lerpColor(a: HSBColor, b: HSBColor, t: number): HSBColor {
  return {
    h: lerpHue(a.h, b.h, t),
    s: a.s + (b.s - a.s) * t,
    b: a.b + (b.b - a.b) * t,
  };
}

// 每帧调用：activePalette 向 targetPalette 步进 t
export function lerpPalette(a: Palette, b: Palette, t: number): Palette {
  return {
    bg:    lerpColor(a.bg, b.bg, t),
    vocal: {
      core:   lerpColor(a.vocal.core,   b.vocal.core,   t),
      glow:   lerpColor(a.vocal.glow,   b.vocal.glow,   t),
      accent: lerpColor(a.vocal.accent, b.vocal.accent, t),
    },
    bowed: lerpColor(a.bowed, b.bowed, t),
    wind:  {
      main: lerpColor(a.wind.main, b.wind.main, t),
      base: lerpColor(a.wind.base, b.wind.base, t),
    },
    plucked: [
      lerpColor(a.plucked[0], b.plucked[0], t),
      lerpColor(a.plucked[1], b.plucked[1], t),
    ],
    perc:  lerpColor(a.perc, b.perc, t),
  };
}

export function deepCopyPalette(p: Palette): Palette {
  return JSON.parse(JSON.stringify(p)) as Palette;
}

// ── Three.js 需要 RGB（0-1），这里做转换 ─────────────────────────────
export function hsbToRgb(c: HSBColor): [number, number, number] {
  const h = c.h / 360, s = c.s / 100, v = c.b / 100;
  let r = 0, g = 0, b = 0;
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    case 5: r = v; g = p; b = q; break;
  }
  return [r, g, b];
}
