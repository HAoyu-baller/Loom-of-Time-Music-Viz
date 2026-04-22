// 岁月织机 — 音频桥接层（v3: 弹簧物理）
// 用阻尼弹簧振荡器替代 EMA，获得「果冻回弹」的物理美感
//   力 = (target - current) * stiffness - velocity * damping
//   velocity += force * dt
//   current  += velocity * dt

import { PALETTES, lerpPalette, deepCopyPalette, hsbToRgb } from '../palettes';
import type { Palette, RegionKey } from '../palettes';
import * as THREE from 'three';

// ── 对外暴露的 mutable state ─────────────────────────────────────────
export const bridge = {
  analysers: {} as Partial<Record<string, AnalyserNode>>,
  targetRegion: 'default' as RegionKey,
  isPlaying: false,
  amp: { vocal: 0, erhu: 0, wind: 0, plucked: 0, perc: 0 },
  pulse: { plucked: 0, perc: 0 },
  palette: deepCopyPalette(PALETTES.default),
  introProgress: 0,    // 0 = 未开始，1 = 播放满 5 秒后完全展开
  percGridScale: 1.0,  // 鼓点触发时弹到 1.08，弹簧衰减回 1.0
  vocalCenterY:  0,    // VocalLine 每帧写入当前中心 Y，供其他层定位
  percHitCount:         0,    // 每首歌累计鼓点次数，新歌开始时重置，驱动背景织入进度
  totalPercHits:        24,   // 本首歌预分析得到的冷却过滤后穿梭次数（loadAndPlay 前写入），0 fallback 到 8
  filteredPercHitCount: 0,    // 实际触发穿梭次数（由 PercFlash.tsx 写入），和冷却门控硬绑定
  weaveProgress:        0,    // [0, 1]，filteredPercHitCount / (totalPercHits - 3)，驱动布匹织入
  weaveGlow:            0,    // [0, 1]，最后 3 次穿梭后亮度提升
  estimatedBpm:   120, // 根据两次鼓点间隔估算，滑动平均稳定，默认 120
};

// ── THREE.Color 缓存（嵌套结构 + 向后兼容别名）─────────────────────
const _vocalMain   = new THREE.Color();
const _vocalAux    = new THREE.Color();
const _vocalAccent = new THREE.Color();
const _windMain    = new THREE.Color();
const _windBase    = new THREE.Color();

export const colors = {
  bg:        new THREE.Color(),
  bowed:     new THREE.Color(),
  plucked0:  new THREE.Color(),
  plucked1:  new THREE.Color(),
  perc:      new THREE.Color(),
  vocal: {
    main:   _vocalMain,
    aux:    _vocalAux,
    accent: _vocalAccent,
  },
  wind: {
    main: _windMain,
    base: _windBase,
  },
  // 向后兼容别名（指向同一 THREE.Color 实例）
  vocalCore: _vocalMain,
  vocalGlow: _vocalAux,
};

function applyColors(p: Palette) {
  colors.bg.setRGB(...hsbToRgb(p.bg));
  colors.vocal.main.setRGB(...hsbToRgb(p.vocal.core));
  colors.vocal.aux.setRGB(...hsbToRgb(p.vocal.glow));
  colors.vocal.accent.setRGB(...hsbToRgb(p.vocal.accent));
  colors.bowed.setRGB(...hsbToRgb(p.bowed));
  colors.wind.main.setRGB(...hsbToRgb(p.wind.main));
  colors.wind.base.setRGB(...hsbToRgb(p.wind.base));
  colors.plucked0.setRGB(...hsbToRgb(p.plucked[0]));
  colors.plucked1.setRGB(...hsbToRgb(p.plucked[1]));
  colors.perc.setRGB(...hsbToRgb(p.perc));
}

// ── RMS ──────────────────────────────────────────────────────────────
const _buf = new Float32Array(2048);
function rms(a: AnalyserNode): number {
  a.getFloatTimeDomainData(_buf);
  let s = 0;
  for (let i = 0; i < _buf.length; i++) s += _buf[i] * _buf[i];
  return Math.sqrt(s / _buf.length);
}

// ── 弹簧状态 ────────────────────────────────────────────────────────
type Ch = 'vocal' | 'erhu' | 'wind' | 'plucked' | 'perc';
const CHANNELS: Ch[] = ['vocal', 'erhu', 'wind', 'plucked', 'perc'];

const _spring: Record<Ch, { vel: number; cur: number }> = {
  vocal:   { vel: 0, cur: 0 },
  erhu:    { vel: 0, cur: 0 },
  wind:    { vel: 0, cur: 0 },
  plucked: { vel: 0, cur: 0 },
  perc:    { vel: 0, cur: 0 },
};

// 不同声部用不同弹簧参数：
//   人声/拉弦 → 柔软（stiffness 低，damping 低 → 缓慢跟随，余韵长）
//   弹拨/打击 → 硬弹（stiffness 高，damping 中 → 快速起跳，果冻回弹）
const _springCfg: Record<Ch, { stiffness: number; damping: number }> = {
  vocal:   { stiffness: 8,  damping: 4 },
  erhu:    { stiffness: 10, damping: 4.5 },
  wind:    { stiffness: 6,  damping: 3.5 },
  plucked: { stiffness: 22, damping: 5 },   // 快攻 + 回弹
  perc:    { stiffness: 28, damping: 5.5 },  // 最硬
};

let _prevPlucked = 0;
let _prevPerc = 0;
let _introStartTime = -1; // 首次播放时刻，-1 = 尚未开始
let _wasPlaying = false;  // 上一帧播放状态，用于检测新歌开始
let _lastPercTime = -1;   // 上一次 perc 脉冲触发时刻（用于 BPM 估算）

// ── 每帧 tick ────────────────────────────────────────────────────────
export function tickBridge(dt: number, elapsed: number) {
  // 限幅防爆（极端 dt 跳帧）
  const safeDt = Math.min(dt, 0.05);

  // 1. 调色板渐变（τ=0.35s，1.5 秒到 99% 收敛）
  const target = PALETTES[bridge.targetRegion] ?? PALETTES.default;
  const tLerp  = 1 - Math.exp(-safeDt / 0.35);
  bridge.palette = lerpPalette(bridge.palette, target, tLerp);
  applyColors(bridge.palette);

  // 2. 读振幅目标
  const idle = !bridge.isPlaying;
  const raw = {
    vocal:   bridge.analysers.vocal   ? rms(bridge.analysers.vocal!)   * 14 : 0,
    erhu:    bridge.analysers.erhu    ? rms(bridge.analysers.erhu!)    * 16 : 0,
    wind:    bridge.analysers.wind    ? rms(bridge.analysers.wind!)    * 16 : 0,
    plucked: bridge.analysers.plucked ? rms(bridge.analysers.plucked!) * 20 : 0,
    perc:    bridge.analysers.perc    ? rms(bridge.analysers.perc!)    * 24 : 0,
  };
  const goal = {
    vocal:   idle ? 0.18 + Math.sin(elapsed * 0.6) * 0.06 : Math.min(1, raw.vocal),
    erhu:    idle ? 0.10 + Math.sin(elapsed * 0.4) * 0.04 : Math.min(1, raw.erhu),
    wind:    idle ? 0.06 : Math.min(1, raw.wind),
    plucked: idle ? 0    : Math.min(1, raw.plucked),
    perc:    idle ? 0    : Math.min(1, raw.perc),
  };

  // 3. 弹簧物理驱动
  for (const k of CHANNELS) {
    const sp = _spring[k];
    const cfg = _springCfg[k];
    const force = (goal[k] - sp.cur) * cfg.stiffness - sp.vel * cfg.damping;
    sp.vel += force * safeDt;
    sp.cur += sp.vel * safeDt;
    bridge.amp[k] = Math.max(0, sp.cur); // 不允许负值
  }

  // 4. introProgress（播放启动后 0→1，持续 5 秒）
  //    同时检测新歌开始（isPlaying false→true），重置背景织入计数
  if (bridge.isPlaying && !_wasPlaying) {
    // 新歌开始：重置 intro + 背景织入计数 + BPM 估算
    _introStartTime = elapsed;
    bridge.introProgress = 0;
    bridge.percHitCount          = 0;
    bridge.filteredPercHitCount  = 0;
    bridge.weaveGlow             = 0;
    bridge.estimatedBpm  = 120;
    _lastPercTime        = -1;
  }
  _wasPlaying = bridge.isPlaying;

  if (bridge.isPlaying && _introStartTime < 0) {
    _introStartTime = elapsed;
  }
  if (_introStartTime >= 0) {
    bridge.introProgress = Math.min(1, (elapsed - _introStartTime) / 5.0);
  }

  // 5. percGridScale 弹簧（鼓点触发时外部写入 1.08，这里每帧衰减回 1.0）
  bridge.percGridScale += (1.0 - bridge.percGridScale) * 0.18;

  // 6. 峰值脉冲
  const PLUCKED_THR = 0.08;   // 降低阈值：更容易捕捉到轻弹拨
  const PERC_THR = 0.15;
  if (bridge.amp.plucked > PLUCKED_THR && _prevPlucked <= PLUCKED_THR) {
    bridge.pulse.plucked = 1;
  }
  if (bridge.amp.perc > PERC_THR && _prevPerc <= PERC_THR) {
    bridge.pulse.perc = 1;
    bridge.percGridScale = 1.08; // 鼓点触发网格收缩脉冲
    bridge.percHitCount++;       // 背景织入计数 +1

    // BPM 估算：用两次鼓点间隔推算瞬时 BPM，指数移动平均平滑
    if (_lastPercTime >= 0) {
      const interval = elapsed - _lastPercTime;
      if (interval > 0.2 && interval < 3.0) { // 有效范围 20~300 BPM
        const instantBpm = 60 / interval;
        // 指数移动平均：新值权重 0.3，历史权重 0.7，让估算稳定不抖动
        bridge.estimatedBpm = bridge.estimatedBpm * 0.7 + instantBpm * 0.3;
      }
    }
    _lastPercTime = elapsed;
  }
  _prevPlucked = bridge.amp.plucked;
  _prevPerc = bridge.amp.perc;

  bridge.pulse.plucked = Math.max(0, bridge.pulse.plucked - safeDt * 1.8);
  bridge.pulse.perc    = Math.max(0, bridge.pulse.perc    - safeDt * 2.5);

  // 7. 织入进度（按冷却过滤后的穿梭总数归一化；最后 3 次用于亮度提升）
  const F        = bridge.totalPercHits > 0 ? bridge.totalPercHits : 1;
  const finishAt = Math.max(1, F - 3);
  const N        = bridge.filteredPercHitCount;
  bridge.weaveProgress = Math.min(1, N / finishAt);

  const remainingAfterFull = F - finishAt;
  bridge.weaveGlow = (remainingAfterFull > 0 && N > finishAt)
    ? Math.min(1, (N - finishAt) / remainingAfterFull)
    : 0;
}
