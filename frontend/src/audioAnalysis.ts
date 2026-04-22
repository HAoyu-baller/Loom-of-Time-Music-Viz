// 岁月织机 — 离线音频分析
// 在歌曲开始播放前，对分离后的 perc.wav 做一次预分析，
// 得到整首歌的鼓点总数，用作 weaveProgress 的精确分母。
//
// 复刻 AudioBridge.tickBridge 的实时逻辑：
//   raw = rms(window) * 24
//   goal = min(1, raw)
//   弹簧（stiffness=28, damping=5.5）：
//     force = (goal - cur) * 28 - vel * 5.5
//     vel  += force * dt
//     cur  += vel * dt
//   amp = max(0, cur)
//   当 amp 从 ≤0.15 穿越到 >0.15 时，触发一次鼓点（percHitCount++）
//
// 步长对齐实时 useFrame（60fps），窗口尺寸 2048（和 AnalyserNode.fftSize 一致）。

const WINDOW_SIZE = 2048;
const PERC_RMS_SCALE = 24;
const PERC_THR = 0.15;
const SPRING_STIFFNESS = 28;
const SPRING_DAMPING = 5.5;
const TARGET_FPS = 60;

/**
 * 统计 AudioBuffer 里能被实时逻辑触发的鼓点次数。
 * 返回值应当接近一首歌播放完毕时 bridge.percHitCount 的最终数。
 */
export function countPercHits(buffer: AudioBuffer): number {
  const sampleRate = buffer.sampleRate;
  // 如果是立体声就取平均；我们的 stems 应该是单声道，但保险起见
  const ch0 = buffer.getChannelData(0);
  const ch1 = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : null;

  // 离线模拟每 1/60 秒一步，和实时 useFrame 对齐
  const stepSize = Math.max(1, Math.floor(sampleRate / TARGET_FPS));
  const stepDt = stepSize / sampleRate;

  let vel = 0, cur = 0;
  let prevAmp = 0;
  let hits = 0;

  for (let start = 0; start + WINDOW_SIZE <= ch0.length; start += stepSize) {
    // 计算窗口 RMS
    let s = 0;
    if (ch1) {
      for (let i = 0; i < WINDOW_SIZE; i++) {
        const v = (ch0[start + i] + ch1[start + i]) * 0.5;
        s += v * v;
      }
    } else {
      for (let i = 0; i < WINDOW_SIZE; i++) {
        const v = ch0[start + i];
        s += v * v;
      }
    }
    const rms = Math.sqrt(s / WINDOW_SIZE);
    const goal = Math.min(1, rms * PERC_RMS_SCALE);

    // 弹簧一步
    const force = (goal - cur) * SPRING_STIFFNESS - vel * SPRING_DAMPING;
    vel += force * stepDt;
    cur += vel * stepDt;
    const amp = Math.max(0, cur);

    // 上升沿判触发
    if (amp > PERC_THR && prevAmp <= PERC_THR) {
      hits++;
    }
    prevAmp = amp;
  }

  return hits;
}

/**
 * 统计经过 2s 冷却过滤后实际会触发的穿梭次数。
 * 复刻 PercFlash.tsx 的冷却门控：两次触发之间至少间隔 minIntervalSec 秒。
 * 返回值即 bridge.totalPercHits 的正确分母。
 */
export function countFilteredPercHits(
  buffer: AudioBuffer,
  minIntervalSec = 2.0,
): number {
  const sampleRate = buffer.sampleRate;
  const ch0 = buffer.getChannelData(0);
  const ch1 = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : null;

  const stepSize = Math.max(1, Math.floor(sampleRate / TARGET_FPS));
  const stepDt   = stepSize / sampleRate;

  let vel = 0, cur = 0;
  let prevAmp = 0;
  let filteredHits = 0;
  let lastFireTime = -Infinity;
  let currentTime  = 0;

  for (let start = 0; start + WINDOW_SIZE <= ch0.length; start += stepSize) {
    let s = 0;
    if (ch1) {
      for (let i = 0; i < WINDOW_SIZE; i++) {
        const v = (ch0[start + i] + ch1[start + i]) * 0.5;
        s += v * v;
      }
    } else {
      for (let i = 0; i < WINDOW_SIZE; i++) {
        const v = ch0[start + i];
        s += v * v;
      }
    }
    const rms  = Math.sqrt(s / WINDOW_SIZE);
    const goal = Math.min(1, rms * PERC_RMS_SCALE);

    const force = (goal - cur) * SPRING_STIFFNESS - vel * SPRING_DAMPING;
    vel += force * stepDt;
    cur += vel * stepDt;
    const amp = Math.max(0, cur);

    if (amp > PERC_THR && prevAmp <= PERC_THR) {
      if (currentTime - lastFireTime >= minIntervalSec) {
        filteredHits++;
        lastFireTime = currentTime;
      }
    }
    prevAmp = amp;
    currentTime += stepDt;
  }

  return filteredHits;
}
