// 岁月织机 — 加载过渡层
//
// Phase 1 "Identifying Regional Style"（后端分离 + 分类推理中）：
//   - 进度条对数曲线缓慢爬到 90%（τ=30s），永远不走满，如实反映后端在处理
//   - 每 2 秒随机切 bridge.targetRegion（5 地区轮播），演"正在识别"效果
//   - 每 700ms 触发 bridge.pulse.plucked = 1，Ripples 自动 spawn 织点
//
// Phase 2 "Separating Instruments"（前端 fetch wav + decodeAudioData 中）：
//   - 进度条对数曲线 0 → 95%（τ=6s），约 15 秒接近 95%
//   - 顶部 reveal 当前识别结果（Detected · Jiangnan · 91%）
//   - bridge.targetRegion 已被 App.tsx 设为识别地区
//   - 每 400ms 触发 Ripples（更紧凑的节奏）
//
// 组件卸载时所有 setInterval/setTimeout/RAF 自动清理

import { useEffect, useRef, useState } from 'react';
import type { RegionKey } from '../palettes';
import { bridge } from '../scene/AudioBridge';

const REGIONS_ROULETTE: RegionKey[] = ['jiangnan', 'shanxi', 'yungui', 'huanan', 'dongbei'];

interface Props {
  phase: 'identifying' | 'separating';
  detectedRegion: { key: RegionKey; en: string; conf: number } | null;
  /** 上传请求开始的时刻（performance.now()），用于 identifying 进度跟踪 */
  uploadStartTime: number;
}

export function LoadingOverlay({ phase, detectedRegion, uploadStartTime }: Props) {
  const [progress, setProgress] = useState(0);
  const startRef = useRef(0);

  // ── 进度条动画（phase 切换时重置）─────────────────────────────────
  useEffect(() => {
    startRef.current = performance.now();
    setProgress(0);
    let raf = 0;
    const tick = () => {
      if (phase === 'identifying') {
        // Phase 1：从上传开始计时，对数曲线缓慢爬到 90%（τ=30s）
        // 永远不走满，如实体现后端还在处理
        const elapsed = (performance.now() - uploadStartTime) / 1000;
        setProgress(0.90 * (1 - Math.exp(-elapsed / 30)));
      } else {
        // Phase 2：从 phase 切换时起算，对数曲线 0 → 95%（τ=6s）
        const elapsed = (performance.now() - startRef.current) / 1000;
        setProgress(0.95 * (1 - Math.exp(-elapsed / 6)));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [phase, uploadStartTime]);

  // ── Ripples 触发：phase 决定频率 ──────────────────────────────────
  useEffect(() => {
    const interval = phase === 'identifying' ? 700 : 400;
    const id = setInterval(() => {
      bridge.pulse.plucked = 1;
    }, interval);
    return () => clearInterval(id);
  }, [phase]);

  // ── Phase 1 专属：每 2 秒随机切一次 region，演"正在识别" ──────────
  useEffect(() => {
    if (phase !== 'identifying') return;
    bridge.targetRegion = REGIONS_ROULETTE[Math.floor(Math.random() * REGIONS_ROULETTE.length)];
    const id = setInterval(() => {
      const r = REGIONS_ROULETTE[Math.floor(Math.random() * REGIONS_ROULETTE.length)];
      bridge.targetRegion = r;
    }, 2000);
    return () => clearInterval(id);
  }, [phase]);

  const label = phase === 'identifying'
    ? 'Identifying Regional Style'
    : 'Separating Instruments';

  return (
    <div className="loading-overlay">
      <div className="loading-content">
        {phase === 'separating' && detectedRegion && (
          <div className="region-reveal">
            <span className="reveal-label">Detected</span>
            <span className="reveal-name">{detectedRegion.en}</span>
            <span className="reveal-conf">
              {Math.round(detectedRegion.conf * 100)}% confidence
            </span>
          </div>
        )}
        <p className="phase-label">{label}</p>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progress * 100}%` }} />
        </div>
      </div>
    </div>
  );
}
