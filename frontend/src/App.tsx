import { useRef, useState, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { LoomScene } from './scene/LoomScene';
import { bridge } from './scene/AudioBridge';
import { countFilteredPercHits } from './audioAnalysis';
import { LoadingOverlay } from './components/LoadingOverlay';
import type { RegionKey } from './palettes';
import './App.css';

const API = 'http://localhost:8000';

interface SeparateResponse {
  job_id: string;
  duration: number;
  stems: Record<string, string>;
  region: string | null;
  region_zh: string | null;
  region_confidence: number | null;
}

const REGION_EN: Record<string, string> = {
  jiangnan: 'Jiangnan',
  shanxi:   'Shaanxi',
  yungui:   'Yungui',
  huanan:   'Huanan',
  dongbei:  'Dongbei',
};

type Phase = 'idle' | 'identifying' | 'separating' | 'playing';

function buildAnalyser(ctx: AudioContext): AnalyserNode {
  const a = ctx.createAnalyser();
  a.fftSize = 2048;
  a.smoothingTimeConstant = 0.6;
  return a;
}

export default function App() {
  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourcesRef = useRef<AudioBufferSourceNode[]>([]);

  const [phase, setPhase] = useState<Phase>('idle');
  const [detectedRegion, setDetectedRegion] = useState<{ key: RegionKey; en: string; conf: number } | null>(null);
  const [uploadStartTime, setUploadStartTime] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');

  const stopAudio = useCallback(() => {
    sourcesRef.current.forEach(s => { try { s.stop(); } catch { /* */ } });
    sourcesRef.current = [];
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    bridge.analysers = {};
    bridge.isPlaying = false;
  }, []);

  // ── 加载 stems → 预分析 perc → 启动播放 ─────────────────────────
  // ctx 必须从外部传入：在 user gesture 同步阶段创建才能避开 Safari autoplay policy
  const loadStemsAndPlay = useCallback(async (data: SeparateResponse, ctx: AudioContext) => {
    const stemUrls = Object.fromEntries(
      Object.entries(data.stems).map(([k, v]) => [k, v.startsWith('/demo/') ? v : `${API}${v}`])
    );

    const stemNames = ['vocal', 'erhu', 'plucked', 'wind', 'perc'] as const;
    const buffers: Record<string, AudioBuffer> = {};
    await Promise.all(stemNames.map(async (stem) => {
      const resp = await fetch(stemUrls[stem]);
      const ab = await resp.arrayBuffer();
      buffers[stem] = await ctx.decodeAudioData(ab);
    }));

    // 预分析 perc.wav：得到整首歌鼓点总数
    const filteredHits = countFilteredPercHits(buffers['perc'], 2.0);
    bridge.totalPercHits = filteredHits > 0 ? filteredHits : 8;

    // Safari / Chrome 可能把长时间闲置的 ctx 置为 suspended → 在 start 前显式 resume
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }

    const newAnalysers: Record<string, AnalyserNode> = {};
    const newSources: AudioBufferSourceNode[] = [];
    const startAt = ctx.currentTime + 0.15;

    for (const stem of stemNames) {
      const analyser = buildAnalyser(ctx);
      analyser.connect(ctx.destination);
      newAnalysers[stem] = analyser;
      const src = ctx.createBufferSource();
      src.buffer = buffers[stem];
      src.connect(analyser);
      src.start(startAt);
      newSources.push(src);
      src.onended = () => {
        if (stem === 'vocal') {
          // 一首歌放完，自动回到 idle
          setPhase('idle');
          setDetectedRegion(null);
          bridge.isPlaying = false;
          bridge.targetRegion = 'default';
          bridge.analysers = {};
        }
      };
    }

    bridge.analysers = newAnalysers;
    bridge.isPlaying = true;
    sourcesRef.current = newSources;
  }, []);

  // ── 核心：上传文件 → 两阶段加载 → 播放 ────────────────────────
  const handleFile = useCallback(async (file: File) => {
    stopAudio();
    setErrorMsg('');
    setDetectedRegion(null);
    setPhase('identifying');

    // ⚠️ 关键：在 user gesture 同步阶段立即创建 AudioContext
    // Safari 严格执行 autoplay policy，延后到 await 之后创建会 → state='suspended' → 没声音
    const ctx = new AudioContext();
    audioCtxRef.current = ctx;

    const form = new FormData();
    form.append('file', file);
    setUploadStartTime(performance.now());

    let data: SeparateResponse;
    try {
      const resp = await fetch(`${API}/separate`, { method: 'POST', body: form });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      data = await resp.json() as SeparateResponse;
    } catch (e) {
      setErrorMsg(`Analysis failed: ${e instanceof Error ? e.message : String(e)}`);
      setPhase('idle');
      bridge.targetRegion = 'default';
      return;
    }

    // 切到识别地区 + reveal
    if (data.region) {
      bridge.targetRegion = data.region as RegionKey;
      setDetectedRegion({
        key: data.region as RegionKey,
        en:  REGION_EN[data.region] ?? data.region,
        conf: data.region_confidence ?? 0,
      });
    }
    setPhase('separating');

    try {
      await loadStemsAndPlay(data, ctx);
    } catch (e) {
      setErrorMsg(`Load failed: ${e instanceof Error ? e.message : String(e)}`);
      setPhase('idle');
      setDetectedRegion(null);
      bridge.targetRegion = 'default';
      return;
    }
    setPhase('playing');
  }, [stopAudio, loadStemsAndPlay]);

  const handleDemo = useCallback(async () => {
    stopAudio();
    setErrorMsg('');
    setDetectedRegion(null);
    setPhase('identifying');

    const ctx = new AudioContext();
    audioCtxRef.current = ctx;

    const base = '/demo';
    const demoRegion: RegionKey = 'jiangnan';
    const fakeData: SeparateResponse = {
      job_id: 'demo',
      duration: 0,
      stems: {
        vocal:   `${base}/vocal.wav`,
        erhu:    `${base}/erhu.wav`,
        plucked: `${base}/plucked.wav`,
        wind:    `${base}/wind.wav`,
        perc:    `${base}/perc.wav`,
      },
      region: demoRegion,
      region_zh: demoRegion,
      region_confidence: 0.94,
    };

    setUploadStartTime(performance.now());
    await new Promise<void>(res => setTimeout(res, 1500));

    bridge.targetRegion = demoRegion;
    setDetectedRegion({ key: demoRegion, en: REGION_EN[demoRegion] ?? demoRegion, conf: 0.94 });
    setPhase('separating');

    try {
      await loadStemsAndPlay(fakeData, ctx);
    } catch (e) {
      setErrorMsg(`Demo load failed: ${e instanceof Error ? e.message : String(e)}`);
      setPhase('idle');
      setDetectedRegion(null);
      bridge.targetRegion = 'default';
      return;
    }
    setPhase('playing');
  }, [stopAudio, loadStemsAndPlay]);

  const handleBack = useCallback(() => {
    stopAudio();
    setPhase('idle');
    setDetectedRegion(null);
    setErrorMsg('');
    bridge.targetRegion = 'default';
  }, [stopAudio]);

  const onFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
    e.target.value = '';
  }, [handleFile]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);
  const onDragOver = (e: React.DragEvent) => e.preventDefault();

  return (
    <>
      {/* Three.js Canvas — 永不卸载 */}
      <div className="canvas-mount">
        <Canvas
          orthographic
          camera={{ zoom: 100, near: -10, far: 10, position: [0, 0, 5] }}
          gl={{ antialias: true, alpha: false }}
          dpr={[1, 2]}
        >
          <LoomScene />
        </Canvas>
      </div>

      {/* DOM overlay */}
      <div className="overlay">
        {/* Landing screen */}
        {phase === 'idle' && (
          <div className="center-ui" onDrop={onDrop} onDragOver={onDragOver}>
            <h1 className="title">Loom of Time</h1>
            <p className="subtitle">An Interactive Weaving of Chinese Folk Music</p>
            <label className="upload-btn" htmlFor="file-input">
              Upload Your Song
            </label>
            <input
              id="file-input"
              type="file"
              accept=".mp3,.wav,.flac,.ogg,.m4a"
              style={{ display: 'none' }}
              onChange={onFileChange}
            />
            <p className="hint">MP3 &middot; WAV &middot; FLAC &middot; OGG &middot; M4A</p>
            <button className="demo-btn" onClick={handleDemo}>
              ▸ Listen to Demo
            </button>
            {errorMsg && <p className="error-msg">{errorMsg}</p>}
          </div>
        )}

        {/* Loading overlay */}
        {(phase === 'identifying' || phase === 'separating') && (
          <LoadingOverlay phase={phase} detectedRegion={detectedRegion} uploadStartTime={uploadStartTime} />
        )}

        {/* Back button — 非 idle 态常驻左上 */}
        {phase !== 'idle' && (
          <button className="back-btn" onClick={handleBack} aria-label="Back">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M19 12H5M12 19l-7-7 7-7" />
            </svg>
          </button>
        )}
      </div>
    </>
  );
}
