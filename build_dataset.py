"""
岁月织机 — 数据集构建脚本
============================
从 music/instruments_clean/ 读取所有音频，切分为固定长度片段，
输出到 music/dataset/<track>/，并生成 manifest.csv 溯源表。

输入目录映射（自动合并 Bilibili 下载源）：
  erhu    ← instruments_clean/erhu/**
  wind    ← instruments_clean/wind/** + dizi/** + suona/** + sheng/**
  plucked ← instruments_clean/plucked/** + tanbo/**
  perc    ← instruments_clean/perc/** + dagu/**
  vocal   ← instruments_clean/vocal/**

输出：
  music/dataset/erhu/erhu_000000.wav  ...
  music/dataset/wind/wind_000000.wav  ...
  music/dataset/plucked/plucked_000000.wav  ...
  music/dataset/perc/perc_000000.wav  ...
  music/dataset/vocal/vocal_000000.wav  ...
  music/dataset/manifest.csv

用法：
  python build_dataset.py                     # 构建全部
  python build_dataset.py --track perc        # 单轨道
  python build_dataset.py --chunk-sec 6 --hop-sec 3
"""

import argparse
import csv
import multiprocessing as mp
import pathlib
import time

import numpy as np
import soundfile as sf
from tqdm import tqdm

# ─── 参数配置 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT  = pathlib.Path(__file__).parent
IN_ROOT       = PROJECT_ROOT / "music" / "instruments_clean"
OUT_ROOT      = PROJECT_ROOT / "music" / "dataset"
MANIFEST_PATH = OUT_ROOT / "manifest.csv"

TARGET_SR     = 22050
CHUNK_SEC     = 6
HOP_SEC       = 3
MIN_SEC       = 2          # 丢弃短于此长度的尾段
SILENCE_RMS   = 1e-4       # RMS 低于此值视为静音片段，跳过

# 轨道 → 源目录列表（相对于 IN_ROOT）
TRACK_SOURCES = {
    "erhu":    ["erhu"],
    "wind":    ["wind", "dizi", "suona", "sheng"],
    "plucked": ["plucked", "tanbo"],
    "perc":    ["perc", "dagu"],
    "vocal":   ["vocal"],
}

# ─── Perc 子池分类（均衡采样用）────────────────────────────────────────────────
# 鼓类：transient 冲击，低中频宽带噪声
PERC_DRUM = {
    "ctis_中国大鼓", "ctis_南鼓", "ctis_小堂鼓", "ctis_板鼓", "ctis_花盆鼓",
    "ctis_压脚鼓", "ctis_上杖鼓", "ctis_杖鼓", "ctis_引鼓", "ctis_手鼓",
    "ctis_渔鼓", "ctis_扁鼓", "ctis_五音排鼓", "ctis_川剧堂鼓",
    "ctis_宜春三星鼓单铛", "ctis_宜春三星鼓双铛", "ctis_宜春三星鼓寿鼓老鼓",
    "ctis_宜春三星鼓禄鼓老鼓", "ctis_宜春三星鼓福鼓老鼓",
}
# 锣钹类：metallic ring，中高频谐波/宽带，含云锣编钟等有音高打击
PERC_METAL = {
    "ctis_包锣", "ctis_低音大锣", "ctis_小锣", "ctis_小锣2", "ctis_川大锣",
    "ctis_川小锣", "ctis_小叫锣", "ctis_武锣", "ctis_草锣", "ctis_曲锣",
    "ctis_斗锣", "ctis_抄锣", "ctis_蛮锣", "ctis_马锣", "ctis_虎音锣",
    "ctis_圆锣", "ctis_锣仔",
    "ctis_钹", "ctis_小钹", "ctis_铙", "ctis_铙钹", "ctis_大镲", "ctis_小镲",
    "ctis_宜春三星鼓镲",
    "ctis_云锣", "ctis_编钟", "ctis_编磬",
    "ctis_双铃", "ctis_大铛铛", "ctis_小铛铛", "ctis_响盏", "ctis_响盏2",
    "ctis_萨巴依",
}
# 板木类：短促木质/竹质冲击，高频
PERC_WOOD = {
    "ctis_拍板", "ctis_简板", "ctis_提手板", "ctis_北梆子", "ctis_南梆子",
    "ctis_脚梆子", "ctis_盖板传统", "ctis_盖板新D调", "ctis_木鱼",
}


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def collect_wav_files(track: str) -> list[pathlib.Path]:
    """收集某轨道所有源目录下的 WAV 文件"""
    files = []
    for src in TRACK_SOURCES[track]:
        d = IN_ROOT / src
        if d.exists():
            files.extend(sorted(d.rglob("*.wav")))
    return files


def collect_wav_files_perc_balanced() -> list[pathlib.Path]:
    """
    Perc 均衡采样：将子目录分为鼓/锣钹/板木三池，
    文件按池交错排列，使三类声学特性在切片中均衡分布。
    """
    pools = {"drum": [], "metal": [], "wood": [], "other": []}
    pool_map = {}
    for name in PERC_DRUM:  pool_map[name] = "drum"
    for name in PERC_METAL: pool_map[name] = "metal"
    for name in PERC_WOOD:  pool_map[name] = "wood"

    for src in TRACK_SOURCES["perc"]:
        d = IN_ROOT / src
        if not d.exists():
            continue
        for wav in sorted(d.rglob("*.wav")):
            pool_key = pool_map.get(wav.parent.name, "other")
            pools[pool_key].append(wav)

    # 打印各池统计
    for key, files in pools.items():
        dur = sum(sf.info(str(f)).duration for f in files if f.exists())
        print(f"    perc [{key:6s}]: {len(files):4d} 个文件  {dur/3600:.2f}h")

    # 交错排列：drum/metal/wood/other 轮流取文件
    result = []
    iters = {k: iter(v) for k, v in pools.items() if v}
    active_keys = list(iters.keys())
    while active_keys:
        next_keys = []
        for k in active_keys:
            try:
                result.append(next(iters[k]))
                next_keys.append(k)
            except StopIteration:
                pass
        active_keys = next_keys

    return result


def chunk_one_file(args):
    """
    处理单个 WAV 文件，切成片段后写出。
    返回 list of (out_path, src_path, start_sec, duration_sec)
    """
    src_path, out_dir, track, start_idx, chunk_samples, hop_samples, min_samples = args

    results = []
    try:
        audio, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
    except Exception:
        return results

    # 多声道 → 单声道
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # 如果采样率不匹配（理论上 clean 后都是 22050）
    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR,
                                 res_type="soxr_hq")

    total_samples = len(audio)
    idx = start_idx
    pos = 0

    while pos + min_samples <= total_samples:
        end = pos + chunk_samples
        chunk = audio[pos:end]

        # 最后一段不足 chunk_samples：如果够 min_samples 就保留，否则丢弃
        if len(chunk) < chunk_samples:
            if len(chunk) < min_samples:
                break
            # 不足 chunk 则零填充到标准长度
            pad = np.zeros(chunk_samples - len(chunk), dtype=np.float32)
            chunk = np.concatenate([chunk, pad])

        # 跳过静音片段
        if np.sqrt(np.mean(chunk ** 2)) < SILENCE_RMS:
            pos += hop_samples
            continue

        out_name = f"{track}_{idx:06d}.wav"
        out_path = out_dir / out_name
        sf.write(str(out_path), chunk, samplerate=TARGET_SR, subtype="PCM_16")

        start_sec = pos / TARGET_SR
        results.append((str(out_path.relative_to(PROJECT_ROOT)),
                        str(src_path.relative_to(PROJECT_ROOT)),
                        round(start_sec, 3),
                        round(len(chunk) / TARGET_SR, 3)))

        idx += 1
        pos += hop_samples

    return results


def build_track(track: str, chunk_sec: float, hop_sec: float,
                min_sec: float, workers: int):
    """构建单个轨道的数据集"""
    print(f"\n{'='*60}")
    print(f"  轨道: {track}")
    print(f"{'='*60}")

    if track == "perc":
        print("  [perc 均衡采样模式]")
        files = collect_wav_files_perc_balanced()
    else:
        files = collect_wav_files(track)

    if not files:
        print(f"  [跳过] 未找到任何 WAV 文件")
        return []

    print(f"  源文件数: {len(files)}")

    out_dir = OUT_ROOT / track
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_samples = int(chunk_sec * TARGET_SR)
    hop_samples   = int(hop_sec   * TARGET_SR)
    min_samples   = int(min_sec   * TARGET_SR)

    # 预先统计已有片段数量用于续编号
    existing = len(list(out_dir.glob(f"{track}_*.wav")))
    if existing > 0:
        print(f"  已有 {existing} 个片段，从 {existing:06d} 续编")

    # 为每个文件分配起始 idx（串行预分配以保证连续）
    # 先快速统计每个文件预计产生多少片段（按时长估算）
    task_args = []
    cur_idx = existing
    for f in files:
        try:
            info = sf.info(str(f))
            n_est = max(1, int((info.duration - min_sec) / hop_sec) + 1)
        except Exception:
            n_est = 1
        task_args.append((f, out_dir, track, cur_idx,
                          chunk_samples, hop_samples, min_samples))
        cur_idx += n_est

    # 并行处理
    all_rows = []
    with mp.Pool(workers) as pool:
        for rows in tqdm(pool.imap_unordered(chunk_one_file, task_args),
                         total=len(task_args), desc=track, unit="file"):
            all_rows.extend(rows)

    print(f"  → 生成片段: {len(all_rows)}")
    return all_rows


# ─── 主入口 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="岁月织机 — 数据集构建")
    parser.add_argument("--track", "-t",
                        choices=list(TRACK_SOURCES.keys()) + ["all"],
                        default="all")
    parser.add_argument("--chunk-sec",  type=float, default=CHUNK_SEC)
    parser.add_argument("--hop-sec",    type=float, default=HOP_SEC)
    parser.add_argument("--min-sec",    type=float, default=MIN_SEC)
    parser.add_argument("--workers",    type=int,
                        default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

    tracks = list(TRACK_SOURCES.keys()) if args.track == "all" else [args.track]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  岁月织机 — 数据集构建器")
    print(f"  片段长度: {args.chunk_sec}s  步长: {args.hop_sec}s  "
          f"最短: {args.min_sec}s")
    print(f"  并行工作进程: {args.workers}")
    print(f"  输出目录: {OUT_ROOT}")
    print("=" * 60)

    t0 = time.time()
    all_manifest_rows = []

    for track in tracks:
        rows = build_track(track, args.chunk_sec, args.hop_sec,
                           args.min_sec, args.workers)
        all_manifest_rows.extend(rows)

    # 写 manifest（追加模式，重复运行不覆盖旧记录）
    write_header = not MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["path", "source", "start_sec", "duration_sec"])
        writer.writerows(all_manifest_rows)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  完成！共生成 {len(all_manifest_rows)} 个片段")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  manifest: {MANIFEST_PATH}")

    # 最终统计
    print(f"\n  各轨道片段数：")
    for track in tracks:
        d = OUT_ROOT / track
        if d.exists():
            n = len(list(d.glob(f"{track}_*.wav")))
            print(f"    {track:10s}: {n:6d} 片段")
    print("=" * 60)


if __name__ == "__main__":
    main()
