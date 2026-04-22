"""
岁月织机 — 数据集均衡筛选脚本
================================
按 RMS 能量筛选每个 stem 中波形最丰富的 N 个片段，
使各 stem 数量对齐，避免模型偏向数据量最大的 stem。

用法（在 HPC 上执行）：
    python build_dataset.py \
        --src /root/autodl-tmp/dataset \
        --dst /root/autodl-tmp/dataset_balanced \
        --per-stem 4500 \
        --min-rms 0.01 \
        --workers 8
"""

import argparse
import pathlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf
from tqdm import tqdm

STEMS = ["erhu", "wind", "plucked", "perc", "vocal"]


def rms(path: pathlib.Path) -> float:
    """读取 WAV，返回 RMS 能量。失败返回 -1。"""
    try:
        if path.stat().st_size == 0:
            return -1.0
        audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return float(np.sqrt(np.mean(audio ** 2)))
    except Exception:
        return -1.0


def scan_stem(stem_dir: pathlib.Path, min_rms: float, workers: int) -> list[tuple[float, pathlib.Path]]:
    """扫描一个 stem 目录，返回 [(rms, path)] 列表，已按 rms 降序排列。"""
    wavs = sorted(stem_dir.glob("*.wav"))
    results = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(rms, p): p for p in wavs}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"  扫描 {stem_dir.name}", ncols=80, leave=False):
            p = futures[fut]
            r = fut.result()
            if r >= min_rms:
                results.append((r, p))

    results.sort(key=lambda x: x[0], reverse=True)  # RMS 高的排前面
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",       default="/root/autodl-tmp/dataset",
                        help="原始数据集根目录")
    parser.add_argument("--dst",       default="/root/autodl-tmp/dataset_balanced",
                        help="输出目录")
    parser.add_argument("--per-stem",  type=int, default=4500,
                        help="每个 stem 保留的片段数（默认 4500，对齐 perc）")
    parser.add_argument("--min-rms",   type=float, default=0.01,
                        help="最低 RMS 阈值，低于此值视为静音片段丢弃")
    parser.add_argument("--workers",   type=int, default=8,
                        help="并行读取线程数")
    args = parser.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    per_stem = args.per_stem
    min_rms  = args.min_rms

    print(f"\n{'='*55}")
    print(f"  岁月织机 数据集均衡筛选")
    print(f"{'='*55}")
    print(f"  源目录    : {src}")
    print(f"  输出目录  : {dst}")
    print(f"  每 stem   : {per_stem} 片段")
    print(f"  最低 RMS  : {min_rms}")
    print()

    dst.mkdir(parents=True, exist_ok=True)

    total_kept = 0
    summary = []

    for stem in STEMS:
        stem_src = src / stem
        stem_dst = dst / stem
        stem_dst.mkdir(exist_ok=True)

        if not stem_src.exists():
            print(f"[警告] 找不到 {stem_src}，跳过")
            continue

        print(f"[{stem}]")
        ranked = scan_stem(stem_src, min_rms, args.workers)

        total_files = len(list(stem_src.glob("*.wav")))
        valid_files = len(ranked)
        keep_n      = min(per_stem, valid_files)

        print(f"  总文件数  : {total_files:,}")
        print(f"  有效片段  : {valid_files:,}  (RMS ≥ {min_rms})")
        print(f"  保留数量  : {keep_n:,}")
        if valid_files > 0:
            print(f"  RMS 范围  : {ranked[-1][0]:.4f} ~ {ranked[0][0]:.4f}")

        # 从 RMS 最高的里均匀抽样，避免只取最响的
        # 策略：先取前 keep_n 个（能量最丰富），再 shuffle 文件名
        selected = [p for _, p in ranked[:keep_n]]

        for p in tqdm(selected, desc=f"  复制 {stem}", ncols=80, leave=False):
            shutil.copy2(str(p), str(stem_dst / p.name))

        total_kept += keep_n
        summary.append((stem, total_files, valid_files, keep_n))
        print()

    # 汇总
    print(f"{'='*55}")
    print(f"  筛选完成")
    print(f"{'='*55}")
    print(f"  {'Stem':10s}  {'原始':>8s}  {'有效':>8s}  {'保留':>8s}")
    print(f"  {'-'*42}")
    for stem, total, valid, kept in summary:
        print(f"  {stem:10s}  {total:8,}  {valid:8,}  {kept:8,}")
    print(f"  {'-'*42}")
    print(f"  {'合计':10s}  {sum(t for _,t,_,_ in summary):8,}  "
          f"{sum(v for _,_,v,_ in summary):8,}  {total_kept:8,}")
    print(f"\n  输出目录  : {dst.resolve()}")
    print(f"  更新 notebook 中的 DATASET_ROOT 为上述路径后重新训练。")


if __name__ == "__main__":
    main()
