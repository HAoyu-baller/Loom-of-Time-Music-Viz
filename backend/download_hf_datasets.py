"""
岁月织机 — HuggingFace 数据集下载脚本
======================================
下载以下三个数据集并整理到 music/instruments_clean/ 对应目录：

  1. ccmusic-database/CTIS        → 按乐器分类分发到各轨道目录
  2. ccmusic-database/erhu_playing_tech → erhu/
  3. ccmusic-database/Guzheng_Tech99    → plucked/

CTIS 乐器 → 轨道 映射规则（基于 cname/pinyin）：
  erhu    ← 二胡、高胡、板胡、京胡、坠胡、马头琴 等拉弦
  wind    ← 竹笛、梆笛、曲笛、唢呐、笙、箫、埙、巴乌 等吹管
  plucked ← 琵琶、古筝、古琴、柳琴、扬琴、三弦、阮 等弹拨
  perc    ← 大鼓、排鼓、板鼓、锣、铙钹、木鱼 等打击
  （其余乐器跳过）

用法：
  python download_hf_datasets.py              # 下载全部三个
  python download_hf_datasets.py --dataset ctis
  python download_hf_datasets.py --dataset erhu_tech
  python download_hf_datasets.py --dataset guzheng99
"""

import argparse
import pathlib
import sys
import numpy as np
import soundfile as sf

PROJECT_ROOT = pathlib.Path(__file__).parent
OUT_ROOT     = PROJECT_ROOT / "music" / "instruments_clean"

TARGET_SR    = 22050
NORM_DBFS    = -1.0

# ─── CTIS 乐器分类映射 ────────────────────────────────────────────────────────
# key: 目标轨道名；value: cname 中包含的关键词列表（任一匹配即归入该轨道）
CTIS_MAPPING = {
    "erhu": [
        "二胡", "高胡", "板胡", "京胡", "坠胡", "坠琴",
        "马头琴", "四胡", "擂琴", "中胡", "低胡",
        "革胡", "大提", "低音拉", "胡琴",
    ],
    "wind": [
        "竹笛", "梆笛", "曲笛", "笛子", "笛",
        "唢呐", "海笛", "喇叭",
        "笙", "排笙", "葫芦笙",
        "箫", "洞箫", "尺八",
        "埙", "巴乌", "葫芦丝",
        "管子", "双管", "喉管",
        "口笛", "竖笛",
    ],
    "plucked": [
        "琵琶", "南琵琶", "柳琴",
        "古筝", "筝",
        "古琴", "七弦琴",
        "扬琴",
        "三弦", "四弦",
        "阮", "中阮", "大阮", "小阮",
        "月琴", "秦琴",
        "冬不拉", "热瓦普", "都塔尔",
        "弦子", "马骨胡", "天琴",
    ],
    "perc": [
        "大鼓", "堂鼓", "排鼓", "板鼓", "腰鼓",
        "手鼓", "铃鼓", "定音鼓",
        "锣", "大锣", "小锣",
        "钹", "铙", "铙钹",
        "木鱼", "梆子", "板",
        "云锣", "编钟", "磬",
        "沙锤", "响板",
    ],
}

def classify_ctis(cname: str) -> str | None:
    """根据中文乐器名判断归属轨道，返回轨道名或 None（跳过）"""
    for track, keywords in CTIS_MAPPING.items():
        for kw in keywords:
            if kw in cname:
                return track
    return None


# ─── 音频处理工具 ─────────────────────────────────────────────────────────────
def process_audio(array: np.ndarray, sr: int) -> np.ndarray | None:
    """重采样 → 单声道 → 峰值归一化"""
    import librosa

    # 确保 float32
    audio = array.astype(np.float32)

    # 多声道 → 单声道
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # 重采样
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq")

    # 峰值归一化
    peak = np.abs(audio).max()
    if peak < 1e-8:
        return None
    target_linear = 10 ** (NORM_DBFS / 20.0)
    audio = np.clip(audio * (target_linear / peak), -1.0, 1.0)
    return audio


def save_wav(audio: np.ndarray, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, samplerate=TARGET_SR, subtype="PCM_16")


# ─── 数据集下载函数 ───────────────────────────────────────────────────────────

def download_ctis():
    """下载 CTIS，按乐器分类存入各轨道目录"""
    from datasets import load_dataset
    from tqdm import tqdm

    print("\n" + "="*60)
    print("  [1/3] 下载 CTIS（中国传统乐器音库）")
    print("="*60)

    ds = load_dataset(
        "ccmusic-database/CTIS",
        name="default", split="train",
        trust_remote_code=True,
    )
    print(f"  共 {len(ds)} 条记录")

    # 统计各轨道分配情况
    track_counts = {t: 0 for t in CTIS_MAPPING}
    skip_count   = 0
    fail_count   = 0

    for i, sample in enumerate(tqdm(ds, desc="CTIS", unit="clip")):
        cname  = sample["cname"]
        pinyin = sample["pinyin"]
        track  = classify_ctis(cname)

        if track is None:
            skip_count += 1
            continue

        audio_data = sample["audio"]
        array = np.array(audio_data["array"], dtype=np.float32)
        sr    = audio_data["sampling_rate"]

        processed = process_audio(array, sr)
        if processed is None:
            fail_count += 1
            continue

        # 文件名：拼音_序号.wav
        out_dir  = OUT_ROOT / track / f"ctis_{pinyin}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # 统计该目录已有多少文件，以续编号
        existing = len(list(out_dir.glob("*.wav")))
        out_path = out_dir / f"{existing:04d}.wav"

        save_wav(processed, out_path)
        track_counts[track] += 1

    print(f"\n  CTIS 完成：")
    for track, n in track_counts.items():
        print(f"    {track:10s}: {n} 条")
    print(f"    跳过（非目标乐器）: {skip_count} 条")
    print(f"    失败: {fail_count} 条")


def download_erhu_tech():
    """下载二胡演奏技法数据集 → erhu/"""
    from datasets import load_dataset
    from tqdm import tqdm

    print("\n" + "="*60)
    print("  [2/3] 下载 erhu_playing_tech（二胡演奏技法）")
    print("="*60)

    ds = load_dataset(
        "ccmusic-database/erhu_playing_tech",
        name="default", split="train",
        trust_remote_code=True,
    )
    print(f"  共 {len(ds)} 条记录")

    ok, fail = 0, 0
    out_base = OUT_ROOT / "erhu" / "erhu_tech"
    out_base.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(tqdm(ds, desc="erhu_tech", unit="clip")):
        audio_data = sample["audio"]
        array = np.array(audio_data["array"], dtype=np.float32)
        sr    = audio_data["sampling_rate"]

        # 按技法分子目录
        tech_label = str(sample.get("label", sample.get("tech", i)))
        sub_dir = out_base / str(tech_label)
        sub_dir.mkdir(parents=True, exist_ok=True)

        processed = process_audio(array, sr)
        if processed is None:
            fail += 1
            continue

        existing  = len(list(sub_dir.glob("*.wav")))
        out_path  = sub_dir / f"{existing:04d}.wav"
        save_wav(processed, out_path)
        ok += 1

    print(f"  erhu_playing_tech 完成：成功 {ok} | 失败 {fail}")


def download_guzheng99():
    """下载古筝99曲数据集 → plucked/"""
    from datasets import load_dataset
    from tqdm import tqdm

    print("\n" + "="*60)
    print("  [3/3] 下载 Guzheng_Tech99（古筝独奏曲99首）")
    print("="*60)

    ds = load_dataset(
        "ccmusic-database/Guzheng_Tech99",
        name="default", split="train",
        trust_remote_code=True,
    )
    print(f"  共 {len(ds)} 条记录")

    ok, fail = 0, 0
    out_base = OUT_ROOT / "plucked" / "guzheng_tech99"
    out_base.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(tqdm(ds, desc="Guzheng99", unit="clip")):
        audio_data = sample["audio"]
        array = np.array(audio_data["array"], dtype=np.float32)
        sr    = audio_data["sampling_rate"]

        processed = process_audio(array, sr)
        if processed is None:
            fail += 1
            continue

        out_path = out_base / f"{i:04d}.wav"
        save_wav(processed, out_path)
        ok += 1

    print(f"  Guzheng_Tech99 完成：成功 {ok} | 失败 {fail}")


# ─── 主入口 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="岁月织机 — HuggingFace 数据集下载")
    parser.add_argument(
        "--dataset", "-d",
        choices=["ctis", "erhu_tech", "guzheng99", "all"],
        default="all",
        help="指定下载哪个数据集（默认 all）",
    )
    args = parser.parse_args()

    print("="*60)
    print("  岁月织机 — HuggingFace 数据集下载器")
    print(f"  输出目录：{OUT_ROOT}")
    print("="*60)

    if args.dataset in ("ctis", "all"):
        download_ctis()
    if args.dataset in ("erhu_tech", "all"):
        download_erhu_tech()
    if args.dataset in ("guzheng99", "all"):
        download_guzheng99()

    print("\n" + "="*60)
    print("  全部下载完成！")

    # 打印最终各目录文件数
    for track in ["erhu", "wind", "plucked", "perc"]:
        d = OUT_ROOT / track
        if d.exists():
            n = sum(1 for _ in d.rglob("*.wav"))
            print(f"  {track:10s}: {n} 个 WAV")
    print("="*60)


if __name__ == "__main__":
    main()
