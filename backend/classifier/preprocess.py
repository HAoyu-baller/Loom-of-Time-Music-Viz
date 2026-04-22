"""
岁月织机 (Loom of Time) — 数据预处理管线
=========================================
读取各地区文件夹中的 MP3，切片为 5 秒片段，
生成纯净梅尔频谱图，按以下结构存放：

  dataset/
  ├── huanan/
  │   ├── caiyunzhuiyue/
  │   │   ├── clip_0001.png
  │   │   ├── clip_0002.png
  │   │   └── ...
  │   └── ...
  ├── shanxi/
  ├── yungui/
  ├── jiangnan/
  └── dongbei/

用法：
  python preprocess.py            # 处理全部5个地区
  python preprocess.py --cls huanan jiangnan  # 只处理指定地区
"""

import os
import re
import pathlib
import argparse
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pypinyin import lazy_pinyin   # 歌名转拼音

# ============================================================
# 全局配置
# ============================================================
SAMPLE_RATE   = 22050
CLIP_DURATION = 5                        # 切片时长（秒）
CLIP_SAMPLES  = SAMPLE_RATE * CLIP_DURATION

N_MELS        = 128
HOP_LENGTH    = 512
N_FFT         = 2048
IMG_PX        = 224                      # 频谱图分辨率

CLASSES       = ['huanan', 'shanxi', 'yungui', 'jiangnan', 'dongbei']
SUPPORTED_EXT = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}

# 音频源文件夹：项目根目录下的各地区文件夹
PROJECT_ROOT  = pathlib.Path(__file__).parent.parent
# 输出根目录
DATASET_DIR   = pathlib.Path(__file__).parent / 'data' / 'spectrograms'


# ============================================================
# 工具函数
# ============================================================

def to_pinyin_slug(chinese: str) -> str:
    """
    将中文字符串转为拼音连写，例如：
    '彩云追月' → 'caiyunzhuiyue'
    非中文字符保留（字母数字），空格转下划线。
    """
    # 只取" - "右边的歌名部分（去掉艺术家名）
    if ' - ' in chinese:
        chinese = chinese.split(' - ', 1)[1]

    # 转拼音
    pinyin_parts = lazy_pinyin(chinese)
    slug = ''.join(pinyin_parts)

    # 只保留字母数字，其余替换为空
    slug = re.sub(r'[^a-z0-9]', '', slug.lower())
    return slug or 'unknown'


def clip_to_png(audio_clip: np.ndarray, out_path: pathlib.Path):
    """
    音频片段 → 梅尔频谱图(dB) → 纯净 PNG（无坐标轴/边框/白边）
    保存的图像是 100% 纯粹的频谱像素点阵，直接可供 CNN 读取。
    """
    mel = librosa.feature.melspectrogram(
        y=audio_clip,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig_inch = IMG_PX / 100.0
    fig = plt.figure(figsize=(fig_inch, fig_inch), dpi=100)
    ax  = fig.add_axes([0, 0, 1, 1])        # 填满整个画布，零留白
    ax.imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(str(out_path), bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)


# ============================================================
# 核心处理逻辑
# ============================================================

def preprocess_class(cls: str) -> dict:
    """
    处理单个地区文件夹下的所有音频文件。
    返回 {歌名: 片段数} 的字典。
    """
    src_dir = PROJECT_ROOT / cls
    if not src_dir.exists():
        print(f"[警告] 找不到源文件夹：{src_dir}，跳过")
        return {}

    audio_files = sorted([f for f in src_dir.iterdir()
                           if f.suffix.lower() in SUPPORTED_EXT])
    if not audio_files:
        print(f"[警告] {src_dir} 中没有支持的音频文件，跳过")
        return {}

    print(f"\n[{cls}] 发现 {len(audio_files)} 个音频文件")
    stats = {}

    for audio_path in tqdm(audio_files, desc=f"  {cls}", unit="file"):
        # 歌名 → 拼音文件夹名
        slug = to_pinyin_slug(audio_path.stem)
        out_dir = DATASET_DIR / cls / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        # 加载音频（统一重采样，单声道）
        try:
            audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"\n  [错误] 无法读取 {audio_path.name}：{e}")
            continue

        # 切片
        num_clips = len(audio) // CLIP_SAMPLES
        if num_clips == 0:
            print(f"\n  [跳过] {audio_path.name} 时长不足 {CLIP_DURATION} 秒")
            continue

        clip_count = 0
        for i in range(num_clips):
            out_png = out_dir / f"clip_{i+1:04d}.png"
            if out_png.exists():          # 增量：已存在则跳过
                clip_count += 1
                continue
            clip = audio[i * CLIP_SAMPLES : (i + 1) * CLIP_SAMPLES]
            clip_to_png(clip, out_png)
            clip_count += 1

        stats[slug] = clip_count
        tqdm.write(f"    {audio_path.name[:40]:40s} → {slug}/  ({clip_count} 张)")

    total = sum(stats.values())
    print(f"  [{cls}] 完成，共 {len(stats)} 首 / {total} 张频谱图")
    return stats


def preprocess_all(classes: list) -> dict:
    """处理全部指定地区，打印汇总"""
    all_stats = {}
    for cls in classes:
        all_stats[cls] = preprocess_class(cls)

    print("\n" + "=" * 55)
    print("预处理汇总")
    print("=" * 55)
    grand_total = 0
    for cls, songs in all_stats.items():
        n_songs = len(songs)
        n_clips = sum(songs.values())
        grand_total += n_clips
        print(f"  {cls:10s}: {n_songs:3d} 首  {n_clips:5d} 张频谱图")
    print(f"  {'合计':10s}:       {grand_total:5d} 张")
    print("=" * 55)
    print(f"\n输出目录：{DATASET_DIR.resolve()}")
    return all_stats


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='岁月织机 — 音频预处理：MP3切片 → 梅尔频谱图')
    parser.add_argument(
        '--cls', nargs='+', choices=CLASSES, default=CLASSES,
        metavar='CLASS',
        help=f'指定处理哪些地区（默认全部）。可选：{CLASSES}')
    parser.add_argument(
        '--clip-duration', type=int, default=CLIP_DURATION,
        help=f'切片时长（秒），默认 {CLIP_DURATION}')
    args = parser.parse_args()

    if args.clip_duration != CLIP_DURATION:
        CLIP_DURATION = args.clip_duration
        CLIP_SAMPLES  = SAMPLE_RATE * CLIP_DURATION
        print(f"[配置] 切片时长已设置为 {CLIP_DURATION} 秒")

    print("=" * 55)
    print("  岁月织机 (Loom of Time) — 数据预处理管线")
    print(f"  处理地区：{args.cls}")
    print("=" * 55)

    preprocess_all(args.cls)
