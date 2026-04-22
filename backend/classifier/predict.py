"""
岁月织机 (Loom of Time) — 推理脚本
====================================
用法：
  python predict.py 你的歌曲.mp3
  python predict.py 你的歌曲.mp3 --plot   # 同时显示置信度图表

流程：MP3 → 5秒切片 → 梅尔频谱图 → 模型预测 → 投票 → 输出地区
"""

import sys
import pathlib
import argparse
import tempfile
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

# ── 与训练时完全一致的配置 ──────────────────────────────────
SAMPLE_RATE   = 22050
CLIP_DURATION = 5
CLIP_SAMPLES  = SAMPLE_RATE * CLIP_DURATION
N_MELS        = 128
HOP_LENGTH    = 512
N_FFT         = 2048
IMG_PX        = 224

CLASSES   = ['huanan', 'shanxi', 'yungui', 'jiangnan', 'dongbei']
CLASS_ZH  = {
    'huanan':   '华南（粤桂琼闽）',
    'shanxi':   '陕西（陕北关中）',
    'yungui':   '云贵（云南贵州）',
    'jiangnan': '江南（江浙沪皖）',
    'dongbei':  '东北（黑吉辽）',
}
CKPT_PATH = pathlib.Path(__file__).parent / 'models' / 'loom_best.pth'

# ── 设备 ────────────────────────────────────────────────────
DEVICE = (torch.device('mps')  if torch.backends.mps.is_available()  else
          torch.device('cuda') if torch.cuda.is_available() else
          torch.device('cpu'))

# ── 图像预处理（与验证集 transform 完全一致）────────────────
transform = T.Compose([
    T.Resize((IMG_PX, IMG_PX)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])


# ── 模型加载 ─────────────────────────────────────────────────
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(True),
        nn.Dropout(0.3),
        nn.Linear(256, len(CLASSES)),
    )
    model.load_state_dict(torch.load(str(CKPT_PATH), map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


# ── 音频片段 → PNG（内存中，不写磁盘）──────────────────────
def clip_to_tensor(audio_clip: np.ndarray) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=audio_clip, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 生成纯净频谱图到内存 buffer
    fig_inch = IMG_PX / 100.0
    fig = plt.figure(figsize=(fig_inch, fig_inch), dpi=100)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
    ax.axis('off')

    # 保存到临时文件再读回 PIL（最稳定的方式）
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    plt.savefig(tmp_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)

    img = Image.open(tmp_path).convert('RGB')
    pathlib.Path(tmp_path).unlink()
    return transform(img)


# ── 核心推理函数 ─────────────────────────────────────────────
def predict(mp3_path: str, verbose: bool = True) -> dict:
    """
    输入 MP3 路径，返回：
    {
        'region':     'jiangnan',           # 预测地区（英文）
        'region_zh':  '江南（江浙沪皖）',    # 预测地区（中文）
        'confidence': 0.73,                 # 投票后的归一化置信度
        'votes':      {'huanan':2, ...},    # 各类别得票数
        'probs':      {'huanan':0.12, ...}, # 平均 softmax 概率
        'n_clips':    12,                   # 有效片段数
    }
    """
    if verbose:
        print(f'\n{"─"*50}')
        print(f'  岁月织机 推理引擎')
        print(f'{"─"*50}')
        print(f'  文件：{pathlib.Path(mp3_path).name}')

    # 1. 加载音频
    audio, _ = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True)
    duration  = len(audio) / SAMPLE_RATE
    n_clips   = len(audio) // CLIP_SAMPLES

    if verbose:
        print(f'  时长：{duration:.1f} 秒  →  {n_clips} 个 {CLIP_DURATION}s 片段')

    if n_clips == 0:
        raise ValueError(f'音频时长不足 {CLIP_DURATION} 秒，无法分析')

    # 2. 逐片段推理，累加 softmax 概率
    model      = load_model()
    sum_probs  = np.zeros(len(CLASSES))
    vote_count = np.zeros(len(CLASSES), dtype=int)

    for i in range(n_clips):
        clip   = audio[i * CLIP_SAMPLES : (i + 1) * CLIP_SAMPLES]
        tensor = clip_to_tensor(clip).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0].cpu().numpy()

        sum_probs           += probs
        vote_count[probs.argmax()] += 1

        if verbose:
            top = CLASSES[probs.argmax()]
            print(f'  片段 {i+1:2d}/{n_clips}  →  '
                  f'{CLASS_ZH[top]}  ({probs.max()*100:.1f}%)')

    # 3. 投票决定最终类别，平均概率作为置信度参考
    avg_probs      = sum_probs / n_clips
    final_idx      = vote_count.argmax()
    final_region   = CLASSES[final_idx]
    vote_conf      = vote_count[final_idx] / n_clips   # 投票置信度

    result = {
        'region':     final_region,
        'region_zh':  CLASS_ZH[final_region],
        'confidence': float(vote_conf),
        'votes':      dict(zip(CLASSES, vote_count.tolist())),
        'probs':      dict(zip(CLASSES, avg_probs.tolist())),
        'n_clips':    n_clips,
    }

    if verbose:
        print(f'\n{"─"*50}')
        print(f'  【预测结果】{result["region_zh"]}')
        print(f'  投票置信度：{vote_conf*100:.1f}%'
              f'  （{vote_count[final_idx]}/{n_clips} 片段一致）')
        print(f'\n  各地区得票：')
        for cls, votes in sorted(result["votes"].items(),
                                  key=lambda x: -x[1]):
            bar = '█' * votes + '░' * (n_clips - votes)
            print(f'    {CLASS_ZH[cls][:8]:8s}  {bar}  {votes} 票'
                  f'  (avg {result["probs"][cls]*100:.1f}%)')
        print(f'{"─"*50}\n')

    return result


# ── 可选：输出置信度柱状图 ────────────────────────────────────
def plot_result(result: dict, mp3_path: str):
    labels   = [CLASS_ZH[c][:5] for c in CLASSES]
    votes    = [result['votes'][c] for c in CLASSES]
    probs    = [result['probs'][c] * 100 for c in CLASSES]
    colors   = ['#E74C3C' if CLASSES[i] == result['region']
                else '#95A5A6' for i in range(len(CLASSES))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'《{pathlib.Path(mp3_path).stem}》\n'
        f'预测地区：{result["region_zh"]}  '
        f'（投票置信度 {result["confidence"]*100:.1f}%）',
        fontsize=13, fontweight='bold'
    )

    ax1.bar(labels, votes, color=colors, edgecolor='white', linewidth=1.5)
    ax1.set(title='各地区得票数', ylabel='票数', ylim=(0, result['n_clips']+1))
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(votes):
        ax1.text(i, v + 0.1, str(v), ha='center', fontsize=11, fontweight='bold')

    ax2.bar(labels, probs, color=colors, edgecolor='white', linewidth=1.5)
    ax2.set(title='平均 Softmax 概率 (%)', ylabel='概率 (%)', ylim=(0, 105))
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(probs):
        ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'sans-serif']
    plt.tight_layout()

    out_path = pathlib.Path(mp3_path).with_suffix('.predict.png')
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  图表已保存：{out_path}')


# ── 主入口 ───────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='岁月织机 — 民歌地区识别')
    parser.add_argument('mp3', help='输入的 MP3 文件路径')
    parser.add_argument('--plot', action='store_true',
                        help='输出置信度柱状图')
    args = parser.parse_args()

    if not pathlib.Path(args.mp3).exists():
        print(f'[错误] 找不到文件：{args.mp3}')
        sys.exit(1)

    result = predict(args.mp3)

    if args.plot:
        matplotlib.use('TkAgg')   # 切换到交互式后端显示图表
        import matplotlib.pyplot as plt
        plot_result(result, args.mp3)
