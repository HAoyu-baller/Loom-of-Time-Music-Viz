"""
本地快速测试脚本 — 用真实民乐 MP3 测试分离效果
用法：
    cd training/
    python test_local.py

输出：test_output/ 目录下生成 5 个分离 WAV 文件
"""

import pathlib
import numpy as np
import soundfile as sf
import torch
from bs_roformer import BSRoformer

# ── 配置 ──────────────────────────────────────────────────────────────────────
STEMS        = ["erhu", "wind", "plucked", "perc", "vocal"]
SR           = 22050
CHUNK        = 132300    # 6s 推理块
OVERLAP      = 22050     # 1s 重叠
CKPT_PATH    = pathlib.Path("checkpoints/best.pt")
INPUT_MP3    = pathlib.Path("../VipSongsDownload/宋祖英 - 苗岭飞歌.mp3")
OUT_DIR      = pathlib.Path("test_output")

# 必须与训练时一致
MODEL_DIM    = 256
MODEL_DEPTH  = 6

# ── 设备 ──────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"设备: {device}")

# ── 加载音频 ──────────────────────────────────────────────────────────────────
print(f"\n[1/3] 读取音频: {INPUT_MP3.name}")
try:
    import librosa
    audio, _ = librosa.load(str(INPUT_MP3), sr=SR, mono=True)
except ImportError:
    # fallback: soundfile + soxr
    import soxr
    audio, file_sr = sf.read(str(INPUT_MP3), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if file_sr != SR:
        audio = soxr.resample(audio, file_sr, SR)

audio = audio.astype(np.float32)
duration = len(audio) / SR
print(f"  时长: {duration:.1f}s  采样数: {len(audio):,}")

OUT_DIR.mkdir(exist_ok=True)
sf.write(str(OUT_DIR / "input.wav"), audio, SR)

# ── 加载模型 ──────────────────────────────────────────────────────────────────
print(f"\n[2/3] 加载模型: {CKPT_PATH}")
if not CKPT_PATH.exists():
    raise FileNotFoundError(
        f"找不到 checkpoint: {CKPT_PATH}\n"
        "请先从 HPC 下载 best.pt 到 training/checkpoints/"
    )

ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

model = BSRoformer(
    dim                    = MODEL_DIM,
    depth                  = MODEL_DEPTH,
    heads                  = 8,
    dim_head               = 32,
    num_stems              = len(STEMS),
    time_transformer_depth = 2,
    freq_transformer_depth = 1,
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

epoch    = ckpt.get("epoch", "?")
val_loss = ckpt.get("val_loss", float("nan"))
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  epoch={epoch}  val_loss={val_loss:.4f}  参数量={n_params:.1f}M")

# ── 分块推理 ──────────────────────────────────────────────────────────────────
print(f"\n[3/3] 分离中 (chunk={CHUNK/SR:.0f}s, overlap={OVERLAP/SR:.0f}s)...")

hop       = CHUNK - OVERLAP
length    = len(audio)
out_sum   = {s: np.zeros(length, dtype=np.float64) for s in STEMS}
out_count = np.zeros(length, dtype=np.float64)
window    = np.hanning(CHUNK).astype(np.float64)

starts = list(range(0, length, hop))
for idx, start in enumerate(starts):
    end         = start + CHUNK
    chunk_audio = audio[start:end]
    if len(chunk_audio) < CHUNK:
        chunk_audio = np.pad(chunk_audio, (0, CHUNK - len(chunk_audio)))

    x = torch.from_numpy(chunk_audio).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)   # [1, num_stems, 1, T_pred]

    pred_len = pred.shape[-1]
    use_len  = min(CHUNK, pred_len, length - start)

    for i, stem in enumerate(STEMS):
        stem_np = pred[0, i, 0, :use_len].cpu().numpy().astype(np.float64)
        out_sum[stem][start:start + use_len] += stem_np * window[:use_len]
    out_count[start:start + use_len] += window[:use_len]

    print(f"  块 {idx+1}/{len(starts)}  {start/SR:.1f}s-{min(end,length)/SR:.1f}s", end="\r")

print()

safe_count = np.where(out_count > 1e-8, out_count, 1.0)

# ── 保存结果 ──────────────────────────────────────────────────────────────────
print("\n─── 分离结果 ───────────────────────────────────────────────")
print(f"{'Stem':10s}  {'RMS':8s}  {'文件'}")
print("─" * 45)

for stem in STEMS:
    result = (out_sum[stem] / safe_count).astype(np.float32)
    out_path = OUT_DIR / f"{stem}.wav"
    sf.write(str(out_path), result, SR)
    rms = float(np.sqrt(np.mean(result ** 2)))
    print(f"{stem:10s}  {rms:.4f}    {out_path.name}")

print("─" * 45)
print(f"\n完成！文件保存至: {OUT_DIR.resolve()}/")
print("\n用 QuickTime 打开对比：")
print(f"  原始混合: test_output/input.wav")
for s in STEMS:
    print(f"  {s}: test_output/{s}.wav")
