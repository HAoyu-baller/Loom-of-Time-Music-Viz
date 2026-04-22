"""
岁月织机 — 音源分离推理脚本
用法：
    python separate.py --input <混合音频.wav> --checkpoint checkpoints/best.pt
    python separate.py --input mix.wav  # 自动找 checkpoints/best.pt

输出：在 --output 目录（默认 separated/）下生成：
    erhu.wav / wind.wav / plucked.wav / perc.wav / vocal.wav
"""

import argparse
import pathlib

import numpy as np
import soundfile as sf
import torch
from bs_roformer import BSRoformer

# ── 常量（必须与训练时一致）────────────────────────────────────────────────────
STEMS  = ["erhu", "wind", "plucked", "perc", "vocal"]
SR     = 22050
CHUNK  = 132300        # 6s，与训练片段等长
OVERLAP = 22050        # 1s 重叠，减少拼接边界伪影


def load_checkpoint(ckpt_path: pathlib.Path, device: torch.device) -> tuple[BSRoformer, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 从 checkpoint 读取训练时的模型规模（如果有）
    # 旧版 checkpoint 可能没有 model_cfg，手动指定
    model_cfg = ckpt.get("model_cfg", {
        "dim": 64, "depth": 2, "heads": 8, "dim_head": 32,
        "time_transformer_depth": 2, "freq_transformer_depth": 1,
    })

    model = BSRoformer(
        dim                    = model_cfg["dim"],
        depth                  = model_cfg["depth"],
        heads                  = model_cfg.get("heads", 8),
        dim_head               = model_cfg.get("dim_head", 32),
        num_stems              = len(STEMS),
        time_transformer_depth = model_cfg.get("time_transformer_depth", 2),
        freq_transformer_depth = model_cfg.get("freq_transformer_depth", 1),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def load_audio(path: pathlib.Path) -> np.ndarray:
    """读取任意音频，转 mono float32，重采样到 SR。"""
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)   # stereo → mono

    if file_sr != SR:
        import soxr
        audio = soxr.resample(audio, file_sr, SR)

    return audio


def separate(
    model: BSRoformer,
    audio: np.ndarray,
    device: torch.device,
    chunk: int = CHUNK,
    overlap: int = OVERLAP,
) -> dict[str, np.ndarray]:
    """
    对任意长度音频做分块推理，重叠相加后取平均。
    返回 dict[stem_name → ndarray float32]
    """
    hop    = chunk - overlap
    length = len(audio)

    # 输出累加器
    out_sum   = {s: np.zeros(length, dtype=np.float64) for s in STEMS}
    out_count = np.zeros(length, dtype=np.float64)

    # 汉宁窗，平滑拼接边界
    window = np.hanning(chunk).astype(np.float64)

    start = 0
    while start < length:
        end   = start + chunk
        chunk_audio = audio[start:end]

        # 末尾补零
        if len(chunk_audio) < chunk:
            chunk_audio = np.pad(chunk_audio, (0, chunk - len(chunk_audio)))

        x = torch.from_numpy(chunk_audio).float().unsqueeze(0).to(device)  # [1, T]

        with torch.no_grad():
            pred = model(x)   # [1, num_stems, 1, T_pred]

        pred_len = pred.shape[-1]
        use_len  = min(chunk, pred_len, length - start)

        for i, stem in enumerate(STEMS):
            stem_np = pred[0, i, 0, :use_len].cpu().numpy().astype(np.float64)
            out_sum[stem][start:start + use_len] += stem_np * window[:use_len]

        out_count[start:start + use_len] += window[:use_len]

        start += hop
        if start >= length:
            break

    # 归一化
    safe_count = np.where(out_count > 1e-8, out_count, 1.0)
    result = {s: (out_sum[s] / safe_count).astype(np.float32) for s in STEMS}
    return result


def main():
    parser = argparse.ArgumentParser(description="岁月织机 音源分离")
    parser.add_argument("--input",      "-i", required=True,  help="输入混合音频路径")
    parser.add_argument("--checkpoint", "-c", default="checkpoints/best.pt",
                        help="模型权重路径（默认 checkpoints/best.pt）")
    parser.add_argument("--output",     "-o", default="separated",
                        help="输出目录（默认 separated/）")
    parser.add_argument("--dim",   type=int, default=None,
                        help="MODEL_DIM（若 checkpoint 不含 model_cfg 时手动指定）")
    parser.add_argument("--depth", type=int, default=None,
                        help="MODEL_DEPTH（若 checkpoint 不含 model_cfg 时手动指定）")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    args = parser.parse_args()

    # ── 设备 ──────────────────────────────────────────────────────────────────
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"设备: {device}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    ckpt_path = pathlib.Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{ckpt_path}")

    print(f"加载模型：{ckpt_path}")
    model, ckpt = load_checkpoint(ckpt_path, device)

    # 允许命令行覆盖 dim/depth（用于旧版 checkpoint）
    if args.dim is not None or args.depth is not None:
        print("  重新构建模型（使用命令行指定的 dim/depth）...")
        dim   = args.dim   or 64
        depth = args.depth or 2
        model = BSRoformer(
            dim=dim, depth=depth, heads=8, dim_head=32,
            num_stems=len(STEMS), time_transformer_depth=2, freq_transformer_depth=1,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"  checkpoint epoch={epoch}  val_loss={val_loss:.4f}")

    # ── 加载音频 ──────────────────────────────────────────────────────────────
    input_path = pathlib.Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    print(f"读取音频：{input_path}")
    audio = load_audio(input_path)
    duration = len(audio) / SR
    print(f"  时长：{duration:.1f}s  采样数：{len(audio)}")

    # ── 推理 ──────────────────────────────────────────────────────────────────
    print("开始分离...")
    stems = separate(model, audio, device)

    # ── 输出 ──────────────────────────────────────────────────────────────────
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem_name, stem_audio in stems.items():
        out_path = out_dir / f"{stem_name}.wav"
        sf.write(str(out_path), stem_audio, SR)
        rms = float(np.sqrt(np.mean(stem_audio ** 2)))
        print(f"  {stem_name:10s} → {out_path}  (RMS={rms:.4f})")

    print(f"\n完成！分离结果已保存至 {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
