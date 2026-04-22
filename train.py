"""
岁月织机 — 训练脚本
====================
使用 bs-roformer (num_stems=5) + auraloss 多分辨率 STFT Loss
在 DynamicMixDataset 上训练中国民乐音源分离模型。

依赖：
    pip install bs-roformer auraloss torch torchaudio tensorboard

用法：
    python train.py                          # 默认配置
    python train.py --epochs 100 --batch 16  # 自定义参数
    python train.py --resume checkpoints/last.pt  # 断点续训
"""

import argparse
import pathlib
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import STEMS, make_loaders

# ─── 默认超参 ──────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    dataset_root = "music/dataset",
    ckpt_dir     = "checkpoints",
    log_dir      = "runs",
    epochs       = 100,
    batch_size   = 8,
    lr           = 3e-4,
    weight_decay = 1e-2,
    val_ratio    = 0.05,
    num_workers  = 4,
    grad_clip    = 5.0,
    save_every   = 5,       # 每 N epoch 保存一次 checkpoint
    sr           = 22050,
)


# ─── 模型构建 ──────────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> nn.Module:
    """
    构建 BS-RoFormer，num_stems=5 对应 erhu/wind/plucked/perc/vocal。
    参数规模约 15-20M，适合单 GPU 训练。
    """
    from bs_roformer import BSRoformer  # pip install bs-roformer

    model = BSRoformer(
        dim                    = 256,      # 主干宽度
        depth                  = 6,        # Time-Transformer 层数
        heads                  = 8,
        dim_head               = 32,
        num_stems              = len(STEMS),  # 5
        time_transformer_depth = 2,
        freq_transformer_depth = 1,
        # freqs_per_bands 使用库默认值（与 stft_n_fft=2048 对应，总和=1025）
    )
    return model.to(device)


# ─── 损失函数 ──────────────────────────────────────────────────────────────────

def build_loss():
    """
    多分辨率 STFT Loss，三个尺度覆盖瞬态精度到音高精度。
    不使用 mel 缩放（避免 auraloss 内部尺寸不对齐问题），
    用对数幅度谱代替，效果等价。
    """
    from auraloss.freq import MultiResolutionSTFTLoss
    return MultiResolutionSTFTLoss(
        fft_sizes   = [512, 2048, 8192],
        hop_sizes   = [128,  512, 2048],
        win_lengths = [512, 2048, 8192],
    )


# ─── 训练工具 ──────────────────────────────────────────────────────────────────

def compute_loss(loss_fn, pred_stems, true_stems, active, device):
    """
    对每个激活的 stem 计算 STFT loss，取平均。
    静音 stem（active=False）跳过，避免惩罚模型正确输出零。
    """
    total, count = torch.tensor(0.0, device=device), 0
    for i, stem in enumerate(STEMS):
        mask = torch.tensor(active[stem], dtype=torch.bool, device=device)
        if not mask.any():
            continue
        # pred_stems: [B, num_stems, 1, T_pred]（BSRoformer STFT padding 导致 T_pred <= T）
        pred = pred_stems[:, i].squeeze(1)[mask].unsqueeze(1)  # [N, 1, T_pred]
        true = true_stems[stem][mask]                          # [N, 1, T]
        # trim 到相同长度（取较短的一侧）
        t = min(pred.shape[-1], true.shape[-1])
        pred, true = pred[..., :t], true[..., :t]
        # auraloss MultiResolutionSTFTLoss 期望 [B, C, T]
        total = total + loss_fn(pred, true)
        count += 1
    return total / max(count, 1)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "loss":       loss,
        "stems":      STEMS,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["loss"]


# ─── 主训练循环 ────────────────────────────────────────────────────────────────

def train(args):
    # 设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    # 目录
    ckpt_dir = pathlib.Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir  = pathlib.Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 数据
    print("加载数据集...")
    train_loader, val_loader = make_loaders(
        dataset_root = args.dataset_root,
        batch_size   = args.batch_size,
        val_ratio    = args.val_ratio,
        num_workers  = args.num_workers,
    )
    print(f"  训练集: {len(train_loader.dataset)} 样本  "
          f"({len(train_loader)} batches)")
    print(f"  验证集: {len(val_loader.dataset)} 样本  "
          f"({len(val_loader)} batches)")

    # 模型 / 损失 / 优化器
    print("构建模型...")
    model     = build_model(device)
    loss_fn   = build_loss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  参数量: {n_params:.1f}M")

    # 断点续训
    start_epoch = 0
    if args.resume:
        print(f"恢复训练：{args.resume}")
        start_epoch, _ = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        start_epoch += 1

    writer = SummaryWriter(log_dir=str(log_dir))

    # ── Epoch 循环 ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for step, (mixture, stems_true, active) in enumerate(train_loader):
            mixture = mixture.to(device)                      # [B, 1, T]
            stems_true = {s: v.to(device) for s, v in stems_true.items()}

            # BS-RoFormer 期望输入 [B, T]（无通道维），输出 [B, num_stems, T]
            pred = model(mixture.squeeze(1))                  # [B, 5, T]

            loss = compute_loss(loss_fn, pred, stems_true, active, device)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item()

            if step % 50 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch:03d}  step {step:04d}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  lr={lr_now:.2e}")

        scheduler.step()
        avg_train = train_loss / len(train_loader)

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mixture, stems_true, active in val_loader:
                mixture    = mixture.to(device)
                stems_true = {s: v.to(device) for s, v in stems_true.items()}
                pred       = model(mixture.squeeze(1))
                loss       = compute_loss(loss_fn, pred, stems_true, active, device)
                val_loss  += loss.item()
        avg_val = val_loss / len(val_loader)

        elapsed = time.time() - t_epoch
        print(f"\nEpoch {epoch:03d}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  "
              f"time={elapsed:.0f}s\n")

        # TensorBoard
        writer.add_scalar("loss/train", avg_train, epoch)
        writer.add_scalar("loss/val",   avg_val,   epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, avg_val,
                        ckpt_dir / "last.pt")
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, avg_val,
                            ckpt_dir / f"epoch_{epoch:03d}.pt")

    writer.close()
    print("训练完成！")


# ─── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="岁月织机 — 训练")
    parser.add_argument("--dataset-root", default=DEFAULTS["dataset_root"])
    parser.add_argument("--ckpt-dir",     default=DEFAULTS["ckpt_dir"])
    parser.add_argument("--log-dir",      default=DEFAULTS["log_dir"])
    parser.add_argument("--epochs",       type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size",   type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--lr",           type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--val-ratio",    type=float, default=DEFAULTS["val_ratio"])
    parser.add_argument("--num-workers",  type=int,   default=DEFAULTS["num_workers"])
    parser.add_argument("--grad-clip",    type=float, default=DEFAULTS["grad_clip"])
    parser.add_argument("--save-every",   type=int,   default=DEFAULTS["save_every"])
    parser.add_argument("--resume",       type=str,   default=None,
                        help="checkpoint 路径，用于断点续训")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
