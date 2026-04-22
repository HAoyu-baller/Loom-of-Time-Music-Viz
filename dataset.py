"""
岁月织机 — DynamicMixDataset
==============================
每次 __getitem__ 从五个轨道各随机取一个 6s 片段，
实时叠加成混合音频，同时返回各干净 stem。

Silence Dropout：每个轨道以 p=0.3 的概率被静音，
模拟真实混音中乐器不总是同时出现的情况。
vocal 轨道静音概率提高到 p=0.5（民乐中纯器乐段落多）。

返回：
    mixture : Tensor [1, T]        # 混合波形（mono）
    stems   : dict[str, Tensor]    # 各轨道干净波形 [1, T]
    active  : dict[str, bool]      # 该轨道是否被静音

用法：
    from dataset import DynamicMixDataset, STEMS
    ds = DynamicMixDataset("music/dataset")
    loader = DataLoader(ds, batch_size=8, num_workers=4, shuffle=True)
    mixture, stems, active = next(iter(loader))
"""

import pathlib
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

# ─── 常量 ────────────────────────────────────────────────────────────────────
STEMS   = ["erhu", "wind", "plucked", "perc", "vocal"]
SR      = 22050
SAMPLES = 132300          # 6s × 22050
MIX_GAIN_DB_RANGE = (-3, 3)   # 混音前对每个 stem 随机施加增益，单位 dB

SILENCE_PROB = {
    "erhu":    0.30,
    "wind":    0.30,
    "plucked": 0.30,
    "perc":    0.40,   # 打击乐在民乐中有时缺席
    "vocal":   0.50,   # 纯器乐段落多
}


class DynamicMixDataset(Dataset):
    """
    动态混音数据集。

    Args:
        dataset_root : music/dataset/ 目录路径
        stems        : 使用的轨道列表，默认全部5个
        silence_prob : 各轨道静音概率字典（覆盖默认值）
        augment      : 是否启用增益抖动增强
    """

    def __init__(
        self,
        dataset_root: str | pathlib.Path,
        stems: list[str] = None,
        silence_prob: dict[str, float] = None,
        augment: bool = True,
    ):
        self.root    = pathlib.Path(dataset_root)
        self.stems   = stems or STEMS
        self.augment = augment
        self.sil_prob = {**SILENCE_PROB, **(silence_prob or {})}

        # 每个轨道的文件路径列表
        self.pools: dict[str, list[pathlib.Path]] = {}
        for stem in self.stems:
            files = sorted((self.root / stem).glob("*.wav"))
            if not files:
                raise FileNotFoundError(
                    f"未找到轨道 '{stem}' 的 WAV 文件，路径：{self.root / stem}"
                )
            self.pools[stem] = files

        # dataset 长度 = 最大轨道的文件数（其他轨道循环取样）
        self._len = max(len(v) for v in self.pools.values())

    def __len__(self) -> int:
        return self._len

    def _load(self, path: pathlib.Path) -> np.ndarray:
        """读取 WAV，返回 float32 ndarray [T]，长度固定为 SAMPLES。"""
        audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
        # 理论上都是 SAMPLES，但防御性处理一下
        if len(audio) < SAMPLES:
            audio = np.pad(audio, (0, SAMPLES - len(audio)))
        elif len(audio) > SAMPLES:
            audio = audio[:SAMPLES]
        return audio

    def __getitem__(self, idx: int):
        stems_audio: dict[str, np.ndarray] = {}
        active: dict[str, bool] = {}

        for stem in self.stems:
            pool  = self.pools[stem]
            # 用 idx % len(pool) 保证小轨道也能均匀覆盖
            fpath = pool[idx % len(pool)]
            audio = self._load(fpath)

            # Silence Dropout
            if random.random() < self.sil_prob[stem]:
                audio  = np.zeros(SAMPLES, dtype=np.float32)
                active[stem] = False
            else:
                active[stem] = True

            # 增益抖动（±3dB）
            if self.augment and active[stem]:
                gain_db  = random.uniform(*MIX_GAIN_DB_RANGE)
                audio   *= 10 ** (gain_db / 20.0)

            stems_audio[stem] = audio

        # 混合
        mixture = sum(stems_audio.values())   # ndarray [T]

        # 峰值限幅，防止混合爆音
        peak = np.abs(mixture).max()
        if peak > 1.0:
            scale   = 1.0 / peak
            mixture = mixture * scale
            for stem in self.stems:
                stems_audio[stem] = stems_audio[stem] * scale

        # ndarray → Tensor [1, T]
        mixture_t = torch.from_numpy(mixture).unsqueeze(0)
        stems_t   = {s: torch.from_numpy(a).unsqueeze(0)
                     for s, a in stems_audio.items()}

        return mixture_t, stems_t, active


# ─── 工厂函数 ─────────────────────────────────────────────────────────────────

def make_loaders(
    dataset_root: str | pathlib.Path,
    batch_size: int  = 8,
    val_ratio: float = 0.05,
    num_workers: int = 4,
    seed: int        = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    按文件索引切分训练集 / 验证集，返回两个 DataLoader。

    验证集取每个轨道最后 val_ratio 比例的文件（不打乱，可复现）。
    训练集取其余文件，每个 epoch 打乱。
    """
    full_ds = DynamicMixDataset(dataset_root, augment=True)
    total   = len(full_ds)
    n_val   = max(1, int(total * val_ratio))
    n_train = total - n_val

    # 固定随机种子切分
    rng      = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=rng
    )

    # 验证集用独立实例，关闭增强（不能直接改 full_ds，否则训练集也受影响）
    val_ds_clean = DynamicMixDataset(dataset_root, augment=False)
    val_indices  = val_ds.indices   # type: ignore
    val_ds       = torch.utils.data.Subset(val_ds_clean, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
        persistent_workers = num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
        persistent_workers = num_workers > 0,
    )
    return train_loader, val_loader


# ─── 快速自测 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    root = pathlib.Path(__file__).parent / "music" / "dataset"
    ds   = DynamicMixDataset(root)

    print(f"Dataset 长度: {len(ds)}")
    print(f"轨道: {ds.stems}")
    print(f"各轨道文件数: { {s: len(ds.pools[s]) for s in ds.stems} }")

    # 取一个样本
    t0 = time.perf_counter()
    mixture, stems, active = ds[0]
    elapsed = time.perf_counter() - t0

    print(f"\n单样本读取耗时: {elapsed*1000:.1f} ms")
    print(f"mixture shape : {mixture.shape}  dtype={mixture.dtype}")
    print(f"mixture range : [{mixture.min():.3f}, {mixture.max():.3f}]")
    print(f"\nStems:")
    for stem in ds.stems:
        t = stems[stem]
        print(f"  {stem:10s}: shape={t.shape}  "
              f"rms={t.pow(2).mean().sqrt():.4f}  "
              f"active={active[stem]}")

    # DataLoader 速度测试
    print("\nDataLoader 速度测试 (batch=8, workers=4, 10 batches)...")
    loader = DataLoader(ds, batch_size=8, num_workers=4, shuffle=True,
                        drop_last=True)
    t0 = time.perf_counter()
    for i, (mix, stms, act) in enumerate(loader):
        if i >= 9:
            break
    elapsed = time.perf_counter() - t0
    print(f"10 batches 耗时: {elapsed:.2f}s  ({elapsed/10*1000:.0f} ms/batch)")
    print(f"mixture batch shape: {mix.shape}")
