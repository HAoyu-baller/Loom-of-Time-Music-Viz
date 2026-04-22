"""
岁月织机 (Loom of Time) — 音频数据清洗管线
==========================================
将 music/instruments/ 下所有音频清洗为格式统一的 WAV：
  · 采样率：22050 Hz
  · 声道数：单声道 (Mono)
  · 去静音：首尾 -50 dBFS 以下的静音段
  · 峰值归一化：-1.0 dBFS
  · 输出编码：16-bit PCM WAV

用法：
  python clean_audio.py                        # 处理全部
  python clean_audio.py --instrument erhu      # 只处理某个乐器目录
  python clean_audio.py --workers 4            # 指定并行进程数（默认 CPU 核数）
"""

import argparse
import multiprocessing
import pathlib
import traceback

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ─── 全局配置 ────────────────────────────────────────────────────────────────
PROJECT_ROOT   = pathlib.Path(__file__).parent
SRC_ROOT       = PROJECT_ROOT / "music" / "instruments"
DST_ROOT       = PROJECT_ROOT / "music" / "instruments_clean"

TARGET_SR      = 22050          # 目标采样率 (Hz)
TRIM_DB        = 50             # 静音阈值（librosa 用正数，即 -50 dBFS）
NORM_DBFS      = -1.0           # 峰值归一化目标（dBFS）
SUPPORTED_EXT  = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}


# ─── 核心处理函数（每个文件独立调用，方便并行） ───────────────────────────────
def process_one(src_path: pathlib.Path) -> tuple[bool, str]:
    """
    清洗单个音频文件。
    返回 (成功与否, 错误信息或空字符串)。
    """
    # 计算输出路径：保持 SRC_ROOT 下的相对目录结构
    rel = src_path.relative_to(SRC_ROOT)
    dst_path = (DST_ROOT / rel).with_suffix(".wav")
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # 已存在则跳过（增量模式）
    if dst_path.exists():
        return True, ""

    try:
        # ── 步骤 1：读取并重采样 ──────────────────────────────────────────
        # mono=True：librosa 会自动把多声道混合为单声道（左右声道平均）
        # res_type='kaiser_best'：高质量重采样（略慢但保真度更好）
        audio, _ = librosa.load(
            str(src_path),
            sr=TARGET_SR,
            mono=True,
            res_type="soxr_hq",   # 高质量重采样，无需 resampy
        )

        # ── 步骤 2：去除首尾静音 ─────────────────────────────────────────
        # top_db=50 即 -50 dBFS 阈值；frame_length/hop_length 决定检测精度
        audio_trimmed, _ = librosa.effects.trim(
            audio,
            top_db=TRIM_DB,
            frame_length=2048,
            hop_length=512,
        )

        # 去静音后若音频为空（极端情况），跳过
        if len(audio_trimmed) == 0:
            return False, f"去静音后为空：{src_path.name}"

        # ── 步骤 3：峰值归一化 ───────────────────────────────────────────
        peak = np.abs(audio_trimmed).max()
        if peak < 1e-8:
            # 全静音文件，跳过
            return False, f"音频几乎全静音（峰值={peak:.2e}）：{src_path.name}"

        # -1.0 dBFS 对应线性幅度 = 10^(-1/20) ≈ 0.89125
        target_linear = 10 ** (NORM_DBFS / 20.0)
        audio_normalized = audio_trimmed * (target_linear / peak)

        # ── 步骤 4：导出为 16-bit PCM WAV ────────────────────────────────
        # 将 float32 [-1, 1] 安全截断到 [-1, 1]，再由 soundfile 转为 int16
        audio_clipped = np.clip(audio_normalized, -1.0, 1.0)
        sf.write(
            str(dst_path),
            audio_clipped,
            samplerate=TARGET_SR,
            subtype="PCM_16",
        )

        return True, ""

    except Exception:
        # 任何异常都捕获，绝不让进程崩溃
        err = traceback.format_exc().splitlines()[-1]
        return False, f"{src_path.name}：{err}"


# ─── 多进程包装（tqdm 不能直接在子进程里更新，所以用主进程收集结果） ────────
def _worker(args):
    """multiprocessing.Pool.imap 的包装函数（必须是顶层函数才能被 pickle）"""
    return process_one(args)


# ─── 收集待处理文件 ──────────────────────────────────────────────────────────
def collect_files(instrument: str | None) -> list[pathlib.Path]:
    """
    扫描 SRC_ROOT 下所有支持格式的音频文件。
    若指定了 instrument，只扫描对应子目录。
    """
    if instrument:
        search_root = SRC_ROOT / instrument
        if not search_root.exists():
            raise FileNotFoundError(f"找不到目录：{search_root}")
    else:
        search_root = SRC_ROOT

    files = [
        f for f in search_root.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    ]
    return sorted(files)


# ─── 主入口 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="岁月织机 — 音频数据清洗管线"
    )
    parser.add_argument(
        "--instrument", "-i",
        default=None,
        metavar="NAME",
        help="只处理指定乐器子目录，如 erhu / vocal。默认处理全部。",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        metavar="N",
        help="并行进程数，默认 CPU 核数 - 1。",
    )
    args = parser.parse_args()

    # ── 收集文件 ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  岁月织机 — 音频数据清洗管线")
    print(f"  源目录   ：{SRC_ROOT}")
    print(f"  输出目录 ：{DST_ROOT}")
    print(f"  目标采样率：{TARGET_SR} Hz  |  单声道  |  16-bit PCM")
    print(f"  静音阈值 ：-{TRIM_DB} dBFS  |  峰值归一化：{NORM_DBFS} dBFS")
    print("=" * 60)

    try:
        files = collect_files(args.instrument)
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        return

    if not files:
        print("[警告] 没有找到任何音频文件，退出。")
        return

    print(f"共找到 {len(files)} 个音频文件，使用 {args.workers} 个进程处理...\n")

    # ── 并行处理 + 进度条 ─────────────────────────────────────────────────
    ok_count   = 0
    fail_count = 0
    fail_logs  = []

    # vocal 目录有 2 万个小 WAV，多进程效果好；
    # 少量大文件（如大鼓 1 小时合集）单进程也快
    pool_size = min(args.workers, len(files))

    with multiprocessing.Pool(pool_size) as pool:
        with tqdm(total=len(files), unit="file", dynamic_ncols=True) as pbar:
            for src, (success, errmsg) in zip(
                files,
                pool.imap(_worker, files, chunksize=4)
            ):
                if success:
                    ok_count += 1
                else:
                    fail_count += 1
                    fail_logs.append(errmsg)
                    tqdm.write(f"  [跳过] {errmsg}")
                pbar.update(1)
                pbar.set_postfix(ok=ok_count, fail=fail_count)

    # ── 最终报告 ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  清洗完成！")
    print(f"  成功：{ok_count} 首  |  跳过/失败：{fail_count} 首")
    print(f"  输出目录：{DST_ROOT.resolve()}")
    if fail_logs:
        print(f"\n  失败详情（共 {len(fail_logs)} 条）：")
        for msg in fail_logs[:20]:   # 最多打印前 20 条
            print(f"    · {msg}")
        if len(fail_logs) > 20:
            print(f"    ... 还有 {len(fail_logs) - 20} 条，已省略")
    print("=" * 60)


if __name__ == "__main__":
    # macOS 多进程必须加这一行，否则在 spawn 模式下会报错
    multiprocessing.freeze_support()
    main()
