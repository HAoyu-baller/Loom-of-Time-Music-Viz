"""
岁月织机 — B站音频批量下载脚本
==============================
用法：
  python download_bilibili.py --instrument erhu URL1 URL2 URL3 ...

功能：
  1. 自动检测/下载 BBDown（macOS arm64）
  2. 用 BBDown --audio-only 下载 B 站音频
  3. 用 ffmpeg 转换为 MP3（VBR ~190kbps）
  4. 输出到 music/instruments/<乐器名>/

示例：
  python download_bilibili.py --instrument erhu "https://www.bilibili.com/video/BV1xx..."
  python download_bilibili.py --instrument pipa  "https://b23.tv/xxxxx" "https://..."
"""

import argparse
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import urllib.request

# ─── 配置 ────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent
BBDOWN_PATH  = PROJECT_ROOT / "tools" / "BBDown"

# BBDown macOS arm64 release（可手动替换为最新版 URL）
BBDOWN_RELEASE_URL = (
    "https://github.com/nilaoda/BBDown/releases/download/1.6.3/"
    "BBDown_1.6.3_20240814_osx-arm64.zip"
)


# ─── BBDown 安装 ──────────────────────────────────────────────────────
def ensure_bbdown() -> pathlib.Path:
    """确保 BBDown 可执行文件存在，否则自动下载安装。"""
    # 1. 优先使用 PATH 里的
    if shutil.which("BBDown"):
        return pathlib.Path(shutil.which("BBDown"))

    # 2. 使用项目内的 tools/BBDown
    if BBDOWN_PATH.exists():
        BBDOWN_PATH.chmod(BBDOWN_PATH.stat().st_mode | stat.S_IEXEC)
        return BBDOWN_PATH

    # 3. 自动下载
    print("[BBDown] 未检测到 BBDown，开始自动下载...")
    BBDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)

    zip_path = BBDOWN_PATH.parent / "BBDown.zip"
    print(f"  下载自：{BBDOWN_RELEASE_URL}")
    try:
        urllib.request.urlretrieve(BBDOWN_RELEASE_URL, zip_path)
    except Exception as e:
        print(f"\n[错误] 下载失败：{e}")
        print("请手动下载 BBDown：https://github.com/nilaoda/BBDown/releases")
        print(f"并将可执行文件放到：{BBDOWN_PATH}")
        sys.exit(1)

    # 解压
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(BBDOWN_PATH.parent)
    zip_path.unlink()

    # 找到可执行文件
    candidates = list(BBDOWN_PATH.parent.glob("BBDown*"))
    exe = next((f for f in candidates if not f.suffix and f.is_file()), None)
    if exe is None:
        exe = next((f for f in candidates if f.is_file()), None)

    if exe is None:
        print("[错误] 解压后找不到 BBDown 可执行文件，请手动处理")
        sys.exit(1)

    if exe != BBDOWN_PATH:
        exe.rename(BBDOWN_PATH)

    BBDOWN_PATH.chmod(BBDOWN_PATH.stat().st_mode | stat.S_IEXEC)
    print(f"[BBDown] 安装完成 → {BBDOWN_PATH}")
    return BBDOWN_PATH


# ─── 工具函数 ─────────────────────────────────────────────────────────
def sanitize_filename(name: str) -> str:
    """移除文件名中的非法字符"""
    return re.sub(r'[\\/:*?"<>|]', '_', name).strip()


def find_audio_file(directory: pathlib.Path) -> pathlib.Path | None:
    """在目录中找到下载的音频文件（m4a / flac / mp3 / aac / ogg）"""
    for ext in (".m4a", ".flac", ".aac", ".mp3", ".ogg", ".wav"):
        files = sorted(directory.glob(f"*{ext}"))
        if files:
            return files[0]
    return None


def to_mp3(src: pathlib.Path, dst: pathlib.Path) -> bool:
    """用 ffmpeg 将任意音频转为 MP3（VBR ~190kbps）"""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-q:a", "2",      # VBR quality 2 ≈ 190kbps
        "-map_metadata", "0",
        str(dst)
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  [ffmpeg 错误] {result.stderr.decode()[:300]}")
        return False
    return True


# ─── 核心下载逻辑 ─────────────────────────────────────────────────────
def download_one(bbdown: pathlib.Path, url: str, out_dir: pathlib.Path) -> bool:
    """
    下载单个 B 站链接的音频，转为 MP3，存入 out_dir。
    返回 True 表示成功。
    """
    # BBDown 会在当前目录生成文件，用临时子目录隔离
    import tempfile
    with tempfile.TemporaryDirectory(dir=out_dir, prefix="bbdown_tmp_") as tmp_str:
        tmp_dir = pathlib.Path(tmp_str)

        cmd = [
            str(bbdown),
            "--audio-only",
            "--work-dir", str(tmp_dir),
            url
        ]
        print(f"\n  → BBDown 下载中：{url}")
        result = subprocess.run(cmd, capture_output=False)  # 实时输出到终端

        if result.returncode != 0:
            print(f"  [错误] BBDown 退出码 {result.returncode}")
            return False

        # 找下载产物
        audio_file = find_audio_file(tmp_dir)
        if audio_file is None:
            # 递归搜索子目录（BBDown 有时会建子文件夹）
            for f in tmp_dir.rglob("*"):
                if f.suffix.lower() in {".m4a", ".flac", ".aac", ".mp3", ".ogg", ".wav"}:
                    audio_file = f
                    break

        if audio_file is None:
            print("  [错误] 找不到下载的音频文件")
            return False

        # 目标 MP3 文件名
        stem    = sanitize_filename(audio_file.stem)
        mp3_dst = out_dir / f"{stem}.mp3"

        # 避免重名
        counter = 1
        while mp3_dst.exists():
            mp3_dst = out_dir / f"{stem}_{counter}.mp3"
            counter += 1

        if audio_file.suffix.lower() == ".mp3":
            shutil.move(str(audio_file), str(mp3_dst))
            print(f"  ✓ 已保存：{mp3_dst.name}")
        else:
            print(f"  → 转换 {audio_file.suffix} → MP3 ...")
            if to_mp3(audio_file, mp3_dst):
                print(f"  ✓ 已保存：{mp3_dst.name}")
            else:
                # ffmpeg 失败时直接移动原文件
                fallback = out_dir / audio_file.name
                shutil.move(str(audio_file), str(fallback))
                print(f"  ! ffmpeg 失败，保留原格式：{fallback.name}")
                return False

    return True


# ─── 主入口 ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="岁月织机 — B站音频批量下载（指定乐器类型）"
    )
    parser.add_argument(
        "--instrument", "-i",
        required=True,
        metavar="NAME",
        help="乐器名称（英文或拼音），如 erhu / pipa / guqin / dizi"
    )
    parser.add_argument(
        "urls",
        nargs="+",
        metavar="URL",
        help="B站视频链接，支持多个"
    )
    args = parser.parse_args()

    instrument = args.instrument.strip().lower().replace(" ", "_")
    out_dir = PROJECT_ROOT / "music" / "instruments" / instrument
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  岁月织机 — B站音频下载器")
    print(f"  乐器：{instrument}")
    print(f"  输出：{out_dir.relative_to(PROJECT_ROOT)}")
    print(f"  共 {len(args.urls)} 个链接")
    print("=" * 55)

    bbdown = ensure_bbdown()

    ok, fail = 0, 0
    for i, url in enumerate(args.urls, 1):
        print(f"\n[{i}/{len(args.urls)}] {url}")
        if download_one(bbdown, url, out_dir):
            ok += 1
        else:
            fail += 1

    print(f"\n{'─'*55}")
    print(f"完成：成功 {ok} | 失败 {fail}")
    print(f"文件位置：{out_dir}")


if __name__ == "__main__":
    main()
