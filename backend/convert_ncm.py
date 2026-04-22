"""
NCM → MP3 批量转换脚本
用法：python convert_ncm.py
会扫描 huanan / shanxi / yungui / jiangnan / dongbei 五个文件夹，
把所有 .ncm 文件解密并原地转换为同名 .mp3，转换成功后删除 .ncm。
"""

import sys, os, pathlib, shutil, subprocess

# ── 确保 site-packages 在路径里（兼容 anaconda 环境）─────────────────
SP = "/opt/anaconda3/lib/python3.12/site-packages"
if SP not in sys.path:
    sys.path.insert(0, SP)

import ncmdump  # NCM 解密核心

# ── 配置 ─────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent
FOLDERS  = ["huanan", "shanxi", "yungui", "jiangnan", "dongbei"]

def convert_ncm(ncm_path: pathlib.Path) -> bool:
    """
    解密单个 .ncm 文件。
    ncmdump 会在同目录输出同名的 .mp3 或 .flac（由文件内元数据决定）。
    返回 True 表示成功。
    """
    out_dir = ncm_path.parent
    try:
        # ncmdump 的公开 API：NCMFile(path).convert(output_dir)
        ncm_file = ncmdump.NCMFile(str(ncm_path))
        ncm_file.convert(str(out_dir))
        return True
    except Exception as e:
        print(f"  [错误] {ncm_path.name}: {e}")
        return False

def find_converted(ncm_path: pathlib.Path) -> pathlib.Path | None:
    """找到同名的 .mp3 或 .flac（ncmdump 输出文件名由元数据决定，不一定和 ncm 同名）"""
    stem = ncm_path.stem
    for ext in (".mp3", ".flac"):
        p = ncm_path.with_suffix(ext)
        if p.exists():
            return p
    return None

def ensure_mp3(converted: pathlib.Path) -> pathlib.Path:
    """如果输出是 .flac，用 ffmpeg 再转一次 mp3"""
    if converted.suffix.lower() == ".mp3":
        return converted
    mp3_path = converted.with_suffix(".mp3")
    cmd = [
        "ffmpeg", "-y", "-i", str(converted),
        "-q:a", "2",          # VBR ~190kbps
        str(mp3_path)
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0:
        converted.unlink()    # 删除 .flac 中间文件
        return mp3_path
    else:
        print(f"  [ffmpeg 错误] {result.stderr.decode()[:200]}")
        return converted      # 保留 flac

def main():
    total_ok, total_fail, total_skip = 0, 0, 0

    for folder in FOLDERS:
        folder_path = BASE_DIR / folder
        ncm_files = sorted(folder_path.glob("*.ncm"))

        if not ncm_files:
            print(f"[{folder}] 没有 .ncm 文件，跳过")
            continue

        print(f"\n[{folder}] 找到 {len(ncm_files)} 个 .ncm 文件")

        for ncm in ncm_files:
            # 如果 mp3 已存在则跳过
            if ncm.with_suffix(".mp3").exists():
                print(f"  ⟳ 已存在 mp3，跳过：{ncm.name}")
                total_skip += 1
                continue

            print(f"  → 转换：{ncm.name}", end=" ... ", flush=True)
            ok = convert_ncm(ncm)

            if not ok:
                total_fail += 1
                continue

            converted = find_converted(ncm)
            if converted is None:
                print("失败（未找到输出文件）")
                total_fail += 1
                continue

            # flac → mp3（如有必要）
            final = ensure_mp3(converted)
            print(f"完成 → {final.name}")

            # 删除原始 .ncm
            ncm.unlink()
            total_ok += 1

    print(f"\n{'─'*50}")
    print(f"转换完成：成功 {total_ok} | 失败 {total_fail} | 跳过 {total_skip}")

if __name__ == "__main__":
    main()
