"""
岁月织机 — CTIS 数据集提取脚本
================================
读取本地已下载的 CTIS arrow 分片，
按乐器精确分类，提取 WAV 到 music/instruments_clean/

输入：/Users/ruanhaoyu/Downloads/data-XXXXX-of-00030.arrow（30个）
输出：music/instruments_clean/{erhu,wind,plucked,perc}/ctis_{cname}/XXXX.wav

用法：
  python extract_ctis.py
"""

import pathlib
import numpy as np
import pyarrow as pa
import soundfile as sf
import librosa
from tqdm import tqdm

# ─── 路径配置 ─────────────────────────────────────────────────
ARROW_DIR  = pathlib.Path("/Users/ruanhaoyu/Downloads")
OUT_ROOT   = pathlib.Path(__file__).parent / "music" / "instruments_clean"
TARGET_SR  = 22050
NORM_DBFS  = -1.0

# ─── 精确白名单映射（cname → 轨道）─────────────────────────────
# 用完整 cname 匹配，避免关键字误伤

ERHU_NAMES = {
    # 正统二胡族
    "二胡","中胡","高胡","六角高胡","扁八角高胡","大胡","低音伬胡","伬胡",
    # 板胡族
    "板胡","中音板胡","高音板胡","绍剧板胡","宛梆子梆胡","莱芜梆子-梆胡",
    # 坠胡/坠琴
    "坠胡","吕剧坠琴",
    # 各地方剧种胡琴
    "仕胡","工胡","广西彩调主胡","扬剧主胡","扬剧主胡(小西皮)","扬剧主胡F调",
    "锡剧主胡","渔胡","越胡","赣胡","高腔赣胡","高腔赣胡第2代",
    "陇剧陇胡(传统)","陇剧陇胡(改良)D调","滇胡","滇葫(小二胡)","黔胡",
    "椰胡","雷胡","花胡","花胡2","壮剧土胡","壮剧土胡2","壮剧马骨胡D调",
    "葫芦琴","牛腿琴","牛角胡","襄阳专用胡琴","齐琴",
    # 奚琴
    "奚琴(传统)","奚琴(改良)","中音奚琴(改良)",
    # 朝鲜/韩国拉弦
    "伽倻琴","伽倻琴(改良)","玄琴","二馨",
    # 独弦琴（拉弦类）
    "佤族独弦琴","独弦琴",
    # 其他拉弦
    "雷琴","丝弦","云南花灯丝弦","二股弦","四股弦","六角弦","壳仔弦",
    "二弦","四弦","南嗳仔",
    # 晋剧胡琴
    "晋剧晋胡","晋剧二股弦",
}

WIND_NAMES = {
    # 笛子
    "A调曲笛","G调新笛","G调梆笛","中音横笛","低音横笛","高音横笛",
    "小闷笛","侗笛","口笛",
    # 唢呐族
    "唢呐","唢呐2","中音加键唢呐","低音加键唢呐","低音唢呐","高音唢呐","长唢呐",
    "大吹","大笒",
    # 笙族
    "中音笙","低音笙","传统笙","高音键笙","大芦笙","小芦笙","芦笙","拉祜族葫芦笙",
    # 箫/洞箫
    "洞箫","南音洞箫","短箫","短箫(传统)",
    # 埙
    "埙",
    # 管子/筚篥
    "管子","中音筚篥","低音筚篥","小筚篥",
    # 葫芦丝/巴乌
    "葫芦丝","F调巴乌",
    # 尖子号/长号
    "尖子号","尖子号2","长号","老长号",
}

PLUCKED_NAMES = {
    # 琵琶
    "琵琶","南音琵琶",
    # 古筝
    "古筝","雅筝",
    # 柳琴
    "柳琴",
    # 扬琴
    "扬琴","扬琴2","扬琴3","扬琴4",
    # 月琴/八角月琴
    "月琴","八角月琴",
    # 三弦
    "三弦","三弦2","南音三弦","澜沧小三弦",
    # 阮
    "中阮","大阮",
    # 热瓦普/都塔尔
    "低音热瓦普","民间热瓦普","都它尔","弹拨尔",
    # 双清/冬不拉
    "双清","冬不拉","陶布舒尔",
    # 竹排琴
    "竹排琴",
    # 箜篌/雅托嘎
    "箜篌","雅托嘎",
}

PERC_NAMES = {
    # 鼓类
    "中国大鼓","南鼓","小堂鼓","板鼓","花盆鼓","压脚鼓","上杖鼓","杖鼓",
    "引鼓","手鼓","渔鼓","扁鼓","五音排鼓","川剧堂鼓",
    "宜春三星鼓单铛","宜春三星鼓双铛","宜春三星鼓寿鼓老鼓",
    "宜春三星鼓禄鼓老鼓","宜春三星鼓福鼓老鼓","宜春三星鼓镲",
    # 锣类
    "包锣","低音大锣","小锣","小锣2","川大锣","川小锣","小叫锣",
    "武锣","草锣","曲锣","斗锣","抄锣","蛮锣","马锣","虎音锣","圆锣","锣仔",
    # 钹/铙
    "钹","小钹","铙","铙钹","大镲","小镲",
    # 板/梆
    "拍板","简板","提手(板)","北梆子","南梆子","脚梆子","盖板(传统)","盖板(新)D调",
    # 编钟/编磬
    "编钟","编磬","云锣","木鱼",
    # 其他打击
    "萨巴依","双铃","大铛铛","小铛铛","响盏","响盏2",
}

# 构建 cname → track 的查找字典
CNAME_TO_TRACK = {}
for name in ERHU_NAMES:    CNAME_TO_TRACK[name] = "erhu"
for name in WIND_NAMES:    CNAME_TO_TRACK[name] = "wind"
for name in PLUCKED_NAMES: CNAME_TO_TRACK[name] = "plucked"
for name in PERC_NAMES:    CNAME_TO_TRACK[name] = "perc"


# ─── 音频处理 ─────────────────────────────────────────────────
def process_audio(array: np.ndarray, sr: int) -> np.ndarray | None:
    audio = array.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq")
    peak = np.abs(audio).max()
    if peak < 1e-8:
        return None
    target_linear = 10 ** (NORM_DBFS / 20.0)
    return np.clip(audio * (target_linear / peak), -1.0, 1.0)


def save_wav(audio: np.ndarray, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, samplerate=TARGET_SR, subtype="PCM_16")


# ─── 主逻辑 ──────────────────────────────────────────────────
def main():
    arrow_files = sorted(ARROW_DIR.glob("data-*-of-00030.arrow"))
    print(f"找到 {len(arrow_files)} 个 arrow 分片")

    track_counts  = {"erhu": 0, "wind": 0, "plucked": 0, "perc": 0}
    skip_count    = 0
    fail_count    = 0
    unmatched_set = set()

    # 统计已有文件数（用于续编号）
    dir_counters = {}

    for arrow_path in tqdm(arrow_files, desc="分片", unit="shard"):
        with pa.ipc.open_stream(str(arrow_path)) as reader:
            table = reader.read_all()

        cnames  = table["cname"].to_pylist()
        arrays  = table["audio"].to_pylist()   # 每条是 dict{bytes, path, ...} 或 struct

        for cname, audio_struct in zip(cnames, arrays):
            track = CNAME_TO_TRACK.get(cname)
            if track is None:
                unmatched_set.add(cname)
                skip_count += 1
                continue

            # 从 arrow struct 取 audio bytes（sr 固定 44100，从 bytes 自动读取）
            try:
                raw_bytes = audio_struct["bytes"]
                if raw_bytes is None:
                    # 部分 HF 数据集用 path 而非 bytes
                    fail_count += 1
                    continue
                import io, soundfile as sf2
                audio_np, sr = sf2.read(io.BytesIO(raw_bytes))
            except Exception as e:
                fail_count += 1
                continue

            processed = process_audio(audio_np, sr)
            if processed is None:
                fail_count += 1
                continue

            # 输出目录：instruments_clean/<track>/ctis_<cname>/
            safe_name = cname.replace("/", "_").replace("(", "").replace(")", "")
            out_dir   = OUT_ROOT / track / f"ctis_{safe_name}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # 续编号
            key = str(out_dir)
            if key not in dir_counters:
                dir_counters[key] = len(list(out_dir.glob("*.wav")))
            idx = dir_counters[key]
            dir_counters[key] += 1

            save_wav(processed, out_dir / f"{idx:04d}.wav")
            track_counts[track] += 1

    # ─── 汇总报告 ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  CTIS 提取完成")
    print("=" * 55)
    for track, n in track_counts.items():
        print(f"  {track:10s}: {n} 条")
    print(f"  跳过（未匹配）: {skip_count} 条")
    print(f"  失败         : {fail_count} 条")
    if unmatched_set:
        print(f"\n  未匹配乐器（共{len(unmatched_set)}种）:")
        for name in sorted(unmatched_set):
            print(f"    {name}")
    print("=" * 55)


if __name__ == "__main__":
    main()
