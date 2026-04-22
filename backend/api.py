"""
岁月织机 — FastAPI 后端
启动: cd backend && uvicorn api:app --reload --port 8000
"""

import sys
import pathlib
import uuid
import shutil
import asyncio
from typing import Literal

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# 将 backend/ 目录加入 sys.path，以便引用 classifier 子包
sys.path.insert(0, str(pathlib.Path(__file__).parent))

# ── 配置 ──────────────────────────────────────────────────────────────────────
STEMS      = ["erhu", "wind", "plucked", "perc", "vocal"]
SR         = 22050
CHUNK      = 132300   # 6s 推理块
OVERLAP    = 22050    # 1s 重叠
CKPT_PATH  = pathlib.Path(__file__).parent / "training" / "checkpoints" / "best.pt"
UPLOAD_DIR = pathlib.Path(__file__).parent / "uploads"
OUTPUT_DIR = pathlib.Path(__file__).parent / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 设备 ──────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ── 懒加载：分离模型 ──────────────────────────────────────────────────────────
_sep_model = None

def get_sep_model():
    global _sep_model
    if _sep_model is None:
        from bs_roformer import BSRoformer
        if not CKPT_PATH.exists():
            raise RuntimeError(f"找不到分离模型: {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        cfg  = ckpt.get("model_cfg", {})
        _sep_model = BSRoformer(
            dim                    = cfg.get("dim", 256),
            depth                  = cfg.get("depth", 6),
            heads                  = cfg.get("heads", 8),
            dim_head               = cfg.get("dim_head", 32),
            num_stems              = len(STEMS),
            time_transformer_depth = cfg.get("time_transformer_depth", 2),
            freq_transformer_depth = cfg.get("freq_transformer_depth", 1),
        ).to(device)
        _sep_model.load_state_dict(ckpt["model"])
        _sep_model.eval()
        ep  = ckpt.get("epoch", "?")
        val = ckpt.get("val_loss", float("nan"))
        print(f"[分离模型] 加载完成  epoch={ep}  val_loss={val:.4f}  device={device}")
    return _sep_model

# ── 懒加载：地区分类模型 ──────────────────────────────────────────────────────
_cls_available = None  # True / False，首次调用后确定

def classify_region(mp3_path: str) -> dict | None:
    """
    调用 classifier 识别民歌地区。
    若模型文件不存在或推理失败，返回 None（不中断主流程）。
    """
    global _cls_available
    if _cls_available is False:
        return None
    try:
        from classifier.predict import predict
        result = predict(mp3_path, verbose=False)
        _cls_available = True
        return {
            "region":     result["region"],
            "region_zh":  result["region_zh"],
            "confidence": result["confidence"],
        }
    except Exception as e:
        print(f"[分类器] 跳过：{e}")
        _cls_available = False
        return None

# ── 音源分离推理 ──────────────────────────────────────────────────────────────
def separate(audio: np.ndarray) -> dict[str, np.ndarray]:
    model  = get_sep_model()
    hop    = CHUNK - OVERLAP
    length = len(audio)
    out_sum   = {s: np.zeros(length, dtype=np.float64) for s in STEMS}
    out_count = np.zeros(length, dtype=np.float64)
    window    = np.hanning(CHUNK).astype(np.float64)

    for start in range(0, length, hop):
        chunk = audio[start:start + CHUNK]
        if len(chunk) < CHUNK:
            chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
        x = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x)
        pred_len = pred.shape[-1]
        use_len  = min(CHUNK, pred_len, length - start)
        for i, stem in enumerate(STEMS):
            out_sum[stem][start:start + use_len] += (
                pred[0, i, 0, :use_len].cpu().numpy().astype(np.float64)
                * window[:use_len]
            )
        out_count[start:start + use_len] += window[:use_len]

    safe = np.where(out_count > 1e-8, out_count, 1.0)
    return {s: (out_sum[s] / safe).astype(np.float32) for s in STEMS}

# ── FastAPI 应用 ──────────────────────────────────────────────────────────────
app = FastAPI(title="岁月织机 API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 开发阶段允许所有来源
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    # 预热分离模型（如有）
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, get_sep_model)
    except Exception as e:
        print(f"[启动] 分离模型预热跳过：{e}")

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "stems": STEMS}

@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    suffix = pathlib.Path(file.filename or "input.mp3").suffix.lower()
    if suffix not in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}:
        raise HTTPException(400, "仅支持 mp3/wav/flac/ogg/m4a")

    job_id   = uuid.uuid4().hex
    tmp_path = UPLOAD_DIR / f"{job_id}{suffix}"
    out_dir  = OUTPUT_DIR / job_id
    out_dir.mkdir()

    try:
        # 保存上传文件
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 读取音频
        import librosa
        audio, _ = librosa.load(str(tmp_path), sr=SR, mono=True)
        audio     = audio.astype(np.float32)
        duration  = round(len(audio) / SR, 2)

        # 并行：分离 + 地区识别
        loop = asyncio.get_event_loop()

        sep_task = loop.run_in_executor(None, separate, audio)
        cls_task = loop.run_in_executor(None, classify_region, str(tmp_path))

        results, region_info = await asyncio.gather(sep_task, cls_task)

        # 写入各声部 wav
        for stem, data in results.items():
            sf.write(str(out_dir / f"{stem}.wav"), data, SR)

        # 构造响应
        response = {
            "job_id":   job_id,
            "duration": duration,
            "stems": {
                "vocal":   f"/outputs/{job_id}/vocal.wav",
                "erhu":    f"/outputs/{job_id}/erhu.wav",
                "plucked": f"/outputs/{job_id}/plucked.wav",
                "wind":    f"/outputs/{job_id}/wind.wav",
                "perc":    f"/outputs/{job_id}/perc.wav",
            },
            "region":            region_info["region"]     if region_info else None,
            "region_zh":         region_info["region_zh"]  if region_info else None,
            "region_confidence": region_info["confidence"] if region_info else None,
        }
        return response

    finally:
        tmp_path.unlink(missing_ok=True)

@app.get("/outputs/{job_id}/{filename}")
def get_output_file(job_id: str, filename: str):
    # 安全校验：只允许字母数字和 .wav
    if not filename.replace(".wav", "").replace("_", "").isalnum():
        raise HTTPException(400, "非法文件名")
    path = OUTPUT_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(404, "文件不存在")
    return FileResponse(str(path), media_type="audio/wav")
