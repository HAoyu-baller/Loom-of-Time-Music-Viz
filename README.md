<div align="center">

# 岁月织机 · Loom of Time

**Real-time Chinese Folk Music Decomposition & Generative Weaving Visualization**

[![ACM Multimedia 2026](https://img.shields.io/badge/ACM%20MM%202026-Interactive%20Art%20Track-red?style=flat-square)](https://2026.acmmm.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![React](https://img.shields.io/badge/React-18-61dafb?style=flat-square&logo=react)](https://react.dev)
[![Three.js](https://img.shields.io/badge/Three.js-r165-black?style=flat-square&logo=threedotjs)](https://threejs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)

*Upload any Chinese folk song. Watch it unravel into five instrument threads and weave itself into a living tapestry.*

</div>

---

## Abstract

**Loom of Time** is an interactive art installation that fuses deep-learning audio analysis with real-time generative graphics. A custom Band-Split Rotary Transformer (BSRoformer) separates uploaded Chinese folk music into five semantic stems — vocal, erhu, plucked strings, wind, and percussion — while a ResNet18 classifier identifies the regional origin (Jiangnan, Shaanxi, Yungui, Huanan, or Dongbei). Each stem drives a distinct visual layer rendered with custom WebGL shaders: percussion beats advance a fabric-weaving progress bar, vocal amplitude animates region-specific ribbon systems, and wind energy scatters procedural particle fields. The result is a continuously evolving textile whose colour palette, motion vocabulary, and weaving rhythm are uniquely determined by the music's geographic soul.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (React + Three.js)               │
│                                                                  │
│  ┌──────────┐   drag / upload    ┌────────────────────────────┐ │
│  │ Landing  │ ────────────────▶  │       FastAPI Backend      │ │
│  │  Screen  │                    │                            │ │
│  └──────────┘                    │  ┌──────────────────────┐  │ │
│                                  │  │  BSRoformer (15-20M)  │  │ │
│  ┌──────────────────────────┐    │  │  5-stem separation   │  │ │
│  │      WebGL Canvas        │    │  └──────────┬───────────┘  │ │
│  │                          │    │             │ parallel      │ │
│  │  L1 FabricReveal  ░░░░░  │    │  ┌──────────▼───────────┐  │ │
│  │  L2 WarpThreads   │││││  │◀───┤  │ ResNet18 Classifier  │  │ │
│  │  L3 Particles     ·····  │    │  │  region detection    │  │ │
│  │  L4 Ripples       ~~~~   │    │  └──────────────────────┘  │ │
│  │  L5 VocalLine     ≈≈≈≈≈  │    │                            │ │
│  │                          │    │  ← stems as WAV → client   │ │
│  │  AudioBridge (RMS→spring)│    └────────────────────────────┘ │
│  └──────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Audio → Visual signal path:** Per-stem RMS → Euler spring-damper (k/d tuned per stem) → `bridge.amp[]` → shader uniforms, updated every animation frame at 60 fps.

---

## Features

| Feature | Detail |
|---|---|
| Source separation | BSRoformer, 5 stems, dim=256 / depth=6 / heads=8 |
| Regional classification | ResNet18, mel-spectrogram input, chunk-based hard voting |
| Visual layers | 5 independent WebGL ShaderMaterial layers, AdditiveBlending |
| Fabric weave sync | Shuttle drops hard-bound to percussion onset cooldown (2 s) |
| Regional palettes | HSB interpolation with shortest-arc hue transition (τ = 0.35 s) |
| Demo mode | Bundled Jiangnan demo stems, no server needed |
| Browser support | Chrome 110+, Firefox 115+, Safari 16.4+ (autoplay-safe) |

---

## Installation

### Prerequisites

- Node.js ≥ 18
- Python ≥ 3.10
- CUDA 11.8+ (GPU) *or* CPU-only (slower inference)

### 1 — Frontend

```bash
cd frontend
npm install
npm run dev          # development server at http://localhost:5173
# or
npm run build        # production build → frontend/dist/
```

### 2 — Backend

```bash
cd backend
pip install -r requirements_api.txt

# Place model weights (see "Model Weights" below), then:
python api.py        # FastAPI server at http://localhost:8000
```

### 3 — Run Locally (both together)

Open two terminals:
```bash
# Terminal 1
cd backend && python api.py

# Terminal 2
cd frontend && npm run dev
```

Then open `http://localhost:5173` and upload a Chinese folk music file (MP3 / WAV / FLAC / OGG / M4A).

---

## Model Weights

Model checkpoints are too large for GitHub and are hosted on Hugging Face:

**🤗 [ruanhaoyu/loom-of-time-models](https://huggingface.co/ruanhaoyu/loom-of-time-models)**  *(link active at submission time)*

Download and place files as follows:

```
backend/
├── training/checkpoints/
│   └── best.pt                        ← BSRoformer separator
└── classifier/models/
    └── loom_best.pth                  ← ResNet18 region classifier
```

Or use the Hugging Face CLI:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('ruanhaoyu/loom-of-time-models', 'separator/best.pt',
                local_dir='backend/training/checkpoints')
hf_hub_download('ruanhaoyu/loom-of-time-models', 'classifier/loom_best.pth',
                local_dir='backend/classifier/models')
"
```

---

## Training Your Own Models

All training code lives in `backend/training/` and the root-level `build_dataset.py` / `dataset.py`.

### Data Pipeline (Separator)

The separator is trained on single-instrument audio clips from public datasets. **You do not need to provide any audio manually** — the download script handles instrument stems automatically.

| Dataset | HF ID | Stems |
|---|---|---|
| Chinese Traditional Instrument Sounds | [`ccmusic-database/CTIS`](https://huggingface.co/datasets/ccmusic-database/CTIS) | erhu, wind, plucked, perc |
| Erhu Playing Techniques | [`ccmusic-database/erhu_playing_tech`](https://huggingface.co/datasets/ccmusic-database/erhu_playing_tech) | erhu |
| Guzheng Tech 99 | [`ccmusic-database/Guzheng_Tech99`](https://huggingface.co/datasets/ccmusic-database/Guzheng_Tech99) | plucked |
| M4Singer | [`M4Singer/M4Singer`](https://huggingface.co/datasets/M4Singer/M4Singer) | vocal |

**Step 1 — Download and preprocess instrument stems:**
```bash
pip install datasets librosa soundfile tqdm
python backend/download_hf_datasets.py          # downloads CTIS, erhu_tech, Guzheng99
# Output: music/instruments_clean/{erhu,wind,plucked,perc}/

# For vocal stem, download M4Singer separately and place under:
# music/instruments_clean/vocal/
```

**Step 2 — Quality filtering and dataset balancing:**

Raw clips are filtered and balanced by `build_dataset.py` through a two-pass pipeline:

1. **Silence rejection** — each clip is read and its RMS energy is computed; clips with RMS < 0.01 are discarded (near-silence, failed recordings, very quiet passages)
2. **Energy ranking** — surviving clips are sorted by RMS descending; the top-N energy-richest clips are kept per stem, ensuring the model trains on musically active material
3. **Cross-stem balancing** — all stems are capped at the same clip count (4,500, limited by the perc stem which is the smallest pool), eliminating per-stem bias

After filtering, the final training set is:

| Stem | Clips kept | Duration |
|---|---|---|
| erhu | 4,500 | ~3.75 h |
| wind | 4,500 | ~3.75 h |
| plucked | 4,500 | ~3.75 h |
| perc | 4,500 | ~3.75 h |
| vocal | 4,500 | ~3.75 h |
| **Total** | **22,500** | **~18.75 h** |

Each clip is 3 seconds at 22,050 Hz mono. At runtime, `DynamicMixDataset` randomly draws one clip per stem per batch item and mixes them on-the-fly with ±3 dB gain jitter — this means the effective training diversity is orders of magnitude larger than the raw clip count.

```bash
python backend/training/build_dataset.py \
    --src music/instruments_clean \
    --dst data/dataset_balanced \
    --per-stem 4500 \
    --min-rms 0.01 \
    --workers 8
```

**Step 3 — Train:**
```bash
cd backend/training
# open train.ipynb, or:
python -c "import train; train.main()"

# Key hyperparameters (dataset.py):
#   batch_size=8, lr=3e-4, epochs=100
#   Silence Dropout: vocal 50%, perc 40%
#   Multi-Resolution STFT Loss: fft_sizes=[512, 2048, 8192]
#   Chunk size: 132300 samples (6 s @ 22050 Hz)
```

### Classifier (ResNet18)

```bash
# 1. Generate spectrogram dataset
python backend/classifier/preprocess.py \
    --audio_dir data/regional/ \
    --out_dir backend/classifier/data/spectrograms/

# 2. Train
# Open backend/classifier/train.ipynb
```

---

## Project Structure

```
.
├── frontend/                  # React + Three.js + Vite
│   ├── src/
│   │   ├── App.tsx            # Main app, upload flow, audio context
│   │   ├── scene/
│   │   │   ├── AudioBridge.ts # Shared state: amps, region, progress
│   │   │   ├── LoomScene.tsx  # Scene root + FabricReveal layer
│   │   │   ├── VocalLine.tsx  # L5: region-specific vocal ribbons
│   │   │   ├── WarpThreads.tsx# L2: vertical warp thread strips
│   │   │   ├── Particles.tsx  # L3: wind-driven particle field
│   │   │   ├── Ripples.tsx    # L4: plucked-string water ripples
│   │   │   └── PercFlash.tsx  # Percussion shuttle + flash
│   │   ├── audioAnalysis.ts   # Offline perc onset pre-counting
│   │   └── palettes.ts        # 5-region HSB colour palettes
│   ├── package.json
│   └── vite.config.ts
│
├── backend/
│   ├── api.py                 # FastAPI: /separate endpoint
│   ├── requirements_api.txt
│   ├── classifier/
│   │   ├── predict.py         # Region inference (ResNet18)
│   │   ├── preprocess.py      # Spectrogram dataset builder
│   │   └── train.ipynb        # Classifier training notebook
│   └── training/
│       ├── separate.py        # BSRoformer inference wrapper
│       ├── dataset.py         # DynamicMixDataset
│       ├── build_dataset.py   # Offline stem mixing pipeline
│       ├── train.ipynb        # Separator training notebook
│       ├── requirements.txt
│       ├── setup_env.sh       # SLURM environment setup
│       └── submit_slurm.sh    # HPC job submission script
│
├── build_dataset.py           # Root-level dataset builder
├── dataset.py                 # Root-level dataset module
├── .gitignore
└── README.md
```

---

## Demo

The frontend ships with a built-in Jiangnan demo (no backend required):

1. Open the app
2. Click **▸ Listen to Demo**
3. Watch the loom weave — percussion beats drive shuttle drops, vocal energy flows through the water-sleeve ribbon, wind particles scatter in Lissajous paths

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{ruan2026loom,
  title     = {Loom of Time: Real-Time Chinese Folk Music Decomposition
               and Generative Weaving Visualization},
  author    = {Ruan, Haoyu},
  booktitle = {Proceedings of the 34th ACM International Conference on
               Multimedia (Interactive Art Track)},
  year      = {2026},
  publisher = {ACM},
  address   = {Dublin, Ireland},
  note      = {\url{https://github.com/ruanhaoyu/loom-of-time}}
}
```

---

## License

Code: [MIT License](LICENSE)  
Demo audio stems: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — non-commercial use only  
Model weights: released for research / non-commercial use

---

<div align="center">
Made with obsessive attention to detail · ACM MM 2026 Interactive Art Track
</div>
