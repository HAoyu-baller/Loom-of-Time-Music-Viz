#!/bin/bash
# 岁月织机 — HPC 环境一键配置脚本
# 用法：bash setup_env.sh
# 在 HPC 登录节点执行，建立 conda 虚拟环境并安装所有依赖

set -e  # 任意命令失败则退出

ENV_NAME="suoyuezhiji"
PYTHON_VER="3.12"

echo "=========================================="
echo "  岁月织机 HPC 环境配置"
echo "=========================================="

# ── Step 1: 创建 conda 环境 ────────────────────────────────────────────────────
echo ""
echo "[1/4] 创建 conda 环境: ${ENV_NAME} (Python ${PYTHON_VER})"
conda create -n ${ENV_NAME} python=${PYTHON_VER} -y
conda activate ${ENV_NAME}

# ── Step 2: 安装 PyTorch（CUDA 12.1 版本，根据集群 CUDA 版本调整）─────────────
echo ""
echo "[2/4] 安装 PyTorch + torchaudio (CUDA 12.1)"
echo "      如集群 CUDA 版本不同，请修改此命令"
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── Step 3: 安装其他依赖 ────────────────────────────────────────────────────────
echo ""
echo "[3/4] 安装项目依赖"
pip install -r requirements.txt

# ── Step 4: 验证安装 ────────────────────────────────────────────────────────────
echo ""
echo "[4/4] 验证安装"
python - <<'EOF'
import torch, torchaudio, bs_roformer, auraloss, soundfile, librosa
print(f"  torch       : {torch.__version__}")
print(f"  torchaudio  : {torchaudio.__version__}")
print(f"  bs_roformer : {bs_roformer.__version__ if hasattr(bs_roformer,'__version__') else 'ok'}")
print(f"  auraloss    : {auraloss.__version__ if hasattr(auraloss,'__version__') else 'ok'}")
print(f"  CUDA 可用   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
EOF

echo ""
echo "=========================================="
echo "  安装完成！激活环境：conda activate ${ENV_NAME}"
echo "=========================================="
