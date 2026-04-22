#!/bin/bash
#SBATCH --job-name=suoyuezhiji          # 作业名
#SBATCH --partition=gpu                  # 分区名（根据集群改）
#SBATCH --gres=gpu:1                     # 申请 1 块 GPU
#SBATCH --cpus-per-task=8               # CPU 核数（对应 NUM_WORKERS=8）
#SBATCH --mem=64G                        # 内存
#SBATCH --time=24:00:00                 # 最大运行时间 24h
#SBATCH --output=logs/slurm_%j.out      # 标准输出日志
#SBATCH --error=logs/slurm_%j.err       # 错误日志
#SBATCH --mail-type=END,FAIL            # 完成/失败时发邮件（可选）
# #SBATCH --mail-user=your@email.com    # 邮件地址（取消注释并填写）

# ── 环境配置 ───────────────────────────────────────────────────────────────────
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Start time : $(date)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

mkdir -p logs

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh   # 根据集群 conda 路径调整
conda activate suoyuezhiji

# ── 数据路径（修改为 HPC 上的实际路径）────────────────────────────────────────
# 方式一：直接修改 notebook 中的 DATASET_ROOT，然后 nbconvert 运行
# 方式二：用环境变量传入（需要 notebook 里读取 os.environ）

DATASET_ROOT="/path/to/music/dataset"   # ← 改成你在 HPC 上的实际路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 运行方式一：将 notebook 转成 Python 脚本运行（推荐，无需 Jupyter server）──
echo ""
echo "将 notebook 转换为 Python 脚本..."
jupyter nbconvert --to script train.ipynb --output train_script

# 替换 DATASET_ROOT 路径
sed -i "s|DATASET_ROOT = pathlib.Path(\"../music/dataset\")|DATASET_ROOT = pathlib.Path(\"${DATASET_ROOT}\")|" train_script.py

echo "开始训练..."
python train_script.py

# ── 运行方式二：直接执行 notebook（保留输出，但需要更多内存）────────────────
# jupyter nbconvert --to notebook --execute train.ipynb \
#     --output train_executed.ipynb \
#     --ExecutePreprocessor.timeout=864000

echo ""
echo "训练完成：$(date)"
echo "Checkpoint 路径：${SCRIPT_DIR}/checkpoints/"
