#!/bin/bash
#SBATCH --partition=h200x8-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=CKV
#SBATCH --output=logs/ckv_%j.out
#SBATCH --error=logs/ckv_%j.err

# Create logs directory
mkdir -p /lustre/nvwulf/scratch/jungshwang/submission_folder/logs

# Use absolute path to python to avoid environment issues in subshells
PYTHON=/lustre/nvwulf/scratch/jungshwang/miniforge3/envs/env/bin/python

# Setup environment
module load cuda12.8/toolkit/12.8.0
export CUDA_HOME=$CUDA_ROOT
export HF_TOKEN=""
export PATH=/lustre/nvwulf/scratch/jungshwang/miniforge3/envs/env/bin:$PATH

cd /lustre/nvwulf/scratch/jungshwang/submission_folder

# === BUILD STEP (only runs once) ===
echo "=== Building CircuitKV ==="
$PYTHON -m pip install --no-build-isolation -e ./CircuitKV

echo "=== Building KVCache-Factory CUDA extensions ==="
cd KVCache-Factory/csrc && $PYTHON build.py install && cd ../..

# === EXPERIMENT ===
cd /lustre/nvwulf/scratch/jungshwang/submission_folder/KVCache-Factory

echo "Starting CircuitKV experiment at $(date)"
$PYTHON run_longbench.py \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --method circuitkv \
  --max_capacity_prompts 2048 \
  --attn_implementation flash_attention_2 \
  --save_dir results/CKV \
  2>&1 | tee ../longbench_CKV.log

echo "=============================================="
echo "All experiments completed at $(date)"
echo "=============================================="

scp jungshwang@login.nvwulf.stonybrook.edu:/lustre/nvwulf/scratch/jungshwang/submission_folder/KVCache-Factory/res.tar.gz .

