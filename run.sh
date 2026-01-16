#!/bin/bash
#SBATCH --partition=h200x4-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --job-name=streamingllm_qwen
#SBATCH --output=logs/streamingllm_qwen_%j.out
#SBATCH --error=logs/streamingllm_qwen_%j.err

# Create logs directory
mkdir -p /lustre/nvwulf/scratch/jungshwang/submission_folder/logs

# Use absolute path to python to avoid environment issues in subshells
PYTHON=/lustre/nvwulf/scratch/jungshwang/envs/env_HIQI/bin/python

# Setup environment
module load cuda12.8/toolkit/12.8.0
export CUDA_HOME=$CUDA_ROOT
export HF_TOKEN=""
export PATH=/lustre/nvwulf/scratch/jungshwang/envs/env_HIQI/bin:$PATH

# === EXPERIMENT ===
cd /lustre/nvwulf/scratch/jungshwang/submission_folder/KVCache-Factory

echo "Starting StreamingLLM Qwen2.5-7B experiments at $(date)"

# Run all budgets sequentially on single GPU
for BUDGET in 2048 1024 512 256 128; do
    echo "StreamingLLM Qwen2.5-7B budget=$BUDGET starting at $(date)"
    $PYTHON run_longbench.py \
        --model_path Qwen/Qwen2.5-7B-Instruct \
        --method streamingllm \
        --max_capacity_prompts $BUDGET \
        --attn_implementation flash_attention_2 \
        --save_dir results/streamingllm_qwen_${BUDGET} \
        2>&1 | tee ../logs/longbench_streamingllm_qwen_${BUDGET}.log
    echo "StreamingLLM Qwen2.5-7B budget=$BUDGET finished at $(date)"
done

echo "=============================================="
echo "All experiments completed at $(date)"
echo "=============================================="
