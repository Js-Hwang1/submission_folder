#!/bin/bash
#SBATCH --job-name=ablation_figs
#SBATCH --output=/lustre/nvwulf/scratch/jungshwang/logs/ablation_figures_%j.out
#SBATCH --error=/lustre/nvwulf/scratch/jungshwang/logs/ablation_figures_%j.err
#SBATCH --partition=b40x4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Generate persuasive ablation figures for CircuitKV paper
# Requires GPU for model inference

mkdir -p /lustre/nvwulf/scratch/jungshwang/logs

SIF=/lustre/nvwulf/scratch/jungshwang/pytorch_25.03-py3.sif
PIPDIR=/lustre/nvwulf/scratch/jungshwang/pip_packages_25.03

echo "=========================================="
echo "CircuitKV Ablation Figure Generator"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo ""

singularity exec --nv \
    --bind /lustre:/lustre \
    $SIF \
    bash -c "
        export PYTHONUSERBASE=$PIPDIR
        export PATH=\$PYTHONUSERBASE/bin:\$PATH
        export PYTHONPATH=\$PYTHONUSERBASE/lib/python3.12/site-packages:\$PYTHONPATH
        export HF_HOME=/lustre/nvwulf/scratch/jungshwang/.cache/huggingface

        cd /lustre/nvwulf/scratch/jungshwang/submission_folder/KVCache-Factory

        # Pull latest code
        git pull

        # Install deps
        pip install --user matplotlib scipy

        # Create output directory
        mkdir -p figures/ablation

        echo ''
        echo '=== Running ablation figure generation ==='
        python scripts/generate_ablation_figures.py \
            --model_path Qwen/Qwen2.5-7B-Instruct \
            --output_dir figures/ablation \
            --num_examples 10

        echo ''
        echo '=== Output files ==='
        ls -la figures/ablation/
    "

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
