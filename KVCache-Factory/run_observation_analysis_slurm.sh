#!/bin/bash
#SBATCH --job-name=obs_analysis
#SBATCH --output=/lustre/nvwulf/scratch/jungshwang/logs/obs_analysis_%j.out
#SBATCH --error=/lustre/nvwulf/scratch/jungshwang/logs/obs_analysis_%j.err
#SBATCH --partition=b40x4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Observation Analysis for CircuitKV Paper
# Generates figures for Section 3 (Observations)

mkdir -p /lustre/nvwulf/scratch/jungshwang/logs

SIF=/lustre/nvwulf/scratch/jungshwang/pytorch_25.03-py3.sif
PIPDIR=/lustre/nvwulf/scratch/jungshwang/pip_packages_25.03

echo "============================================================"
echo "CircuitKV Observation Analysis"
echo "Started at $(date)"
echo "============================================================"

singularity exec --nv \
    --bind /lustre:/lustre \
    $SIF \
    bash -c "
        export PYTHONUSERBASE=$PIPDIR
        export PATH=\$PYTHONUSERBASE/bin:\$PATH
        export PYTHONPATH=\$PYTHONUSERBASE/lib/python3.12/site-packages:\$PYTHONPATH
        # HF_TOKEN should be set in environment or .bashrc

        python -c 'import torch; print(f\"PyTorch: {torch.__version__}, GPU: {torch.cuda.get_device_name()}\")'

        cd /lustre/nvwulf/scratch/jungshwang/submission_folder/KVCache-Factory

        # Pull latest code
        git pull

        # Install matplotlib if needed
        pip install --user matplotlib seaborn

        # Create output directories
        mkdir -p figures/observations/qasper
        mkdir -p figures/observations/trec

        echo ''
        echo '=== Analyzing qasper dataset (retrieval QA) ==='
        python scripts/observation_analysis.py \
            --model_path Qwen/Qwen2.5-7B-Instruct \
            --dataset qasper \
            --num_examples 30 \
            --max_length 2048 \
            --output_dir figures/observations/qasper

        echo ''
        echo '=== Analyzing trec dataset (classification) ==='
        python scripts/observation_analysis.py \
            --model_path Qwen/Qwen2.5-7B-Instruct \
            --dataset trec \
            --num_examples 30 \
            --max_length 2048 \
            --output_dir figures/observations/trec

        echo ''
        echo '=== Generated figures ==='
        ls -la figures/observations/qasper/
        ls -la figures/observations/trec/
    "

echo "============================================================"
echo "Observation analysis completed at $(date)"
echo "Figures saved to: figures/observations/"
echo "============================================================"
