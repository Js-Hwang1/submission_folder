#!/bin/bash
#SBATCH --job-name=intuitive_figs
#SBATCH --output=/lustre/nvwulf/scratch/jungshwang/logs/intuitive_figures_%j.out
#SBATCH --error=/lustre/nvwulf/scratch/jungshwang/logs/intuitive_figures_%j.err
#SBATCH --partition=b40x4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Generate intuitive observation figures for CircuitKV paper
# Output: 3 separate PDFs + SVGs in figures/intuitive/

mkdir -p /lustre/nvwulf/scratch/jungshwang/logs

SIF=/lustre/nvwulf/scratch/jungshwang/pytorch_25.03-py3.sif
PIPDIR=/lustre/nvwulf/scratch/jungshwang/pip_packages_25.03

echo "=========================================="
echo "CircuitKV Intuitive Figure Generator"
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

        cd /lustre/nvwulf/scratch/jungshwang/submission_folder/KVCache-Factory

        # Pull latest code
        git pull

        # Install matplotlib if needed
        pip install --user matplotlib scipy

        # Create output directory
        mkdir -p figures/intuitive

        echo ''
        echo '=== Running intuitive figure generation ==='
        python scripts/generate_intuitive_figures.py --output_dir figures/intuitive

        echo ''
        echo '=== Output files ==='
        ls -la figures/intuitive/
    "

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
