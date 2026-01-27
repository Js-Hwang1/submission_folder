#!/bin/bash
#SBATCH --job-name=intuitive_figs
#SBATCH --output=/lustre/nvwulf/scratch/jungshwang/submission_folder/KVCache-Factory/logs/intuitive_figures_%j.out
#SBATCH --error=/lustre/nvwulf/scratch/jungshwang/submission_folder/KVCache-Factory/logs/intuitive_figures_%j.err
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G

# Generate intuitive observation figures for CircuitKV paper
# Output: 3 separate PDFs + SVGs in figures/intuitive/

echo "=========================================="
echo "CircuitKV Intuitive Figure Generator"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo ""

cd /lustre/nvwulf/scratch/jungshwang/submission_folder/KVCache-Factory

# Create output directory
mkdir -p figures/intuitive
mkdir -p logs

# Run with singularity
echo "Running figure generation..."
singularity exec --nv \
    --bind /lustre/nvwulf/scratch/jungshwang:/lustre/nvwulf/scratch/jungshwang \
    /lustre/nvwulf/scratch/jungshwang/containers/pytorch_24.09.sif \
    python scripts/generate_intuitive_figures.py --output_dir figures/intuitive

echo ""
echo "=========================================="
echo "Output files:"
ls -la figures/intuitive/
echo "=========================================="
echo "Completed: $(date)"
