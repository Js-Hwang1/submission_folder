#!/bin/bash
# Run observation analysis to generate figures for CircuitKV paper
#
# Usage:
#   bash scripts/run_observation_analysis.sh
#
# This script:
# 1. Creates output directory for figures
# 2. Runs analysis on qasper dataset (retrieval QA)
# 3. Runs analysis on trec dataset (classification)
# 4. Combines results for paper figures

set -e

cd /root/submission_folder/KVCache-Factory

# Create output directories
mkdir -p figures/observations/qasper
mkdir -p figures/observations/trec
mkdir -p figures/observations/combined

echo "=========================================="
echo "CircuitKV Observation Analysis"
echo "=========================================="

# Install matplotlib if needed
pip install matplotlib seaborn -q

# Run on qasper (retrieval QA - shows multi-hop importance)
echo ""
echo "[1/2] Analyzing qasper dataset..."
python scripts/observation_analysis.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --dataset qasper \
    --num_examples 20 \
    --max_length 2048 \
    --output_dir figures/observations/qasper

# Run on trec (classification - shows hub importance)
echo ""
echo "[2/2] Analyzing trec dataset..."
python scripts/observation_analysis.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --dataset trec \
    --num_examples 20 \
    --max_length 2048 \
    --output_dir figures/observations/trec

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "Figures saved to:"
echo "  - figures/observations/qasper/"
echo "  - figures/observations/trec/"
echo "=========================================="

# List generated files
echo ""
echo "Generated files:"
ls -la figures/observations/qasper/
ls -la figures/observations/trec/
