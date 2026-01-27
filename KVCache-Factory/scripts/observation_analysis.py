#!/usr/bin/env python3
"""
Observation Analysis Script for CircuitKV Paper

This script collects attention patterns from Qwen model on LongBench examples
and generates figures for the Observations section of the paper.

Observations to validate:
1. Multi-hop information flow (one-hop vs two-hop reachability)
2. Hub token existence (column sums of attention)
3. Attention entropy variation across heads

Usage:
    python scripts/observation_analysis.py --output_dir figures/observations
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def compute_attention_entropy(attn_weights):
    """
    Compute entropy of attention distribution for each head.

    Args:
        attn_weights: [num_heads, seq_len, seq_len] or [num_heads, seq_len]

    Returns:
        entropy: [num_heads] entropy values
    """
    if attn_weights.dim() == 3:
        # Use last row (query position)
        attn = attn_weights[:, -1, :]  # [num_heads, seq_len]
    else:
        attn = attn_weights

    # Compute entropy: H = -sum(p * log(p))
    eps = 1e-10
    entropy = -(attn * (attn + eps).log()).sum(dim=-1)
    return entropy


def compute_hub_scores(attn_weights):
    """
    Compute hub scores (column sums) for each position.

    Args:
        attn_weights: [num_heads, seq_len, seq_len]

    Returns:
        hub_scores: [seq_len] averaged hub scores across heads
    """
    # Column sum: how much attention each position receives
    # attn_weights[h, i, j] = attention from position i to position j
    # hub_score[j] = sum over i of attn[i, j]
    hub_per_head = attn_weights.sum(dim=1)  # [num_heads, seq_len]
    hub_scores = hub_per_head.mean(dim=0)  # [seq_len]
    return hub_scores


def compute_multihop_reachability(attn_weights, query_idx=-1, gamma=0.5):
    """
    Compute one-hop and two-hop reachability from query position.

    Args:
        attn_weights: [num_heads, seq_len, seq_len]
        query_idx: query position (default: last position)
        gamma: weight for two-hop term

    Returns:
        one_hop: [seq_len] direct attention from query
        two_hop: [seq_len] indirect attention via intermediate tokens
        combined: [seq_len] one_hop + gamma * two_hop
    """
    num_heads, seq_len, _ = attn_weights.shape

    # Average across heads
    attn_mean = attn_weights.mean(dim=0)  # [seq_len, seq_len]

    # One-hop: direct attention from query
    one_hop = attn_mean[query_idx, :]  # [seq_len]

    # Two-hop: query -> intermediate -> target
    # two_hop[j] = sum_k attn[query, k] * attn[k, j]
    two_hop = torch.matmul(one_hop.unsqueeze(0), attn_mean).squeeze(0)  # [seq_len]

    # Combined
    combined = one_hop + gamma * two_hop

    return one_hop, two_hop, combined


def get_attention_weights(model, tokenizer, text, max_length=2048):
    """
    Get attention weights from model for given text.

    Returns:
        attn_weights: dict of layer_idx -> [num_heads, seq_len, seq_len]
        tokens: list of token strings
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Collect attention weights from all layers
    attn_weights = {}
    for layer_idx, attn in enumerate(outputs.attentions):
        # attn: [batch, num_heads, seq_len, seq_len]
        attn_weights[layer_idx] = attn[0].cpu()  # Remove batch dim

    return attn_weights, tokens


def analyze_example(model, tokenizer, example, max_length=2048):
    """
    Analyze a single example and return statistics.
    """
    context = example.get("context", example.get("input", ""))
    question = example.get("question", example.get("instruction", ""))

    # Combine context and question
    if question:
        text = f"{context}\n\nQuestion: {question}"
    else:
        text = context

    # Get attention weights
    attn_weights, tokens = get_attention_weights(model, tokenizer, text, max_length)

    results = {
        "tokens": tokens,
        "num_tokens": len(tokens),
        "layers": {}
    }

    for layer_idx, attn in attn_weights.items():
        # Compute statistics for this layer
        entropy = compute_attention_entropy(attn)
        hub_scores = compute_hub_scores(attn)
        one_hop, two_hop, combined = compute_multihop_reachability(attn)

        results["layers"][layer_idx] = {
            "entropy_per_head": entropy.numpy(),
            "entropy_mean": entropy.mean().item(),
            "entropy_std": entropy.std().item(),
            "hub_scores": hub_scores.numpy(),
            "one_hop": one_hop.numpy(),
            "two_hop": two_hop.numpy(),
            "combined": combined.numpy(),
        }

    return results


def plot_entropy_distribution(all_results, output_dir):
    """
    Figure 2c: Distribution of attention entropy across heads.
    """
    # Collect all entropy values
    entropy_by_layer = defaultdict(list)
    all_entropy = []

    for result in all_results:
        for layer_idx, layer_data in result["layers"].items():
            # Filter out NaN values
            valid_entropy = [e for e in layer_data["entropy_per_head"].tolist() if not np.isnan(e)]
            entropy_by_layer[layer_idx].extend(valid_entropy)
            all_entropy.extend(valid_entropy)

    if not all_entropy:
        print("Warning: No valid entropy values found!")
        return {"entropy_min": 0, "entropy_max": 0, "entropy_mean": 0, "entropy_std": 0}

    # Plot 1: Overall entropy distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of all entropy values
    ax = axes[0]
    ax.hist(all_entropy, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(all_entropy), color='red', linestyle='--', label=f'Mean: {np.mean(all_entropy):.2f}')
    ax.axvline(np.median(all_entropy), color='green', linestyle='--', label=f'Median: {np.median(all_entropy):.2f}')
    ax.set_xlabel('Attention Entropy (nats)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Attention Entropy Across All Heads', fontsize=14)
    ax.legend()

    # Entropy by layer
    ax = axes[1]
    layer_indices = sorted(entropy_by_layer.keys())
    layer_means = [np.nanmean(entropy_by_layer[l]) if entropy_by_layer[l] else 0 for l in layer_indices]
    layer_stds = [np.nanstd(entropy_by_layer[l]) if entropy_by_layer[l] else 0 for l in layer_indices]

    ax.errorbar(layer_indices, layer_means, yerr=layer_stds, marker='o', capsize=3)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Mean Entropy (nats)', fontsize=12)
    ax.set_title('Attention Entropy by Layer', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_distribution.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'entropy_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Entropy range: {np.min(all_entropy):.2f} - {np.max(all_entropy):.2f}")
    print(f"Entropy mean: {np.mean(all_entropy):.2f}, std: {np.std(all_entropy):.2f}")

    return {
        "entropy_min": float(np.min(all_entropy)),
        "entropy_max": float(np.max(all_entropy)),
        "entropy_mean": float(np.mean(all_entropy)),
        "entropy_std": float(np.std(all_entropy)),
    }


def plot_hub_analysis(all_results, all_tokens, output_dir, sink_size=4):
    """
    Figure 2b: Hub token analysis.

    Args:
        sink_size: Number of initial tokens to exclude (attention sink tokens)
    """
    # Collect hub scores and token types
    hub_by_token_type = defaultdict(list)
    all_hub_scores = []

    for result, tokens in zip(all_results, all_tokens):
        # Use middle layer for hub analysis
        mid_layer = len(result["layers"]) // 2
        hub_scores = result["layers"][mid_layer]["hub_scores"]

        # Skip sink tokens (first sink_size positions)
        for i, (token, hub_score) in enumerate(zip(tokens, hub_scores)):
            if i < sink_size:
                continue  # Skip attention sink tokens
            all_hub_scores.append(hub_score)

            # Categorize token type (skip sink tokens)
            if i < sink_size:
                continue
            if token in [".", ",", "!", "?", ":", ";"]:
                hub_by_token_type["Punctuation"].append(hub_score)
            elif token in ["<s>", "</s>", "<|endoftext|>", "[CLS]", "[SEP]"]:
                hub_by_token_type["Special"].append(hub_score)
            elif token.startswith("Ġ") or token.startswith("▁"):
                # Word start
                clean = token[1:] if token[0] in ["Ġ", "▁"] else token
                if clean.lower() in ["the", "a", "an", "is", "are", "was", "were", "be"]:
                    hub_by_token_type["Function Words"].append(hub_score)
                elif clean[0].isupper() if clean else False:
                    hub_by_token_type["Capitalized"].append(hub_score)
                else:
                    hub_by_token_type["Content Words"].append(hub_score)
            else:
                hub_by_token_type["Subwords"].append(hub_score)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box plot by token type
    ax = axes[0]
    token_types = ["Punctuation", "Special", "Function Words", "Capitalized", "Content Words", "Subwords"]
    data = [hub_by_token_type[t] for t in token_types if t in hub_by_token_type]
    labels = [t for t in token_types if t in hub_by_token_type]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Hub Score (Column Sum)', fontsize=12)
    ax.set_title('Hub Scores by Token Type', fontsize=14)
    ax.tick_params(axis='x', rotation=45)

    # Hub score distribution
    ax = axes[1]
    ax.hist(all_hub_scores, bins=50, edgecolor='black', alpha=0.7)

    # Mark top 10% as "hubs"
    threshold = np.percentile(all_hub_scores, 90)
    ax.axvline(threshold, color='red', linestyle='--', label=f'Top 10% threshold: {threshold:.2f}')
    ax.set_xlabel('Hub Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Hub Scores', fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hub_analysis.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'hub_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Print statistics
    print("\nHub score by token type:")
    for t in token_types:
        if t in hub_by_token_type and hub_by_token_type[t]:
            print(f"  {t}: mean={np.mean(hub_by_token_type[t]):.3f}, std={np.std(hub_by_token_type[t]):.3f}")

    return hub_by_token_type


def plot_multihop_analysis(all_results, output_dir, sink_size=4):
    """
    Figure 2a/2d: Multi-hop reachability analysis.

    Args:
        sink_size: Number of initial tokens to exclude (attention sink tokens)
    """
    # Collect one-hop vs two-hop statistics
    one_hop_all = []
    two_hop_all = []
    improvement_ratios = []

    for result in all_results:
        mid_layer = len(result["layers"]) // 2
        layer_data = result["layers"][mid_layer]

        one_hop = layer_data["one_hop"]
        two_hop = layer_data["two_hop"]

        # Exclude sink tokens
        one_hop_no_sink = one_hop[sink_size:]
        two_hop_no_sink = two_hop[sink_size:]

        # For positions where one-hop is low, check if two-hop is higher
        if len(one_hop_no_sink) > 0:
            low_onehop_mask = one_hop_no_sink < np.percentile(one_hop_no_sink, 50)

            if low_onehop_mask.sum() > 0:
                avg_onehop_low = one_hop_no_sink[low_onehop_mask].mean()
                avg_twohop_low = two_hop_no_sink[low_onehop_mask].mean()
                if avg_onehop_low > 1e-8:
                    improvement_ratios.append(avg_twohop_low / avg_onehop_low)

        one_hop_all.extend(one_hop_no_sink.tolist())
        two_hop_all.extend(two_hop_no_sink.tolist())

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: one-hop vs two-hop
    ax = axes[0]
    # Subsample for visualization
    n_points = min(5000, len(one_hop_all))
    indices = np.random.choice(len(one_hop_all), n_points, replace=False)
    x = np.array(one_hop_all)[indices]
    y = np.array(two_hop_all)[indices]

    ax.scatter(x, y, alpha=0.3, s=5)
    ax.plot([0, max(x)], [0, max(x)], 'r--', label='y=x')
    ax.set_xlabel('One-Hop Attention Score', fontsize=12)
    ax.set_ylabel('Two-Hop Reachability Score', fontsize=12)
    ax.set_title('One-Hop vs Two-Hop Reachability', fontsize=14)
    ax.legend()

    # Correlation
    corr = np.corrcoef(one_hop_all, two_hop_all)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Improvement histogram
    ax = axes[1]
    if improvement_ratios:
        ax.hist(improvement_ratios, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(1.0, color='red', linestyle='--', label='No improvement')
        ax.axvline(np.mean(improvement_ratios), color='green', linestyle='--',
                   label=f'Mean: {np.mean(improvement_ratios):.2f}x')
        ax.set_xlabel('Two-Hop / One-Hop Ratio (for low one-hop positions)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Multi-Hop Improvement for Indirect Tokens', fontsize=14)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multihop_analysis.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'multihop_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nOne-hop vs Two-hop correlation: {corr:.3f}")
    if improvement_ratios:
        print(f"Mean improvement ratio for low-attention positions: {np.mean(improvement_ratios):.2f}x")

    return {"correlation": corr, "improvement_ratios": improvement_ratios}


def plot_hero_figure(result, tokens, output_dir, sink_size=4):
    """
    Generate a detailed hero figure showing the key concepts.

    Args:
        sink_size: Number of initial tokens to exclude (attention sink tokens)
    """
    mid_layer = len(result["layers"]) // 2
    layer_data = result["layers"][mid_layer]

    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel (a): Attention entropy per head
    ax1 = fig.add_subplot(gs[0, 0])
    entropy = layer_data["entropy_per_head"]
    num_heads = len(entropy)

    # Handle NaN values in entropy
    valid_entropy = entropy[~np.isnan(entropy)]
    if len(valid_entropy) > 0:
        entropy_max = np.nanmax(entropy)
        entropy_min = np.nanmin(entropy)
        # Normalize for colors, handling NaN
        entropy_normalized = np.nan_to_num((entropy - entropy_min) / (entropy_max - entropy_min + 1e-8), nan=0.5)
        colors = plt.cm.RdYlGn_r(entropy_normalized)
        median_val = np.nanmedian(entropy)
    else:
        colors = plt.cm.RdYlGn_r(np.zeros(num_heads))
        median_val = 0

    bars = ax1.bar(range(num_heads), np.nan_to_num(entropy), color=colors, edgecolor='black', linewidth=0.5)
    ax1.axhline(median_val, color='blue', linestyle='--', label=f'Median: {median_val:.2f}')
    ax1.set_xlabel('Head Index', fontsize=11)
    ax1.set_ylabel('Entropy (nats)', fontsize=11)
    ax1.set_title('(a) Attention Entropy Varies Across Heads', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)

    # Add annotations for sharp vs diffuse heads
    valid_mask = ~np.isnan(entropy)
    if valid_mask.any():
        entropy_safe = np.where(valid_mask, entropy, np.inf)
        sharp_idx = np.argmin(entropy_safe)
        entropy_safe = np.where(valid_mask, entropy, -np.inf)
        diffuse_idx = np.argmax(entropy_safe)

        if not np.isnan(entropy[sharp_idx]):
            ax1.annotate('Sharp', (sharp_idx, entropy[sharp_idx]),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9,
                         arrowprops=dict(arrowstyle='->', color='green'))
        if not np.isnan(entropy[diffuse_idx]):
            ax1.annotate('Diffuse', (diffuse_idx, entropy[diffuse_idx]),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9,
                         arrowprops=dict(arrowstyle='->', color='red'))

    # Panel (b): Hub scores - EXCLUDE sink tokens for visualization
    ax2 = fig.add_subplot(gs[0, 1])
    hub_scores = layer_data["hub_scores"]
    # Exclude sink tokens for better visualization
    hub_scores_no_sink = hub_scores[sink_size:]
    seq_len = min(200, len(hub_scores_no_sink))
    ax2.bar(range(sink_size, sink_size + seq_len), hub_scores_no_sink[:seq_len], alpha=0.7, width=1.0)
    threshold = np.percentile(hub_scores_no_sink, 90)
    ax2.axhline(threshold, color='red', linestyle='--', label=f'Top 10% threshold')
    ax2.set_xlabel('Token Position (excluding sink tokens)', fontsize=11)
    ax2.set_ylabel('Hub Score (Column Sum)', fontsize=11)
    ax2.set_title('(b) Hub Tokens Receive High Attention From Many Positions', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)

    # Panel (c): One-hop vs Combined - EXCLUDE sink tokens
    ax3 = fig.add_subplot(gs[1, 0])
    one_hop = layer_data["one_hop"]
    combined = layer_data["combined"]
    # Exclude sink tokens
    one_hop_no_sink = one_hop[sink_size:]
    combined_no_sink = combined[sink_size:]
    seq_len = min(200, len(one_hop_no_sink))

    x = np.arange(sink_size, sink_size + seq_len)
    width = 0.35
    ax3.bar(x - width/2, one_hop_no_sink[:seq_len], width, label='One-Hop (H2O)', alpha=0.7)
    ax3.bar(x + width/2, combined_no_sink[:seq_len], width, label='Combined (Ours)', alpha=0.7)
    ax3.set_xlabel('Token Position (excluding sink tokens)', fontsize=11)
    ax3.set_ylabel('Importance Score', fontsize=11)
    ax3.set_title('(c) Multi-Hop Captures Indirect Dependencies', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)

    # Panel (d): Concept diagram
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Draw concept diagram
    concept_text = """
    CircuitKV: Dual Importance via Markov Chains

    ┌─────────────────────────────────────────┐
    │                                         │
    │  Query Importance (QI):                 │
    │  • Multi-hop reachability from query    │
    │  • Captures indirect dependencies       │
    │  • QI = attn + γ·(attn × attn)         │
    │                                         │
    │  Hub Importance (HI):                   │
    │  • Structural centrality               │
    │  • Tokens attended by many positions    │
    │  • HI = Σᵢ attn[i, j]                  │
    │                                         │
    │  Entropy-Adaptive Weighting:            │
    │  • Sharp heads (low H) → trust QI      │
    │  • Diffuse heads (high H) → trust HI   │
    │  • Score = w·QI + (1-w)·HI             │
    │                                         │
    └─────────────────────────────────────────┘
    """
    ax4.text(0.5, 0.5, concept_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax4.set_title('(d) Method Overview', fontsize=12, fontweight='bold')

    plt.savefig(os.path.join(output_dir, 'hero_figure.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'hero_figure.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("Hero figure saved!")


def main():
    parser = argparse.ArgumentParser(description='Generate observation figures for CircuitKV paper')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Path to model')
    parser.add_argument('--output_dir', type=str, default='figures/observations',
                        help='Output directory for figures')
    parser.add_argument('--num_examples', type=int, default=20,
                        help='Number of examples to analyze')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--dataset', type=str, default='qasper',
                        choices=['qasper', 'narrativeqa', 'multifieldqa_en', 'trec'],
                        help='Dataset to use')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"  # Need eager for attention weights
    )
    model.eval()

    print(f"Loading dataset: {args.dataset}")
    # Load LongBench dataset
    dataset = load_dataset('THUDM/LongBench', args.dataset, split='test')

    # Analyze examples
    all_results = []
    all_tokens = []

    print(f"Analyzing {args.num_examples} examples...")
    for i, example in enumerate(tqdm(dataset.select(range(min(args.num_examples, len(dataset)))))):
        try:
            result = analyze_example(model, tokenizer, example, args.max_length)
            all_results.append(result)
            all_tokens.append(result["tokens"])
        except Exception as e:
            print(f"Error on example {i}: {e}")
            continue

    if not all_results:
        print("No examples analyzed successfully!")
        return

    print(f"\nAnalyzed {len(all_results)} examples successfully")
    print(f"Average sequence length: {np.mean([r['num_tokens'] for r in all_results]):.0f}")

    # Generate figures
    print("\n=== Generating Figures ===\n")

    print("1. Entropy distribution...")
    entropy_stats = plot_entropy_distribution(all_results, args.output_dir)

    print("\n2. Hub analysis...")
    hub_stats = plot_hub_analysis(all_results, all_tokens, args.output_dir)

    print("\n3. Multi-hop analysis...")
    multihop_stats = plot_multihop_analysis(all_results, args.output_dir)

    print("\n4. Hero figure...")
    plot_hero_figure(all_results[0], all_tokens[0], args.output_dir)

    # Save statistics to JSON
    stats = {
        "num_examples": len(all_results),
        "dataset": args.dataset,
        "model": args.model_path,
        "entropy": entropy_stats,
        "multihop": {
            "correlation": multihop_stats["correlation"],
            "mean_improvement": np.mean(multihop_stats["improvement_ratios"]) if multihop_stats["improvement_ratios"] else None
        }
    }

    with open(os.path.join(args.output_dir, 'statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Done! Figures saved to {args.output_dir} ===")


if __name__ == "__main__":
    main()
