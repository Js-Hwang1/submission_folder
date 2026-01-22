#!/usr/bin/env python3
"""
Proof of Concept: Neumann Series Score Distribution Analysis

This script analyzes how token importance scores evolve across Neumann iterations k=1 to k=10.
Key questions:
1. What distribution do scores follow at each k? (Power law like H2O?)
2. Do high-scoring tokens at k=1 remain high at k=2, k=3, etc.?
3. Is progressive filtering viable?
4. How different are QI vs HI vs H2O in token selection?

Outputs:
- neumann_distribution_k{1-10}.png: Score distributions at each k
- neumann_rank_persistence.png: How top tokens persist across k
- neumann_score_heatmap.png: Token scores across all k values
- qi_hi_h2o_comparison.png: Detailed comparison of QI, HI, and H2O
- neumann_analysis_report.txt: Numerical analysis with QI/HI/H2O comparison

Usage:
    python poc_neumann_distribution.py --model Qwen/Qwen2.5-7B-Instruct --num_samples 5
"""

import argparse
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to non-interactive backend for server
plt.switch_backend('Agg')


def compute_neumann_scores_per_iteration(
    attention: torch.Tensor,
    query_idx: int,
    sink_size: int = 4,
    max_k: int = 10,
) -> dict:
    """
    Compute Neumann series scores at each iteration k=1 to max_k.

    Returns dict with:
    - 'qi_scores': [max_k, n] tensor of QI scores at each k
    - 'hi_scores': [max_k, n] tensor of HI scores at each k
    - 'cumulative_qi': [max_k, n] cumulative QI (N = I + Q + Q^2 + ...)
    - 'cumulative_hi': [max_k, n] cumulative HI
    """
    n = attention.shape[0]
    device = attention.device

    # Convert to float32 for numerical stability
    attention = attention.float()

    # Build transition matrix P from attention
    row_sums = attention.sum(dim=1, keepdim=True).clamp(min=1e-8)
    P = attention / row_sums

    # Extract Q (transient-to-transient)
    n_transient = n - sink_size
    Q = P[sink_size:, sink_size:].contiguous()

    # Query position in transient space
    query_transient_idx = query_idx - sink_size
    if query_transient_idx < 0:
        query_transient_idx = n_transient - 1

    # Initialize vectors
    # QI: start from query position
    v_qi = torch.zeros(n_transient, device=device, dtype=torch.float32)
    v_qi[query_transient_idx] = 1.0

    # HI: start from uniform (average over all starting positions)
    v_hi = torch.ones(n_transient, device=device, dtype=torch.float32) / n_transient

    # Storage for each iteration
    qi_per_k = []  # Score contribution at each k
    hi_per_k = []
    cumulative_qi = []  # Cumulative sum up to k
    cumulative_hi = []

    # k=0: Identity (starting point)
    cum_qi = v_qi.clone()
    cum_hi = v_hi.clone()

    for k in range(1, max_k + 1):
        # One iteration: v = Q.T @ v
        v_qi = torch.mv(Q.t(), v_qi)
        v_hi = torch.mv(Q.t(), v_hi)

        # Store per-iteration contribution
        qi_per_k.append(v_qi.clone())
        hi_per_k.append(v_hi.clone())

        # Accumulate
        cum_qi = cum_qi + v_qi
        cum_hi = cum_hi + v_hi

        # Store cumulative (pad to full size)
        full_qi = torch.zeros(n, device=device, dtype=torch.float32)
        full_hi = torch.zeros(n, device=device, dtype=torch.float32)
        full_qi[sink_size:] = cum_qi
        full_hi[sink_size:] = cum_hi
        full_qi[:sink_size] = cum_qi.mean() * 0.1  # Small value for sink
        full_hi[:sink_size] = cum_hi.mean() * 0.1

        cumulative_qi.append(full_qi)
        cumulative_hi.append(full_hi)

    return {
        'cumulative_qi': torch.stack(cumulative_qi),  # [max_k, n]
        'cumulative_hi': torch.stack(cumulative_hi),  # [max_k, n]
        'n': n,
        'sink_size': sink_size,
    }


def compute_h2o_scores(attention: torch.Tensor) -> torch.Tensor:
    """
    Compute H2O-style importance scores: column sums of attention matrix.

    H2O uses "attention received" as the importance metric - tokens that
    receive more attention from other tokens are considered more important.

    This is equivalent to k=1 Neumann with uniform starting distribution,
    but computed directly from the attention matrix (not transition matrix P).
    """
    attention = attention.float()
    # H2O: sum of attention weights received by each token
    # For causal attention: column sum (how much attention each position receives)
    h2o_scores = attention.sum(dim=0)  # Sum over query positions (rows)
    return h2o_scores


def compare_token_selection(
    scores1: torch.Tensor,
    scores2: torch.Tensor,
    name1: str,
    name2: str,
    budget_ratios: list = [0.05, 0.1, 0.2, 0.3, 0.5]
) -> dict:
    """
    Compare token selection between two scoring methods at various budgets.

    Returns:
    - overlap_at_budget: Dict of budget_ratio -> overlap fraction
    - unique_to_1: Tokens selected by method 1 but not method 2
    - unique_to_2: Tokens selected by method 2 but not method 1
    - rank_correlation: Spearman correlation between full rankings
    """
    n = len(scores1)

    result = {
        'name1': name1,
        'name2': name2,
        'n': n,
        'overlap_at_budget': {},
        'unique_to_1_count': {},
        'unique_to_2_count': {},
    }

    # Get ranked indices for each method
    _, indices1 = torch.sort(scores1, descending=True)
    _, indices2 = torch.sort(scores2, descending=True)

    for ratio in budget_ratios:
        k = max(10, int(n * ratio))  # At least 10 tokens

        top_k_1 = set(indices1[:k].cpu().tolist())
        top_k_2 = set(indices2[:k].cpu().tolist())

        overlap = len(top_k_1 & top_k_2)
        result['overlap_at_budget'][ratio] = overlap / k
        result['unique_to_1_count'][ratio] = len(top_k_1 - top_k_2)
        result['unique_to_2_count'][ratio] = len(top_k_2 - top_k_1)

    # Spearman rank correlation
    ranks1 = torch.argsort(torch.argsort(scores1, descending=True)).float()
    ranks2 = torch.argsort(torch.argsort(scores2, descending=True)).float()

    corr_matrix = torch.corrcoef(torch.stack([ranks1, ranks2]))
    corr = corr_matrix[0, 1].item()
    if np.isnan(corr):
        corr = 1.0
    result['rank_correlation'] = corr

    return result


def analyze_qi_hi_differences(
    cumulative_qi: torch.Tensor,
    cumulative_hi: torch.Tensor,
    h2o_scores: torch.Tensor,
    sink_size: int = 4,
    max_k: int = 10
) -> dict:
    """
    Detailed analysis of QI vs HI vs H2O differences.

    Returns analysis of:
    - Where QI and HI diverge most
    - Token characteristics of QI-preferred vs HI-preferred tokens
    - Comparison with H2O baseline
    """
    n = len(h2o_scores)

    # Use final cumulative scores (k=max_k)
    qi_final = cumulative_qi[-1]
    hi_final = cumulative_hi[-1]

    # Also get k=1 scores for H2O-like comparison
    qi_k1 = cumulative_qi[0]
    hi_k1 = cumulative_hi[0]

    result = {
        'comparisons': {},
        'position_analysis': {},
        'evolution_analysis': {},
    }

    # 1. Pairwise comparisons at final k
    result['comparisons']['qi_vs_hi_final'] = compare_token_selection(
        qi_final, hi_final, 'QI_k10', 'HI_k10'
    )
    result['comparisons']['qi_vs_h2o'] = compare_token_selection(
        qi_final, h2o_scores, 'QI_k10', 'H2O'
    )
    result['comparisons']['hi_vs_h2o'] = compare_token_selection(
        hi_final, h2o_scores, 'HI_k10', 'H2O'
    )

    # Compare k=1 to H2O (should be most similar)
    result['comparisons']['qi_k1_vs_h2o'] = compare_token_selection(
        qi_k1, h2o_scores, 'QI_k1', 'H2O'
    )
    result['comparisons']['hi_k1_vs_h2o'] = compare_token_selection(
        hi_k1, h2o_scores, 'HI_k1', 'H2O'
    )

    # 2. Position analysis: Where in the sequence do QI vs HI diverge?
    # Get top 10% by each method
    budget = max(10, int(n * 0.1))
    _, qi_top_idx = torch.topk(qi_final, budget)
    _, hi_top_idx = torch.topk(hi_final, budget)
    _, h2o_top_idx = torch.topk(h2o_scores, budget)

    qi_only = set(qi_top_idx.cpu().tolist()) - set(hi_top_idx.cpu().tolist())
    hi_only = set(hi_top_idx.cpu().tolist()) - set(qi_top_idx.cpu().tolist())
    h2o_only = set(h2o_top_idx.cpu().tolist()) - set(qi_top_idx.cpu().tolist()) - set(hi_top_idx.cpu().tolist())

    # Position statistics (excluding sink)
    def position_stats(indices, n, sink_size):
        positions = [i for i in indices if i >= sink_size]
        if not positions:
            return {'mean_relative_pos': 0, 'std': 0, 'early_frac': 0, 'late_frac': 0}
        positions = np.array(positions)
        relative_pos = (positions - sink_size) / (n - sink_size)
        return {
            'mean_relative_pos': float(np.mean(relative_pos)),  # 0=start, 1=end
            'std': float(np.std(relative_pos)),
            'early_frac': float(np.mean(relative_pos < 0.33)),  # First third
            'late_frac': float(np.mean(relative_pos > 0.67)),   # Last third
        }

    result['position_analysis']['qi_only_tokens'] = position_stats(qi_only, n, sink_size)
    result['position_analysis']['hi_only_tokens'] = position_stats(hi_only, n, sink_size)
    result['position_analysis']['h2o_only_tokens'] = position_stats(h2o_only, n, sink_size)
    result['position_analysis']['qi_only_count'] = len(qi_only)
    result['position_analysis']['hi_only_count'] = len(hi_only)
    result['position_analysis']['h2o_only_count'] = len(h2o_only)

    # 3. Evolution analysis: How do QI and HI evolve differently?
    # Compute how much each method changes from k=1 to k=10
    qi_evolution = []
    hi_evolution = []

    for k in range(1, max_k):
        # Rank correlation between k and k+1
        qi_corr = compare_token_selection(
            cumulative_qi[k-1], cumulative_qi[k], f'QI_k{k}', f'QI_k{k+1}'
        )['rank_correlation']
        hi_corr = compare_token_selection(
            cumulative_hi[k-1], cumulative_hi[k], f'HI_k{k}', f'HI_k{k+1}'
        )['rank_correlation']
        qi_evolution.append(qi_corr)
        hi_evolution.append(hi_corr)

    result['evolution_analysis']['qi_stability'] = qi_evolution
    result['evolution_analysis']['hi_stability'] = hi_evolution
    result['evolution_analysis']['qi_avg_stability'] = float(np.mean(qi_evolution))
    result['evolution_analysis']['hi_avg_stability'] = float(np.mean(hi_evolution))

    return result


# =============================================================================
# SELECTION STRATEGY COMPARISON
# =============================================================================

def select_tokens_raw(scores: torch.Tensor, budget: int, name: str) -> dict:
    """Raw selection: top-k by a single scoring method."""
    _, indices = torch.topk(scores, budget)
    return {
        'name': name,
        'indices': set(indices.cpu().tolist()),
        'method': 'raw',
    }


def select_tokens_union(
    qi_scores: torch.Tensor,
    h2o_scores: torch.Tensor,
    budget: int,
    qi_ratio: float = 0.5
) -> dict:
    """
    Idea 2: Union Selection - guarantee coverage from both methods.
    Split budget between QI and H2O, take union.
    """
    qi_budget = int(budget * qi_ratio)
    h2o_budget = budget - qi_budget

    _, qi_top = torch.topk(qi_scores, qi_budget)
    _, h2o_top = torch.topk(h2o_scores, h2o_budget)

    # Union (may have overlap, so actual count could be less)
    qi_set = set(qi_top.cpu().tolist())
    h2o_set = set(h2o_top.cpu().tolist())
    union_set = qi_set | h2o_set

    # If union is smaller than budget due to overlap, add more from H2O
    if len(union_set) < budget:
        remaining = budget - len(union_set)
        _, h2o_all = torch.topk(h2o_scores, budget + remaining)
        for idx in h2o_all.cpu().tolist():
            if idx not in union_set:
                union_set.add(idx)
                if len(union_set) >= budget:
                    break

    return {
        'name': f'Union(QI:{qi_ratio:.0%}, H2O:{1-qi_ratio:.0%})',
        'indices': union_set,
        'method': 'union',
        'qi_ratio': qi_ratio,
        'qi_contribution': len(qi_set - h2o_set),
        'h2o_contribution': len(h2o_set - qi_set),
        'overlap': len(qi_set & h2o_set),
    }


def select_tokens_two_stage(
    qi_scores: torch.Tensor,
    h2o_scores: torch.Tensor,
    budget: int,
    candidate_ratio: float = 2.0
) -> dict:
    """
    Idea 3: Two-Stage Filtering.
    Stage 1: H2O selects candidate pool (candidate_ratio * budget)
    Stage 2: QI re-ranks within candidates to select final budget
    """
    candidate_budget = int(budget * candidate_ratio)

    # Stage 1: H2O candidates
    _, h2o_candidates = torch.topk(h2o_scores, candidate_budget)

    # Stage 2: QI re-rank within candidates
    qi_candidate_scores = qi_scores[h2o_candidates]
    _, top_within = torch.topk(qi_candidate_scores, budget)

    final_indices = h2o_candidates[top_within]

    return {
        'name': f'TwoStage(H2O→QI, {candidate_ratio:.1f}x)',
        'indices': set(final_indices.cpu().tolist()),
        'method': 'two_stage',
        'candidate_ratio': candidate_ratio,
        'candidates_from_h2o': candidate_budget,
    }


def compute_attention_mass_captured(
    attention: torch.Tensor,
    selected_indices: set,
    window_size: int = 64
) -> dict:
    """
    Compute what fraction of attention mass the selected tokens capture.

    This measures: if we keep only these tokens, how much of the
    query's attention is preserved?
    """
    n = attention.shape[0]

    # Get attention from last window_size positions (observation window)
    window_start = max(0, n - window_size)
    window_attn = attention[window_start:, :]  # [window, n]

    # Total attention mass (should be ~window_size due to softmax)
    total_mass = window_attn.sum().item()

    # Attention mass to selected tokens
    selected_mask = torch.zeros(n, device=attention.device)
    for idx in selected_indices:
        if idx < n:
            selected_mask[idx] = 1.0

    captured_mass = (window_attn * selected_mask.unsqueeze(0)).sum().item()

    return {
        'total_mass': total_mass,
        'captured_mass': captured_mass,
        'capture_ratio': captured_mass / (total_mass + 1e-10),
    }


def compare_selection_strategies(
    qi_scores: torch.Tensor,
    hi_scores: torch.Tensor,
    h2o_scores: torch.Tensor,
    attention: torch.Tensor,
    budget_ratios: list = [0.05, 0.1, 0.2],
    sink_size: int = 4,
    window_size: int = 64
) -> dict:
    """
    Compare different selection strategies across budget levels.

    Strategies:
    1. QI-only (raw)
    2. HI-only (raw)
    3. H2O-only (raw)
    4. Union 50/50 (QI + H2O)
    5. Union 30/70 (more H2O)
    6. Two-Stage 2x (H2O candidates → QI re-rank)
    7. Two-Stage 3x
    8. Current: MAX(QI, HI) approximation
    """
    n = len(qi_scores)
    results = {
        'budget_ratios': budget_ratios,
        'strategies': {},
    }

    for ratio in budget_ratios:
        budget = max(10, int(n * ratio))
        # Reserve sink and window
        effective_budget = budget - sink_size - window_size
        if effective_budget < 10:
            effective_budget = budget  # Small sequences

        budget_key = f'{int(ratio*100)}%'
        results['strategies'][budget_key] = {}

        # 1. Raw methods
        for name, scores in [('QI-only', qi_scores), ('HI-only', hi_scores), ('H2O-only', h2o_scores)]:
            selection = select_tokens_raw(scores, budget, name)
            attn_capture = compute_attention_mass_captured(attention, selection['indices'], window_size)
            results['strategies'][budget_key][name] = {
                'indices': selection['indices'],
                'attention_captured': attn_capture['capture_ratio'],
            }

        # 2. Union strategies
        for qi_ratio in [0.5, 0.3]:
            selection = select_tokens_union(qi_scores, h2o_scores, budget, qi_ratio)
            attn_capture = compute_attention_mass_captured(attention, selection['indices'], window_size)
            results['strategies'][budget_key][selection['name']] = {
                'indices': selection['indices'],
                'attention_captured': attn_capture['capture_ratio'],
                'qi_contribution': selection['qi_contribution'],
                'h2o_contribution': selection['h2o_contribution'],
            }

        # 3. Two-stage strategies
        for candidate_ratio in [2.0, 3.0]:
            selection = select_tokens_two_stage(qi_scores, h2o_scores, budget, candidate_ratio)
            attn_capture = compute_attention_mass_captured(attention, selection['indices'], window_size)
            results['strategies'][budget_key][selection['name']] = {
                'indices': selection['indices'],
                'attention_captured': attn_capture['capture_ratio'],
            }

        # 4. Current approach approximation: MAX(rank(QI), rank(HI))
        qi_ranks = torch.argsort(torch.argsort(qi_scores, descending=True)).float()
        hi_ranks = torch.argsort(torch.argsort(hi_scores, descending=True)).float()
        # Lower rank = better, so we want MIN of ranks for MAX importance
        combined_ranks = torch.min(qi_ranks, hi_ranks)
        _, max_qi_hi_indices = torch.topk(-combined_ranks, budget)  # Negate for topk
        max_qi_hi_set = set(max_qi_hi_indices.cpu().tolist())
        attn_capture = compute_attention_mass_captured(attention, max_qi_hi_set, window_size)
        results['strategies'][budget_key]['MAX(QI,HI)'] = {
            'indices': max_qi_hi_set,
            'attention_captured': attn_capture['capture_ratio'],
        }

    return results


def compute_strategy_diversity(strategy_results: dict, budget_key: str) -> dict:
    """Compute how different each strategy's selection is from others."""
    strategies = strategy_results['strategies'][budget_key]
    strategy_names = list(strategies.keys())

    diversity = {}
    for name in strategy_names:
        indices = strategies[name]['indices']
        overlaps = {}
        for other_name in strategy_names:
            if other_name != name:
                other_indices = strategies[other_name]['indices']
                overlap = len(indices & other_indices) / len(indices)
                overlaps[other_name] = overlap
        diversity[name] = {
            'avg_overlap_with_others': np.mean(list(overlaps.values())),
            'overlaps': overlaps,
        }

    return diversity


def plot_strategy_comparison(
    all_strategy_results: list,
    output_dir: str
):
    """Plot comparison of selection strategies across all samples."""
    if not all_strategy_results:
        return

    # Aggregate results
    budget_keys = all_strategy_results[0]['budget_ratios']
    strategy_names = list(all_strategy_results[0]['strategies'][f'{int(budget_keys[0]*100)}%'].keys())

    # Collect attention capture for each strategy at each budget
    fig, axes = plt.subplots(1, len(budget_keys), figsize=(6*len(budget_keys), 6))
    if len(budget_keys) == 1:
        axes = [axes]

    fig.suptitle('Selection Strategy Comparison: Attention Mass Captured', fontsize=14)

    for ax_idx, ratio in enumerate(budget_keys):
        budget_key = f'{int(ratio*100)}%'
        ax = axes[ax_idx]

        # Collect data
        strategy_captures = {name: [] for name in strategy_names}
        for result in all_strategy_results:
            for name in strategy_names:
                if name in result['strategies'][budget_key]:
                    strategy_captures[name].append(
                        result['strategies'][budget_key][name]['attention_captured']
                    )

        # Plot bars
        names = list(strategy_captures.keys())
        means = [np.mean(strategy_captures[n]) for n in names]
        stds = [np.std(strategy_captures[n]) for n in names]

        # Color code by type
        colors = []
        for name in names:
            if 'QI-only' in name:
                colors.append('blue')
            elif 'HI-only' in name:
                colors.append('red')
            elif 'H2O-only' in name:
                colors.append('green')
            elif 'Union' in name:
                colors.append('purple')
            elif 'TwoStage' in name:
                colors.append('orange')
            elif 'MAX' in name:
                colors.append('brown')
            else:
                colors.append('gray')

        x = np.arange(len(names))
        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=3)

        ax.set_xlabel('Strategy', fontsize=11)
        ax.set_ylabel('Attention Mass Captured', fontsize=11)
        ax.set_title(f'Budget = {budget_key}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'), dpi=150)
    plt.close()

    # Also create a summary table plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.set_title('Selection Strategy Summary (Attention Captured)', fontsize=14)

    # Build table data
    header = ['Strategy'] + [f'{int(r*100)}%' for r in budget_keys]
    table_data = [header]

    for name in strategy_names:
        row = [name]
        for ratio in budget_keys:
            budget_key = f'{int(ratio*100)}%'
            captures = [r['strategies'][budget_key][name]['attention_captured']
                       for r in all_strategy_results if name in r['strategies'][budget_key]]
            if captures:
                row.append(f'{np.mean(captures):.3f}±{np.std(captures):.3f}')
            else:
                row.append('N/A')
        table_data.append(row)

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color header
    for i in range(len(header)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Highlight best in each column
    for col_idx in range(1, len(header)):
        values = []
        for row_idx in range(1, len(table_data)):
            cell_text = table_data[row_idx][col_idx]
            if cell_text != 'N/A':
                values.append((row_idx, float(cell_text.split('±')[0])))
        if values:
            best_row = max(values, key=lambda x: x[1])[0]
            table[(best_row, col_idx)].set_facecolor('#90EE90')  # Light green

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_summary_table.png'), dpi=150)
    plt.close()


def analyze_distribution(scores: torch.Tensor, name: str) -> dict:
    """Analyze the distribution of scores."""
    scores_np = scores.cpu().numpy()
    scores_np = scores_np[scores_np > 1e-10]  # Remove zeros

    if len(scores_np) < 10:
        return {'name': name, 'valid': False}

    # Log-transform for power law analysis
    log_scores = np.log10(scores_np + 1e-10)

    # Basic stats
    result = {
        'name': name,
        'valid': True,
        'mean': float(np.mean(scores_np)),
        'std': float(np.std(scores_np)),
        'min': float(np.min(scores_np)),
        'max': float(np.max(scores_np)),
        'median': float(np.median(scores_np)),
        'p90': float(np.percentile(scores_np, 90)),
        'p99': float(np.percentile(scores_np, 99)),
        'skewness': float(stats.skew(scores_np)),
        'kurtosis': float(stats.kurtosis(scores_np)),
    }

    # Test for power law: fit log-log regression
    sorted_scores = np.sort(scores_np)[::-1]
    ranks = np.arange(1, len(sorted_scores) + 1)

    # Fit power law: score ~ rank^(-alpha)
    log_ranks = np.log10(ranks)
    log_sorted = np.log10(sorted_scores + 1e-10)

    # Linear regression on log-log
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_sorted)
    result['power_law_alpha'] = -slope  # Power law exponent
    result['power_law_r2'] = r_value ** 2
    result['power_law_pvalue'] = p_value

    return result


def compute_rank_persistence(cumulative_scores: torch.Tensor, top_k: int = 100) -> dict:
    """
    Analyze how top-k tokens at iteration 1 persist in later iterations.

    cumulative_scores: [max_iter, n] tensor
    """
    max_iter, n = cumulative_scores.shape

    # Get top-k indices at each iteration
    top_indices_per_k = []
    for k in range(max_iter):
        _, indices = torch.topk(cumulative_scores[k], min(top_k, n))
        top_indices_per_k.append(set(indices.cpu().tolist()))

    # Compute persistence: what fraction of k=1 top tokens are still in top at k=i
    k1_top = top_indices_per_k[0]
    persistence = []
    for k in range(max_iter):
        overlap = len(k1_top & top_indices_per_k[k])
        persistence.append(overlap / len(k1_top))

    # Compute pairwise Jaccard similarity
    jaccard_matrix = np.zeros((max_iter, max_iter))
    for i in range(max_iter):
        for j in range(max_iter):
            intersection = len(top_indices_per_k[i] & top_indices_per_k[j])
            union = len(top_indices_per_k[i] | top_indices_per_k[j])
            jaccard_matrix[i, j] = intersection / union if union > 0 else 0

    # Rank correlation between consecutive iterations
    rank_correlations = []
    for k in range(max_iter - 1):
        ranks_k = torch.argsort(torch.argsort(cumulative_scores[k], descending=True)).float()
        ranks_k1 = torch.argsort(torch.argsort(cumulative_scores[k + 1], descending=True)).float()
        corr_matrix = torch.corrcoef(torch.stack([ranks_k, ranks_k1]))
        corr = corr_matrix[0, 1].item()
        # Handle NaN (can happen if all values are identical)
        if np.isnan(corr):
            corr = 1.0  # Perfect correlation if no variance
        rank_correlations.append(corr)

    return {
        'persistence_from_k1': persistence,
        'jaccard_matrix': jaccard_matrix,
        'rank_correlations': rank_correlations,
    }


def plot_distributions(cumulative_qi: torch.Tensor, cumulative_hi: torch.Tensor,
                       output_dir: str, sample_id: int):
    """Plot score distributions for each k."""
    max_k = cumulative_qi.shape[0]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'Score Distributions at Each Neumann Iteration (Sample {sample_id})', fontsize=14)

    for k in range(max_k):
        ax = axes[k // 5, k % 5]

        qi = cumulative_qi[k].cpu().numpy()
        hi = cumulative_hi[k].cpu().numpy()

        # Remove near-zero values for log scale
        qi = qi[qi > 1e-8]
        hi = hi[hi > 1e-8]

        # Plot histograms
        ax.hist(np.log10(qi + 1e-10), bins=50, alpha=0.6, label='QI', density=True)
        ax.hist(np.log10(hi + 1e-10), bins=50, alpha=0.6, label='HI', density=True)
        ax.set_xlabel('log10(score)')
        ax.set_ylabel('Density')
        ax.set_title(f'k={k+1}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'neumann_distribution_sample{sample_id}.png'), dpi=150)
    plt.close()


def plot_rank_score_curve(cumulative_hi: torch.Tensor, output_dir: str, sample_id: int):
    """Plot rank vs score curve (like H2O paper) for each k."""
    max_k = cumulative_hi.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, max_k))

    for k in range(max_k):
        scores = cumulative_hi[k].cpu().numpy()
        sorted_scores = np.sort(scores)[::-1]
        ranks = np.arange(1, len(sorted_scores) + 1)

        # Subsample for plotting
        step = max(1, len(ranks) // 500)
        ax.loglog(ranks[::step], sorted_scores[::step],
                  color=colors[k], label=f'k={k+1}', alpha=0.8, linewidth=1.5)

    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Rank-Score Curve (Power Law Test) - Sample {sample_id}', fontsize=14)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'neumann_rank_score_sample{sample_id}.png'), dpi=150)
    plt.close()


def plot_persistence(persistence_data: dict, output_dir: str, sample_id: int):
    """Plot rank persistence analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Persistence from k=1
    ax = axes[0]
    k_values = range(1, len(persistence_data['persistence_from_k1']) + 1)
    ax.plot(k_values, persistence_data['persistence_from_k1'], 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Neumann Iteration k', fontsize=12)
    ax.set_ylabel('Fraction of k=1 Top-100 Still in Top-100', fontsize=10)
    ax.set_title('Top Token Persistence from k=1', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()

    # 2. Jaccard similarity heatmap
    ax = axes[1]
    im = ax.imshow(persistence_data['jaccard_matrix'], cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('Iteration k', fontsize=12)
    ax.set_ylabel('Iteration k', fontsize=12)
    ax.set_title('Jaccard Similarity of Top-100 Sets', fontsize=12)
    ax.set_xticks(range(len(persistence_data['jaccard_matrix'])))
    ax.set_yticks(range(len(persistence_data['jaccard_matrix'])))
    ax.set_xticklabels(range(1, len(persistence_data['jaccard_matrix']) + 1))
    ax.set_yticklabels(range(1, len(persistence_data['jaccard_matrix']) + 1))
    plt.colorbar(im, ax=ax, label='Jaccard Index')

    # 3. Rank correlation between consecutive iterations
    ax = axes[2]
    k_values = range(1, len(persistence_data['rank_correlations']) + 1)
    ax.bar(k_values, persistence_data['rank_correlations'], color='steelblue', alpha=0.8)
    ax.set_xlabel('Transition (k → k+1)', fontsize=12)
    ax.set_ylabel('Spearman Rank Correlation', fontsize=10)
    ax.set_title('Rank Stability Between Iterations', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'neumann_persistence_sample{sample_id}.png'), dpi=150)
    plt.close()


def plot_score_heatmap(cumulative_hi: torch.Tensor, output_dir: str, sample_id: int):
    """Plot heatmap of scores across k and token positions."""
    max_k, n = cumulative_hi.shape

    # Subsample tokens for visualization
    step = max(1, n // 200)
    scores_subset = cumulative_hi[:, ::step].cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Normalize each row for better visualization
    scores_norm = scores_subset / (scores_subset.max(axis=1, keepdims=True) + 1e-10)

    im = ax.imshow(scores_norm, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Token Position (subsampled)', fontsize=12)
    ax.set_ylabel('Neumann Iteration k', fontsize=12)
    ax.set_title(f'Normalized HI Scores Across Iterations - Sample {sample_id}', fontsize=14)
    ax.set_yticks(range(max_k))
    ax.set_yticklabels(range(1, max_k + 1))
    plt.colorbar(im, ax=ax, label='Normalized Score')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'neumann_heatmap_sample{sample_id}.png'), dpi=150)
    plt.close()


def plot_qi_hi_h2o_comparison(
    cumulative_qi: torch.Tensor,
    cumulative_hi: torch.Tensor,
    h2o_scores: torch.Tensor,
    comparison_result: dict,
    output_dir: str,
    sample_id: int
):
    """Plot detailed comparison of QI vs HI vs H2O."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'QI vs HI vs H2O Comparison - Sample {sample_id}', fontsize=14)

    n = len(h2o_scores)
    qi_final = cumulative_qi[-1].cpu().numpy()
    hi_final = cumulative_hi[-1].cpu().numpy()
    h2o = h2o_scores.cpu().numpy()

    # 1. Scatter plot: QI vs HI
    ax = axes[0, 0]
    # Subsample for visualization
    step = max(1, n // 500)
    ax.scatter(qi_final[::step], hi_final[::step], alpha=0.3, s=10)
    corr = comparison_result['comparisons']['qi_vs_hi_final']['rank_correlation']
    ax.set_xlabel('QI Score (k=10)', fontsize=11)
    ax.set_ylabel('HI Score (k=10)', fontsize=11)
    ax.set_title(f'QI vs HI (rank corr={corr:.3f})', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 2. Scatter plot: QI vs H2O
    ax = axes[0, 1]
    ax.scatter(qi_final[::step], h2o[::step], alpha=0.3, s=10, color='green')
    corr = comparison_result['comparisons']['qi_vs_h2o']['rank_correlation']
    ax.set_xlabel('QI Score (k=10)', fontsize=11)
    ax.set_ylabel('H2O Score', fontsize=11)
    ax.set_title(f'QI vs H2O (rank corr={corr:.3f})', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 3. Scatter plot: HI vs H2O
    ax = axes[0, 2]
    ax.scatter(hi_final[::step], h2o[::step], alpha=0.3, s=10, color='orange')
    corr = comparison_result['comparisons']['hi_vs_h2o']['rank_correlation']
    ax.set_xlabel('HI Score (k=10)', fontsize=11)
    ax.set_ylabel('H2O Score', fontsize=11)
    ax.set_title(f'HI vs H2O (rank corr={corr:.3f})', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 4. Overlap at different budgets
    ax = axes[1, 0]
    budgets = sorted(comparison_result['comparisons']['qi_vs_hi_final']['overlap_at_budget'].keys())
    qi_hi_overlap = [comparison_result['comparisons']['qi_vs_hi_final']['overlap_at_budget'][b] for b in budgets]
    qi_h2o_overlap = [comparison_result['comparisons']['qi_vs_h2o']['overlap_at_budget'][b] for b in budgets]
    hi_h2o_overlap = [comparison_result['comparisons']['hi_vs_h2o']['overlap_at_budget'][b] for b in budgets]

    x = np.arange(len(budgets))
    width = 0.25
    ax.bar(x - width, qi_hi_overlap, width, label='QI vs HI', color='blue', alpha=0.7)
    ax.bar(x, qi_h2o_overlap, width, label='QI vs H2O', color='green', alpha=0.7)
    ax.bar(x + width, hi_h2o_overlap, width, label='HI vs H2O', color='orange', alpha=0.7)
    ax.set_xlabel('Budget Ratio', fontsize=11)
    ax.set_ylabel('Overlap Fraction', fontsize=11)
    ax.set_title('Token Selection Overlap at Different Budgets', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(b*100)}%' for b in budgets])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Position distribution of unique tokens
    ax = axes[1, 1]
    pos_analysis = comparison_result['position_analysis']
    categories = ['QI-only', 'HI-only', 'H2O-only']
    early_fracs = [pos_analysis['qi_only_tokens']['early_frac'],
                   pos_analysis['hi_only_tokens']['early_frac'],
                   pos_analysis['h2o_only_tokens']['early_frac']]
    late_fracs = [pos_analysis['qi_only_tokens']['late_frac'],
                  pos_analysis['hi_only_tokens']['late_frac'],
                  pos_analysis['h2o_only_tokens']['late_frac']]

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, early_fracs, width, label='Early (first 33%)', color='lightblue')
    ax.bar(x + width/2, late_fracs, width, label='Late (last 33%)', color='salmon')
    ax.set_xlabel('Method-Unique Tokens', fontsize=11)
    ax.set_ylabel('Fraction', fontsize=11)
    ax.set_title('Position Distribution of Unique Tokens', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Stability over iterations (QI vs HI)
    ax = axes[1, 2]
    evol = comparison_result['evolution_analysis']
    k_values = range(1, len(evol['qi_stability']) + 1)
    ax.plot(k_values, evol['qi_stability'], 'b-o', label=f"QI (avg={evol['qi_avg_stability']:.3f})", linewidth=2)
    ax.plot(k_values, evol['hi_stability'], 'r-s', label=f"HI (avg={evol['hi_avg_stability']:.3f})", linewidth=2)
    ax.set_xlabel('Transition (k → k+1)', fontsize=11)
    ax.set_ylabel('Rank Correlation', fontsize=11)
    ax.set_title('Score Stability: QI vs HI', fontsize=12)
    ax.legend()
    ax.set_ylim(0.9, 1.005)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'qi_hi_h2o_comparison_sample{sample_id}.png'), dpi=150)
    plt.close()


def plot_score_ranks_venn(
    cumulative_qi: torch.Tensor,
    cumulative_hi: torch.Tensor,
    h2o_scores: torch.Tensor,
    output_dir: str,
    sample_id: int,
    top_ratio: float = 0.1
):
    """Plot showing which tokens are selected by each method."""
    n = len(h2o_scores)
    budget = max(10, int(n * top_ratio))

    qi_final = cumulative_qi[-1]
    hi_final = cumulative_hi[-1]

    _, qi_top = torch.topk(qi_final, budget)
    _, hi_top = torch.topk(hi_final, budget)
    _, h2o_top = torch.topk(h2o_scores, budget)

    qi_set = set(qi_top.cpu().tolist())
    hi_set = set(hi_top.cpu().tolist())
    h2o_set = set(h2o_top.cpu().tolist())

    # Calculate overlaps
    qi_only = len(qi_set - hi_set - h2o_set)
    hi_only = len(hi_set - qi_set - h2o_set)
    h2o_only = len(h2o_set - qi_set - hi_set)
    qi_hi = len((qi_set & hi_set) - h2o_set)
    qi_h2o = len((qi_set & h2o_set) - hi_set)
    hi_h2o = len((hi_set & h2o_set) - qi_set)
    all_three = len(qi_set & hi_set & h2o_set)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Venn-style summary as table
    ax = axes[0]
    ax.axis('off')
    ax.set_title(f'Token Selection Overlap (Top {int(top_ratio*100)}% = {budget} tokens)', fontsize=12)

    table_data = [
        ['Category', 'Count', '% of Budget'],
        ['All three (QI ∩ HI ∩ H2O)', str(all_three), f'{all_three/budget*100:.1f}%'],
        ['QI ∩ HI only', str(qi_hi), f'{qi_hi/budget*100:.1f}%'],
        ['QI ∩ H2O only', str(qi_h2o), f'{qi_h2o/budget*100:.1f}%'],
        ['HI ∩ H2O only', str(hi_h2o), f'{hi_h2o/budget*100:.1f}%'],
        ['QI only', str(qi_only), f'{qi_only/budget*100:.1f}%'],
        ['HI only', str(hi_only), f'{hi_only/budget*100:.1f}%'],
        ['H2O only', str(h2o_only), f'{h2o_only/budget*100:.1f}%'],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Color header
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # 2. Visualization: Score profiles of differently-selected tokens
    ax = axes[1]

    # Normalize scores for comparison
    qi_norm = (qi_final - qi_final.min()) / (qi_final.max() - qi_final.min() + 1e-10)
    hi_norm = (hi_final - hi_final.min()) / (hi_final.max() - hi_final.min() + 1e-10)
    h2o_norm = (h2o_scores - h2o_scores.min()) / (h2o_scores.max() - h2o_scores.min() + 1e-10)

    # Sample tokens from each category
    def get_avg_scores(token_set, qi, hi, h2o):
        if not token_set:
            return [0, 0, 0]
        tokens = list(token_set)[:50]  # Max 50 for average
        return [
            float(qi[tokens].mean()),
            float(hi[tokens].mean()),
            float(h2o[tokens].mean())
        ]

    categories = ['All Three', 'QI∩HI only', 'QI∩H2O only', 'HI∩H2O only', 'QI only', 'HI only', 'H2O only']
    token_sets = [
        qi_set & hi_set & h2o_set,
        (qi_set & hi_set) - h2o_set,
        (qi_set & h2o_set) - hi_set,
        (hi_set & h2o_set) - qi_set,
        qi_set - hi_set - h2o_set,
        hi_set - qi_set - h2o_set,
        h2o_set - qi_set - hi_set,
    ]

    avg_scores = [get_avg_scores(ts, qi_norm, hi_norm, h2o_norm) for ts in token_sets]

    x = np.arange(len(categories))
    width = 0.25
    ax.bar(x - width, [s[0] for s in avg_scores], width, label='QI (norm)', color='blue', alpha=0.7)
    ax.bar(x, [s[1] for s in avg_scores], width, label='HI (norm)', color='red', alpha=0.7)
    ax.bar(x + width, [s[2] for s in avg_scores], width, label='H2O (norm)', color='green', alpha=0.7)

    ax.set_xlabel('Token Category', fontsize=11)
    ax.set_ylabel('Average Normalized Score', fontsize=11)
    ax.set_title('Score Profile by Selection Category', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'qi_hi_h2o_venn_sample{sample_id}.png'), dpi=150)
    plt.close()

    return {
        'all_three': all_three,
        'qi_hi': qi_hi,
        'qi_h2o': qi_h2o,
        'hi_h2o': hi_h2o,
        'qi_only': qi_only,
        'hi_only': hi_only,
        'h2o_only': h2o_only,
        'budget': budget,
    }


def plot_aggregated_qi_hi_h2o(all_comparison_results: list, output_dir: str):
    """Plot aggregated QI vs HI vs H2O comparison across all samples."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Aggregated QI vs HI vs H2O Analysis', fontsize=14)

    # 1. Average overlap at different budgets
    ax = axes[0, 0]
    budgets = sorted(all_comparison_results[0]['comparisons']['qi_vs_hi_final']['overlap_at_budget'].keys())

    qi_hi_overlaps = []
    qi_h2o_overlaps = []
    hi_h2o_overlaps = []

    for b in budgets:
        qi_hi_overlaps.append([r['comparisons']['qi_vs_hi_final']['overlap_at_budget'][b] for r in all_comparison_results])
        qi_h2o_overlaps.append([r['comparisons']['qi_vs_h2o']['overlap_at_budget'][b] for r in all_comparison_results])
        hi_h2o_overlaps.append([r['comparisons']['hi_vs_h2o']['overlap_at_budget'][b] for r in all_comparison_results])

    x = np.arange(len(budgets))
    width = 0.25
    ax.bar(x - width, [np.mean(o) for o in qi_hi_overlaps], width,
           yerr=[np.std(o) for o in qi_hi_overlaps], label='QI vs HI', color='blue', alpha=0.7, capsize=3)
    ax.bar(x, [np.mean(o) for o in qi_h2o_overlaps], width,
           yerr=[np.std(o) for o in qi_h2o_overlaps], label='QI vs H2O', color='green', alpha=0.7, capsize=3)
    ax.bar(x + width, [np.mean(o) for o in hi_h2o_overlaps], width,
           yerr=[np.std(o) for o in hi_h2o_overlaps], label='HI vs H2O', color='orange', alpha=0.7, capsize=3)
    ax.set_xlabel('Budget Ratio', fontsize=11)
    ax.set_ylabel('Overlap Fraction (± std)', fontsize=11)
    ax.set_title('Average Token Selection Overlap', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(b*100)}%' for b in budgets])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Rank correlations
    ax = axes[0, 1]
    comparisons = ['QI vs HI', 'QI vs H2O', 'HI vs H2O', 'QI_k1 vs H2O', 'HI_k1 vs H2O']
    corr_keys = ['qi_vs_hi_final', 'qi_vs_h2o', 'hi_vs_h2o', 'qi_k1_vs_h2o', 'hi_k1_vs_h2o']

    corr_values = []
    corr_stds = []
    for key in corr_keys:
        corrs = [r['comparisons'][key]['rank_correlation'] for r in all_comparison_results]
        corr_values.append(np.mean(corrs))
        corr_stds.append(np.std(corrs))

    colors = ['blue', 'green', 'orange', 'lightgreen', 'lightsalmon']
    x = np.arange(len(comparisons))
    bars = ax.bar(x, corr_values, yerr=corr_stds, color=colors, alpha=0.7, capsize=5)
    ax.set_xlabel('Comparison', fontsize=11)
    ax.set_ylabel('Rank Correlation (± std)', fontsize=11)
    ax.set_title('Score Rank Correlations', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(comparisons, rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, corr_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 3. Evolution stability comparison
    ax = axes[1, 0]
    all_qi_stab = np.array([r['evolution_analysis']['qi_stability'] for r in all_comparison_results])
    all_hi_stab = np.array([r['evolution_analysis']['hi_stability'] for r in all_comparison_results])

    k_values = range(1, all_qi_stab.shape[1] + 1)
    ax.errorbar(k_values, all_qi_stab.mean(axis=0), yerr=all_qi_stab.std(axis=0),
                fmt='b-o', label='QI stability', linewidth=2, capsize=3)
    ax.errorbar(k_values, all_hi_stab.mean(axis=0), yerr=all_hi_stab.std(axis=0),
                fmt='r-s', label='HI stability', linewidth=2, capsize=3)
    ax.set_xlabel('Transition (k → k+1)', fontsize=11)
    ax.set_ylabel('Rank Correlation (± std)', fontsize=11)
    ax.set_title('Score Evolution: QI vs HI', fontsize=12)
    ax.legend()
    ax.set_ylim(0.9, 1.005)
    ax.grid(True, alpha=0.3)

    # 4. Position analysis summary
    ax = axes[1, 1]
    qi_early = [r['position_analysis']['qi_only_tokens']['early_frac'] for r in all_comparison_results]
    qi_late = [r['position_analysis']['qi_only_tokens']['late_frac'] for r in all_comparison_results]
    hi_early = [r['position_analysis']['hi_only_tokens']['early_frac'] for r in all_comparison_results]
    hi_late = [r['position_analysis']['hi_only_tokens']['late_frac'] for r in all_comparison_results]
    h2o_early = [r['position_analysis']['h2o_only_tokens']['early_frac'] for r in all_comparison_results]
    h2o_late = [r['position_analysis']['h2o_only_tokens']['late_frac'] for r in all_comparison_results]

    categories = ['QI-only', 'HI-only', 'H2O-only']
    early_means = [np.mean(qi_early), np.mean(hi_early), np.mean(h2o_early)]
    early_stds = [np.std(qi_early), np.std(hi_early), np.std(h2o_early)]
    late_means = [np.mean(qi_late), np.mean(hi_late), np.mean(h2o_late)]
    late_stds = [np.std(qi_late), np.std(hi_late), np.std(h2o_late)]

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, early_means, width, yerr=early_stds, label='Early (first 33%)',
           color='lightblue', capsize=3)
    ax.bar(x + width/2, late_means, width, yerr=late_stds, label='Late (last 33%)',
           color='salmon', capsize=3)
    ax.set_xlabel('Method-Unique Tokens', fontsize=11)
    ax.set_ylabel('Fraction (± std)', fontsize=11)
    ax.set_title('Position Bias of Method-Specific Tokens', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qi_hi_h2o_aggregated.png'), dpi=150)
    plt.close()


def plot_aggregated_analysis(all_analyses: list, output_dir: str):
    """Plot aggregated analysis across all samples."""

    # Aggregate power law exponents
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Power law alpha across k
    ax = axes[0, 0]
    for sample_data in all_analyses:
        alphas = [sample_data['hi_analysis'][k]['power_law_alpha']
                  for k in range(len(sample_data['hi_analysis']))
                  if sample_data['hi_analysis'][k]['valid']]
        ax.plot(range(1, len(alphas) + 1), alphas, 'o-', alpha=0.5)
    ax.set_xlabel('Neumann Iteration k', fontsize=12)
    ax.set_ylabel('Power Law Exponent (alpha)', fontsize=12)
    ax.set_title('Power Law Exponent vs Iteration', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 2. R² of power law fit across k
    ax = axes[0, 1]
    for sample_data in all_analyses:
        r2s = [sample_data['hi_analysis'][k]['power_law_r2']
               for k in range(len(sample_data['hi_analysis']))
               if sample_data['hi_analysis'][k]['valid']]
        ax.plot(range(1, len(r2s) + 1), r2s, 'o-', alpha=0.5)
    ax.set_xlabel('Neumann Iteration k', fontsize=12)
    ax.set_ylabel('Power Law R²', fontsize=12)
    ax.set_title('Power Law Fit Quality vs Iteration', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # 3. Average persistence from k=1
    ax = axes[1, 0]
    all_persistence = np.array([d['persistence']['persistence_from_k1'] for d in all_analyses])
    mean_persistence = all_persistence.mean(axis=0)
    std_persistence = all_persistence.std(axis=0)
    k_values = range(1, len(mean_persistence) + 1)
    ax.errorbar(k_values, mean_persistence, yerr=std_persistence, fmt='bo-',
                linewidth=2, markersize=8, capsize=5)
    ax.fill_between(k_values, mean_persistence - std_persistence,
                    mean_persistence + std_persistence, alpha=0.2)
    ax.set_xlabel('Neumann Iteration k', fontsize=12)
    ax.set_ylabel('Persistence from k=1', fontsize=12)
    ax.set_title('Average Top-100 Persistence (± std)', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # 4. Average rank correlation
    ax = axes[1, 1]
    all_corr = np.array([d['persistence']['rank_correlations'] for d in all_analyses])
    mean_corr = all_corr.mean(axis=0)
    std_corr = all_corr.std(axis=0)
    k_values = range(1, len(mean_corr) + 1)
    ax.bar(k_values, mean_corr, yerr=std_corr, color='steelblue', alpha=0.8, capsize=5)
    ax.set_xlabel('Transition (k → k+1)', fontsize=12)
    ax.set_ylabel('Rank Correlation', fontsize=12)
    ax.set_title('Average Rank Stability (± std)', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neumann_aggregated_analysis.png'), dpi=150)
    plt.close()


def get_attention_from_model(model, tokenizer, text: str, device: str) -> torch.Tensor:
    """Get attention matrix from model for given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Get attention from middle layer, average across heads
    # attentions is tuple of [batch, heads, seq, seq] for each layer
    num_layers = len(outputs.attentions)
    mid_layer = num_layers // 2

    # Average across heads, squeeze batch
    attn = outputs.attentions[mid_layer][0].mean(dim=0)  # [seq, seq]

    # Detach and move to CPU to free GPU memory for next sample
    return attn.detach().cpu()


def main():
    parser = argparse.ArgumentParser(description='Neumann Series Distribution Analysis')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Model to use for attention extraction')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to analyze')
    parser.add_argument('--output_dir', type=str, default='neumann_analysis_output',
                        help='Output directory for plots and reports')
    parser.add_argument('--max_k', type=int, default=10,
                        help='Maximum Neumann iterations')
    parser.add_argument('--sink_size', type=int, default=4,
                        help='Absorbing boundary size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Use synthetic attention matrices instead of real model')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=" * 60)
    print(f"Neumann Series Distribution Analysis")
    print(f"=" * 60)
    print(f"Model: {args.model}")
    print(f"Samples: {args.num_samples}")
    print(f"Max k: {args.max_k}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print(f"=" * 60)

    if args.use_synthetic:
        print("\nUsing synthetic attention matrices...")
        attention_matrices = []
        for i in range(args.num_samples):
            n = np.random.randint(500, 2000)
            # Create causal attention-like matrix
            attn = torch.zeros(n, n)
            for row in range(n):
                # Causal: only attend to previous tokens
                weights = torch.softmax(torch.randn(row + 1) * 2, dim=0)
                attn[row, :row + 1] = weights
            attention_matrices.append(attn.to(args.device))
            print(f"  Generated synthetic attention matrix {i+1}: [{n}, {n}]")
    else:
        print("\nLoading model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map=args.device,
            trust_remote_code=True,
            attn_implementation="eager",  # Need eager for attention outputs
        )
        model.eval()

        print("Loading dataset samples...")
        # Use pg19 (Project Gutenberg) - standard parquet format, no loading scripts
        dataset = load_dataset("emozilla/pg19", split="train", streaming=True)

        attention_matrices = []
        for i, sample in enumerate(dataset):
            if i >= args.num_samples:
                break
            text = sample['text'][:8000]  # Truncate for memory
            print(f"  Processing sample {i+1}: {len(text)} chars...")
            attn = get_attention_from_model(model, tokenizer, text, args.device)
            attention_matrices.append(attn)
            print(f"    Attention shape: {attn.shape}")

        # Free model memory
        del model
        torch.cuda.empty_cache()

    # Analyze each sample
    all_analyses = []
    all_comparison_results = []
    all_venn_results = []
    all_strategy_results = []

    for sample_id, attn in enumerate(attention_matrices):
        print(f"\nAnalyzing sample {sample_id + 1}/{len(attention_matrices)}...")
        try:
            n = attn.shape[0]
            query_idx = n - 1

            # Move attention to compute device for analysis
            attn_device = attn.to(args.device)

            # Compute H2O scores first (needs original attention)
            print(f"  Computing H2O scores...")
            h2o_scores = compute_h2o_scores(attn_device)

            # Compute Neumann scores at each k
            print(f"  Computing Neumann scores (k=1 to k={args.max_k})...")
            scores = compute_neumann_scores_per_iteration(
                attn_device, query_idx,
                sink_size=args.sink_size,
                max_k=args.max_k
            )

            # Free GPU memory
            del attn_device
            if args.device == 'cuda':
                torch.cuda.empty_cache()

            # Analyze distributions
            hi_analysis = []
            qi_analysis = []
            h2o_analysis = analyze_distribution(h2o_scores, 'H2O')

            for k in range(args.max_k):
                hi_analysis.append(analyze_distribution(scores['cumulative_hi'][k], f'HI_k{k+1}'))
                qi_analysis.append(analyze_distribution(scores['cumulative_qi'][k], f'QI_k{k+1}'))

            # Analyze persistence
            persistence = compute_rank_persistence(scores['cumulative_hi'], top_k=100)

            # NEW: Analyze QI vs HI vs H2O differences
            print(f"  Analyzing QI vs HI vs H2O differences...")
            comparison_result = analyze_qi_hi_differences(
                scores['cumulative_qi'],
                scores['cumulative_hi'],
                h2o_scores,
                sink_size=args.sink_size,
                max_k=args.max_k
            )
            all_comparison_results.append(comparison_result)

            # Store analysis
            sample_analysis = {
                'sample_id': sample_id,
                'n': n,
                'hi_analysis': hi_analysis,
                'qi_analysis': qi_analysis,
                'h2o_analysis': h2o_analysis,
                'persistence': persistence,
            }
            all_analyses.append(sample_analysis)

            # Generate plots for this sample
            print(f"  Generating plots...")
            plot_distributions(scores['cumulative_qi'], scores['cumulative_hi'],
                              args.output_dir, sample_id)
            plot_rank_score_curve(scores['cumulative_hi'], args.output_dir, sample_id)
            plot_persistence(persistence, args.output_dir, sample_id)
            plot_score_heatmap(scores['cumulative_hi'], args.output_dir, sample_id)

            # NEW: QI vs HI vs H2O comparison plots
            print(f"  Generating QI vs HI vs H2O comparison plots...")
            plot_qi_hi_h2o_comparison(
                scores['cumulative_qi'],
                scores['cumulative_hi'],
                h2o_scores,
                comparison_result,
                args.output_dir,
                sample_id
            )
            venn_result = plot_score_ranks_venn(
                scores['cumulative_qi'],
                scores['cumulative_hi'],
                h2o_scores,
                args.output_dir,
                sample_id
            )
            all_venn_results.append(venn_result)

            # NEW: Selection strategy comparison
            print(f"  Comparing selection strategies...")
            strategy_result = compare_selection_strategies(
                qi_scores=scores['cumulative_qi'][-1],  # Final QI (k=10)
                hi_scores=scores['cumulative_hi'][-1],  # Final HI (k=10)
                h2o_scores=h2o_scores,
                attention=attn.to(scores['cumulative_qi'].device),
                budget_ratios=[0.05, 0.1, 0.2],
                sink_size=args.sink_size,
                window_size=64
            )
            all_strategy_results.append(strategy_result)

        except Exception as e:
            print(f"  ERROR processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate aggregated analysis
    if len(all_analyses) == 0:
        print("\nERROR: No samples were successfully analyzed!")
        return

    print(f"\nGenerating aggregated analysis from {len(all_analyses)} samples...")
    plot_aggregated_analysis(all_analyses, args.output_dir)

    # NEW: Aggregated QI vs HI vs H2O analysis
    if len(all_comparison_results) > 0:
        print(f"Generating aggregated QI vs HI vs H2O analysis...")
        plot_aggregated_qi_hi_h2o(all_comparison_results, args.output_dir)

    # NEW: Selection strategy comparison
    if len(all_strategy_results) > 0:
        print(f"Generating selection strategy comparison...")
        plot_strategy_comparison(all_strategy_results, args.output_dir)

    # Write report
    report_path = os.path.join(args.output_dir, 'neumann_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("NEUMANN SERIES DISTRIBUTION ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("CONFIGURATION:\n")
        f.write(f"  Model: {args.model}\n")
        f.write(f"  Samples: {args.num_samples}\n")
        f.write(f"  Max k: {args.max_k}\n")
        f.write(f"  Sink size: {args.sink_size}\n\n")

        # Aggregate statistics
        f.write("=" * 70 + "\n")
        f.write("POWER LAW ANALYSIS (HI Scores)\n")
        f.write("=" * 70 + "\n\n")

        for k in range(args.max_k):
            alphas = [d['hi_analysis'][k]['power_law_alpha'] for d in all_analyses if d['hi_analysis'][k]['valid']]
            r2s = [d['hi_analysis'][k]['power_law_r2'] for d in all_analyses if d['hi_analysis'][k]['valid']]
            if alphas:
                f.write(f"k={k+1}:\n")
                f.write(f"  Power Law Alpha: {np.mean(alphas):.3f} +/- {np.std(alphas):.3f}\n")
                f.write(f"  Power Law R²:    {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("RANK PERSISTENCE ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        all_persistence = np.array([d['persistence']['persistence_from_k1'] for d in all_analyses])
        mean_persistence = all_persistence.mean(axis=0)

        f.write("Fraction of k=1 Top-100 tokens remaining in Top-100 at each k:\n")
        for k in range(args.max_k):
            f.write(f"  k={k+1}: {mean_persistence[k]:.3f}\n")

        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("QI vs HI vs H2O COMPARISON\n")
        f.write("=" * 70 + "\n\n")

        if len(all_comparison_results) > 0:
            # Rank correlations
            f.write("RANK CORRELATIONS (averaged across samples):\n")
            corr_pairs = [
                ('qi_vs_hi_final', 'QI_k10 vs HI_k10'),
                ('qi_vs_h2o', 'QI_k10 vs H2O'),
                ('hi_vs_h2o', 'HI_k10 vs H2O'),
                ('qi_k1_vs_h2o', 'QI_k1 vs H2O'),
                ('hi_k1_vs_h2o', 'HI_k1 vs H2O'),
            ]
            for key, label in corr_pairs:
                corrs = [r['comparisons'][key]['rank_correlation'] for r in all_comparison_results]
                f.write(f"  {label}: {np.mean(corrs):.3f} +/- {np.std(corrs):.3f}\n")

            # Token overlap at 10% budget
            f.write("\nTOKEN SELECTION OVERLAP (at 10% budget):\n")
            for key, label in [('qi_vs_hi_final', 'QI vs HI'), ('qi_vs_h2o', 'QI vs H2O'), ('hi_vs_h2o', 'HI vs H2O')]:
                overlaps = [r['comparisons'][key]['overlap_at_budget'][0.1] for r in all_comparison_results]
                f.write(f"  {label}: {np.mean(overlaps):.1%} +/- {np.std(overlaps):.1%}\n")

            # Position analysis
            f.write("\nPOSITION BIAS OF METHOD-UNIQUE TOKENS:\n")
            for method, key_prefix in [('QI-only', 'qi_only_tokens'), ('HI-only', 'hi_only_tokens'), ('H2O-only', 'h2o_only_tokens')]:
                early = [r['position_analysis'][key_prefix]['early_frac'] for r in all_comparison_results]
                late = [r['position_analysis'][key_prefix]['late_frac'] for r in all_comparison_results]
                f.write(f"  {method}: early={np.mean(early):.1%}, late={np.mean(late):.1%}\n")

            # Evolution stability
            f.write("\nSCORE EVOLUTION STABILITY (k→k+1 transitions):\n")
            qi_stab = [r['evolution_analysis']['qi_avg_stability'] for r in all_comparison_results]
            hi_stab = [r['evolution_analysis']['hi_avg_stability'] for r in all_comparison_results]
            f.write(f"  QI average stability: {np.mean(qi_stab):.4f}\n")
            f.write(f"  HI average stability: {np.mean(hi_stab):.4f}\n")

        if len(all_venn_results) > 0:
            f.write("\nTOKEN SELECTION VENN ANALYSIS (at 10% budget):\n")
            avg_all_three = np.mean([v['all_three']/v['budget'] for v in all_venn_results])
            avg_qi_only = np.mean([v['qi_only']/v['budget'] for v in all_venn_results])
            avg_hi_only = np.mean([v['hi_only']/v['budget'] for v in all_venn_results])
            avg_h2o_only = np.mean([v['h2o_only']/v['budget'] for v in all_venn_results])
            f.write(f"  All three (QI ∩ HI ∩ H2O): {avg_all_three:.1%}\n")
            f.write(f"  QI-only (not in HI or H2O): {avg_qi_only:.1%}\n")
            f.write(f"  HI-only (not in QI or H2O): {avg_hi_only:.1%}\n")
            f.write(f"  H2O-only (not in QI or HI): {avg_h2o_only:.1%}\n")

        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 70 + "\n\n")

        # Determine if progressive filtering is viable
        avg_persistence_k10 = mean_persistence[-1]
        if avg_persistence_k10 > 0.7:
            f.write("FINDING 1: HIGH PERSISTENCE\n")
            f.write(f"  {avg_persistence_k10:.1%} of k=1 top tokens remain in top at k=10\n")
            f.write("  -> Progressive filtering is VIABLE: early filtering won't remove important tokens\n\n")
        elif avg_persistence_k10 > 0.4:
            f.write("FINDING 1: MODERATE PERSISTENCE\n")
            f.write(f"  {avg_persistence_k10:.1%} of k=1 top tokens remain in top at k=10\n")
            f.write("  -> Progressive filtering needs CAREFUL thresholding\n\n")
        else:
            f.write("FINDING 1: LOW PERSISTENCE\n")
            f.write(f"  Only {avg_persistence_k10:.1%} of k=1 top tokens remain in top at k=10\n")
            f.write("  -> Scores change significantly: k=1 and k=10 capture DIFFERENT tokens\n")
            f.write("  -> This suggests multi-hop importance is distinct from single-hop\n\n")

        # Check power law
        valid_r2s = [d['hi_analysis'][0]['power_law_r2'] for d in all_analyses if d['hi_analysis'][0]['valid']]
        avg_r2_k1 = np.mean(valid_r2s) if valid_r2s else 0.0
        if avg_r2_k1 > 0.9:
            f.write("FINDING 2: STRONG POWER LAW at k=1\n")
            f.write(f"  R² = {avg_r2_k1:.3f} (similar to H2O observation)\n")
            f.write("  -> Validates connection between k=1 Neumann and H2O\n\n")

        # NEW: QI vs HI insight
        if len(all_comparison_results) > 0:
            qi_h2o_corrs = [r['comparisons']['qi_vs_h2o']['rank_correlation'] for r in all_comparison_results]
            hi_h2o_corrs = [r['comparisons']['hi_vs_h2o']['rank_correlation'] for r in all_comparison_results]
            avg_qi_h2o = np.mean(qi_h2o_corrs)
            avg_hi_h2o = np.mean(hi_h2o_corrs)

            f.write("FINDING 3: QI vs HI RELATIONSHIP TO H2O\n")
            if avg_hi_h2o > avg_qi_h2o + 0.05:
                f.write(f"  HI is MORE similar to H2O (corr={avg_hi_h2o:.3f}) than QI (corr={avg_qi_h2o:.3f})\n")
                f.write("  -> HI captures 'global hub' importance similar to H2O's attention-received metric\n")
                f.write("  -> QI provides DIFFERENT information: query-specific transitive importance\n\n")
            elif avg_qi_h2o > avg_hi_h2o + 0.05:
                f.write(f"  QI is MORE similar to H2O (corr={avg_qi_h2o:.3f}) than HI (corr={avg_hi_h2o:.3f})\n")
                f.write("  -> QI captures direct query importance similar to H2O\n")
                f.write("  -> HI provides DIFFERENT information: global hub structure\n\n")
            else:
                f.write(f"  QI and HI are SIMILARLY correlated with H2O (QI={avg_qi_h2o:.3f}, HI={avg_hi_h2o:.3f})\n")
                f.write("  -> Both capture similar information to H2O's attention-based importance\n\n")

            # Check what's unique
            if len(all_venn_results) > 0:
                avg_qi_only = np.mean([v['qi_only']/v['budget'] for v in all_venn_results])
                avg_hi_only = np.mean([v['hi_only']/v['budget'] for v in all_venn_results])

                f.write("FINDING 4: METHOD-UNIQUE TOKEN SELECTION\n")
                f.write(f"  QI-only tokens: {avg_qi_only:.1%} of budget\n")
                f.write(f"  HI-only tokens: {avg_hi_only:.1%} of budget\n")
                if avg_qi_only > 0.05 or avg_hi_only > 0.05:
                    f.write("  -> There are tokens that ONLY one method captures!\n")
                    f.write("  -> Combining QI and HI may capture complementary information\n\n")
                else:
                    f.write("  -> QI and HI largely select the SAME tokens\n")
                    f.write("  -> Little benefit from combining both methods\n\n")

        # NEW: Selection strategy comparison
        if len(all_strategy_results) > 0:
            f.write("=" * 70 + "\n")
            f.write("SELECTION STRATEGY COMPARISON\n")
            f.write("=" * 70 + "\n\n")

            f.write("Strategies tested:\n")
            f.write("  1. QI-only: Top-k by Query Importance\n")
            f.write("  2. HI-only: Top-k by Hub Importance\n")
            f.write("  3. H2O-only: Top-k by attention column sums\n")
            f.write("  4. Union(50/50): Half from QI, half from H2O\n")
            f.write("  5. Union(30/70): 30% QI, 70% H2O\n")
            f.write("  6. TwoStage(2x): H2O selects 2x candidates, QI re-ranks\n")
            f.write("  7. TwoStage(3x): H2O selects 3x candidates, QI re-ranks\n")
            f.write("  8. MAX(QI,HI): Current approach (element-wise max of ranks)\n\n")

            f.write("ATTENTION MASS CAPTURED (averaged across samples):\n\n")

            budget_keys = [f'{int(r*100)}%' for r in all_strategy_results[0]['budget_ratios']]
            strategy_names = list(all_strategy_results[0]['strategies'][budget_keys[0]].keys())

            for budget_key in budget_keys:
                f.write(f"Budget = {budget_key}:\n")
                strategy_scores = []
                for name in strategy_names:
                    captures = [r['strategies'][budget_key][name]['attention_captured']
                               for r in all_strategy_results if name in r['strategies'][budget_key]]
                    if captures:
                        mean_cap = np.mean(captures)
                        std_cap = np.std(captures)
                        strategy_scores.append((name, mean_cap, std_cap))

                # Sort by mean capture (descending)
                strategy_scores.sort(key=lambda x: -x[1])

                for rank, (name, mean_cap, std_cap) in enumerate(strategy_scores, 1):
                    marker = " <-- BEST" if rank == 1 else ""
                    f.write(f"  {rank}. {name}: {mean_cap:.3f} ± {std_cap:.3f}{marker}\n")
                f.write("\n")

            # Determine winner
            best_strategies = []
            for budget_key in budget_keys:
                strategy_scores = []
                for name in strategy_names:
                    captures = [r['strategies'][budget_key][name]['attention_captured']
                               for r in all_strategy_results if name in r['strategies'][budget_key]]
                    if captures:
                        strategy_scores.append((name, np.mean(captures)))
                if strategy_scores:
                    best = max(strategy_scores, key=lambda x: x[1])
                    best_strategies.append((budget_key, best[0], best[1]))

            f.write("BEST STRATEGY PER BUDGET:\n")
            for budget_key, best_name, best_score in best_strategies:
                f.write(f"  {budget_key}: {best_name} ({best_score:.3f})\n")

            f.write("\n")
            f.write("RECOMMENDATION:\n")
            # Check if two-stage or union wins
            two_stage_wins = sum(1 for _, name, _ in best_strategies if 'TwoStage' in name)
            union_wins = sum(1 for _, name, _ in best_strategies if 'Union' in name)
            if two_stage_wins > len(best_strategies) / 2:
                f.write("  -> Two-Stage filtering (H2O candidates → QI re-rank) performs best!\n")
                f.write("  -> This validates Idea 3: use H2O for candidate generation, QI for selection\n")
            elif union_wins > len(best_strategies) / 2:
                f.write("  -> Union selection (split budget between QI and H2O) performs best!\n")
                f.write("  -> This validates Idea 2: guaranteed coverage from both methods\n")
            else:
                f.write("  -> Results are mixed; may need task-specific tuning\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("GENERATED FILES\n")
        f.write("=" * 70 + "\n\n")
        f.write("Per-sample plots:\n")
        for i in range(len(all_analyses)):
            f.write(f"  - neumann_distribution_sample{i}.png\n")
            f.write(f"  - neumann_rank_score_sample{i}.png\n")
            f.write(f"  - neumann_persistence_sample{i}.png\n")
            f.write(f"  - neumann_heatmap_sample{i}.png\n")
            f.write(f"  - qi_hi_h2o_comparison_sample{i}.png\n")
            f.write(f"  - qi_hi_h2o_venn_sample{i}.png\n")
        f.write("\nAggregated analysis:\n")
        f.write("  - neumann_aggregated_analysis.png\n")
        f.write("  - qi_hi_h2o_aggregated.png\n")
        f.write("  - strategy_comparison.png\n")
        f.write("  - strategy_summary_table.png\n")
        f.write("  - neumann_analysis_report.txt (this file)\n")

    print(f"\n{'=' * 60}")
    print(f"Analysis complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"Report: {report_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
