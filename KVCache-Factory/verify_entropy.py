 #!/usr/bin/env python3
"""
Entropy-Aware Markov Construction PoC (v6.5.0)

Hypothesis: Multi-hop reasoning relies on "Induction Heads" which are sharp (low entropy).
By averaging ALL heads, we drown the "Bridge Signal" in noise.

Solution: Use top-K sharpest heads (lowest entropy) for QI, all heads for HI.

This script compares:
  A) Baseline (v6.2.0): P = Average(All Heads)
  B) Experiment (v6.5.0): P_qi = Average(Top-8 Sharpest), P_hi = Average(All)

Usage:
    python verify_entropy.py --model_path Qwen/Qwen2.5-7B-Instruct --n_samples 20

For RTX 6000 (48GB), should handle 7B models comfortably.
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EntropyConfig:
    """Configuration for entropy-aware Markov construction."""
    top_k_heads: int = 8          # Number of sharpest heads to use for QI
    sink_size: int = 4            # Absorbing boundary (first N tokens)
    window_size: int = 64         # Local attention window
    neumann_iterations: int = 10  # Iterations for Neumann series
    budget: int = 2048            # KV cache budget
    temperature: float = 1.0      # Attention sharpening (lower = sharper)


# ============================================================================
# Core Entropy-Aware Functions
# ============================================================================

def compute_head_entropy(attn_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute entropy for each attention head.

    Sharp heads (induction heads) have LOW entropy - they focus on few tokens.
    Diffuse heads have HIGH entropy - they spread attention uniformly.

    Args:
        attn_weights: [bsz, num_heads, window_size, seq_len] attention after softmax
        eps: Small value to avoid log(0)

    Returns:
        entropy: [num_heads] entropy for each head (averaged over batch and queries)
    """
    # CRITICAL: Convert to float32 to avoid NaN from float16 log operations
    attn_f32 = attn_weights.float()

    # Clamp to avoid log(0) - use larger eps for numerical stability
    attn_clamped = attn_f32.clamp(min=eps)

    # H = -sum(p * log(p)) for each query position
    # Shape: [bsz, num_heads, window_size]
    log_attn = torch.log(attn_clamped)
    entropy_per_query = -(attn_clamped * log_attn).sum(dim=-1)

    # Average over batch and query positions
    # Shape: [num_heads]
    entropy_per_head = entropy_per_query.mean(dim=(0, 2))

    return entropy_per_head


def select_sharp_heads(entropy: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Select the top-K sharpest heads (lowest entropy).

    Args:
        entropy: [num_heads] entropy for each head
        top_k: Number of heads to select

    Returns:
        indices: [top_k] indices of selected heads
    """
    # Lower entropy = sharper attention
    _, indices = torch.topk(entropy, k=top_k, largest=False)
    return indices


def build_transition_matrix(
    attn_weights: torch.Tensor,
    head_indices: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Build transition matrix P from attention weights.

    If head_indices is provided, only use those heads.
    Otherwise, use all heads.

    Args:
        attn_weights: [bsz, num_heads, window_size, seq_len]
        head_indices: Optional [k] indices of heads to use
        temperature: Sharpening factor (lower = sharper)

    Returns:
        P: [window_size, seq_len] transition matrix (averaged)
    """
    if head_indices is not None:
        # Select only specified heads
        attn_subset = attn_weights[:, head_indices, :, :]
    else:
        attn_subset = attn_weights

    # Average over batch and selected heads
    # Shape: [window_size, seq_len]
    P = attn_subset.mean(dim=(0, 1))

    # Apply temperature sharpening
    if temperature != 1.0 and temperature > 0:
        P = P ** (1.0 / temperature)
        # Re-normalize rows
        P = P / (P.sum(dim=-1, keepdim=True).clamp(min=1e-8))

    return P


def compute_neumann_importance(
    full_attn: torch.Tensor,
    query_idx: int,
    sink_size: int = 4,
    num_iterations: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute QI and HI using Neumann series on the transition matrix.

    This is the core absorbing Markov chain computation.

    Args:
        full_attn: [seq_len, seq_len] full attention/transition matrix
        query_idx: Query position (usually seq_len - 1)
        sink_size: First N tokens are absorbing states
        num_iterations: Neumann series iterations

    Returns:
        qi_scores: [seq_len] Query Importance scores
        hi_scores: [seq_len] Hub Importance scores
    """
    n = full_attn.shape[0]
    device = full_attn.device

    if n <= sink_size:
        uniform = torch.ones(n, device=device, dtype=torch.float32) / n
        return uniform, uniform

    # Normalize to transition matrix
    row_sums = full_attn.sum(dim=1, keepdim=True).clamp(min=1e-8)
    P = full_attn / row_sums

    # Extract Q (transient-to-transient)
    n_transient = n - sink_size
    Q = P[sink_size:, sink_size:].contiguous()

    # Query in transient space
    query_transient_idx = query_idx - sink_size
    if query_transient_idx < 0:
        uniform = torch.ones(n, device=device, dtype=torch.float32) / n
        return uniform, uniform

    # Initialize QI (one-hot at query)
    v_qi = torch.zeros(n_transient, device=device, dtype=torch.float32)
    v_qi[query_transient_idx] = 1.0
    result_qi = v_qi.clone()

    # Initialize HI (uniform)
    v_hi = torch.ones(n_transient, device=device, dtype=torch.float32) / n_transient
    result_hi = v_hi.clone()

    # Neumann series: N ≈ I + Q + Q² + ...
    for _ in range(num_iterations):
        v_qi = torch.mv(Q.t(), v_qi)
        v_hi = torch.mv(Q.t(), v_hi)
        result_qi = result_qi + v_qi
        result_hi = result_hi + v_hi

        if v_qi.abs().max().item() < 1e-8 and v_hi.abs().max().item() < 1e-8:
            break

    # Map back to full sequence
    qi_scores = torch.zeros(n, device=device, dtype=torch.float32)
    qi_scores[sink_size:] = result_qi
    qi_scores[:sink_size] = result_qi.sum() * 0.01

    hi_scores = torch.zeros(n, device=device, dtype=torch.float32)
    hi_scores[sink_size:] = result_hi
    hi_scores[:sink_size] = result_hi.sum() * 0.01

    # Normalize to [0, 1]
    qi_max = qi_scores.max()
    if qi_max > 0:
        qi_scores = qi_scores / qi_max

    hi_max = hi_scores.max()
    if hi_max > 0:
        hi_scores = hi_scores / hi_max

    return qi_scores, hi_scores


def rank_normalize(scores: torch.Tensor) -> torch.Tensor:
    """Convert scores to rank-based percentiles."""
    n = scores.numel()
    if n == 0:
        return scores
    ranks = torch.argsort(torch.argsort(scores)).float()
    return ranks / (n - 1) if n > 1 else ranks


def select_tokens_by_importance(
    qi_scores: torch.Tensor,
    hi_scores: torch.Tensor,
    budget: int,
    sink_size: int,
    window_size: int,
) -> torch.Tensor:
    """
    Select top tokens using MAX(rank(QI), rank(HI)).

    Returns:
        keep_mask: [seq_len] boolean mask of tokens to keep
    """
    n = qi_scores.shape[0]
    device = qi_scores.device

    # Always keep sink + local window
    keep_mask = torch.zeros(n, dtype=torch.bool, device=device)
    keep_mask[:sink_size] = True
    local_start = max(sink_size, n - window_size)
    keep_mask[local_start:] = True

    current_kept = keep_mask.sum().item()
    remaining = max(0, budget - current_kept)

    if remaining == 0:
        return keep_mask

    # Combined score: MAX(rank(QI), rank(HI))
    qi_rank = rank_normalize(qi_scores)
    hi_rank = rank_normalize(hi_scores)
    combined = torch.maximum(qi_rank, hi_rank)

    # Mask already kept positions
    combined[keep_mask] = -float('inf')

    # Select top remaining
    _, topk_indices = torch.topk(combined, min(remaining, (~keep_mask).sum().item()))
    keep_mask[topk_indices] = True

    return keep_mask


# ============================================================================
# Entropy-Aware Markov (v6.5.0) - The Novel Contribution
# ============================================================================

def entropy_aware_importance(
    attn_weights: torch.Tensor,
    seq_len: int,
    config: EntropyConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    v6.5.0: Entropy-Aware Markov Construction.

    Key insight: Use DIFFERENT head subsets for QI vs HI.
    - QI (Query Importance): Use top-K SHARPEST heads (low entropy)
      → Captures induction/reasoning paths
    - HI (Hub Importance): Use ALL heads
      → Captures global context/hubs

    Args:
        attn_weights: [bsz, num_heads, window_size, seq_len] raw attention
        seq_len: Current sequence length
        config: EntropyConfig with parameters

    Returns:
        qi_scores: [seq_len] Query Importance from sharp heads
        hi_scores: [seq_len] Hub Importance from all heads
        diagnostics: Dict with entropy stats for analysis
    """
    device = attn_weights.device
    num_heads = attn_weights.shape[1]
    window_size = attn_weights.shape[2]

    # Step 1: Compute entropy for each head
    head_entropy = compute_head_entropy(attn_weights)

    # Step 2: Select sharpest heads for QI
    top_k = min(config.top_k_heads, num_heads)
    sharp_indices = select_sharp_heads(head_entropy, top_k)

    # Step 3: Build separate transition matrices
    # P_qi: Only sharp heads (for bridge detection)
    P_qi_window = build_transition_matrix(
        attn_weights, sharp_indices, config.temperature
    )

    # P_hi: All heads (for global context)
    P_hi_window = build_transition_matrix(
        attn_weights, None, config.temperature
    )

    # Step 4: Build full attention matrices
    # For positions outside window, use H2O-weighted transitions
    full_attn_qi = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)
    full_attn_hi = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)

    # Fill window portion
    full_attn_qi[-window_size:, :] = P_qi_window
    full_attn_hi[-window_size:, :] = P_hi_window

    # For prefix, use H2O-based transitions
    n_prefix = seq_len - window_size
    if n_prefix > 1:
        # Use all-heads P for H2O scores (global importance)
        h2o_scores = P_hi_window.sum(dim=0)[:n_prefix].clone()
        h2o_scores = h2o_scores.clamp(min=1e-6)

        cumsum = h2o_scores.cumsum(dim=0)
        denom = torch.zeros(n_prefix, device=device, dtype=torch.float32)
        denom[1:] = cumsum[:-1]
        denom[0] = 1.0

        h2o_expanded = h2o_scores.unsqueeze(0).expand(n_prefix, n_prefix)
        denom_expanded = denom.unsqueeze(1).expand(n_prefix, n_prefix)
        h2o_trans = h2o_expanded / (denom_expanded + 1e-8)

        mask = torch.tril(torch.ones(n_prefix, n_prefix, device=device), diagonal=-1)
        full_attn_qi[:n_prefix, :n_prefix] = h2o_trans * mask
        full_attn_hi[:n_prefix, :n_prefix] = h2o_trans * mask

    # Step 5: Compute importance scores
    query_idx = seq_len - 1

    qi_scores, _ = compute_neumann_importance(
        full_attn_qi, query_idx, config.sink_size, config.neumann_iterations
    )

    _, hi_scores = compute_neumann_importance(
        full_attn_hi, query_idx, config.sink_size, config.neumann_iterations
    )

    # Diagnostics
    diagnostics = {
        'head_entropy': head_entropy.cpu().tolist(),
        'sharp_head_indices': sharp_indices.cpu().tolist(),
        'entropy_min': head_entropy.min().item(),
        'entropy_max': head_entropy.max().item(),
        'entropy_mean': head_entropy.mean().item(),
        'sharp_entropy_mean': head_entropy[sharp_indices].mean().item(),
        'diffuse_entropy_mean': head_entropy.mean().item(),  # All heads
    }

    return qi_scores, hi_scores, diagnostics


def baseline_importance(
    attn_weights: torch.Tensor,
    seq_len: int,
    config: EntropyConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    v6.2.0 Baseline: Use all heads for both QI and HI.

    This is the current approach - average ALL heads.
    """
    device = attn_weights.device
    window_size = attn_weights.shape[2]

    # Build single P from all heads
    P_window = build_transition_matrix(attn_weights, None, config.temperature)

    # Build full attention matrix
    full_attn = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)
    full_attn[-window_size:, :] = P_window

    n_prefix = seq_len - window_size
    if n_prefix > 1:
        h2o_scores = P_window.sum(dim=0)[:n_prefix].clone()
        h2o_scores = h2o_scores.clamp(min=1e-6)

        cumsum = h2o_scores.cumsum(dim=0)
        denom = torch.zeros(n_prefix, device=device, dtype=torch.float32)
        denom[1:] = cumsum[:-1]
        denom[0] = 1.0

        h2o_expanded = h2o_scores.unsqueeze(0).expand(n_prefix, n_prefix)
        denom_expanded = denom.unsqueeze(1).expand(n_prefix, n_prefix)
        h2o_trans = h2o_expanded / (denom_expanded + 1e-8)

        mask = torch.tril(torch.ones(n_prefix, n_prefix, device=device), diagonal=-1)
        full_attn[:n_prefix, :n_prefix] = h2o_trans * mask

    query_idx = seq_len - 1
    qi_scores, hi_scores = compute_neumann_importance(
        full_attn, query_idx, config.sink_size, config.neumann_iterations
    )

    return qi_scores, hi_scores


# ============================================================================
# Data Loading
# ============================================================================

def load_longbench_data(dataset_name: str, data_dir: str, n_samples: int = 20) -> List[Dict]:
    """Load samples from LongBench dataset."""
    filepath = Path(data_dir) / "data" / "LongBench" / f"{dataset_name}.jsonl"

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            samples.append(json.loads(line))

    return samples


def format_prompt(sample: Dict, dataset_name: str) -> str:
    """Format sample into prompt string."""
    prompts = {
        "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
    }

    template = prompts.get(dataset_name, "{context}\n\n{input}")
    return template.format(context=sample['context'], input=sample['input'])


# ============================================================================
# Main Analysis
# ============================================================================

class AttentionCaptureHook:
    """Hook to capture attention weights from a single layer (memory-efficient)."""

    def __init__(self):
        self.attention = None

    def __call__(self, module, args, kwargs, output):  # noqa: ARG002
        # output is (hidden_states, attn_weights, past_key_value) when output_attentions=True
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            # Clone and detach to avoid holding computation graph
            self.attention = output[1].detach()
        return output

    def clear(self):
        if self.attention is not None:
            del self.attention
        self.attention = None


@torch.no_grad()
def analyze_sample(
    model,
    tokenizer,
    sample: Dict,
    dataset_name: str,
    config: EntropyConfig,
    attention_hook: AttentionCaptureHook,
    _target_layer_idx: int,  # Used for documentation; hook already registered
) -> Dict:
    """
    Analyze a single sample, comparing baseline vs entropy-aware approach.

    Returns detailed diagnostics about attention patterns and token selection.

    MEMORY FIX: Uses a hook to capture attention from ONLY the target layer,
    instead of output_attentions=True which stores ALL layers.
    """
    # Format and tokenize
    prompt = format_prompt(sample, dataset_name)

    # Truncate to reasonable length to avoid OOM
    # We need enough context for multi-hop but not so much we OOM
    max_len = 4096  # Reduced from 8192 for memory safety
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs.input_ids.to(model.device)
    seq_len = input_ids.shape[1]

    if seq_len < config.budget:
        return {
            'skipped': True,
            'reason': f'seq_len ({seq_len}) < budget ({config.budget})',
            'seq_len': seq_len,
        }

    # Clear previous attention capture
    attention_hook.clear()

    # Forward pass - hook captures attention from target layer only
    # We enable output_attentions but the hook captures before it accumulates
    try:
        _ = model(
            input_ids,
            output_attentions=True,  # Needed for hook to receive attention
            use_cache=False,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            'skipped': True,
            'reason': f'OOM at seq_len={seq_len}',
            'seq_len': seq_len,
        }

    # Get captured attention from hook
    if attention_hook.attention is None:
        return {
            'skipped': True,
            'reason': 'Hook failed to capture attention',
            'seq_len': seq_len,
        }

    attn = attention_hook.attention  # [bsz, num_heads, seq_len, seq_len]

    # Extract window portion for analysis
    window_size = min(config.window_size, seq_len)
    attn_window = attn[:, :, -window_size:, :]  # [bsz, heads, window, seq_len]

    # Clear to free memory
    attention_hook.clear()
    torch.cuda.empty_cache()

    # Baseline (v6.2.0): All heads
    qi_baseline, hi_baseline = baseline_importance(attn_window, seq_len, config)

    # Entropy-aware (v6.5.0): Sharp heads for QI
    qi_entropy, hi_entropy, diagnostics = entropy_aware_importance(attn_window, seq_len, config)

    # Compare token selection
    keep_baseline = select_tokens_by_importance(
        qi_baseline, hi_baseline, config.budget, config.sink_size, config.window_size
    )
    keep_entropy = select_tokens_by_importance(
        qi_entropy, hi_entropy, config.budget, config.sink_size, config.window_size
    )

    # Compute overlap and differences
    both_keep = (keep_baseline & keep_entropy).sum().item()
    only_baseline = (keep_baseline & ~keep_entropy).sum().item()
    only_entropy = (~keep_baseline & keep_entropy).sum().item()

    # Analyze which positions differ
    diff_positions = (keep_baseline != keep_entropy).nonzero(as_tuple=True)[0]

    # Score correlation
    qi_corr = torch.corrcoef(torch.stack([qi_baseline, qi_entropy]))[0, 1].item()
    hi_corr = torch.corrcoef(torch.stack([hi_baseline, hi_entropy]))[0, 1].item()

    return {
        'skipped': False,
        'seq_len': seq_len,
        'num_kept_baseline': keep_baseline.sum().item(),
        'num_kept_entropy': keep_entropy.sum().item(),
        'overlap': both_keep,
        'only_baseline': only_baseline,
        'only_entropy': only_entropy,
        'jaccard_similarity': both_keep / (both_keep + only_baseline + only_entropy + 1e-8),
        'qi_correlation': qi_corr,
        'hi_correlation': hi_corr,
        'entropy_diagnostics': diagnostics,
        'diff_positions': diff_positions.cpu().tolist()[:20],  # First 20 diff positions
    }


def main():
    parser = argparse.ArgumentParser(description="Entropy-Aware Markov Construction PoC")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model to use")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of samples per dataset")
    parser.add_argument("--top_k_heads", type=int, default=8,
                        help="Number of sharpest heads for QI")
    parser.add_argument("--budget", type=int, default=2048,
                        help="KV cache budget")
    parser.add_argument("--data_dir", type=str, default=".",
                        help="Directory containing data/LongBench/")
    parser.add_argument("--output", type=str, default="entropy_analysis.json",
                        help="Output file for results")
    args = parser.parse_args()

    print("=" * 70)
    print("Entropy-Aware Markov Construction PoC (v6.5.0)")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Top-K Sharp Heads: {args.top_k_heads}")
    print(f"Budget: {args.budget}")
    print(f"Samples per dataset: {args.n_samples}")
    print("=" * 70)

    # Config
    config = EntropyConfig(
        top_k_heads=args.top_k_heads,
        budget=args.budget,
    )

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Need eager for attention weights
    )
    model.eval()

    # Set up attention capture hook on middle layer
    # Qwen2.5-7B has 28 layers, so middle is layer 14
    n_layers = len(model.model.layers)
    target_layer_idx = n_layers // 2
    print(f"Capturing attention from layer {target_layer_idx} of {n_layers}")

    attention_hook = AttentionCaptureHook()
    hook_handle = model.model.layers[target_layer_idx].self_attn.register_forward_hook(
        attention_hook, with_kwargs=True
    )

    # Datasets to analyze
    datasets_to_test = [
        ("musique", "Multi-hop QA (TARGET - should improve)"),
        ("repobench-p", "Code completion (CONTROL - should stay stable)"),
    ]

    all_results = {}

    for dataset_name, description in datasets_to_test:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"Description: {description}")
        print("=" * 70)

        try:
            samples = load_longbench_data(dataset_name, args.data_dir, args.n_samples)
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            continue

        results = []
        entropy_stats = defaultdict(list)

        for i, sample in enumerate(tqdm(samples, desc=f"Analyzing {dataset_name}")):
            result = analyze_sample(
                model, tokenizer, sample, dataset_name, config,
                attention_hook, target_layer_idx
            )
            results.append(result)

            if not result['skipped']:
                entropy_stats['jaccard'].append(result['jaccard_similarity'])
                entropy_stats['qi_corr'].append(result['qi_correlation'])
                entropy_stats['hi_corr'].append(result['hi_correlation'])
                entropy_stats['only_entropy'].append(result['only_entropy'])

                diag = result['entropy_diagnostics']
                entropy_stats['entropy_min'].append(diag['entropy_min'])
                entropy_stats['entropy_max'].append(diag['entropy_max'])
                entropy_stats['sharp_mean'].append(diag['sharp_entropy_mean'])

        # Summary statistics
        n_analyzed = len([r for r in results if not r['skipped']])

        print(f"\n{dataset_name} Summary ({n_analyzed} samples analyzed):")
        print("-" * 50)

        if n_analyzed > 0:
            avg_jaccard = sum(entropy_stats['jaccard']) / n_analyzed
            avg_qi_corr = sum(entropy_stats['qi_corr']) / n_analyzed
            avg_hi_corr = sum(entropy_stats['hi_corr']) / n_analyzed
            avg_only_entropy = sum(entropy_stats['only_entropy']) / n_analyzed
            avg_entropy_min = sum(entropy_stats['entropy_min']) / n_analyzed
            avg_entropy_max = sum(entropy_stats['entropy_max']) / n_analyzed
            avg_sharp_mean = sum(entropy_stats['sharp_mean']) / n_analyzed

            print(f"Token Selection Overlap (Jaccard): {avg_jaccard:.3f}")
            print(f"QI Score Correlation: {avg_qi_corr:.3f}")
            print(f"HI Score Correlation: {avg_hi_corr:.3f}")
            print(f"Avg tokens ONLY in entropy-aware: {avg_only_entropy:.1f}")
            print(f"Head Entropy Range: [{avg_entropy_min:.3f}, {avg_entropy_max:.3f}]")
            print(f"Sharp Heads Avg Entropy: {avg_sharp_mean:.3f}")

            # Key insight
            entropy_gap = avg_entropy_max - avg_entropy_min
            print(f"\n>>> Entropy Gap (max - min): {entropy_gap:.3f}")
            if entropy_gap > 1.0:
                print("    SIGNIFICANT head specialization detected!")
                print("    Sharp heads may carry distinct 'bridge' signal.")
            else:
                print("    Heads are relatively uniform in entropy.")

        all_results[dataset_name] = {
            'samples': results,
            'summary': dict(entropy_stats),
        }

    # Save results
    print(f"\n{'='*70}")
    print(f"Saving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Final recommendation
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("""
Key Questions Answered:

1. Does building the Markov Chain on "Sharp Heads" alone allow the Query
   to find the Bridge Entity in 'musique'?

   → Check if 'only_entropy' count is significant for musique
   → High count = entropy-aware finds DIFFERENT tokens (potential bridges)
   → Look at 'diff_positions' to see WHERE selections differ

2. Does this preserve the "Pure Markov" nature of the work?

   → YES. We modify ONLY the graph construction (which heads to average).
   → No layer-wise heuristics, no external knowledge.
   → The Neumann series and absorbing chain math remain unchanged.
   → This is "topology refinement" within the Markov framework.

Next Steps:
- If musique shows high differentiation and repobench-p shows stability,
  implement entropy_aware_P in pyramidkv_utils.py for full benchmark.
- Run v6.5.0 on H200 with full LongBench.
""")

    # Cleanup
    hook_handle.remove()


if __name__ == "__main__":
    main()
