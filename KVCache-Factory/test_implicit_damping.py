#!/usr/bin/env python3
"""
Test Gemini's "Implicit Damping" Hypothesis

Hypothesis: Heavy sink attention causes Q (transient-to-transient) to have
row sums ~0.1, which exponentially attenuates multi-hop signals:
- Q   ~ 0.1  (1-hop)
- Q²  ~ 0.01 (2-hop bridge)
- Q³  ~ 0.001 (3-hop)

This script empirically measures:
1. Row sums of Q matrix
2. Spectral radius of Q
3. Frobenius norms of Q^k for k=1,2,3,...
4. Contribution of each Neumann term to total N

Usage:
    python test_implicit_damping.py --model_path Qwen/Qwen2.5-7B-Instruct --n_samples 5
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict


# ============================================================================
# Attention Capture
# ============================================================================

class AttentionCaptureHook:
    """Hook to capture attention weights from a single layer."""

    def __init__(self):
        self.attention = None

    def __call__(self, module, args, kwargs, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            self.attention = output[1].detach()
        return output

    def clear(self):
        if self.attention is not None:
            del self.attention
        self.attention = None


# ============================================================================
# Markov Chain Analysis
# ============================================================================

def analyze_Q_matrix(attn_weights: torch.Tensor, sink_size: int = 4, window_size: int = 32):
    """
    Analyze the transient-to-transient matrix Q from attention weights.

    Args:
        attn_weights: [bsz, num_heads, seq_len, seq_len] attention after softmax
        sink_size: Number of sink tokens at start
        window_size: Number of local window tokens at end

    Returns:
        Dictionary with analysis results
    """
    # Average over batch and heads
    # Shape: [seq_len, seq_len]
    P = attn_weights.float().mean(dim=(0, 1))
    seq_len = P.shape[0]

    # Define absorbing and transient states
    # Absorbing: sink tokens (first sink_size) + local window (last window_size)
    # Transient: everything in between

    local_start = max(sink_size, seq_len - window_size)

    # Transient indices: from sink_size to local_start
    if local_start <= sink_size:
        # No transient tokens
        return {
            'valid': False,
            'reason': 'No transient tokens (seq too short)',
            'seq_len': seq_len,
        }

    transient_indices = list(range(sink_size, local_start))
    absorbing_indices = list(range(sink_size)) + list(range(local_start, seq_len))

    n_transient = len(transient_indices)
    n_absorbing = len(absorbing_indices)

    # Extract Q matrix: transient -> transient transitions
    # Q[i,j] = P[transient_i, transient_j]
    Q = P[transient_indices, :][:, transient_indices].clone()

    # Extract R matrix: transient -> absorbing transitions
    R = P[transient_indices, :][:, absorbing_indices].clone()

    # Analysis 1: Row sums of Q (should be ~0.1 if Gemini is right)
    Q_row_sums = Q.sum(dim=-1)
    R_row_sums = R.sum(dim=-1)

    # Row sums should approximately sum to 1 (Q + R)
    total_row_sums = Q_row_sums + R_row_sums

    # Analysis 2: Spectral radius of Q
    try:
        eigenvalues = torch.linalg.eigvals(Q)
        spectral_radius = eigenvalues.abs().max().item()
    except:
        spectral_radius = float('nan')

    # Analysis 3: Frobenius norms of Q^k
    Q_norms = []
    Q_power = torch.eye(n_transient, device=Q.device, dtype=Q.dtype)
    for k in range(11):  # Q^0 to Q^10
        Q_norms.append(Q_power.norm().item())
        Q_power = Q_power @ Q

    # Analysis 4: Neumann series contribution
    # N = I + Q + Q² + Q³ + ...
    # Compute contribution of each term to ||N||
    N_cumulative = torch.zeros_like(Q)
    Q_power = torch.eye(n_transient, device=Q.device, dtype=Q.dtype)
    neumann_contributions = []

    for k in range(15):
        N_cumulative = N_cumulative + Q_power
        neumann_contributions.append({
            'k': k,
            'term_norm': Q_power.norm().item(),
            'cumulative_norm': N_cumulative.norm().item(),
        })
        Q_power = Q_power @ Q

    # Analysis 5: Per-head analysis (are some heads less damped?)
    head_Q_row_sums = []
    n_heads = attn_weights.shape[1]
    for h in range(n_heads):
        P_h = attn_weights[0, h].float()  # Single head
        Q_h = P_h[transient_indices, :][:, transient_indices]
        head_Q_row_sums.append(Q_h.sum(dim=-1).mean().item())

    return {
        'valid': True,
        'seq_len': seq_len,
        'n_transient': n_transient,
        'n_absorbing': n_absorbing,

        # Q row sum statistics (Gemini claims ~0.1)
        'Q_row_sum_mean': Q_row_sums.mean().item(),
        'Q_row_sum_std': Q_row_sums.std().item(),
        'Q_row_sum_min': Q_row_sums.min().item(),
        'Q_row_sum_max': Q_row_sums.max().item(),

        # R row sums (attention to absorbing states)
        'R_row_sum_mean': R_row_sums.mean().item(),

        # Total should be ~1
        'total_row_sum_mean': total_row_sums.mean().item(),

        # Spectral radius
        'spectral_radius': spectral_radius,

        # Q^k norms (exponential decay?)
        'Q_power_norms': Q_norms,

        # Neumann series analysis
        'neumann_contributions': neumann_contributions,

        # Per-head analysis
        'head_Q_row_sums': head_Q_row_sums,
        'head_Q_row_sum_min': min(head_Q_row_sums),
        'head_Q_row_sum_max': max(head_Q_row_sums),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test Implicit Damping Hypothesis")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--target_layer", type=int, default=None, help="Layer to analyze (default: middle)")
    args = parser.parse_args()

    print("=" * 70)
    print("IMPLICIT DAMPING HYPOTHESIS TEST")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Samples: {args.n_samples}")
    print(f"Sink size: {args.sink_size}, Window size: {args.window_size}")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Need eager for attention weights
    )
    model.eval()

    # Set up attention hook
    n_layers = len(model.model.layers)
    target_layer = args.target_layer if args.target_layer else n_layers // 2
    print(f"Capturing attention from layer {target_layer} of {n_layers}")

    hook = AttentionCaptureHook()
    handle = model.model.layers[target_layer].self_attn.register_forward_hook(
        hook, with_kwargs=True
    )

    # Load dataset
    print("Loading musique dataset...")
    try:
        data = load_dataset("THUDM/LongBench", "musique", split="test")
        samples = list(data)[:args.n_samples]
    except Exception as e:
        print(f"Could not load dataset: {e}")
        print("Using synthetic test...")
        samples = [{'context': 'A ' * 1000, 'input': 'What is A?', 'answers': ['A']}]

    # Analyze samples
    all_results = []
    aggregated = defaultdict(list)

    for i, sample in enumerate(tqdm(samples, desc="Analyzing")):
        prompt = f"{sample['context']}\n\nQuestion: {sample['input']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = inputs.input_ids.to(model.device)

        hook.clear()

        with torch.no_grad():
            _ = model(input_ids, output_attentions=True, use_cache=False)

        if hook.attention is None:
            continue

        result = analyze_Q_matrix(
            hook.attention,
            sink_size=args.sink_size,
            window_size=args.window_size
        )

        if result['valid']:
            all_results.append(result)
            aggregated['Q_row_sum_mean'].append(result['Q_row_sum_mean'])
            aggregated['R_row_sum_mean'].append(result['R_row_sum_mean'])
            aggregated['spectral_radius'].append(result['spectral_radius'])
            aggregated['head_Q_row_sum_min'].append(result['head_Q_row_sum_min'])
            aggregated['head_Q_row_sum_max'].append(result['head_Q_row_sum_max'])

        hook.clear()
        torch.cuda.empty_cache()

    handle.remove()

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    if not all_results:
        print("No valid results!")
        return

    print(f"\nAnalyzed {len(all_results)} samples")
    print()

    # Key metric: Q row sums
    avg_Q_row_sum = np.mean(aggregated['Q_row_sum_mean'])
    avg_R_row_sum = np.mean(aggregated['R_row_sum_mean'])
    avg_spectral = np.mean([r for r in aggregated['spectral_radius'] if not np.isnan(r)])

    print("=" * 50)
    print("GEMINI'S HYPOTHESIS TEST")
    print("=" * 50)
    print(f"Claim: Q row sums should be ~0.1 (90% to sinks)")
    print(f"Measured: Q row sum mean = {avg_Q_row_sum:.4f}")
    print(f"          R row sum mean = {avg_R_row_sum:.4f} (attention to absorbing)")
    print(f"          Total          = {avg_Q_row_sum + avg_R_row_sum:.4f} (should be ~1)")
    print()

    if avg_Q_row_sum < 0.2:
        print("VERDICT: HYPOTHESIS SUPPORTED")
        print(f"  Q row sums are low ({avg_Q_row_sum:.3f} < 0.2)")
        print("  Multi-hop signals will be heavily attenuated")
    elif avg_Q_row_sum < 0.5:
        print("VERDICT: HYPOTHESIS PARTIALLY SUPPORTED")
        print(f"  Q row sums are moderate ({avg_Q_row_sum:.3f})")
        print("  Some attenuation but not as severe as claimed")
    else:
        print("VERDICT: HYPOTHESIS NOT SUPPORTED")
        print(f"  Q row sums are high ({avg_Q_row_sum:.3f} >= 0.5)")
        print("  Multi-hop signals should propagate")

    print()
    print("=" * 50)
    print("MULTI-HOP ATTENUATION")
    print("=" * 50)

    # Show Q^k decay from first sample
    if all_results:
        norms = all_results[0]['Q_power_norms']
        print("Q^k Frobenius norms (exponential decay):")
        for k in range(min(6, len(norms))):
            decay = norms[k] / norms[0] if norms[0] > 0 else 0
            print(f"  k={k}: ||Q^{k}|| = {norms[k]:.4f} (relative: {decay:.4f})")

    print()
    print("=" * 50)
    print("NEUMANN SERIES CONTRIBUTION")
    print("=" * 50)

    if all_results:
        contribs = all_results[0]['neumann_contributions']
        print("Term contributions to N = I + Q + Q² + ...:")
        for c in contribs[:8]:
            pct = 100 * c['term_norm'] / contribs[0]['term_norm'] if contribs[0]['term_norm'] > 0 else 0
            print(f"  k={c['k']}: term_norm={c['term_norm']:.4f} ({pct:.1f}% of identity)")

    print()
    print("=" * 50)
    print("PER-HEAD VARIATION")
    print("=" * 50)
    print("(Sharp heads may have higher Q row sums)")
    print(f"  Min head Q row sum: {np.mean(aggregated['head_Q_row_sum_min']):.4f}")
    print(f"  Max head Q row sum: {np.mean(aggregated['head_Q_row_sum_max']):.4f}")

    if all_results:
        head_sums = all_results[0]['head_Q_row_sums']
        sorted_heads = sorted(enumerate(head_sums), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 5 heads with highest Q row sums (least damped):")
        for h, s in sorted_heads[:5]:
            print(f"    Head {h}: {s:.4f}")
        print(f"\n  Bottom 5 heads with lowest Q row sums (most damped):")
        for h, s in sorted_heads[-5:]:
            print(f"    Head {h}: {s:.4f}")

    print()
    print("=" * 50)
    print("SPECTRAL RADIUS")
    print("=" * 50)
    print(f"Average spectral radius of Q: {avg_spectral:.4f}")
    print("(Determines convergence rate of Neumann series)")
    print("If < 1, series converges. If << 1, converges VERY fast (damping).")

    # Save results
    output_file = "implicit_damping_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy to python types for JSON
        save_results = []
        for r in all_results:
            save_r = {}
            for k, v in r.items():
                if isinstance(v, (np.floating, np.integer)):
                    save_r[k] = float(v)
                elif isinstance(v, list) and v and isinstance(v[0], dict):
                    save_r[k] = v
                elif isinstance(v, list):
                    save_r[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                else:
                    save_r[k] = v
            save_results.append(save_r)
        json.dump(save_results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    main()
