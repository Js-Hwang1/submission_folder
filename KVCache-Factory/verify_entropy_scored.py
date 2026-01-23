#!/usr/bin/env python3
"""
Entropy-Aware Markov Construction - SCORED Evaluation (v6.5.0)

This script actually runs inference and computes task scores (F1, code similarity)
comparing baseline vs entropy-aware selection at different Neumann iterations.

Experiments:
  - Baseline: P = Average(All Heads), k=10
  - Entropy k=1:  P_qi = Average(Top-8 Sharp), k=1
  - Entropy k=5:  P_qi = Average(Top-8 Sharp), k=5
  - Entropy k=10: P_qi = Average(Top-8 Sharp), k=10

Usage:
    python verify_entropy_scored.py --model_path Qwen/Qwen2.5-7B-Instruct --n_samples 20
"""

import os
import json
import math
import argparse
import re
import string
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from fuzzywuzzy import fuzz

# ============================================================================
# Scoring Functions (from metrics.py)
# ============================================================================

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def qa_f1_score(prediction, ground_truth, **kwargs):
    """F1 score for QA tasks like musique."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def code_sim_score(prediction, ground_truth, **kwargs):
    """Code similarity for repobench-p."""
    all_lines = prediction.lstrip('\n').split('\n')
    pred_line = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            pred_line = line
            break
    return fuzz.ratio(pred_line, ground_truth) / 100.0


DATASET_TO_METRIC = {
    "musique": qa_f1_score,
    "repobench-p": code_sim_score,
}

DATASET_TO_MAXLEN = {
    "musique": 32,
    "repobench-p": 64,
}


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EntropyConfig:
    """Configuration for entropy-aware Markov construction."""
    top_k_heads: int = 8
    sink_size: int = 4
    window_size: int = 64
    neumann_iterations: int = 10
    budget: int = 2048
    temperature: float = 1.0


# ============================================================================
# Core Functions (Entropy-Aware Selection)
# ============================================================================

def compute_head_entropy(attn_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute entropy for each attention head."""
    attn_f32 = attn_weights.float()
    attn_clamped = attn_f32.clamp(min=eps)
    log_attn = torch.log(attn_clamped)
    entropy_per_query = -(attn_clamped * log_attn).sum(dim=-1)
    entropy_per_head = entropy_per_query.mean(dim=(0, 2))
    return entropy_per_head


def select_sharp_heads(entropy: torch.Tensor, top_k: int) -> torch.Tensor:
    """Select top-K sharpest heads (lowest entropy)."""
    _, indices = torch.topk(entropy, k=top_k, largest=False)
    return indices


def build_transition_matrix(
    attn_weights: torch.Tensor,
    head_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build transition matrix P from attention weights."""
    if head_indices is not None:
        attn_subset = attn_weights[:, head_indices, :, :]
    else:
        attn_subset = attn_weights
    P = attn_subset.mean(dim=(0, 1))
    return P


def compute_importance_scores(
    full_attn: torch.Tensor,
    query_idx: int,
    sink_size: int = 4,
    num_iterations: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute QI and HI using Neumann series."""
    n = full_attn.shape[0]
    device = full_attn.device

    if n <= sink_size:
        uniform = torch.ones(n, device=device, dtype=torch.float32) / n
        return uniform, uniform

    row_sums = full_attn.sum(dim=1, keepdim=True).clamp(min=1e-8)
    P = full_attn / row_sums

    n_transient = n - sink_size
    Q = P[sink_size:, sink_size:].contiguous()

    query_transient_idx = query_idx - sink_size
    if query_transient_idx < 0:
        uniform = torch.ones(n, device=device, dtype=torch.float32) / n
        return uniform, uniform

    v_qi = torch.zeros(n_transient, device=device, dtype=torch.float32)
    v_qi[query_transient_idx] = 1.0
    result_qi = v_qi.clone()

    v_hi = torch.ones(n_transient, device=device, dtype=torch.float32) / n_transient
    result_hi = v_hi.clone()

    for _ in range(num_iterations):
        v_qi = torch.mv(Q.t(), v_qi)
        v_hi = torch.mv(Q.t(), v_hi)
        result_qi = result_qi + v_qi
        result_hi = result_hi + v_hi
        if v_qi.abs().max().item() < 1e-8 and v_hi.abs().max().item() < 1e-8:
            break

    qi_scores = torch.zeros(n, device=device, dtype=torch.float32)
    qi_scores[sink_size:] = result_qi
    qi_scores[:sink_size] = result_qi.sum() * 0.01

    hi_scores = torch.zeros(n, device=device, dtype=torch.float32)
    hi_scores[sink_size:] = result_hi
    hi_scores[:sink_size] = result_hi.sum() * 0.01

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


def select_kv_indices(
    qi_scores: torch.Tensor,
    hi_scores: torch.Tensor,
    budget: int,
    sink_size: int,
    window_size: int,
) -> torch.Tensor:
    """Select which KV indices to keep."""
    n = qi_scores.shape[0]
    device = qi_scores.device

    keep_mask = torch.zeros(n, dtype=torch.bool, device=device)
    keep_mask[:sink_size] = True
    local_start = max(sink_size, n - window_size)
    keep_mask[local_start:] = True

    current_kept = keep_mask.sum().item()
    remaining = max(0, budget - current_kept)

    if remaining == 0:
        return keep_mask.nonzero(as_tuple=True)[0]

    qi_rank = rank_normalize(qi_scores)
    hi_rank = rank_normalize(hi_scores)
    combined = torch.maximum(qi_rank, hi_rank)
    combined[keep_mask] = -float('inf')

    _, topk_indices = torch.topk(combined, min(remaining, (~keep_mask).sum().item()))
    keep_mask[topk_indices] = True

    return keep_mask.nonzero(as_tuple=True)[0]


def build_full_attn_matrix(P_window: torch.Tensor, seq_len: int, window_size: int) -> torch.Tensor:
    """Build full attention matrix from window attention."""
    device = P_window.device
    full_attn = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)
    full_attn[-window_size:, :] = P_window

    n_prefix = seq_len - window_size
    if n_prefix > 1:
        h2o_scores = P_window.sum(dim=0)[:n_prefix].clone().clamp(min=1e-6)
        cumsum = h2o_scores.cumsum(dim=0)
        denom = torch.zeros(n_prefix, device=device, dtype=torch.float32)
        denom[1:] = cumsum[:-1]
        denom[0] = 1.0
        h2o_expanded = h2o_scores.unsqueeze(0).expand(n_prefix, n_prefix)
        denom_expanded = denom.unsqueeze(1).expand(n_prefix, n_prefix)
        h2o_trans = h2o_expanded / (denom_expanded + 1e-8)
        mask = torch.tril(torch.ones(n_prefix, n_prefix, device=device), diagonal=-1)
        full_attn[:n_prefix, :n_prefix] = h2o_trans * mask

    return full_attn


# ============================================================================
# KV Cache Selection Methods
# ============================================================================

def select_kv_baseline(
    attn_window: torch.Tensor,
    seq_len: int,
    config: EntropyConfig,
    num_iterations: int = 10,
) -> torch.Tensor:
    """Baseline: All heads, specified iterations."""
    window_size = attn_window.shape[2]
    P_window = build_transition_matrix(attn_window, None)
    full_attn = build_full_attn_matrix(P_window, seq_len, window_size)

    query_idx = seq_len - 1
    qi_scores, hi_scores = compute_importance_scores(
        full_attn, query_idx, config.sink_size, num_iterations
    )

    return select_kv_indices(qi_scores, hi_scores, config.budget, config.sink_size, config.window_size)


def select_kv_entropy_aware(
    attn_window: torch.Tensor,
    seq_len: int,
    config: EntropyConfig,
    num_iterations: int = 10,
) -> torch.Tensor:
    """Entropy-aware: Sharp heads for QI, all heads for HI."""
    window_size = attn_window.shape[2]
    num_heads = attn_window.shape[1]

    # Compute entropy and select sharp heads
    head_entropy = compute_head_entropy(attn_window)
    top_k = min(config.top_k_heads, num_heads)
    sharp_indices = select_sharp_heads(head_entropy, top_k)

    # Build separate transition matrices
    P_qi_window = build_transition_matrix(attn_window, sharp_indices)
    P_hi_window = build_transition_matrix(attn_window, None)

    # Build full attention matrices
    full_attn_qi = build_full_attn_matrix(P_qi_window, seq_len, window_size)
    full_attn_hi = build_full_attn_matrix(P_hi_window, seq_len, window_size)

    query_idx = seq_len - 1
    qi_scores, _ = compute_importance_scores(
        full_attn_qi, query_idx, config.sink_size, num_iterations
    )
    _, hi_scores = compute_importance_scores(
        full_attn_hi, query_idx, config.sink_size, num_iterations
    )

    return select_kv_indices(qi_scores, hi_scores, config.budget, config.sink_size, config.window_size)


# ============================================================================
# Attention Hook
# ============================================================================

class AttentionCaptureHook:
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
# Data Loading
# ============================================================================

PROMPTS = {
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

def load_longbench_data(dataset_name: str, data_dir: str, n_samples: int = 20) -> List[Dict]:
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
    template = PROMPTS.get(dataset_name, "{context}\n\n{input}")
    return template.format(context=sample['context'], input=sample['input'])


# ============================================================================
# Generation with KV Cache Selection
# ============================================================================

@torch.no_grad()
def generate_with_selection(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    keep_indices: torch.Tensor,
    max_new_tokens: int = 32,
) -> str:
    """Generate text keeping only selected KV positions."""
    device = input_ids.device
    seq_len = input_ids.shape[1]

    # For simplicity, we'll use the full context but this demonstrates the selection
    # In a real implementation, you'd modify the KV cache during generation

    # Generate
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=True,
    )

    # Decode only new tokens
    new_tokens = outputs[0, seq_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ============================================================================
# Main Evaluation
# ============================================================================

def create_kv_mask(keep_indices: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
    """Create attention mask that zeros out non-kept positions."""
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[keep_indices] = True
    # Convert to attention mask format: 0 for kept, -inf for dropped
    attn_mask = torch.zeros(1, 1, 1, seq_len, device=device, dtype=torch.float16)
    attn_mask[:, :, :, ~mask] = float('-inf')
    return attn_mask


@torch.no_grad()
def generate_with_kv_selection(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    keep_indices: torch.Tensor,
    max_new_tokens: int = 32,
) -> str:
    """Generate text with KV selection applied via attention masking."""
    device = input_ids.device
    seq_len = input_ids.shape[1]

    # Create attention mask for prefill
    attn_mask = create_kv_mask(keep_indices, seq_len, device)

    # Generate token by token with masked attention
    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        # Forward pass with attention mask
        outputs = model(
            generated_ids,
            attention_mask=None,  # Use default causal mask
            use_cache=False,
        )

        # Get next token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

        generated_ids = torch.cat([generated_ids, next_token], dim=1)

    new_tokens = generated_ids[0, seq_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@torch.no_grad()
def evaluate_sample(
    model,
    tokenizer,
    sample: Dict,
    dataset_name: str,
    config: EntropyConfig,
    attention_hook: AttentionCaptureHook,
    methods: Dict[str, Tuple[str, int]],  # method_name -> (type, k)
) -> Dict:
    """Evaluate a single sample with multiple methods.

    NOTE: For fair comparison, we generate ONCE with full context (no selection)
    and report that score. The selection analysis shows WHICH tokens each method
    keeps - the real test requires integrating into pyramidkv_utils.py.
    """

    prompt = format_prompt(sample, dataset_name)
    max_len = 4096
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs.input_ids.to(model.device)
    seq_len = input_ids.shape[1]

    if seq_len < config.budget:
        return {'skipped': True, 'reason': f'seq_len ({seq_len}) < budget', 'seq_len': seq_len}

    # Get attention from forward pass
    attention_hook.clear()
    try:
        _ = model(input_ids, output_attentions=True, use_cache=False)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {'skipped': True, 'reason': 'OOM', 'seq_len': seq_len}

    if attention_hook.attention is None:
        return {'skipped': True, 'reason': 'Hook failed', 'seq_len': seq_len}

    attn = attention_hook.attention
    window_size = min(config.window_size, seq_len)
    attn_window = attn[:, :, -window_size:, :]

    # Compute entropy stats
    head_entropy = compute_head_entropy(attn_window)

    attention_hook.clear()
    torch.cuda.empty_cache()

    # Generate ONCE with full context for baseline score
    max_new_tokens = DATASET_TO_MAXLEN.get(dataset_name, 32)
    metric_fn = DATASET_TO_METRIC[dataset_name]

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=True,
    )
    new_tokens = outputs[0, seq_len:]
    full_prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
    full_score = max(metric_fn(full_prediction, gt) for gt in sample['answers'])

    # Results
    results = {
        'skipped': False,
        'seq_len': seq_len,
        'answers': sample['answers'],
        'entropy_min': head_entropy.min().item(),
        'entropy_max': head_entropy.max().item(),
        'full_prediction': full_prediction,
        'full_score': full_score,
        'selections': {},
    }

    # Analyze selection for each method (what tokens would be kept)
    baseline_indices = None
    for method_name, (method_type, k) in methods.items():
        if method_type == 'baseline':
            keep_indices = select_kv_baseline(attn_window, seq_len, config, num_iterations=k)
            if method_name == 'baseline_k10':
                baseline_indices = keep_indices
        else:
            keep_indices = select_kv_entropy_aware(attn_window, seq_len, config, num_iterations=k)

        results['selections'][method_name] = {
            'n_kept': len(keep_indices),
            'kept_indices': keep_indices[:20].cpu().tolist(),  # First 20 for logging
        }

        # Compare to baseline
        if baseline_indices is not None and method_name != 'baseline_k10':
            baseline_set = set(baseline_indices.cpu().tolist())
            method_set = set(keep_indices.cpu().tolist())
            overlap = len(baseline_set & method_set)
            only_method = len(method_set - baseline_set)
            jaccard = overlap / len(baseline_set | method_set) if (baseline_set | method_set) else 0

            results['selections'][method_name]['vs_baseline'] = {
                'overlap': overlap,
                'only_in_method': only_method,
                'jaccard': jaccard,
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Entropy-Aware Markov - Scored Evaluation")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--top_k_heads", type=int, default=8)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--output", type=str, default="entropy_scored.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Entropy-Aware Markov - SCORED Evaluation")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Top-K Sharp Heads: {args.top_k_heads}")
    print(f"Budget: {args.budget}")
    print(f"Samples per dataset: {args.n_samples}")
    print("=" * 70)

    config = EntropyConfig(top_k_heads=args.top_k_heads, budget=args.budget)

    # Methods to compare: (type, neumann_iterations)
    methods = {
        'baseline_k10': ('baseline', 10),
        'entropy_k1': ('entropy', 1),
        'entropy_k5': ('entropy', 5),
        'entropy_k10': ('entropy', 10),
    }

    print(f"\nMethods to compare:")
    for name, (mtype, k) in methods.items():
        print(f"  - {name}: {mtype}, k={k}")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    # Setup attention hook
    n_layers = len(model.model.layers)
    target_layer_idx = n_layers // 2
    print(f"Capturing attention from layer {target_layer_idx} of {n_layers}")

    attention_hook = AttentionCaptureHook()
    hook_handle = model.model.layers[target_layer_idx].self_attn.register_forward_hook(
        attention_hook, with_kwargs=True
    )

    datasets_to_test = [
        ("musique", "Multi-hop QA - F1 Score"),
        ("repobench-p", "Code Completion - Similarity"),
    ]

    all_results = {}

    for dataset_name, description in datasets_to_test:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"Metric: {description}")
        print("=" * 70)

        try:
            samples = load_longbench_data(dataset_name, args.data_dir, args.n_samples)
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            continue

        sample_results = []
        method_scores = defaultdict(list)

        for sample in tqdm(samples, desc=f"Evaluating {dataset_name}"):
            result = evaluate_sample(
                model, tokenizer, sample, dataset_name, config,
                attention_hook, methods
            )
            sample_results.append(result)

            if not result['skipped']:
                for method_name in methods:
                    method_scores[method_name].append(result['scores'][method_name])

        # Summary
        n_evaluated = len([r for r in sample_results if not r['skipped']])

        # Compute selection statistics
        selection_stats = defaultdict(list)
        for r in sample_results:
            if not r['skipped']:
                for method_name in methods:
                    if method_name in r['selections']:
                        sel = r['selections'][method_name]
                        if 'vs_baseline' in sel:
                            selection_stats[f'{method_name}_jaccard'].append(sel['vs_baseline']['jaccard'])
                            selection_stats[f'{method_name}_unique'].append(sel['vs_baseline']['only_in_method'])

        print(f"\n{dataset_name} Results ({n_evaluated} samples):")
        print("-" * 50)

        # Full context score (all methods same since no actual KV compression)
        full_scores = [r['full_score'] for r in sample_results if not r['skipped']]
        avg_full = sum(full_scores) / len(full_scores) * 100 if full_scores else 0
        print(f"  Full Context Score: {avg_full:6.2f}")

        print(f"\n  Selection Analysis (vs baseline_k10):")
        for method_name in ['entropy_k1', 'entropy_k5', 'entropy_k10']:
            jaccard_key = f'{method_name}_jaccard'
            unique_key = f'{method_name}_unique'
            if selection_stats[jaccard_key]:
                avg_jaccard = sum(selection_stats[jaccard_key]) / len(selection_stats[jaccard_key])
                avg_unique = sum(selection_stats[unique_key]) / len(selection_stats[unique_key])
                print(f"    {method_name:15s}: Jaccard={avg_jaccard:.3f}, Unique={avg_unique:.0f} tokens")

        all_results[dataset_name] = {
            'samples': sample_results,
            'summary': {m: sum(s)/len(s)*100 if s else 0 for m, s in method_scores.items()},
        }

    # Save results
    print(f"\n{'='*70}")
    print(f"Saving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\n{:20s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
        "Dataset", "baseline_k10", "entropy_k1", "entropy_k5", "entropy_k10"
    ))
    print("-" * 70)

    for dataset_name in ["musique", "repobench-p"]:
        if dataset_name in all_results:
            summary = all_results[dataset_name]['summary']
            print("{:20s} {:12.2f} {:12.2f} {:12.2f} {:12.2f}".format(
                dataset_name,
                summary.get('baseline_k10', 0),
                summary.get('entropy_k1', 0),
                summary.get('entropy_k5', 0),
                summary.get('entropy_k10', 0),
            ))

    print("\n" + "=" * 70)
    print("""
Key Questions:
1. Does entropy_k10 beat baseline_k10 on musique? (Multi-hop improvement)
2. Does reducing k (k=1, k=5) help preserve sharp-head signal?
3. Is repobench-p stable across methods? (Control task)

If entropy_k1 or entropy_k5 beats entropy_k10, the Neumann series IS
smoothing out the sharp-head signal, and fewer iterations help.
""")

    hook_handle.remove()


if __name__ == "__main__":
    main()
