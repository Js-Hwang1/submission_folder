"""
CircuitKV Breakthroughs Module

This module implements the mathematically grounded improvements identified
in the ICML 2026 novelty analysis:

1. Instruction Anchor Detection (Breakthrough 2) - CRITICAL for TREC
2. Fundamental Matrix Normalization (Breakthrough 1) - Core theoretical improvement
3. Multi-Horizon Walk Ensemble (Breakthrough 3) - Adaptive to task structure

These are designed to strengthen CircuitKV's core absorbing random walk
contribution while addressing specific failure modes on LongBench.
"""

import math
from typing import List, Optional, Set, Tuple

import torch
import torch.nn.functional as F


# =============================================================================
# BREAKTHROUGH 2: Instruction Anchor Detection
# =============================================================================

# Pattern tokens that anchor few-shot instruction formats
INSTRUCTION_PATTERNS = {
    # Classification patterns (TREC, LSHT)
    "Type": 1.0,
    "type": 1.0,
    ":": 0.8,
    "Question": 0.9,
    "question": 0.9,
    "Answer": 0.9,
    "answer": 0.9,
    # Few-shot delimiters
    "\n": 0.5,
    "Example": 0.8,
    "example": 0.8,
    # Summarization patterns (samsum)
    "Summary": 0.9,
    "summary": 0.9,
    "Dialogue": 0.8,
    "dialogue": 0.8,
    # Code patterns
    "def": 0.7,
    "class": 0.7,
    "return": 0.6,
}


def detect_instruction_anchors(
    token_ids: torch.Tensor,
    attention_matrix: torch.Tensor,
    tokenizer,
    self_attention_threshold: float = 0.05,
    pattern_score_threshold: float = 0.5,
    max_anchors: int = 32,
    sink_size: int = 4,
) -> Set[int]:
    """
    Detect tokens that anchor instruction patterns in few-shot prompts.

    These tokens are critical for classification tasks like TREC where the
    model needs to learn the "Question: X\nType: Y" pattern from examples.

    Algorithm:
        1. Decode tokens to text
        2. Match against known instruction patterns
        3. Filter by self-attention (structurally important tokens attend to themselves)
        4. Return set of anchor positions

    Args:
        token_ids: Token IDs [seq_len]
        attention_matrix: Attention weights [seq_len, seq_len]
        tokenizer: HuggingFace tokenizer for decoding
        self_attention_threshold: Min self-attention to be considered structural
        pattern_score_threshold: Min pattern match score
        max_anchors: Maximum number of anchors to return
        sink_size: Positions 0 to sink_size-1 are already sinks

    Returns:
        Set of token positions that are instruction anchors

    Theory:
        Instruction anchors are tokens that define the task format.
        They have two properties:
        1. Pattern matching: They match known instruction delimiters
        2. Structural importance: High self-attention (tokens that "stand out")

        By protecting these tokens, we preserve the few-shot learning signal.
    """
    anchors = set()
    seq_len = token_ids.shape[0]

    # Decode tokens (batch decode is faster)
    try:
        # Convert to list for tokenizer
        token_list = token_ids.tolist()
        decoded_tokens = [tokenizer.decode([t]) for t in token_list]
    except Exception:
        # Fallback: no tokenizer or decode error
        return anchors

    # Compute self-attention diagonal
    if attention_matrix.dim() == 2:
        self_attention = torch.diagonal(attention_matrix)
    else:
        self_attention = torch.zeros(seq_len, device=attention_matrix.device)

    # Score each token
    anchor_scores = []
    for i, token_text in enumerate(decoded_tokens):
        if i < sink_size:
            continue  # Skip sink tokens

        # Pattern matching score
        pattern_score = 0.0
        for pattern, weight in INSTRUCTION_PATTERNS.items():
            if pattern in token_text:
                pattern_score = max(pattern_score, weight)

        if pattern_score < pattern_score_threshold:
            continue

        # Self-attention score (normalized by max in neighborhood)
        local_start = max(0, i - 10)
        local_end = min(seq_len, i + 10)
        local_max = self_attention[local_start:local_end].max().item()

        if local_max > 0:
            relative_self_attn = self_attention[i].item() / local_max
        else:
            relative_self_attn = 0.0

        # Combined score
        if relative_self_attn >= self_attention_threshold or pattern_score >= 0.8:
            combined_score = pattern_score * (0.5 + 0.5 * relative_self_attn)
            anchor_scores.append((i, combined_score))

    # Sort by score and take top max_anchors
    anchor_scores.sort(key=lambda x: x[1], reverse=True)
    for pos, score in anchor_scores[:max_anchors]:
        anchors.add(pos)

    return anchors


def expand_absorbing_boundary(
    base_sink_size: int,
    instruction_anchors: Set[int],
    seq_len: int,
) -> torch.Tensor:
    """
    Create an expanded absorbing boundary mask that includes instruction anchors.

    Args:
        base_sink_size: Original sink size (first N tokens)
        instruction_anchors: Set of instruction anchor positions
        seq_len: Sequence length

    Returns:
        Boolean mask [seq_len] where True = absorbing state

    Theory:
        Standard CircuitKV has absorbing boundary at positions 0 to sink_size-1.
        By expanding this to include instruction anchors, walkers will also
        terminate when reaching these critical tokens.

        This ensures:
        1. Instruction anchors are ALWAYS kept (score = 1.0)
        2. Tokens near instruction anchors get higher visit counts
        3. The few-shot pattern structure is preserved
    """
    absorbing_mask = torch.zeros(seq_len, dtype=torch.bool)

    # Base sink
    absorbing_mask[:base_sink_size] = True

    # Instruction anchors
    for pos in instruction_anchors:
        if 0 <= pos < seq_len:
            absorbing_mask[pos] = True

    return absorbing_mask


# =============================================================================
# BREAKTHROUGH 1: Fundamental Matrix Normalization
# =============================================================================

def compute_neumann_normalization(
    attention_matrix: torch.Tensor,
    start_positions: torch.Tensor,
    sink_size: int = 4,
    num_terms: int = 5,
    num_probe_walks: int = 100,
) -> torch.Tensor:
    """
    Compute normalization factors using truncated Neumann series approximation.

    For an absorbing Markov chain with transition matrix Q (transient states),
    the fundamental matrix is:
        N = (I - Q)^{-1} = I + Q + Q^2 + Q^3 + ...

    N[i,j] = expected number of visits to j starting from i.

    We approximate this via Monte Carlo:
        (Q^k)[i,j] ≈ P(reach j in exactly k steps from i)

    Args:
        attention_matrix: Attention weights [seq_len, seq_len]
        start_positions: Walker start positions [num_starts] or single int
        sink_size: Absorbing boundary (positions 0 to sink_size-1)
        num_terms: Number of Neumann series terms (I + Q + ... + Q^{num_terms})
        num_probe_walks: Number of probe walks per start position per term

    Returns:
        expected_visits: Expected visit counts [seq_len]

    Theory:
        Current sqrt(n-p) normalization is heuristic.

        Proper normalization for absorbing chains is:
            normalized[j] = observed_visits[j] / E[visits to j]

        where E[visits to j] = sum_i start_prob[i] * N[i,j]

        This gives the "deviation from expected" which correctly identifies
        tokens that are visited MORE than random chance would predict.
    """
    seq_len = attention_matrix.shape[0]
    device = attention_matrix.device

    # Handle single start position
    if isinstance(start_positions, int):
        start_positions = torch.tensor([start_positions], device=device)

    # Uniform start distribution over provided positions
    num_starts = len(start_positions)
    start_prob = 1.0 / num_starts

    # Accumulate expected visits
    expected_visits = torch.zeros(seq_len, device=device, dtype=torch.float32)

    # For each Neumann term Q^k, estimate reachability via k-step walks
    for k in range(num_terms + 1):
        if k == 0:
            # Q^0 = I: each start position visits itself once
            for start in start_positions:
                if start >= sink_size:  # Only transient states
                    expected_visits[start] += start_prob
        else:
            # Q^k: k-step reachability
            term_visits = torch.zeros(seq_len, device=device, dtype=torch.float32)

            for start in start_positions:
                start_idx = start.item() if isinstance(start, torch.Tensor) else start
                if start_idx < sink_size:
                    continue  # Skip absorbed starts

                # Run probe walks of exactly k steps
                for _ in range(num_probe_walks):
                    pos = start_idx
                    for step in range(k):
                        if pos < sink_size:
                            break  # Absorbed

                        # Sample next position from attention
                        attn_row = attention_matrix[pos, :pos]
                        if attn_row.sum() > 1e-8:
                            probs = attn_row / attn_row.sum()
                            next_pos = torch.multinomial(probs, 1).item()
                            pos = next_pos
                        else:
                            # No valid transitions, jump toward sink
                            pos = max(0, pos - 1)

                    # Record final position if still transient
                    if pos >= sink_size:
                        term_visits[pos] += start_prob / num_probe_walks

            expected_visits += term_visits

    # Clamp to avoid division by zero
    expected_visits = expected_visits.clamp(min=1e-6)

    return expected_visits


def fundamental_matrix_normalize(
    raw_visits: torch.Tensor,
    expected_visits: torch.Tensor,
    sink_size: int = 4,
) -> torch.Tensor:
    """
    Normalize raw visit counts by expected visits from fundamental matrix.

    Args:
        raw_visits: Observed visit counts [seq_len]
        expected_visits: Expected visits from Neumann approximation [seq_len]
        sink_size: Absorbing boundary size

    Returns:
        normalized_scores: Deviation from expected [seq_len]

    Theory:
        score[j] = visits[j] / E[visits[j]]

        Scores > 1.0 indicate tokens visited more than expected (important).
        Scores < 1.0 indicate tokens visited less than expected.

        This is the principled replacement for sqrt(n-p) normalization.
    """
    seq_len = raw_visits.shape[0]
    normalized = raw_visits / expected_visits

    # Sink positions get score 1.0 (always kept)
    normalized[:sink_size] = 1.0

    # Normalize to [0, 1] range
    max_score = normalized[sink_size:].max()
    if max_score > 0:
        normalized[sink_size:] = normalized[sink_size:] / max_score

    return normalized


# =============================================================================
# BREAKTHROUGH 3: Multi-Horizon Walk Ensemble
# =============================================================================

def compute_attention_entropy(attention_row: torch.Tensor) -> float:
    """
    Compute entropy of an attention distribution.

    Args:
        attention_row: Attention weights [seq_len], should sum to 1

    Returns:
        entropy: Entropy in nats (natural log)
    """
    # Clamp to avoid log(0)
    probs = attention_row.clamp(min=1e-10)
    probs = probs / probs.sum()  # Normalize

    entropy = -(probs * torch.log(probs)).sum().item()
    return entropy


def compute_focus_ratio(
    attention_matrix: torch.Tensor,
    query_pos: int,
) -> float:
    """
    Compute focus ratio: how concentrated is the query's attention?

    Args:
        attention_matrix: Attention weights [seq_len, seq_len]
        query_pos: Position of the query token

    Returns:
        focus_ratio: 0.0 (diffuse) to 1.0 (focused)

    Theory:
        focus_ratio = 1 - entropy / max_entropy

        High focus_ratio: attention is concentrated on few tokens (retrieval)
        Low focus_ratio: attention is spread widely (global context needed)
    """
    seq_len = attention_matrix.shape[0]

    # Get query's attention distribution
    query_attn = attention_matrix[query_pos, :query_pos + 1]
    if query_attn.sum() < 1e-8:
        return 0.5  # Default to balanced

    query_attn = query_attn / query_attn.sum()

    # Compute entropy
    entropy = compute_attention_entropy(query_attn)
    max_entropy = math.log(query_pos + 1)  # Uniform distribution

    if max_entropy < 1e-8:
        return 0.5

    focus_ratio = 1.0 - (entropy / max_entropy)
    return max(0.0, min(1.0, focus_ratio))


def get_horizon_weights(focus_ratio: float) -> Tuple[float, float, float]:
    """
    Get weights for short/medium/long horizon walks based on attention focus.

    Args:
        focus_ratio: 0.0 (diffuse) to 1.0 (focused)

    Returns:
        (weight_short, weight_medium, weight_long)

    Theory:
        Focused attention (high ratio): local patterns matter, favor short walks
        Diffuse attention (low ratio): global context matters, favor long walks
    """
    if focus_ratio > 0.7:
        # Focused: favor short walks
        return (0.6, 0.3, 0.1)
    elif focus_ratio < 0.3:
        # Diffuse: favor long walks
        return (0.1, 0.3, 0.6)
    else:
        # Balanced
        return (0.33, 0.34, 0.33)


def multi_horizon_ensemble(
    run_walks_fn,  # Callable that runs walks and returns visit counts
    attention_matrix: torch.Tensor,
    query_pos: int,
    sink_size: int = 4,
    horizons: Tuple[int, int, int] = (10, 50, 200),
    num_walkers: int = 1000,
) -> torch.Tensor:
    """
    Run absorbing walks at multiple time horizons and combine scores.

    Args:
        run_walks_fn: Function(attention, query_pos, max_steps, num_walkers) -> visits
        attention_matrix: Attention weights [seq_len, seq_len]
        query_pos: Position of query token
        sink_size: Absorbing boundary size
        horizons: (short_steps, medium_steps, long_steps)
        num_walkers: Walkers per horizon (divided among horizons)

    Returns:
        combined_scores: Ensemble-weighted scores [seq_len]

    Theory:
        Different walk lengths approximate different powers of Q:
        - Short walks ≈ I + Q + Q² (local patterns)
        - Long walks ≈ (I - Q)^{-1} (full fundamental matrix)

        The ensemble adapts to task structure via attention entropy.
    """
    seq_len = attention_matrix.shape[0]
    device = attention_matrix.device

    # Compute focus ratio for adaptive weighting
    focus_ratio = compute_focus_ratio(attention_matrix, query_pos)
    weights = get_horizon_weights(focus_ratio)

    # Walkers per horizon
    walkers_per_horizon = num_walkers // 3

    # Run walks at each horizon
    all_visits = []
    for max_steps in horizons:
        visits = run_walks_fn(attention_matrix, query_pos, max_steps, walkers_per_horizon)
        all_visits.append(visits)

    # Normalize each horizon's visits to [0, 1]
    normalized_visits = []
    for visits in all_visits:
        max_v = visits[sink_size:].max()
        if max_v > 0:
            norm_v = visits / max_v
        else:
            norm_v = visits
        normalized_visits.append(norm_v)

    # Weighted combination
    combined = torch.zeros(seq_len, device=device, dtype=torch.float32)
    for w, norm_v in zip(weights, normalized_visits):
        combined += w * norm_v

    # Sink positions get score 1.0
    combined[:sink_size] = 1.0

    return combined


# =============================================================================
# COMBINED BREAKTHROUGH: Enhanced CircuitKV Scoring
# =============================================================================

def enhanced_circuitkv_scores(
    attention_matrix: torch.Tensor,
    token_ids: Optional[torch.Tensor],
    tokenizer,
    run_walks_fn,
    query_pos: int,
    h2o_scores: torch.Tensor,
    sink_size: int = 4,
    num_walkers: int = 10000,
    use_instruction_anchors: bool = True,
    use_fundamental_norm: bool = True,
    use_multi_horizon: bool = True,
) -> Tuple[torch.Tensor, Set[int]]:
    """
    Compute enhanced CircuitKV scores with all breakthroughs applied.

    This combines:
    1. Instruction anchor detection (for TREC-like tasks)
    2. Fundamental matrix normalization (principled scoring)
    3. Multi-horizon ensemble (adaptive to task structure)
    4. MAX(H2O, Influence) hedging (robust across task types)

    Args:
        attention_matrix: Attention weights [seq_len, seq_len]
        token_ids: Token IDs [seq_len] (optional, for anchor detection)
        tokenizer: HuggingFace tokenizer (optional)
        run_walks_fn: Function to run absorbing walks
        query_pos: Position of query token
        h2o_scores: H2O scores (column sums) [seq_len]
        sink_size: Absorbing boundary size
        num_walkers: Total number of walkers
        use_instruction_anchors: Whether to detect instruction anchors
        use_fundamental_norm: Whether to use fundamental matrix normalization
        use_multi_horizon: Whether to use multi-horizon ensemble

    Returns:
        combined_scores: Final combined scores [seq_len]
        instruction_anchors: Set of detected anchor positions
    """
    seq_len = attention_matrix.shape[0]
    device = attention_matrix.device

    # Step 1: Detect instruction anchors
    instruction_anchors = set()
    if use_instruction_anchors and token_ids is not None and tokenizer is not None:
        try:
            instruction_anchors = detect_instruction_anchors(
                token_ids, attention_matrix, tokenizer,
                max_anchors=32, sink_size=sink_size
            )
        except Exception:
            pass  # Fallback: no anchors

    # Step 2: Run walks (with optional multi-horizon)
    if use_multi_horizon:
        influence_raw = multi_horizon_ensemble(
            run_walks_fn, attention_matrix, query_pos,
            sink_size=sink_size, num_walkers=num_walkers
        )
    else:
        # Single horizon (default 10 steps)
        influence_raw = run_walks_fn(attention_matrix, query_pos, 10, num_walkers)

    # Step 3: Normalize (with optional fundamental matrix)
    if use_fundamental_norm:
        start_positions = torch.tensor([query_pos], device=device)
        expected_visits = compute_neumann_normalization(
            attention_matrix, start_positions, sink_size=sink_size,
            num_terms=5, num_probe_walks=50
        )
        influence_scores = fundamental_matrix_normalize(
            influence_raw, expected_visits, sink_size=sink_size
        )
    else:
        # Fallback: max normalization
        max_v = influence_raw[sink_size:].max()
        if max_v > 0:
            influence_scores = influence_raw / max_v
        else:
            influence_scores = influence_raw
        influence_scores[:sink_size] = 1.0

    # Step 4: Rank normalize both signals
    def rank_normalize(scores):
        ranks = torch.argsort(torch.argsort(scores)).float()
        return ranks / (len(scores) - 1 + 1e-10)

    h2o_rank = rank_normalize(h2o_scores[:seq_len])
    influence_rank = rank_normalize(influence_scores[:seq_len])

    # Step 5: MAX combination (hedging strategy)
    combined_scores = torch.maximum(h2o_rank, influence_rank)

    # Step 6: Boost instruction anchors to ensure they're kept
    for anchor_pos in instruction_anchors:
        if 0 <= anchor_pos < seq_len:
            combined_scores[anchor_pos] = 1.0

    # Sink positions always get 1.0
    combined_scores[:sink_size] = 1.0

    return combined_scores, instruction_anchors
