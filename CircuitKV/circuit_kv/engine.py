"""
CircuitKV Engine: High-level Python wrapper for the CUDA extension.

This module provides the "Monitor" class that manages the sparse attention graph
and runs absorbing random walks to compute current-flow betweenness scores.

Physics Analogy:
- Query = Battery positive terminal (Source)
- Context Start (tokens 0-3) = Battery negative terminal (Sink)
- Attention weights = Conductance (1/Resistance)
- Visit counts = Current flow through each node
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CircuitKVConfig:
    """Configuration for CircuitKV monitor."""

    # Graph parameters
    max_seq_len: int = 8192  # Maximum sequence length
    top_k: int = 32  # Number of neighbors per token in sparse graph

    # Walker parameters (for C++ compatibility - alpha/num_steps not used by circuit walker)
    num_walkers: int = 1024  # Number of parallel random walkers

    # These are kept for API compatibility but CircuitKV uses internal defaults
    alpha: float = 0.85  # Not used by CircuitKV (no teleport)
    num_steps: int = 100  # CircuitKV uses MAX_STEPS=100 internally

    # Eviction parameters
    sink_size: int = 4  # CircuitKV absorbing boundary (first N tokens)
    local_window: int = 32  # Local window to always keep

    # Observation window for importance scoring (like SnapKV)
    observation_window: int = 1  # Last W tokens for H2O attention + multi-source walks

    # Capacitive CircuitKV: State-Space Model Parameters
    # Physics: Tokens are capacitors that accumulate charge over time
    decay: float = 0.95  # EMA decay for charge accumulation (temporal smoothing)

    # RC+B: Bidirectional Circuit Walks
    # When enabled, runs walks in BOTH directions:
    # - Backward: Query -> Sink (standard direction)
    # - Forward: Sink -> Query (via transpose graph)
    # Tokens visited by BOTH directions get bridge bonus: score = max(b,f) + 0.5*min(b,f)
    # NOTE: Disabled by default - R@65 improved but narrativeqa score dropped (25.10 -> 23.78)
    bidirectional: bool = False

    # Hardware
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


class CircuitKVMonitor:
    """
    The Capacitive CircuitKV Monitor with Observation Window (P0+P1).

    This class wraps the CUDA extension and provides a high-level interface
    for updating the attention graph and computing current-flow scores.

    Physics Model (RC Circuit / State-Space):
    - Tokens are CAPACITORS that accumulate charge over time
    - INITIALIZATION (End of Prefill): Observation Window Strategy
      * P1: Real H2O from softmax(Q[-W:]@K^T).sum() - true attention importance
      * P0: Multi-source walks from ALL W tokens - distributed bridge detection
      * Combined: max(H2O, Circuit) - keeps tokens high on EITHER signal
    - UPDATE (During Generation): Current from CircuitKV walker updates charge via EMA
      Charge_t = decay * Charge_{t-1} + (1-decay) * Current_t

    Key Innovations:
    - P1 (Real H2O): Uses actual attention scores, not key norms
      * Catches true Heavy Hitters that receive high attention
    - P0 (Multi-source): Runs walks from observation window, not single token
      * Catches distributed importance patterns (summarization, few-shot)
      * Single-source only captures last-token dependencies

    Usage:
        config = CircuitKVConfig()
        monitor = CircuitKVMonitor(config)

        # At end of prefill phase (Single-Token Multiplicative Gating):
        monitor.initialize_from_prefill(keys, final_query_state)

        # During generation, after each new token:
        monitor.update(query, keys, current_idx)

        # When you need eviction decisions:
        scores = monitor.get_scores()
        keep_mask = monitor.get_keep_mask(budget_ratio=0.25)
    """

    def __init__(self, config: CircuitKVConfig):
        self.config = config
        self._graph: Optional[object] = None
        self._initialized = False

        # Capacitive State: Accumulated charge for each token
        # This is the "memory" of the RC circuit
        self._accumulated_charge: Optional[torch.Tensor] = None
        self._prefill_initialized = False

    def _lazy_init(self) -> None:
        """Lazily initialize the CUDA circuit manager."""
        if self._initialized:
            return

        # Import here to defer CUDA initialization
        from circuit_kv import get_extension

        ext = get_extension()
        self._graph = ext.CircuitGraph(
            self.config.max_seq_len,
            self.config.top_k,
            self.config.alpha,  # Not used by circuit walker, but needed for C++ init
            self.config.num_walkers,
            self.config.num_steps,  # Not used by circuit walker (uses MAX_STEPS)
        )
        self._initialized = True

    def initialize_from_prefill(
        self,
        keys: torch.Tensor,
        final_query_state: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> None:
        """
        Initialize capacitor charge using Observation Window + Union (Max).

        **P0+P1 Implementation:**
        - P1 (Real H2O): Compute softmax(Q[-W:]@K^T).sum() instead of key norms
          * Uses actual attention importance from observation window queries
          * Captures true "Heavy Hitters" that receive high attention
        - P0 (Multi-source walks): Run walks from ALL tokens in observation window
          * Aggregates visit counts from W parallel walks
          * Captures distributed importance, not just last-token dependencies
        - Combination: max(H2O, Circuit) - keeps tokens high on EITHER signal

        Args:
            keys: Key vectors from prefill. Shape: [N, D] or [B, H, N, D]
            final_query_state: Query vectors. Shape: [W, D] or [B, H, N, D] (uses last W)
                If None, uses last W keys as proxy.
            seq_len: Sequence length (optional, inferred from keys).
        """
        import math
        import torch.nn.functional as F

        self._lazy_init()

        # Handle different input shapes - flatten to [N, D]
        if keys.dim() == 4:
            # [B, H, N, D] -> [N, D] (use first batch, first head)
            keys_flat = keys[0, 0, :, :].contiguous()
        elif keys.dim() == 3:
            # [B, N, D] -> [N, D]
            keys_flat = keys[0, :, :].contiguous()
        else:
            # Already [N, D]
            keys_flat = keys.contiguous()

        actual_seq_len = seq_len if seq_len is not None else keys_flat.shape[0]
        device = keys_flat.device
        head_dim = keys_flat.shape[-1]

        # Observation window size (capped by sequence length)
        W = min(self.config.observation_window, actual_seq_len - self.config.sink_size)
        W = max(W, 1)  # At least 1 token

        # =====================================================================
        # P1: REAL ATTENTION-BASED H2O (Not key norms!)
        # Compute softmax(Q[-W:] @ K^T / sqrt(d)).sum(dim=0) for true importance
        # =====================================================================
        # Get queries from observation window
        if final_query_state is not None:
            # Handle different shapes - extract last W queries
            if final_query_state.dim() == 4:
                # [B, H, N, D] -> [W, D]
                queries_window = final_query_state[0, 0, -W:, :].contiguous()
            elif final_query_state.dim() == 3:
                # [B, N, D] -> [W, D]
                queries_window = final_query_state[0, -W:, :].contiguous()
            elif final_query_state.dim() == 2:
                # [N, D] -> [W, D]
                queries_window = final_query_state[-W:, :].contiguous()
            else:
                # [D] -> [1, D] (single query fallback)
                queries_window = final_query_state.unsqueeze(0).contiguous()
                W = 1
        else:
            # Fallback: use last W keys as proxy for queries
            queries_window = keys_flat[-W:, :].contiguous()

        # Compute attention scores: [W, N] = [W, D] @ [D, N]
        # Only attend to tokens BEFORE each query (causal mask)
        attn_logits = queries_window @ keys_flat[:actual_seq_len].T / math.sqrt(head_dim)

        # Apply causal mask: query at position (seq_len - W + i) can only attend to [0, seq_len - W + i]
        causal_mask = torch.ones(W, actual_seq_len, device=device, dtype=torch.bool)
        for i in range(W):
            query_pos = actual_seq_len - W + i
            causal_mask[i, query_pos + 1:] = False  # Mask future tokens

        attn_logits = attn_logits.masked_fill(~causal_mask, float('-inf'))

        # Softmax and sum across window queries -> H2O importance
        attn_weights = F.softmax(attn_logits.float(), dim=-1)  # [W, N]
        h2o = attn_weights.sum(dim=0)  # [N] - total attention received from window

        # Normalize to [0, 1]
        h2o_max = h2o.max()
        if h2o_max > 0:
            h2o = h2o / h2o_max

        # =====================================================================
        # P0+P3: CIRCUIT WALKS (CUDA-parallelized)
        # Run walks from observation window - either unidirectional or bidirectional
        # =====================================================================
        # Build source indices array
        source_indices = torch.arange(
            actual_seq_len - W, actual_seq_len,
            device=device, dtype=torch.int32
        )

        # Convert queries to FP32 for CUDA kernel
        queries_fp32 = queries_window.float().contiguous()
        keys_fp32 = keys_flat[:actual_seq_len].float().contiguous()

        if self.config.bidirectional:
            # =====================================================================
            # RC+B: BIDIRECTIONAL CIRCUIT WALKS
            # Run walks in BOTH directions:
            # - Backward: Query -> Sink (who does query attend to)
            # - Forward: Sink -> Query (who attends to sink, via transpose graph)
            # Bridge bonus for tokens visited by BOTH directions
            # =====================================================================
            self._graph.update_and_step_circuit_bidirectional(
                queries_fp32, keys_fp32, source_indices
            )

            # Get combined bidirectional scores (includes bridge bonus)
            circuit_total = self._graph.get_bidirectional_scores()[:actual_seq_len].float()
        else:
            # Standard unidirectional multi-source walks
            self._graph.update_and_step_circuit_multi_source(
                queries_fp32, keys_fp32, source_indices
            )

            # Get aggregated visit counts from all W sources
            circuit_total = self._graph.get_scores()[:actual_seq_len].float()

        # =====================================================================
        # FIXED SCALING (NOT dynamic normalization)
        # Scale by (num_walkers * W) so 10% visits across all walks -> ~1.0
        # =====================================================================
        scale_factor = 10.0 / (self.config.num_walkers * W)
        circuit = circuit_total * scale_factor

        # =====================================================================
        # UNION (MAX) STRATEGY: Keep tokens high on EITHER signal
        # - Heavy Hitters (H2O from real attention) are preserved
        # - Circuit Bridges (from multi-source walks) are added
        # =====================================================================
        combined = torch.maximum(h2o, circuit)

        # Initialize accumulated charge buffer
        self._accumulated_charge = torch.zeros(
            self.config.max_seq_len,
            device=device,
            dtype=torch.float32,
        )
        self._accumulated_charge[:actual_seq_len] = combined
        self._prefill_initialized = True

    def update(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        current_idx: int,
    ) -> None:
        """
        Update the graph and accumulate charge via EMA (Generation Step).

        This triggers:
        1. Graph Update Kernel: Compute Top-K neighbors from query-key dot products
        2. Circuit Walker Kernel: Run absorbing walks from current_idx toward sink
        3. Charge Update: EMA accumulation of current flow into capacitor state

        Physics:
        - Current_t = CircuitWalker(query) -> instant current flow
        - Charge_t = decay * Charge_{t-1} + (1-decay) * Current_t

        This ensures:
        - Heavy Hitters maintain charge (high decay preserves history)
        - Reasoning Bridges gain charge immediately when walker visits them

        Args:
            query: Query vector of the new token. Shape: [1, D] or [D]
            keys: Key cache of all past tokens. Shape: [N, D]
            current_idx: Index of the current token (source node).
        """
        self._lazy_init()

        # Ensure correct shapes
        if query.dim() == 1:
            query = query.unsqueeze(0)

        query = query.contiguous()
        keys = keys.contiguous()

        # Run circuit walker to get instant current flow
        self._graph.update_and_step_circuit(query, keys, current_idx)

        # Get instant current flow from walker
        instant_current = self._graph.get_scores()

        # Normalize instant current to [0, 1]
        max_current = instant_current[:current_idx + 1].max()
        if max_current > 0:
            instant_current_norm = instant_current / max_current
        else:
            instant_current_norm = instant_current

        # Initialize accumulated charge if not done (fallback for no prefill init)
        if self._accumulated_charge is None:
            self._accumulated_charge = torch.zeros(
                self.config.max_seq_len,
                device=instant_current.device,
                dtype=torch.float32,
            )

        # EMA Update: Charge_t = decay * Charge_{t-1} + (1-decay) * Current_t
        decay = self.config.decay
        self._accumulated_charge = (
            decay * self._accumulated_charge +
            (1 - decay) * instant_current_norm.float()
        )

    def get_scores(self) -> torch.Tensor:
        """
        Get the accumulated charge scores for all tokens.

        Returns accumulated charge (capacitor state) which combines:
        - Initial H2O scores from prefill (if initialized)
        - EMA-accumulated current flow from generation steps

        Returns:
            Tensor of shape [max_seq_len] with accumulated charge values.
        """
        self._lazy_init()

        # Return accumulated charge if available, otherwise raw walker scores
        if self._accumulated_charge is not None:
            return self._accumulated_charge
        else:
            return self._graph.get_scores()

    def get_instant_current(self) -> torch.Tensor:
        """
        Get the raw instant current flow from the most recent walker run.

        This bypasses the accumulated charge and returns the raw visit counts
        from the circuit walker. Useful for debugging or analysis.

        Returns:
            Tensor of shape [max_seq_len] with raw visit counts.
        """
        self._lazy_init()
        return self._graph.get_scores()

    def get_keep_mask(
        self,
        budget_ratio: float,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Get a boolean mask indicating which tokens to keep.

        The mask respects:
        1. Attention sinks (always kept - these are the absorbing boundary)
        2. Local window (always kept)
        3. Top tokens by current-flow score (up to budget)

        Args:
            budget_ratio: Fraction of total tokens to keep (0.0-1.0)
            seq_len: Current sequence length

        Returns:
            Boolean tensor of shape [seq_len] where True = keep.
        """
        scores = self.get_scores()[:seq_len]
        budget = int(seq_len * budget_ratio)

        # Always keep sinks and local window
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=scores.device)
        keep_mask[: self.config.sink_size] = True  # Sinks (absorbing boundary)
        local_start = max(0, seq_len - self.config.local_window)
        keep_mask[local_start:] = True  # Local window

        # Count how many we already kept
        already_kept = keep_mask.sum().item()
        remaining_budget = max(0, budget - already_kept)

        # Select top tokens by score (excluding already kept)
        if remaining_budget > 0:
            scores_masked = scores.clone()
            scores_masked[keep_mask] = float("-inf")  # Exclude already kept
            _, top_indices = scores_masked.topk(remaining_budget)
            keep_mask[top_indices] = True

        return keep_mask

    def reset(self) -> None:
        """Reset the graph and capacitor state for a new sequence."""
        if self._graph is not None:
            self._graph.reset()

        # Reset capacitive state
        self._accumulated_charge = None
        self._prefill_initialized = False

    def synchronize(self) -> None:
        """Wait for all async operations to complete."""
        if self._graph is not None:
            self._graph.synchronize()
