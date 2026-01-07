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
    local_window: int = 64  # Local window to always keep

    # Capacitive CircuitKV: State-Space Model Parameters
    # Physics: Tokens are capacitors that accumulate charge over time
    decay: float = 0.95  # EMA decay for charge accumulation (temporal smoothing)

    # Hardware
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


@dataclass
class LandmarkWalkerConfig:
    """Configuration for Landmark-Diverse Walker with Positional Normalization.

    Configuration (v0.4.0):
    - Stratified landmark selection (one per segment, best H2O within segment)
    - 32 landmarks for better coverage
    - Positional normalization (default): visits / (1/distance^alpha)
    """

    # Graph parameters
    max_seq_len: int = 8192  # Maximum sequence length
    top_k: int = 32  # Kept for CircuitGraph initialization (not used by landmark walker)

    # Landmark selection parameters (STRATIFIED strategy)
    num_landmarks: int = 32  # Number of diverse landmarks (32 for comprehensive coverage)
    min_spacing: int = 50  # Minimum segment size for stratified sampling

    # Walker parameters
    walkers_per_source: int = 100  # Walkers launched from each source
    query_boost: float = 2.0  # Weight multiplier for query-sourced walkers

    # Normalization mode: True = reachability, False = positional (default)
    use_reachability: bool = False  # Positional normalization: visits / (1/distance^alpha)

    # Positional normalization parameter (only used if use_reachability=False)
    position_alpha: float = 0.6  # Exponent for expected visits (1/distance^alpha)

    # Eviction parameters
    sink_size: int = 4  # Absorbing boundary (first N tokens)
    local_window: int = 64  # Local window to always keep

    # Hardware
    device: str = "cuda"


class CircuitKVMonitor:
    """
    The Capacitive CircuitKV Monitor.

    This class wraps the CUDA extension and provides a high-level interface
    for updating the attention graph and computing current-flow scores.

    Physics Model (RC Circuit / State-Space):
    - Tokens are CAPACITORS that accumulate charge over time
    - INITIALIZATION (End of Prefill): Charge is set from H2O accumulated attention
      This represents the "steady state" charge deposited by prefill tokens.
    - UPDATE (During Generation): Current from CircuitKV walker updates charge via EMA
      Charge_t = decay * Charge_{t-1} + (1-decay) * Current_t

    Key Innovation:
    - Heavy Hitters (NarrativeQA): Start with high charge from prefill attention
    - Reasoning Bridges (HotpotQA): Gain charge immediately when referenced by walker,
      overriding their low historical attention

    Usage:
        config = CircuitKVConfig()
        monitor = CircuitKVMonitor(config)

        # At end of prefill phase:
        monitor.initialize_from_prefill(attention_matrix)

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
        attention_matrix: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> None:
        """
        Initialize capacitor charge from prefill-phase attention (H2O Initialization).

        This sets the "steady state" charge based on accumulated attention during prefill.
        Heavy Hitters (tokens that received lots of attention) start with high charge.

        Physics Rationale:
        - During prefill, ~30k tokens have already "deposited" attention onto keys
        - This accumulated attention represents the equilibrium charge distribution
        - We use column sums of attention matrix (H2O scores) as initial charge

        Args:
            attention_matrix: Attention weights from prefill. Shape: [N, N] or [B, H, N, N]
                              Can also be pre-computed column sums [N] or [B, H, N]
            seq_len: Sequence length (optional, inferred from attention_matrix)
        """
        self._lazy_init()

        # Handle different input shapes
        if attention_matrix.dim() == 1:
            # Already column sums [N]
            h2o_scores = attention_matrix
        elif attention_matrix.dim() == 2:
            # Full attention matrix [N, N] -> column sums
            h2o_scores = attention_matrix.sum(dim=0)
        elif attention_matrix.dim() == 4:
            # Batched [B, H, N, N] -> average over batch and heads, then column sum
            h2o_scores = attention_matrix.mean(dim=(0, 1)).sum(dim=0)
        else:
            # [B, H, N] -> average over batch and heads
            h2o_scores = attention_matrix.mean(dim=(0, 1))

        # Normalize to [0, 1] range
        max_score = h2o_scores.max()
        if max_score > 0:
            h2o_scores = h2o_scores / max_score

        # Initialize accumulated charge
        actual_seq_len = seq_len if seq_len is not None else h2o_scores.shape[0]
        self._accumulated_charge = torch.zeros(
            self.config.max_seq_len,
            device=h2o_scores.device,
            dtype=torch.float32,
        )
        self._accumulated_charge[:actual_seq_len] = h2o_scores[:actual_seq_len].float()
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


class LandmarkWalkerMonitor:
    """
    Landmark-Diverse Walker Monitor with Reachability Normalization.

    This class implements the WINNING CircuitKV approach from PoC ablation:
    1. STRATIFIED landmark selection (one per segment, best H2O within segment)
    2. Launch walkers from ALL sources (landmarks + query) in parallel
    3. Apply REACHABILITY normalization (accounts for walker distribution)

    Key Insight:
        Single-source walks miss important tokens not directly visible from query.
        By launching walks from geographically-diverse landmarks, we discover
        "bridge tokens" through path convergence.

    Reachability Normalization (vs Positional):
        - Positional: normalized[p] = visits[p] / (1/distance^alpha)
        - Reachability: normalized[p] = visits[p] / total_walkers_that_could_reach[p]

        Reachability is better because it accounts for the actual walker
        distribution, not just position. Bridge tokens between landmarks
        get proper credit.

    This is particularly effective for:
    - Multi-hop reasoning (HotpotQA, 2WikiMQA)
    - Long document QA (NarrativeQA, Qasper)
    - Cases where important tokens are not directly attended by the query

    Usage:
        config = LandmarkWalkerConfig()  # Uses 16 landmarks + reachability by default
        monitor = LandmarkWalkerMonitor(config)

        # Given a full attention matrix from the model:
        monitor.update(attention_matrix, current_idx)

        # Get importance scores:
        scores = monitor.get_scores()
        keep_mask = monitor.get_keep_mask(budget_ratio=0.25, seq_len=seq_len)
    """

    def __init__(self, config: LandmarkWalkerConfig):
        self.config = config
        self._graph: Optional[object] = None
        self._initialized = False
        self._current_scores: Optional[torch.Tensor] = None

    def _lazy_init(self) -> None:
        """Lazily initialize the CUDA circuit manager."""
        if self._initialized:
            return

        # Import here to defer CUDA initialization
        from circuit_kv import get_extension

        ext = get_extension()
        # Use placeholder values for parameters not used by landmark walker
        self._graph = ext.CircuitGraph(
            self.config.max_seq_len,
            self.config.top_k,  # Not used by landmark walker
            0.85,  # alpha - not used
            1024,  # num_walkers - uses its own walkers
            100,   # num_steps - uses its own
        )
        self._initialized = True

    def update(
        self,
        attention_matrix: torch.Tensor,
        current_idx: int,
    ) -> None:
        """
        Run landmark-diverse walker on the attention matrix.

        This triggers:
        1. H2O Score Computation: Column sums to find high-attention tokens
        2. Landmark Selection: Greedy selection with spacing constraint
        3. Attention Caching: Cache landmark attention rows
        4. Multi-source Walking: Launch walkers from all landmarks + query
        5. Positional Normalization: Remove -log bias from visit counts

        Args:
            attention_matrix: Full attention matrix [seq_len, seq_len], FP32.
            current_idx: Index of the current token (query position).
        """
        self._lazy_init()

        # Ensure correct dtype and contiguous
        if attention_matrix.dtype != torch.float32:
            attention_matrix = attention_matrix.float()
        attention_matrix = attention_matrix.contiguous()

        # Run landmark walker with configured normalization mode
        self._graph.update_and_step_landmark_walker(
            attention_matrix,
            current_idx,
            self.config.num_landmarks,
            self.config.walkers_per_source,
            self.config.query_boost,
            self.config.min_spacing,
            self.config.position_alpha,
            self.config.use_reachability,
        )

        # Cache the scores
        self._current_scores = self._graph.get_landmark_scores()

    def get_scores(self) -> torch.Tensor:
        """
        Get the landmark walker importance scores.

        Returns positionally-normalized scores that reveal tokens visited
        more than expected, accounting for the natural -log bias of
        absorbing random walks.

        Returns:
            Tensor of shape [seq_len] with normalized importance scores.
        """
        self._lazy_init()

        if self._current_scores is not None:
            return self._current_scores
        else:
            # No update yet, return zeros
            return torch.zeros(self.config.max_seq_len, device=self.config.device)

    def get_keep_mask(
        self,
        budget_ratio: float,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Get a boolean mask indicating which tokens to keep.

        The mask respects:
        1. Attention sinks (always kept - absorbing boundary)
        2. Local window (always kept)
        3. Top tokens by landmark walker score (up to budget)

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
        """Reset the graph for a new sequence."""
        if self._graph is not None:
            self._graph.reset()
        self._current_scores = None

    def synchronize(self) -> None:
        """Wait for all async operations to complete."""
        if self._graph is not None:
            self._graph.synchronize()
