/**
 * CircuitKV - Kernel Launcher Declarations
 *
 * This header declares the host-side launcher functions for our CUDA kernels.
 * These launchers handle block/thread configuration and stream management.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace circuit_kv {

// =============================================================================
// Kernel 1: Graph Update (Top-K Neighbor Selection)
// =============================================================================

/**
 * Launch the graph update kernel.
 *
 * Computes dot products between the query vector and all keys,
 * then selects the Top-K highest-scoring keys as neighbors.
 *
 * Block/Thread Mapping:
 * - 1 block, 256 threads (or adjusted based on seq_len)
 * - Each thread handles multiple keys in a loop
 * - Shared memory for partial Top-K reduction
 *
 * @param query        Query vector of new token [1, head_dim], FP16
 * @param keys         Key cache [seq_len, head_dim], FP16
 * @param adj_list     Output adjacency list [max_seq_len, top_k], INT32
 * @param adj_weights  Output edge weights [max_seq_len, top_k], FP32
 * @param current_idx  Index of the current token (row to update)
 * @param seq_len      Current sequence length (number of valid keys)
 * @param head_dim     Dimension of query/key vectors
 * @param top_k        Number of neighbors to select
 * @param stream       CUDA stream for async execution
 */
void launch_graph_update_kernel(
    const __half* query,
    const __half* keys,
    int32_t* adj_list,
    float* adj_weights,
    int current_idx,
    int seq_len,
    int head_dim,
    int top_k,
    cudaStream_t stream
);

/**
 * FP32 version of graph update kernel for testing/debugging.
 */
void launch_graph_update_kernel_fp32(
    const float* query,
    const float* keys,
    int32_t* adj_list,
    float* adj_weights,
    int current_idx,
    int seq_len,
    int head_dim,
    int top_k,
    cudaStream_t stream
);

// =============================================================================
// Kernel 2: CircuitKV Absorbing Walker (Current-Flow Betweenness)
// =============================================================================

/**
 * Launch the CircuitKV Absorbing Walker kernel.
 *
 * Runs num_walkers absorbing random walks from source_node toward the sink.
 * Walkers flow from SOURCE (query token) to SINK (first SINK_SIZE tokens),
 * simulating electrical current through the attention graph.
 *
 * Key Properties:
 * - NO RESTART: Walkers never teleport. They march until absorbed or stuck.
 * - ABSORBING BOUNDARY: Walk stops when current_pos < SINK_SIZE (default 4).
 * - WEIGHTED SAMPLING: Neighbors sampled proportional to attention weights.
 * - TRANSPORT METRIC: Visit counts measure "current flow" through each token.
 *
 * This captures "bridge tokens" that connect Query to Context but have low
 * degree. Unlike PPR, these tokens accumulate visits because they're on the
 * path from Source to Sink.
 *
 * Block/Thread Mapping:
 * - num_walkers / 256 blocks, 256 threads per block
 * - Each thread is one walker
 * - Visit counts updated via atomicAdd
 *
 * @param adj_list      Adjacency list [max_seq_len, top_k], INT32
 * @param adj_weights   Edge weights (attention scores) [max_seq_len, top_k], FP32
 * @param visit_counts  Output visit counts [max_seq_len], INT32
 * @param rng_states    Pre-initialized PRNG states [num_walkers * 2]
 * @param source_node   Source node (current query position, usually last token)
 * @param seq_len       Current sequence length
 * @param top_k         Neighbors per token
 * @param num_walkers   Total number of parallel walkers
 * @param stream        CUDA stream for async execution
 */
void launch_absorbing_walker_kernel(
    const int32_t* adj_list,
    const float* adj_weights,
    int32_t* visit_counts,
    uint64_t* rng_states,
    int source_node,
    int seq_len,
    int top_k,
    int num_walkers,
    cudaStream_t stream
);

// =============================================================================
// Initialization Kernels
// =============================================================================

/**
 * Initialize PRNG states for random walkers.
 * Should be called once during setup, not every step.
 *
 * @param rng_states   Output PRNG states [num_walkers * 2] (state, inc pairs)
 * @param seed         Base seed for initialization
 * @param num_walkers  Number of walkers
 * @param stream       CUDA stream
 */
void launch_init_rng_kernel(
    uint64_t* rng_states,
    uint64_t seed,
    int num_walkers,
    cudaStream_t stream
);

/**
 * Reset visit counts to zero.
 *
 * @param visit_counts  Visit counts array [max_seq_len]
 * @param max_seq_len   Array size
 * @param stream        CUDA stream
 */
void launch_reset_counts_kernel(
    int32_t* visit_counts,
    int max_seq_len,
    cudaStream_t stream
);

/**
 * Reset adjacency list to -1 (no neighbors).
 *
 * @param adj_list     Adjacency list [max_seq_len, top_k]
 * @param max_seq_len  Maximum sequence length
 * @param top_k        Neighbors per token
 * @param stream       CUDA stream
 */
void launch_reset_graph_kernel(
    int32_t* adj_list,
    int max_seq_len,
    int top_k,
    cudaStream_t stream
);

// =============================================================================
// Kernel 3: Spectral Power Iteration (Eigenvector Centrality)
// =============================================================================

/**
 * Launch sparse matrix-vector multiply: v_out = A @ v_in
 * Uses adjacency list representation for efficient SpMV.
 *
 * @param adj_list     Sparse neighbors [seq_len, top_k]
 * @param adj_weights  Edge weights [seq_len, top_k]
 * @param v_in         Input vector [seq_len]
 * @param v_out        Output vector [seq_len]
 * @param seq_len      Current sequence length
 * @param top_k        Max neighbors per token
 * @param stream       CUDA stream
 */
void launch_spmv_kernel(
    const int32_t* adj_list,
    const float* adj_weights,
    const float* v_in,
    float* v_out,
    int seq_len,
    int top_k,
    cudaStream_t stream
);

/**
 * Launch vector normalization (L2 norm).
 * Computes v = v / ||v||_2 in-place.
 *
 * @param v               Vector to normalize in-place [seq_len]
 * @param partial_sums    Temporary buffer for reduction [num_reduce_blocks]
 * @param norm_out        Temporary scalar for norm [1]
 * @param seq_len         Vector length
 * @param num_reduce_blocks Number of blocks for reduction
 * @param stream          CUDA stream
 */
void launch_normalize_vector_kernel(
    float* v,
    float* partial_sums,
    float* norm_out,
    int seq_len,
    int num_reduce_blocks,
    cudaStream_t stream
);

/**
 * Initialize vector to uniform: v[i] = 1/sqrt(N)
 *
 * @param v         Output vector [seq_len]
 * @param seq_len   Vector length
 * @param stream    CUDA stream
 */
void launch_init_uniform_kernel(
    float* v,
    int seq_len,
    cudaStream_t stream
);

/**
 * Run power iteration to compute dominant eigenvector.
 * This is the main entry point for spectral scoring.
 *
 * Algorithm:
 *   1. Initialize v = uniform
 *   2. For num_iterations:
 *      v = normalize(A @ v)
 *   3. Return v as importance scores
 *
 * @param adj_list       Sparse neighbors [seq_len, top_k]
 * @param adj_weights    Edge weights [seq_len, top_k]
 * @param v              Output eigenvector [seq_len]
 * @param v_temp         Temporary buffer [seq_len]
 * @param partial_sums   Temporary for reduction [256]
 * @param norm_out       Temporary scalar [1]
 * @param seq_len        Current sequence length
 * @param top_k          Max neighbors per token
 * @param num_iterations Number of power iterations (default 10)
 * @param stream         CUDA stream
 */
void launch_power_iteration(
    const int32_t* adj_list,
    const float* adj_weights,
    float* v,
    float* v_temp,
    float* partial_sums,
    float* norm_out,
    int seq_len,
    int top_k,
    int num_iterations,
    cudaStream_t stream
);

/**
 * Convert integer visit counts to float scores.
 *
 * @param visit_counts  Integer visit counts from walker [seq_len]
 * @param scores        Output float scores [seq_len]
 * @param seq_len       Array length
 * @param stream        CUDA stream
 */
void launch_convert_visits_kernel(
    const int32_t* visit_counts,
    float* scores,
    int seq_len,
    cudaStream_t stream
);

/**
 * Normalize scores to [0, 1] range by dividing by max.
 *
 * @param v               Vector to normalize in-place [seq_len]
 * @param partial_max     Temporary for reduction [num_reduce_blocks]
 * @param max_out         Temporary scalar for max [1]
 * @param seq_len         Vector length
 * @param num_reduce_blocks Number of blocks for reduction
 * @param stream          CUDA stream
 */
void launch_normalize_to_unit_max(
    float* v,
    float* partial_max,
    float* max_out,
    int seq_len,
    int num_reduce_blocks,
    cudaStream_t stream
);

/**
 * Combine spectral and walker scores using element-wise MAX.
 *
 * combined[i] = max(spectral[i], walker[i])
 *
 * This preserves BOTH hub tokens (high spectral) AND bridge tokens (high walker).
 *
 * @param spectral_scores  Normalized spectral scores [seq_len]
 * @param walker_scores    Normalized walker scores [seq_len]
 * @param combined_scores  Output combined scores [seq_len]
 * @param seq_len          Array length
 * @param stream           CUDA stream
 */
void launch_max_combine_kernel(
    const float* spectral_scores,
    const float* walker_scores,
    float* combined_scores,
    int seq_len,
    cudaStream_t stream
);

// =============================================================================
// Kernel 4: Landmark Walker (Multi-Source Absorbing Walks)
// =============================================================================

/**
 * Launch the landmark walker kernel.
 *
 * Runs absorbing random walks from multiple geographically-diverse sources:
 * - Landmarks selected via H2O (high column sum) with spacing constraint
 * - Query token (with boosted weight)
 *
 * Key Insight:
 *   Single-source walks miss important tokens not directly visible from query.
 *   Multi-source walks from diverse landmarks discover "bridge tokens" through
 *   path convergence.
 *
 * Block/Thread Mapping:
 * - total_walkers = (num_landmarks + 1) * walkers_per_source
 * - Each thread is one walker assigned to one source
 * - Visit counts shared across all walkers (atomicAdd)
 *
 * @param landmark_attention  Cached attention rows [num_landmarks, seq_len]
 * @param query_attention     Query's attention row [seq_len]
 * @param h2o_scores          H2O scores for fallback transitions [seq_len]
 * @param visit_counts        Output visit counts [seq_len]
 * @param rng_states          PRNG states [total_walkers * 2]
 * @param landmark_positions  Landmark positions [num_landmarks]
 * @param num_landmarks       Number of landmarks (not including query)
 * @param walkers_per_source  Walkers per source
 * @param query_boost         Weight multiplier for query walkers (default 2.0)
 * @param seq_len             Current sequence length
 * @param stream              CUDA stream
 */
void launch_landmark_walker_kernel(
    const float* landmark_attention,
    const float* query_attention,
    const float* h2o_scores,
    int32_t* visit_counts,
    uint64_t* rng_states,
    const int32_t* landmark_positions,
    int num_landmarks,
    int walkers_per_source,
    float query_boost,
    int seq_len,
    cudaStream_t stream
);

/**
 * Launch positional normalization to remove -log bias.
 *
 * Absorbing walks create visits[p] ~ 1/distance because all walkers pass
 * through early positions. This normalizes by expected visits to reveal
 * tokens visited MORE than expected.
 *
 * normalized[p] = visits[p] / (1 / (p - sink_size + 1)^alpha)
 *
 * @param visit_counts       Raw visit counts [seq_len]
 * @param normalized_scores  Output normalized scores [seq_len]
 * @param partial_max        Temp buffer for max reduction [num_blocks]
 * @param seq_len            Sequence length
 * @param sink_size          Sink region size (default 4)
 * @param alpha              Position exponent (default 0.6)
 * @param stream             CUDA stream
 */
void launch_positional_normalize_kernel(
    const int32_t* visit_counts,
    float* normalized_scores,
    float* partial_max,
    int seq_len,
    int sink_size,
    float alpha,
    cudaStream_t stream
);

/**
 * Launch reachability normalization (WINNING STRATEGY from PoC ablation).
 *
 * Normalizes visits by total walkers that COULD REACH each position.
 * This is better than positional normalization because it accounts for
 * the actual walker distribution, not just position.
 *
 * normalized[p] = visits[p] / total_reachable[p]
 * where total_reachable[p] = sum over sources > p of (num_walkers * weight)
 *
 * @param visit_counts       Raw visit counts [seq_len]
 * @param normalized_scores  Output normalized scores [seq_len]
 * @param landmark_positions Landmark positions [num_landmarks]
 * @param num_landmarks      Number of landmarks
 * @param walkers_per_source Walkers per source
 * @param query_boost        Weight for query walkers
 * @param seq_len            Sequence length
 * @param sink_size          Sink region size (default 4)
 * @param stream             CUDA stream
 */
void launch_reachability_normalize_kernel(
    const int32_t* visit_counts,
    float* normalized_scores,
    const int32_t* landmark_positions,
    int num_landmarks,
    int walkers_per_source,
    float query_boost,
    int seq_len,
    int sink_size,
    cudaStream_t stream
);

/**
 * Select diverse landmarks using H2O scores with spacing constraint.
 *
 * Algorithm:
 *   1. Greedily select highest H2O position
 *   2. Exclude positions within min_spacing
 *   3. Repeat until max_landmarks reached
 *
 * @param attention_row_sums  H2O scores (column sums) [seq_len]
 * @param landmark_positions  Output landmark positions [max_landmarks]
 * @param num_landmarks_out   Output number of landmarks selected
 * @param seq_len             Sequence length
 * @param max_landmarks       Maximum landmarks to select
 * @param min_spacing         Minimum spacing between landmarks
 * @param sink_buffer         Extra buffer from sink region
 * @param window_size         Last window_size tokens excluded
 * @param stream              CUDA stream
 */
void launch_select_landmarks_kernel(
    const float* attention_row_sums,
    int32_t* landmark_positions,
    int* num_landmarks_out,
    int seq_len,
    int max_landmarks,
    int min_spacing,
    int sink_buffer,
    int window_size,
    cudaStream_t stream
);

/**
 * Cache attention rows for selected landmarks.
 *
 * @param full_attention       Full attention matrix or provided rows
 * @param landmark_attention   Output cached attention [num_landmarks, seq_len]
 * @param landmark_positions   Landmark positions [num_landmarks]
 * @param num_landmarks        Number of landmarks
 * @param seq_len              Sequence length
 * @param stream               CUDA stream
 */
void launch_cache_landmark_attention_kernel(
    const float* full_attention,
    float* landmark_attention,
    const int32_t* landmark_positions,
    int num_landmarks,
    int seq_len,
    cudaStream_t stream
);

/**
 * Compute H2O scores (column sums of attention matrix).
 *
 * H2O score for position j = sum_i(attention[i, j])
 * This measures how much total attention each key receives.
 *
 * @param attention_matrix   Full attention [seq_len, seq_len] or sparse
 * @param h2o_scores         Output scores [seq_len]
 * @param seq_len            Sequence length
 * @param stream             CUDA stream
 */
void launch_compute_h2o_scores_kernel(
    const float* attention_matrix,
    float* h2o_scores,
    int seq_len,
    cudaStream_t stream
);

// =============================================================================
// Kernel 5: Landmark Absorbing Walker (Landmarks as Sources AND Sinks)
// =============================================================================

/**
 * Launch the landmark absorbing walker kernel.
 *
 * NEW APPROACH: Landmarks are both SOURCES and SINKS.
 * Walkers absorb when reaching ANY landmark (not just tokens 0-3).
 * This creates a "mesh" of current flows between landmarks.
 *
 * Physics Analogy:
 *   - Old: Single battery (Query+ to Sink-)
 *   - New: Mesh network where each landmark can source or sink current
 *
 * Key Benefit:
 *   Captures "local bridges" - tokens that connect ADJACENT landmarks.
 *   A token between landmarks L1 and L2 gets visits from both directions.
 *
 * @param landmark_attention   Cached attention rows [num_landmarks, seq_len]
 * @param query_attention      Query's attention row [seq_len]
 * @param h2o_scores           H2O scores for fallback transitions [seq_len]
 * @param visit_counts         Output visit counts [seq_len]
 * @param rng_states           PRNG states [total_walkers * 2]
 * @param landmark_positions   Landmark positions [num_landmarks]
 * @param num_landmarks        Number of landmarks (not including query)
 * @param walkers_per_source   Walkers per source
 * @param query_boost          Weight multiplier for query walkers
 * @param seq_len              Current sequence length
 * @param absorb_at_landmarks  If true, walkers absorb at landmarks (new behavior)
 * @param stream               CUDA stream
 */
void launch_landmark_absorbing_walker_kernel(
    const float* landmark_attention,
    const float* query_attention,
    const float* h2o_scores,
    int32_t* visit_counts,
    uint64_t* rng_states,
    const int32_t* landmark_positions,
    int num_landmarks,
    int walkers_per_source,
    float query_boost,
    int seq_len,
    bool absorb_at_landmarks,
    cudaStream_t stream
);

/**
 * Launch reachability normalization for landmark-absorbing walks.
 *
 * @param visit_counts        Raw visit counts [seq_len]
 * @param normalized_scores   Output normalized scores [seq_len]
 * @param landmark_positions  Landmark positions [num_landmarks]
 * @param num_landmarks       Number of landmarks
 * @param walkers_per_source  Walkers per source
 * @param query_boost         Weight for query walkers
 * @param seq_len             Sequence length
 * @param sink_size           Sink region size (default 4)
 * @param stream              CUDA stream
 */
void launch_landmark_reachability_normalize_kernel(
    const int32_t* visit_counts,
    float* normalized_scores,
    const int32_t* landmark_positions,
    int num_landmarks,
    int walkers_per_source,
    float query_boost,
    int seq_len,
    int sink_size,
    cudaStream_t stream
);

// =============================================================================
// Kernel 6: Causal Influence Walker (v1.0.0 - VALIDATED BY PoC5)
// =============================================================================

/**
 * Launch the causal influence walker kernel.
 *
 * VALIDATED BY PoC5:
 *   - Influence vs Gen Attn: Spearman r = 0.41 (H2O: -0.02)
 *   - Top-10 overlap with actual generation attention: 70% (H2O: 10%)
 *   - Walker approximates Influence Oracle: Spearman r = 0.94
 *
 * Algorithm:
 *   1. All walkers start at current_idx (generation position)
 *   2. At each step, walker at `pos` samples next from A[pos, :pos+1]
 *   3. Visit weight = cumulative product of attention along path
 *   4. Absorb at sink (first sink_size tokens)
 *
 * Key Insight:
 *   Influence = "How much can token j reach the generation position through
 *   multi-hop attention?" This correlates with actual generation attention
 *   MUCH better than H2O (which just measures degree/popularity).
 *
 * Block/Thread Mapping:
 *   - num_walkers / 256 blocks, 256 threads per block
 *   - Each thread is one walker starting at current_idx
 *   - Weighted visits updated via atomicAdd
 *
 * @param attention      Full attention matrix [seq_len, seq_len], FP32
 * @param visits         Output weighted visit counts [seq_len], FP32
 * @param rng_states     PRNG states [num_walkers * 2]
 * @param seq_len        Sequence length
 * @param current_idx    Generation position (start for all walkers)
 * @param num_walkers    Number of walkers (default 10000)
 * @param max_steps      Max steps per walker (default 10)
 * @param sink_size      Absorbing boundary (default 4)
 * @param stream         CUDA stream
 */
void launch_influence_walker_kernel(
    const float* attention,
    float* visits,
    uint64_t* rng_states,
    int seq_len,
    int current_idx,
    int num_walkers,
    int max_steps,
    int sink_size,
    cudaStream_t stream
);

/**
 * Clear influence visits buffer (float version).
 *
 * @param visits    Float visit buffer [seq_len]
 * @param seq_len   Sequence length
 * @param stream    CUDA stream
 */
void launch_clear_influence_visits_kernel(
    float* visits,
    int seq_len,
    cudaStream_t stream
);

/**
 * Find max and normalize influence scores to [0, 1].
 *
 * Two-pass operation:
 *   1. Find max value across all visits
 *   2. Apply positional opportunity normalization (v1.0.5)
 *   3. Divide all values by max
 *
 * @param visits          Input visit counts [seq_len]
 * @param normalized      Output normalized scores [seq_len]
 * @param partial_max     Temp buffer for reduction [num_blocks]
 * @param seq_len         Sequence length
 * @param num_blocks      Number of reduction blocks
 * @param stream          CUDA stream
 * @param sink_size       Size of sink region for positional adjustment (v1.0.5)
 */
void launch_find_max_and_normalize_kernel(
    float* visits,
    float* normalized,
    float* partial_max,
    int seq_len,
    int num_blocks,
    cudaStream_t stream,
    int sink_size = 4  // v1.0.5: Default to 4 for backwards compatibility
);

}  // namespace circuit_kv
