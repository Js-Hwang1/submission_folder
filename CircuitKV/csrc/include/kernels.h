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

}  // namespace circuit_kv
