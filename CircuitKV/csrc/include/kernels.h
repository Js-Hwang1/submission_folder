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

/**
 * Batched graph update kernel - builds graphs for W sources in parallel.
 * (P3 Optimization)
 *
 * @param queries         Query vectors [num_sources, head_dim], FP32
 * @param keys            Key cache [seq_len, head_dim], FP32
 * @param adj_list        Output adjacency list [max_seq_len, top_k], INT32
 * @param adj_weights     Output edge weights [max_seq_len, top_k], FP32
 * @param source_indices  Indices to update [num_sources], INT32
 * @param num_sources     Number of sources (W)
 * @param seq_len         Current sequence length
 * @param head_dim        Dimension of query/key vectors
 * @param top_k           Number of neighbors to select per source
 * @param stream          CUDA stream
 */
void launch_batched_graph_update_kernel_fp32(
    const float* queries,
    const float* keys,
    int32_t* adj_list,
    float* adj_weights,
    const int* source_indices,
    int num_sources,
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

/**
 * Multi-source absorbing walker kernel - runs walks from W sources in parallel.
 * (P3 Optimization)
 *
 * This kernel runs walks from ALL tokens in the observation window simultaneously,
 * avoiding the Python loop overhead. Total walkers = num_sources * walkers_per_source.
 *
 * @param adj_list           Adjacency list [max_seq_len, top_k], INT32
 * @param adj_weights        Edge weights [max_seq_len, top_k], FP32
 * @param visit_counts       Output visit counts [max_seq_len], INT32 (atomicAdd)
 * @param rng_states         PRNG states [num_sources * walkers_per_source * 2]
 * @param source_nodes       Array of source indices [num_sources], INT32
 * @param num_sources        Number of sources (W)
 * @param walkers_per_source Number of walkers per source
 * @param seq_len            Current sequence length
 * @param top_k              Neighbors per token
 * @param stream             CUDA stream
 */
void launch_multi_source_absorbing_walker_kernel(
    const int32_t* adj_list,
    const float* adj_weights,
    int32_t* visit_counts,
    uint64_t* rng_states,
    const int* source_nodes,
    int num_sources,
    int walkers_per_source,
    int seq_len,
    int top_k,
    cudaStream_t stream
);

// =============================================================================
// Kernel 3: Transpose Graph Construction (For Forward Walks)
// =============================================================================

/**
 * Build transpose graph for forward walks (who attends TO each token).
 *
 * For bidirectional CircuitKV, we need both:
 * - Forward graph: A[i,j] = token i attends to token j (for backward walks)
 * - Transpose graph: A^T[k,j] = token j attends to token k (for forward walks)
 *
 * The transpose graph allows walkers to move from Sink toward Query,
 * following the "who attends to me" direction.
 *
 * @param keys            Key cache [seq_len, head_dim], FP32
 * @param queries         Query cache [seq_len, head_dim], FP32
 * @param rev_adj_list    Output transpose adjacency list [max_seq_len, top_k], INT32
 * @param rev_adj_weights Output transpose edge weights [max_seq_len, top_k], FP32
 * @param seq_len         Current sequence length
 * @param head_dim        Dimension of query/key vectors
 * @param top_k           Number of neighbors to select
 * @param stream          CUDA stream
 */
void launch_build_transpose_graph_kernel_fp32(
    const float* keys,
    const float* queries,
    int32_t* rev_adj_list,
    float* rev_adj_weights,
    int seq_len,
    int head_dim,
    int top_k,
    cudaStream_t stream
);

// =============================================================================
// Kernel 4: Bidirectional Absorbing Walker (RC+B)
// =============================================================================

/**
 * Launch the bidirectional absorbing walker kernel.
 *
 * Runs walkers in BOTH directions simultaneously:
 * - Backward: Query -> Sink (following A[i,j], who i attends to)
 * - Forward: Sink -> Query (following A^T[k,j], who attends to k)
 *
 * Tokens visited by BOTH directions are "true bridges" and get bonus scoring.
 * Bridge score = min(backward_visits, forward_visits)
 * Final score = max(backward, forward) + 0.5 * bridge
 *
 * Block/Thread Mapping:
 * - (total_walkers / 256) blocks, 256 threads per block
 * - total_walkers = (num_query_nodes + num_sink_nodes) * walkers_per_direction
 * - First half of threads run backward walks, second half run forward walks
 *
 * @param adj_list           Forward adjacency list [max_seq_len, top_k], INT32
 * @param adj_weights        Forward edge weights [max_seq_len, top_k], FP32
 * @param rev_adj_list       Transpose adjacency list [max_seq_len, top_k], INT32
 * @param rev_adj_weights    Transpose edge weights [max_seq_len, top_k], FP32
 * @param backward_visits    Output backward visit counts [max_seq_len], INT32
 * @param forward_visits     Output forward visit counts [max_seq_len], INT32
 * @param rng_states         PRNG states [total_walkers * 2]
 * @param query_nodes        Source nodes for backward walks [num_query_nodes], INT32
 * @param sink_nodes         Source nodes for forward walks [num_sink_nodes], INT32
 * @param num_query_nodes    Number of query sources (observation window size)
 * @param num_sink_nodes     Number of sink sources (typically SINK_SIZE=4)
 * @param walkers_per_dir    Walkers per source per direction
 * @param seq_len            Current sequence length
 * @param top_k              Neighbors per token
 * @param query_region_start Start of query region (for forward walk termination)
 * @param stream             CUDA stream
 */
void launch_bidirectional_walker_kernel(
    const int32_t* adj_list,
    const float* adj_weights,
    const int32_t* rev_adj_list,
    const float* rev_adj_weights,
    int32_t* backward_visits,
    int32_t* forward_visits,
    uint64_t* rng_states,
    const int* query_nodes,
    const int* sink_nodes,
    int num_query_nodes,
    int num_sink_nodes,
    int walkers_per_direction,
    int seq_len,
    int top_k,
    int query_region_start,
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

}  // namespace circuit_kv
