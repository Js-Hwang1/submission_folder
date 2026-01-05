/**
 * CircuitKV - Absorbing Walker Kernel
 *
 * Purpose:
 *   Run Monte Carlo random walks on the sparse attention graph to compute
 *   Current-Flow Betweenness scores. Walkers flow from SOURCE (query token)
 *   to SINK (first few tokens), simulating electrical current through a
 *   resistor network.
 *
 * Algorithm (per walker):
 *   1. Start at source_node (the current query token, usually last token)
 *   2. For each step until absorption:
 *      a. Record visit: atomicAdd(&visits[current_pos], 1)
 *      b. Check absorption: if current_pos < SINK_SIZE, STOP (absorbed)
 *      c. Check timeout: if step >= MAX_STEPS, STOP (safety break)
 *      d. Check dead end: if num_neighbors == 0, STOP
 *      e. Sample next node based on attention weights (edge probabilities)
 *      f. Move to sampled neighbor
 *   3. After absorption/termination, visits[] contains "through-traffic" counts
 *
 * Key Properties:
 *   - NO RESTART: Walkers never teleport back to start. They march toward sink.
 *   - ABSORBING BOUNDARY: Walkers stop when reaching sink tokens (indices 0-3).
 *   - TRANSPORT METRIC: We measure "how many walkers passed through token i"
 *     on their way from Question to Context-Start.
 *
 * Physics Analogy:
 *   - Source = Battery positive terminal (Query)
 *   - Sink = Battery negative terminal (Context start)
 *   - Edge weights = Conductance (1/Resistance)
 *   - Visit counts = Current flow through each node
 *
 * CRITICAL: Walk ALONG A (source -> neighbor), NOT A^T!
 *   In causal attention, A[i,j] > 0 means token i attends to token j.
 *   The walker must follow this direction: from query toward past tokens.
 *   Using A^T would cause dead ends (last token has no incoming edges).
 */

#include "kernels/common.cuh"
#include "include/kernels.h"

namespace circuit_kv {

// =============================================================================
// Configuration
// =============================================================================

constexpr int WALKER_BLOCK_SIZE = 256;

// CircuitKV Absorbing Walk Parameters
constexpr int SINK_SIZE = 4;       // Absorbing boundary: tokens 0..(SINK_SIZE-1)
constexpr int MAX_STEPS = 1500;    // Safety timeout to prevent infinite walks

// =============================================================================
// Main Kernel: CircuitKV Absorbing Walker
// =============================================================================

/**
 * CircuitKV Absorbing Walker Kernel - Current-Flow Betweenness via Monte Carlo.
 *
 * Each thread is one walker. Walkers flow from SOURCE (query) toward SINK
 * (first SINK_SIZE tokens), simulating electrical current through the
 * attention graph.
 *
 * This "rescues" bridge tokens: tokens that connect Query to Context
 * but have low degree. They accumulate current because they're on the
 * path from Source to Sink.
 */
__global__ void absorbing_walker_kernel(
    const int32_t* __restrict__ adj_list,
    const float* __restrict__ adj_weights,
    int32_t* __restrict__ visit_counts,
    uint64_t* __restrict__ rng_states,
    int source_node,
    int seq_len,
    int top_k,
    int num_walkers
) {
    int walker_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (walker_id >= num_walkers) return;

    // Load PRNG state from global memory
    PCGState rng;
    rng.state = rng_states[walker_id * 2];
    rng.inc = rng_states[walker_id * 2 + 1];

    // All walkers start at SOURCE (the query position)
    int current_pos = source_node;

    // Absorbing random walk loop
    for (int step = 0; step < MAX_STEPS; ++step) {
        // Record visit at current position (count "current flow" through this node)
        atomicAdd(&visit_counts[current_pos], 1);

        // ABSORPTION CHECK: Stop if we reached the sink (first SINK_SIZE tokens)
        if (current_pos < SINK_SIZE) {
            break;  // Absorbed! Walker terminates successfully.
        }

        // Get neighbors of current position (walking ALONG A, not A^T)
        const int32_t* neighbors = adj_list + current_pos * top_k;
        const float* weights = adj_weights + current_pos * top_k;

        // Compute total weight and count valid neighbors
        float total_weight = 0.0f;
        int num_valid = 0;
        for (int k = 0; k < top_k; ++k) {
            int neighbor = neighbors[k];
            if (neighbor >= 0 && neighbor < seq_len) {
                total_weight += weights[k];
                num_valid++;
            }
        }

        // DEAD END CHECK: No valid neighbors
        if (num_valid == 0 || total_weight <= 0.0f) {
            break;  // Dead end. Walker terminates (not absorbed, but stuck).
        }

        // WEIGHTED TRANSITION: Sample neighbor proportional to attention weight
        float target = pcg_uniform(&rng) * total_weight;
        float cumsum = 0.0f;
        int selected = -1;

        for (int k = 0; k < top_k && selected < 0; ++k) {
            int neighbor = neighbors[k];
            if (neighbor >= 0 && neighbor < seq_len) {
                cumsum += weights[k];
                if (cumsum >= target) {
                    selected = neighbor;
                }
            }
        }

        // Move to selected neighbor (should always find one if total_weight > 0)
        if (selected >= 0) {
            current_pos = selected;
        } else {
            // Fallback: shouldn't happen, but break if it does
            break;
        }
    }

    // Store updated PRNG state back to global memory
    rng_states[walker_id * 2] = rng.state;
    rng_states[walker_id * 2 + 1] = rng.inc;
}

// =============================================================================
// Initialization Kernels
// =============================================================================

/**
 * Initialize PRNG states for all walkers.
 * Each walker gets a unique sequence ID for its RNG stream.
 */
__global__ void init_rng_kernel(
    uint64_t* __restrict__ rng_states,
    uint64_t seed,
    int num_walkers
) {
    int walker_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (walker_id >= num_walkers) return;

    PCGState rng;
    pcg_init(&rng, seed, walker_id);

    // Store state
    rng_states[walker_id * 2] = rng.state;
    rng_states[walker_id * 2 + 1] = rng.inc;
}

/**
 * Reset visit counts to zero.
 */
__global__ void reset_counts_kernel(
    int32_t* __restrict__ visit_counts,
    int max_seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < max_seq_len) {
        visit_counts[idx] = 0;
    }
}

/**
 * Reset adjacency list to -1 (no neighbors).
 */
__global__ void reset_graph_kernel(
    int32_t* __restrict__ adj_list,
    int max_seq_len,
    int top_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = max_seq_len * top_k;
    if (idx < total) {
        adj_list[idx] = -1;
    }
}

// =============================================================================
// Multi-Source Walker Kernel (P3 Optimization)
// =============================================================================

/**
 * Multi-Source Absorbing Walker Kernel - Run walks from W sources in parallel.
 *
 * This kernel runs walks from ALL tokens in the observation window simultaneously,
 * avoiding the Python loop overhead. Each thread handles one walk from one source.
 *
 * Thread Mapping:
 * - Total threads = num_sources * walkers_per_source
 * - walker_id = blockIdx.x * blockDim.x + threadIdx.x
 * - source_idx = walker_id / walkers_per_source
 * - walker_within_source = walker_id % walkers_per_source
 *
 * Visit counts are aggregated across ALL sources (atomicAdd).
 */
__global__ void multi_source_absorbing_walker_kernel(
    const int32_t* __restrict__ adj_list,
    const float* __restrict__ adj_weights,
    int32_t* __restrict__ visit_counts,
    uint64_t* __restrict__ rng_states,
    const int* __restrict__ source_nodes,  // Array of W source indices
    int num_sources,                        // W (observation window size)
    int walkers_per_source,                 // Walkers per source
    int seq_len,
    int top_k
) {
    int global_walker_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_walkers = num_sources * walkers_per_source;
    if (global_walker_id >= total_walkers) return;

    // Determine which source this walker belongs to
    int source_idx = global_walker_id / walkers_per_source;
    int source_node = source_nodes[source_idx];

    // Bounds check on source node
    if (source_node < 0 || source_node >= seq_len) return;

    // Load PRNG state - use global walker id for unique stream
    PCGState rng;
    rng.state = rng_states[global_walker_id * 2];
    rng.inc = rng_states[global_walker_id * 2 + 1];

    // Start walk from this source
    int current_pos = source_node;

    // Absorbing random walk loop
    for (int step = 0; step < MAX_STEPS; ++step) {
        // Record visit at current position
        atomicAdd(&visit_counts[current_pos], 1);

        // ABSORPTION CHECK: Stop if we reached the sink
        if (current_pos < SINK_SIZE) {
            break;
        }

        // Get neighbors of current position
        const int32_t* neighbors = adj_list + current_pos * top_k;
        const float* weights = adj_weights + current_pos * top_k;

        // Compute total weight and count valid neighbors
        float total_weight = 0.0f;
        int num_valid = 0;
        for (int k = 0; k < top_k; ++k) {
            int neighbor = neighbors[k];
            if (neighbor >= 0 && neighbor < seq_len) {
                total_weight += weights[k];
                num_valid++;
            }
        }

        // DEAD END CHECK
        if (num_valid == 0 || total_weight <= 0.0f) {
            break;
        }

        // WEIGHTED TRANSITION
        float target = pcg_uniform(&rng) * total_weight;
        float cumsum = 0.0f;
        int selected = -1;

        for (int k = 0; k < top_k && selected < 0; ++k) {
            int neighbor = neighbors[k];
            if (neighbor >= 0 && neighbor < seq_len) {
                cumsum += weights[k];
                if (cumsum >= target) {
                    selected = neighbor;
                }
            }
        }

        if (selected >= 0) {
            current_pos = selected;
        } else {
            break;
        }
    }

    // Store updated PRNG state
    rng_states[global_walker_id * 2] = rng.state;
    rng_states[global_walker_id * 2 + 1] = rng.inc;
}

// =============================================================================
// Host Launcher Functions
// =============================================================================

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
) {
    int total_walkers = num_sources * walkers_per_source;
    if (total_walkers <= 0 || seq_len <= 0) {
        return;
    }

    dim3 block(WALKER_BLOCK_SIZE);
    dim3 grid((total_walkers + block.x - 1) / block.x);

    multi_source_absorbing_walker_kernel<<<grid, block, 0, stream>>>(
        adj_list, adj_weights, visit_counts, rng_states,
        source_nodes, num_sources, walkers_per_source, seq_len, top_k
    );
}

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
) {
    if (num_walkers <= 0 || seq_len <= 0) {
        return;
    }

    // Clamp source_node to valid range (should be last token / query position)
    if (source_node < 0 || source_node >= seq_len) {
        source_node = seq_len - 1;
    }

    dim3 block(WALKER_BLOCK_SIZE);
    dim3 grid((num_walkers + block.x - 1) / block.x);

    absorbing_walker_kernel<<<grid, block, 0, stream>>>(
        adj_list, adj_weights, visit_counts, rng_states,
        source_node, seq_len, top_k, num_walkers
    );
}

void launch_init_rng_kernel(
    uint64_t* rng_states,
    uint64_t seed,
    int num_walkers,
    cudaStream_t stream
) {
    if (num_walkers <= 0) return;

    dim3 block(WALKER_BLOCK_SIZE);
    dim3 grid((num_walkers + block.x - 1) / block.x);

    init_rng_kernel<<<grid, block, 0, stream>>>(
        rng_states, seed, num_walkers
    );
}

void launch_reset_counts_kernel(
    int32_t* visit_counts,
    int max_seq_len,
    cudaStream_t stream
) {
    if (max_seq_len <= 0) return;

    dim3 block(WALKER_BLOCK_SIZE);
    dim3 grid((max_seq_len + block.x - 1) / block.x);

    reset_counts_kernel<<<grid, block, 0, stream>>>(
        visit_counts, max_seq_len
    );
}

void launch_reset_graph_kernel(
    int32_t* adj_list,
    int max_seq_len,
    int top_k,
    cudaStream_t stream
) {
    int total = max_seq_len * top_k;
    if (total <= 0) return;

    dim3 block(WALKER_BLOCK_SIZE);
    dim3 grid((total + block.x - 1) / block.x);

    reset_graph_kernel<<<grid, block, 0, stream>>>(
        adj_list, max_seq_len, top_k
    );
}

// =============================================================================
// Bidirectional Walker Kernel (RC+B)
// =============================================================================

/**
 * Bidirectional Absorbing Walker Kernel - Run backward AND forward walks.
 *
 * This kernel runs walks in both directions:
 * - Backward (Query→Sink): Uses adj_list (forward graph)
 * - Forward (Sink→Query): Uses rev_adj_list (transpose graph)
 *
 * Results are stored in separate buffers for later combination.
 *
 * Bridge score = min(backward, forward) - tokens visited by BOTH directions
 */
__global__ void bidirectional_walker_kernel(
    const int32_t* __restrict__ adj_list,        // Forward graph [seq_len, top_k]
    const float* __restrict__ adj_weights,       // Forward weights
    const int32_t* __restrict__ rev_adj_list,    // Transpose graph [seq_len, top_k]
    const float* __restrict__ rev_adj_weights,   // Transpose weights
    int32_t* __restrict__ backward_visits,       // Output: backward walk visits
    int32_t* __restrict__ forward_visits,        // Output: forward walk visits
    uint64_t* __restrict__ rng_states,
    const int* __restrict__ query_nodes,         // Query positions for backward walk
    const int* __restrict__ sink_nodes,          // Sink positions for forward walk
    int num_query_nodes,
    int num_sink_nodes,
    int walkers_per_direction,                   // Walkers per direction total
    int seq_len,
    int top_k,
    int query_region_start                       // For forward walk absorption
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_walkers = walkers_per_direction * 2;  // Half backward, half forward
    if (global_id >= total_walkers) return;

    bool is_backward = (global_id < walkers_per_direction);
    int walker_in_dir = global_id % walkers_per_direction;

    // Load PRNG state
    PCGState rng;
    rng.state = rng_states[global_id * 2];
    rng.inc = rng_states[global_id * 2 + 1];

    int current_pos;
    const int32_t* graph;
    const float* weights;
    int32_t* visits;

    if (is_backward) {
        // Backward walk: start at query, walk toward sink
        int source_idx = walker_in_dir % num_query_nodes;
        current_pos = query_nodes[source_idx];
        graph = adj_list;
        weights = adj_weights;
        visits = backward_visits;
    } else {
        // Forward walk: start at sink, walk toward query
        int source_idx = walker_in_dir % num_sink_nodes;
        current_pos = sink_nodes[source_idx];
        graph = rev_adj_list;
        weights = rev_adj_weights;
        visits = forward_visits;
    }

    // Bounds check
    if (current_pos < 0 || current_pos >= seq_len) return;

    // Walk loop
    for (int step = 0; step < MAX_STEPS; ++step) {
        // Record visit
        atomicAdd(&visits[current_pos], 1);

        // Absorption check
        if (is_backward) {
            // Backward: absorb at sink (early tokens)
            if (current_pos < SINK_SIZE) break;
        } else {
            // Forward: absorb at query region (late tokens)
            if (current_pos >= query_region_start) break;
        }

        // Get neighbors
        const int32_t* neighbors = graph + current_pos * top_k;
        const float* edge_weights = weights + current_pos * top_k;

        // Compute total weight
        float total_weight = 0.0f;
        int num_valid = 0;
        for (int k = 0; k < top_k; ++k) {
            int neighbor = neighbors[k];
            if (neighbor >= 0 && neighbor < seq_len) {
                total_weight += edge_weights[k];
                num_valid++;
            }
        }

        if (num_valid == 0 || total_weight <= 0.0f) break;

        // Sample transition
        float target = pcg_uniform(&rng) * total_weight;
        float cumsum = 0.0f;
        int selected = -1;

        for (int k = 0; k < top_k && selected < 0; ++k) {
            int neighbor = neighbors[k];
            if (neighbor >= 0 && neighbor < seq_len) {
                cumsum += edge_weights[k];
                if (cumsum >= target) {
                    selected = neighbor;
                }
            }
        }

        if (selected >= 0) {
            current_pos = selected;
        } else {
            break;
        }
    }

    // Store PRNG state
    rng_states[global_id * 2] = rng.state;
    rng_states[global_id * 2 + 1] = rng.inc;
}

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
) {
    int total_walkers = walkers_per_direction * 2;
    if (total_walkers <= 0 || seq_len <= 0) return;

    dim3 block(WALKER_BLOCK_SIZE);
    dim3 grid((total_walkers + block.x - 1) / block.x);

    bidirectional_walker_kernel<<<grid, block, 0, stream>>>(
        adj_list, adj_weights, rev_adj_list, rev_adj_weights,
        backward_visits, forward_visits, rng_states,
        query_nodes, sink_nodes, num_query_nodes, num_sink_nodes,
        walkers_per_direction, seq_len, top_k, query_region_start
    );
}

}  // namespace circuit_kv
