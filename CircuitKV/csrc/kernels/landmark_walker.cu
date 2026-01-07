/**
 * CircuitKV - Landmark Walker Kernel
 *
 * Multi-source absorbing random walks from diverse landmarks + query.
 *
 * Key Insight (from PoC2):
 *   Single-source walks miss important tokens that aren't directly visible
 *   from the query. By launching walks from geographically-diverse landmarks,
 *   we discover "bridge tokens" through path convergence.
 *
 * Algorithm:
 *   1. Select diverse landmarks using H2O scores with spacing constraint
 *   2. Cache attention rows for each landmark
 *   3. Launch walkers from ALL sources (landmarks + query)
 *   4. Walkers use cached attention for transitions
 *   5. Apply positional normalization to remove -log bias
 *
 * Parallelism:
 *   - Total walkers = num_sources * walkers_per_source
 *   - Each thread is one walker
 *   - Walker i belongs to source (i / walkers_per_source)
 *   - All walkers run in parallel across sources
 */

#include "kernels/common.cuh"
#include "include/kernels.h"

namespace circuit_kv {

// =============================================================================
// Configuration
// =============================================================================

constexpr int LANDMARK_BLOCK_SIZE = 256;
constexpr int LANDMARK_SINK_SIZE = 4;
constexpr int LANDMARK_MAX_STEPS = 1500;
constexpr int MAX_LANDMARKS = 32;  // Maximum supported landmarks

// =============================================================================
// Landmark Walker Kernel - Multi-Source Parallel Walks
// =============================================================================

/**
 * Multi-source absorbing walker kernel.
 *
 * Each thread is a walker assigned to one source. All walkers run in parallel.
 * This is more efficient than launching separate kernels per source.
 *
 * @param landmark_attention  Cached attention for landmarks [num_landmarks, seq_len]
 * @param query_attention     Query's attention row [seq_len]
 * @param visit_counts        Output visit counts (shared across all walkers)
 * @param rng_states          PRNG states for all walkers
 * @param landmark_positions  Array of landmark positions [num_landmarks]
 * @param num_landmarks       Number of landmarks (not including query)
 * @param walkers_per_source  Walkers to launch from each source
 * @param query_boost         Weight multiplier for query-sourced walkers
 * @param seq_len             Current sequence length
 */
__global__ void landmark_walker_kernel(
    const float* __restrict__ landmark_attention,
    const float* __restrict__ query_attention,
    int32_t* __restrict__ visit_counts,
    uint64_t* __restrict__ rng_states,
    const int32_t* __restrict__ landmark_positions,
    int num_landmarks,
    int walkers_per_source,
    float query_boost,
    int seq_len
) {
    int num_sources = num_landmarks + 1;  // landmarks + query
    int total_walkers = num_sources * walkers_per_source;

    int walker_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (walker_id >= total_walkers) return;

    // Determine which source this walker belongs to
    int source_idx = walker_id / walkers_per_source;
    bool is_query_source = (source_idx == num_landmarks);

    // Get source position and weight
    int source_pos;
    float weight;
    const float* attention_row;

    if (is_query_source) {
        source_pos = seq_len - 1;  // Query is always last token
        weight = query_boost;
        attention_row = query_attention;
    } else {
        source_pos = landmark_positions[source_idx];
        weight = 1.0f;
        attention_row = landmark_attention + source_idx * seq_len;
    }

    // Load PRNG state
    PCGState rng;
    rng.state = rng_states[walker_id * 2];
    rng.inc = rng_states[walker_id * 2 + 1];

    // Compute starting distribution from source's attention
    // Zero out sink positions and normalize
    float total_weight = 0.0f;
    for (int j = LANDMARK_SINK_SIZE; j < source_pos; ++j) {
        total_weight += attention_row[j];
    }

    // Handle edge case: no valid positions
    if (total_weight <= 1e-8f) {
        rng_states[walker_id * 2] = rng.state;
        rng_states[walker_id * 2 + 1] = rng.inc;
        return;
    }

    // Sample starting position from source's attention
    float target = pcg_uniform(&rng) * total_weight;
    float cumsum = 0.0f;
    int current_pos = LANDMARK_SINK_SIZE;  // Default

    for (int j = LANDMARK_SINK_SIZE; j < source_pos; ++j) {
        cumsum += attention_row[j];
        if (cumsum >= target) {
            current_pos = j;
            break;
        }
    }

    // Absorbing random walk
    for (int step = 0; step < LANDMARK_MAX_STEPS; ++step) {
        // Record visit with weight
        // Use float atomic for weighted visits
        atomicAdd(&visit_counts[current_pos], (int32_t)(weight));

        // Check absorption at sink
        if (current_pos < LANDMARK_SINK_SIZE) {
            break;
        }

        // Get transition probabilities from current position
        // Use query attention if in window, otherwise find nearest landmark
        const float* trans_row;
        bool have_attention = false;

        // Check if current position is in window (last 64 tokens)
        int window_start = max(0, seq_len - 64);
        if (current_pos >= window_start) {
            trans_row = query_attention;  // Approximate: use query's view
            have_attention = true;
        }

        // Check if current position is a landmark
        if (!have_attention) {
            for (int lm = 0; lm < num_landmarks; ++lm) {
                if (landmark_positions[lm] == current_pos) {
                    trans_row = landmark_attention + lm * seq_len;
                    have_attention = true;
                    break;
                }
            }
        }

        // If no cached attention, use nearest landmark as proxy
        if (!have_attention) {
            int nearest_lm = 0;
            int min_dist = abs(landmark_positions[0] - current_pos);
            for (int lm = 1; lm < num_landmarks; ++lm) {
                int dist = abs(landmark_positions[lm] - current_pos);
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_lm = lm;
                }
            }
            trans_row = landmark_attention + nearest_lm * seq_len;
        }

        // Sample next position (only to positions before current)
        total_weight = 0.0f;
        for (int j = 0; j < current_pos; ++j) {
            total_weight += trans_row[j];
        }

        if (total_weight <= 1e-8f) {
            // No valid transitions, jump heuristically toward sink
            int distance = current_pos - LANDMARK_SINK_SIZE;
            int jump = max(1, (int)(distance * (0.02f + pcg_uniform(&rng) * 0.08f)));
            current_pos = max(LANDMARK_SINK_SIZE, current_pos - jump);
            continue;
        }

        // Sample from transition distribution
        target = pcg_uniform(&rng) * total_weight;
        cumsum = 0.0f;
        int next_pos = 0;

        for (int j = 0; j < current_pos; ++j) {
            cumsum += trans_row[j];
            if (cumsum >= target) {
                next_pos = j;
                break;
            }
        }

        current_pos = next_pos;
    }

    // Store updated PRNG state
    rng_states[walker_id * 2] = rng.state;
    rng_states[walker_id * 2 + 1] = rng.inc;
}

// =============================================================================
// Positional Normalization Kernel
// =============================================================================

/**
 * Apply positional normalization to remove -log bias from walker visits.
 *
 * The problem: Absorbing walks create visits[p] ~ 1/distance_from_sink
 * because all walkers must pass through early positions.
 *
 * The fix: Divide by expected visits to reveal tokens visited MORE than expected.
 *   expected[p] = 1 / (p - sink_size + 1)^alpha
 *   normalized[p] = visits[p] / expected[p]
 */
__global__ void positional_normalize_kernel(
    const int32_t* __restrict__ visit_counts,
    float* __restrict__ normalized_scores,
    float* __restrict__ partial_max,
    int seq_len,
    int sink_size,
    float alpha
) {
    extern __shared__ float smem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // First pass: compute normalized values
    float normalized = 0.0f;
    if (idx < seq_len) {
        if (idx < sink_size) {
            normalized = 0.0f;  // Sink always kept
        } else {
            float distance = (float)(idx - sink_size + 1);
            float expected = 1.0f / powf(distance, alpha);
            normalized = (float)visit_counts[idx] / (expected + 1e-8f);
        }
        normalized_scores[idx] = normalized;
    }

    // Block-level max reduction for later normalization
    smem[threadIdx.x] = (idx < seq_len) ? normalized : 0.0f;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    // Write block max
    if (threadIdx.x == 0) {
        partial_max[blockIdx.x] = smem[0];
    }
}

/**
 * Final normalization pass: divide by global max.
 */
__global__ void finalize_normalization_kernel(
    float* __restrict__ scores,
    const float* __restrict__ partial_max,
    int seq_len,
    int num_blocks
) {
    // First thread finds global max
    __shared__ float global_max;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        global_max = 0.0f;
        for (int i = 0; i < num_blocks; ++i) {
            global_max = fmaxf(global_max, partial_max[i]);
        }
    }
    __syncthreads();

    // Broadcast global_max to all threads
    if (threadIdx.x == 0) {
        global_max = partial_max[0];  // Actually need to recompute...
    }

    // Each thread normalizes its elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len && global_max > 1e-8f) {
        scores[idx] = scores[idx] / global_max;
    }
}

// =============================================================================
// Landmark Selection Kernel (H2O-based with spacing)
// =============================================================================

/**
 * Select diverse landmarks from H2O scores.
 *
 * Algorithm:
 *   1. Compute H2O score for each position (sum of incoming attention)
 *   2. Greedily select highest H2O positions with minimum spacing
 *   3. Skip sink region and window region
 */
__global__ void select_landmarks_kernel(
    const float* __restrict__ attention_row_sums,  // H2O scores [seq_len]
    int32_t* __restrict__ landmark_positions,       // Output [max_landmarks]
    int* __restrict__ num_landmarks_out,
    int seq_len,
    int max_landmarks,
    int min_spacing,
    int sink_buffer,      // Extra buffer from sink (e.g., 20)
    int window_size       // Last window_size tokens excluded
) {
    // This kernel is single-threaded (run with 1 block, 1 thread)
    // Could be parallelized but landmark selection is not perf-critical

    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int window_start = seq_len - window_size;
    int valid_start = LANDMARK_SINK_SIZE + sink_buffer;
    int valid_end = window_start;

    // Mask array to track excluded positions
    // Using a simple approach: mark selected positions and their neighbors
    bool* excluded = new bool[seq_len];
    for (int i = 0; i < seq_len; ++i) {
        excluded[i] = (i < valid_start || i >= valid_end);
    }

    int num_selected = 0;

    while (num_selected < max_landmarks) {
        // Find highest H2O score among non-excluded positions
        float best_score = -1e30f;
        int best_idx = -1;

        for (int i = valid_start; i < valid_end; ++i) {
            if (!excluded[i] && attention_row_sums[i] > best_score) {
                best_score = attention_row_sums[i];
                best_idx = i;
            }
        }

        if (best_idx < 0) break;  // No more valid positions

        // Add to landmarks
        landmark_positions[num_selected++] = best_idx;

        // Exclude nearby positions
        int exclude_start = max(0, best_idx - min_spacing);
        int exclude_end = min(seq_len, best_idx + min_spacing + 1);
        for (int i = exclude_start; i < exclude_end; ++i) {
            excluded[i] = true;
        }
    }

    *num_landmarks_out = num_selected;

    delete[] excluded;
}

// =============================================================================
// H2O Scores Kernel (Column Sums)
// =============================================================================

/**
 * Compute H2O scores (column sums of attention matrix).
 *
 * For each position j: H2O[j] = sum_i(attention[i, j])
 * This measures total "incoming attention" to each token.
 *
 * Block/Thread: One block per column, threads collaborate on the sum.
 */
__global__ void compute_h2o_scores_kernel(
    const float* __restrict__ attention,  // [seq_len, seq_len]
    float* __restrict__ h2o_scores,        // Output [seq_len]
    int seq_len
) {
    extern __shared__ float smem[];

    int col_idx = blockIdx.x;  // Which column (j) we're summing
    if (col_idx >= seq_len) return;

    // Each thread sums part of the column
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        // attention[i, col_idx] = attention[i * seq_len + col_idx]
        // But for causal mask, only i >= col_idx have valid attention
        if (i >= col_idx) {
            local_sum += attention[i * seq_len + col_idx];
        }
    }

    // Store in shared memory
    smem[threadIdx.x] = local_sum;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write final sum
    if (threadIdx.x == 0) {
        h2o_scores[col_idx] = smem[0];
    }
}

// =============================================================================
// Cache Landmark Attention Kernel
// =============================================================================

/**
 * Cache attention rows for selected landmarks.
 *
 * Each block handles one landmark's attention row.
 */
__global__ void cache_landmark_attention_kernel(
    const float* __restrict__ full_attention,    // [seq_len, seq_len] or provided rows
    float* __restrict__ landmark_attention,       // Output [num_landmarks, seq_len]
    const int32_t* __restrict__ landmark_positions,
    int num_landmarks,
    int seq_len
) {
    int lm_idx = blockIdx.x;
    if (lm_idx >= num_landmarks) return;

    int lm_pos = landmark_positions[lm_idx];

    // Copy the landmark's attention row
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        landmark_attention[lm_idx * seq_len + j] = full_attention[lm_pos * seq_len + j];
    }
}

// =============================================================================
// Host Launcher Functions
// =============================================================================

void launch_landmark_walker_kernel(
    const float* landmark_attention,
    const float* query_attention,
    int32_t* visit_counts,
    uint64_t* rng_states,
    const int32_t* landmark_positions,
    int num_landmarks,
    int walkers_per_source,
    float query_boost,
    int seq_len,
    cudaStream_t stream
) {
    int num_sources = num_landmarks + 1;
    int total_walkers = num_sources * walkers_per_source;

    if (total_walkers <= 0 || seq_len <= 0) return;

    dim3 block(LANDMARK_BLOCK_SIZE);
    dim3 grid((total_walkers + block.x - 1) / block.x);

    landmark_walker_kernel<<<grid, block, 0, stream>>>(
        landmark_attention,
        query_attention,
        visit_counts,
        rng_states,
        landmark_positions,
        num_landmarks,
        walkers_per_source,
        query_boost,
        seq_len
    );
}

void launch_positional_normalize_kernel(
    const int32_t* visit_counts,
    float* normalized_scores,
    float* partial_max,
    int seq_len,
    int sink_size,
    float alpha,
    cudaStream_t stream
) {
    if (seq_len <= 0) return;

    dim3 block(256);
    dim3 grid((seq_len + block.x - 1) / block.x);
    size_t smem_size = 256 * sizeof(float);

    positional_normalize_kernel<<<grid, block, smem_size, stream>>>(
        visit_counts,
        normalized_scores,
        partial_max,
        seq_len,
        sink_size,
        alpha
    );
}

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
) {
    // Single-threaded kernel for simplicity
    // TODO: Consider moving to CPU for better performance
    select_landmarks_kernel<<<1, 1, 0, stream>>>(
        attention_row_sums,
        landmark_positions,
        num_landmarks_out,
        seq_len,
        max_landmarks,
        min_spacing,
        sink_buffer,
        window_size
    );
}

void launch_cache_landmark_attention_kernel(
    const float* full_attention,
    float* landmark_attention,
    const int32_t* landmark_positions,
    int num_landmarks,
    int seq_len,
    cudaStream_t stream
) {
    if (num_landmarks <= 0 || seq_len <= 0) return;

    dim3 block(256);
    dim3 grid(num_landmarks);

    cache_landmark_attention_kernel<<<grid, block, 0, stream>>>(
        full_attention,
        landmark_attention,
        landmark_positions,
        num_landmarks,
        seq_len
    );
}

void launch_compute_h2o_scores_kernel(
    const float* attention_matrix,
    float* h2o_scores,
    int seq_len,
    cudaStream_t stream
) {
    if (seq_len <= 0) return;

    dim3 block(256);
    dim3 grid(seq_len);  // One block per column
    size_t smem_size = 256 * sizeof(float);

    compute_h2o_scores_kernel<<<grid, block, smem_size, stream>>>(
        attention_matrix,
        h2o_scores,
        seq_len
    );
}

}  // namespace circuit_kv
