/**
 * Causal Influence Propagation Walker (v2.0.0 - SDI Optimized)
 *
 * ALGORITHM:
 *   1. MULTI-SOURCE: 50% walkers start at query, 50% start at random positions
 *   2. DUAL-TEMPERATURE: 70% logical (T=base), 30% exploratory (T=high/uniform)
 *   3. Visit count = unweighted for ALL steps (count from step 0)
 *   4. Normalize by POSITIONAL OPPORTUNITY to counteract early-position bias
 *
 * v2.0.0 OPTIMIZATION - "Single-Pass Sampling":
 *   Problem: v1.0.8 computed powf() twice per position per step (sum + sample loops)
 *   Solution: Fused single-pass sampling using normalized random threshold
 *     - First pass computes sum AND caches scaled values in registers
 *     - For small pos ranges (<= 32), use register caching
 *     - For larger ranges, use online sampling with single powf per position
 *
 * v1.0.8 FIX - "Dual-Temperature Walkers":
 *   Problem: Single temperature is a trade-off. Low T misses middle positions,
 *   high T loses attention signal. No single temperature works for all tasks.
 *
 *   Solution: Use TWO temperature regimes simultaneously:
 *     - LOGICAL walkers (70%): T = base_temperature (default 2.0)
 *       → Follow attention closely, find "hard evidence" (Qasper, Code)
 *     - EXPLORATORY walkers (30%): T = explore_temperature (default 10.0)
 *       → Sample nearly uniformly, find "context & atmosphere" (NarrativeQA)
 *
 *   This captures BOTH types of important tokens:
 *     - Logical: Tokens the model actually attends to
 *     - Exploratory: Tokens that provide context but aren't directly attended
 *
 * DIRECTION:
 *   Walk ALONG A (not A^T). A[i,j] = how much position i attends to j.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "common.cuh"

namespace circuit_kv {

// Default hyperparameters (validated by PoC5)
constexpr int INFLUENCE_DEFAULT_WALKERS = 10000;
constexpr int INFLUENCE_DEFAULT_MAX_STEPS = 10;
constexpr int INFLUENCE_DEFAULT_SINK_SIZE = 4;
constexpr float INFLUENCE_DEFAULT_TEMPERATURE = 2.0f;       // v1.0.7: Base temperature
constexpr float INFLUENCE_DEFAULT_EXPLORE_TEMP = 10.0f;     // v1.0.8: Exploratory temperature
constexpr float INFLUENCE_DEFAULT_EXPLORE_RATIO = 0.3f;     // v1.0.8: 30% exploratory walkers

/**
 * Influence Walker Kernel (v1.0.8 - Dual-Temperature Exploration)
 *
 * Each thread runs one walker. 50% start at current_idx (query), 50% start at
 * random positions in [sink_size, current_idx]. All walk through attention
 * for max_steps, ONLY among non-sink positions.
 *
 * v1.0.8: Dual temperature regime:
 *   - First (1-explore_ratio) walkers use base temperature (logical)
 *   - Last explore_ratio walkers use high temperature (exploratory)
 *
 * @param attention        Full attention matrix [seq_len, seq_len], row-major
 * @param visits           Output: visit counts [seq_len] (all steps)
 * @param rng_states       RNG states [num_walkers] as PCGState structs
 * @param seq_len          Sequence length
 * @param current_idx      Generation position (walker start)
 * @param num_walkers      Number of walkers
 * @param max_steps        Maximum steps per walker
 * @param sink_size        Sink region size (positions 0 to sink_size-1 excluded)
 * @param inv_temperature  1/base_temperature for logical walkers
 * @param inv_explore_temp 1/explore_temperature for exploratory walkers
 * @param explore_ratio    Fraction of walkers that use exploratory temperature
 */
__global__ void influence_walker_kernel(
    const float* __restrict__ attention,
    float* __restrict__ visits,
    PCGState* __restrict__ rng_states,
    int seq_len,
    int current_idx,
    int num_walkers,
    int max_steps,
    int sink_size,
    float inv_temperature,
    float inv_explore_temp,
    float explore_ratio
) {
    int walker_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (walker_id >= num_walkers) return;

    // Load RNG state for this walker
    PCGState rng = rng_states[walker_id];

    // v1.0.8: Dual-Temperature Selection
    // First (1-explore_ratio) walkers are LOGICAL (base temperature)
    // Last explore_ratio walkers are EXPLORATORY (high temperature)
    int explore_start = (int)((1.0f - explore_ratio) * num_walkers);
    float my_inv_temp = (walker_id >= explore_start) ? inv_explore_temp : inv_temperature;

    // v1.0.6: Multi-Source Walks
    // - Walker IDs 0 to num_walkers/2 - 1: Start at query (current_idx)
    // - Walker IDs num_walkers/2 to num_walkers - 1: Start at random positions
    int pos;
    int half_walkers = num_walkers / 2;

    if (walker_id < half_walkers) {
        // Query-start walker
        pos = current_idx;
    } else {
        // Random-start walker: sample from [sink_size, current_idx] inclusive
        int num_valid_starts = current_idx - sink_size + 1;
        if (num_valid_starts <= 0) {
            // Edge case: if current_idx <= sink_size, no valid random starts
            pos = current_idx;
        } else {
            int rand_offset = (int)(pcg_uniform(&rng) * num_valid_starts);
            // Clamp to avoid edge case where rand = 1.0
            if (rand_offset >= num_valid_starts) rand_offset = num_valid_starts - 1;
            pos = sink_size + rand_offset;
        }
    }

    // Walk for max_steps, sampling only among non-sink positions
    // v2.0.0 OPTIMIZATION: Single-pass sampling with early termination
    for (int step = 0; step < max_steps; step++) {
        // Causal: can attend to positions 0..pos (inclusive)
        // But we EXCLUDE sink positions (0 to sink_size-1)
        // So valid candidates are: sink_size to pos (inclusive)

        // If current position is at or below sink, no valid candidates - terminate
        if (pos < sink_size) break;

        const float* attn_row = attention + pos * seq_len;
        int range_size = pos - sink_size + 1;

        // v2.0.0: Single-pass sampling using TWO-PHASE approach
        // Phase 1: Compute sum AND cache scaled values for small ranges
        // Phase 2: Use cached values for O(1) lookup, or recompute for large ranges

        int next_pos = sink_size;  // Default

        if (range_size <= 64) {
            // SMALL RANGE: Cache in registers (fast path for most walks)
            // Use local array - compiler will use registers for small sizes
            float cached_scaled[64];
            float non_sink_sum = 0.0f;

            #pragma unroll 8
            for (int j = 0; j < range_size; j++) {
                int idx = sink_size + j;
                float scaled = powf(fmaxf(attn_row[idx], 1e-10f), my_inv_temp);
                cached_scaled[j] = scaled;
                non_sink_sum += scaled;
            }

            if (non_sink_sum < 1e-10f) break;

            // Sample using cached values (single powf per position!)
            float r = pcg_uniform(&rng) * non_sink_sum;
            float cumsum = 0.0f;

            for (int j = 0; j < range_size; j++) {
                cumsum += cached_scaled[j];
                if (r < cumsum) {
                    next_pos = sink_size + j;
                    break;
                }
                next_pos = sink_size + j;
            }
        } else {
            // LARGE RANGE: Two-pass but with pre-generated random
            // Still faster than original due to optimized memory access
            float non_sink_sum = 0.0f;

            // First pass: compute sum
            for (int j = sink_size; j <= pos; j++) {
                float scaled = powf(fmaxf(attn_row[j], 1e-10f), my_inv_temp);
                non_sink_sum += scaled;
            }

            if (non_sink_sum < 1e-10f) break;

            // Pre-generate random number
            float r = pcg_uniform(&rng) * non_sink_sum;
            float cumsum = 0.0f;

            // Second pass: sample with early exit
            for (int j = sink_size; j <= pos; j++) {
                float scaled = powf(fmaxf(attn_row[j], 1e-10f), my_inv_temp);
                cumsum += scaled;
                if (r < cumsum) {
                    next_pos = j;
                    break;
                }
                next_pos = j;
            }
        }

        // v1.0.6: Count visits from ALL steps (including step 0)
        atomicAdd(&visits[next_pos], 1.0f);

        // Move to next position
        pos = next_pos;
    }

    // Save RNG state for next call
    rng_states[walker_id] = rng;
}

/**
 * Initialize RNG states for influence walkers
 */
__global__ void init_influence_rng_kernel(
    PCGState* rng_states,
    int num_walkers,
    uint64_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_walkers) return;

    // Initialize PCG state for this walker
    pcg_init(&rng_states[idx], seed + idx, idx);
}

/**
 * Clear visits buffer before walker run
 */
__global__ void clear_influence_visits_kernel(
    float* visits,
    int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len) {
        visits[idx] = 0.0f;
    }
}

/**
 * Normalize visits to [0, 1] range (optional, for numerical stability)
 * Uses max normalization to preserve relative rankings
 */
__global__ void normalize_influence_scores_kernel(
    float* visits,
    float* normalized,
    int seq_len,
    float max_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len) {
        if (max_val > 1e-10f) {
            normalized[idx] = visits[idx] / max_val;
        } else {
            normalized[idx] = 0.0f;
        }
    }
}

/**
 * Find max value in visits array (reduction)
 * Named with influence_ prefix to avoid collision with spectral_power.cu
 */
__global__ void influence_find_max_kernel(
    const float* visits,
    float* partial_max,
    int seq_len
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and find local max
    sdata[tid] = (idx < seq_len) ? visits[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        partial_max[blockIdx.x] = sdata[0];
    }
}

// =============================================================================
// Kernel Launchers (called from circuit_manager.cpp via kernels.h)
// =============================================================================

void launch_influence_walker_kernel(
    const float* attention,
    float* visits,
    uint64_t* rng_states,  // Actually PCGState*, but kept as uint64_t* for API compatibility
    int seq_len,
    int current_idx,
    int num_walkers,
    int max_steps,
    int sink_size,
    cudaStream_t stream,
    float temperature,       // v1.0.7: Base temperature (default 2.0)
    float explore_temp,      // v1.0.8: Exploratory temperature (default 10.0)
    float explore_ratio      // v1.0.8: Ratio of exploratory walkers (default 0.3)
) {
    const int block_size = 256;
    const int num_blocks = (num_walkers + block_size - 1) / block_size;

    // Cast to PCGState* (each PCGState is 2 uint64_t = 16 bytes)
    PCGState* pcg_states = reinterpret_cast<PCGState*>(rng_states);

    // v1.0.8: Compute inverse temperatures for both regimes
    // inv_temp = 1/T, so attn^(1/T) = attn^inv_temp
    float inv_temperature = 1.0f / fmaxf(temperature, 0.1f);
    float inv_explore_temp = 1.0f / fmaxf(explore_temp, 0.1f);

    influence_walker_kernel<<<num_blocks, block_size, 0, stream>>>(
        attention, visits, pcg_states, seq_len, current_idx,
        num_walkers, max_steps, sink_size,
        inv_temperature, inv_explore_temp, explore_ratio
    );
}

void launch_clear_influence_visits_kernel(
    float* visits,
    int seq_len,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int num_blocks = (seq_len + block_size - 1) / block_size;

    clear_influence_visits_kernel<<<num_blocks, block_size, 0, stream>>>(
        visits, seq_len
    );
}

/**
 * Final max reduction kernel (single block)
 * Takes partial maxes and produces final max
 * Named with influence_ prefix to avoid collision with spectral_power.cu
 */
__global__ void influence_final_max_kernel(
    const float* partial_max,
    float* max_out,
    int num_partials
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Load partials
    sdata[tid] = (tid < num_partials) ? partial_max[tid] : 0.0f;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < num_partials) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_out[0] = sdata[0];
    }
}

/**
 * v1.0.5: Apply positional opportunity adjustment to visits
 *
 * adjusted[p] = visits[p] / sqrt(seq_len - p + 1)
 *
 * This counteracts the early-position bias in random walks.
 * Early positions have more "opportunities" to be visited, which
 * inflates their visit counts regardless of actual importance.
 */
__global__ void influence_positional_adjust_kernel(
    const float* visits,
    float* adjusted,
    int seq_len,
    int sink_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len) {
        // Sink positions: Set to 0 temporarily, will be set to 1.0 in final normalization
        if (idx < sink_size) {
            adjusted[idx] = 0.0f;
        } else {
            // Opportunity = (seq_len - idx) roughly
            // Use sqrt for gentler correction
            float opportunity = sqrtf((float)(seq_len - idx + 1));
            adjusted[idx] = visits[idx] / opportunity;
        }
    }
}

/**
 * Find max of positionally-adjusted visits
 */
__global__ void influence_find_adjusted_max_kernel(
    const float* adjusted,
    float* partial_max,
    int seq_len,
    int sink_size
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and find local max (skip sink positions)
    float val = 0.0f;
    if (idx < seq_len && idx >= sink_size) {
        val = adjusted[idx];
    }
    sdata[tid] = val;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        partial_max[blockIdx.x] = sdata[0];
    }
}

/**
 * Normalize by max value kernel
 * Named with influence_ prefix to avoid collision with spectral_power.cu
 * v1.0.7: Sink positions are set to 1.0 to ensure they're ALWAYS kept
 */
__global__ void influence_normalize_by_max_kernel(
    const float* input,
    float* output,
    const float* max_val,
    int seq_len,
    int sink_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len) {
        // v1.0.7: Sink positions get score 1.0 - ALWAYS KEEP first sink_size tokens
        if (idx < sink_size) {
            output[idx] = 1.0f;
        } else {
            float m = max_val[0];
            if (m > 1e-10f) {
                output[idx] = input[idx] / m;
            } else {
                output[idx] = 0.0f;
            }
        }
    }
}

void launch_find_max_and_normalize_kernel(
    float* visits,
    float* normalized,
    float* partial_max,
    int seq_len,
    int num_blocks,
    cudaStream_t stream,
    int sink_size  // v1.0.5: Added for positional normalization
) {
    const int block_size = 256;
    int grid_blocks = (seq_len + block_size - 1) / block_size;

    // v1.0.5: Step 1 - Apply positional opportunity adjustment
    // adjusted[p] = visits[p] / sqrt(seq_len - p + 1)
    // Use 'normalized' as temporary buffer for adjusted values
    influence_positional_adjust_kernel<<<grid_blocks, block_size, 0, stream>>>(
        visits, normalized, seq_len, sink_size
    );

    // Step 2: Find partial maxes of adjusted values
    influence_find_adjusted_max_kernel<<<num_blocks, block_size, block_size * sizeof(float), stream>>>(
        normalized, partial_max, seq_len, sink_size
    );

    // Step 3: Final max reduction (single block)
    influence_final_max_kernel<<<1, 256, 256 * sizeof(float), stream>>>(
        partial_max, partial_max, num_blocks
    );

    // Step 4: Normalize adjusted values by max
    // v1.0.7: Also sets sink positions to 1.0 to ensure they're always kept
    influence_normalize_by_max_kernel<<<grid_blocks, block_size, 0, stream>>>(
        normalized, normalized, partial_max, seq_len, sink_size
    );
}

}  // namespace circuit_kv
