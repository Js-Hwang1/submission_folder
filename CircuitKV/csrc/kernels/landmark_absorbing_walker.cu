/**
 * CircuitKV - Landmark Absorbing Walker Kernel
 *
 * NEW APPROACH: Landmarks are both SOURCES and SINKS.
 *
 * Key Insight:
 *   Instead of all walkers flowing to tokens 0-3, walkers absorb at
 *   ANY landmark (or sink). This creates a "mesh" of current flows
 *   between landmarks, potentially capturing local bridge tokens better.
 *
 * Physics Analogy:
 *   - Old: Single battery (Query+ to Sink-)
 *   - New: Multiple batteries in a mesh network. Each landmark is a node
 *          that can source or sink current depending on the walker's origin.
 *
 * Algorithm:
 *   1. Walker starts at source S (a landmark or query)
 *   2. Walk backward (toward earlier tokens) using attention weights
 *   3. STOP if: current_pos is in sink OR current_pos is a DIFFERENT landmark
 *   4. Record visits along the path
 *
 * This captures "local bridges" - tokens that connect adjacent landmarks.
 */

#include "kernels/common.cuh"
#include "include/kernels.h"

namespace circuit_kv {

// =============================================================================
// Configuration
// =============================================================================

constexpr int LAW_BLOCK_SIZE = 256;      // Threads per block
constexpr int LAW_SINK_SIZE = 4;         // Tokens 0-3 always absorb
constexpr int LAW_MAX_STEPS = 1500;      // Safety timeout
constexpr int LAW_MAX_LANDMARKS = 64;    // Max landmarks for shared memory

// =============================================================================
// Device Helper: Check if position is an absorbing landmark
// =============================================================================

/**
 * Check if position is a landmark (and not the walker's own source).
 * Uses shared memory for fast lookup.
 */
__device__ __forceinline__ bool is_absorbing_landmark(
    int pos,
    int source_landmark_idx,  // -1 if source is query
    const int32_t* __restrict__ landmark_positions,
    int num_landmarks
) {
    for (int i = 0; i < num_landmarks; ++i) {
        if (landmark_positions[i] == pos && i != source_landmark_idx) {
            return true;  // Hit a different landmark -> absorb
        }
    }
    return false;
}

// =============================================================================
// Main Kernel: Landmark Absorbing Walker
// =============================================================================

/**
 * Multi-source absorbing walker where landmarks are also absorbing states.
 *
 * @param landmark_attention  Cached attention for landmarks [num_landmarks, seq_len]
 * @param query_attention     Query's attention row [seq_len]
 * @param h2o_scores          H2O scores for fallback transitions [seq_len]
 * @param visit_counts        Output visit counts (atomic, shared across walkers)
 * @param rng_states          PRNG states [total_walkers * 2]
 * @param landmark_positions  Positions of landmarks [num_landmarks]
 * @param num_landmarks       Number of landmarks (query is additional source)
 * @param walkers_per_source  Walkers launched from each source
 * @param query_boost         Weight multiplier for query-sourced walker visits
 * @param seq_len             Current sequence length
 * @param absorb_at_landmarks If true, landmarks absorb walkers (new behavior)
 */
__global__ void landmark_absorbing_walker_kernel(
    const float* __restrict__ landmark_attention,
    const float* __restrict__ query_attention,
    const float* __restrict__ h2o_scores,
    int32_t* __restrict__ visit_counts,
    uint64_t* __restrict__ rng_states,
    const int32_t* __restrict__ landmark_positions,
    int num_landmarks,
    int walkers_per_source,
    float query_boost,
    int seq_len,
    bool absorb_at_landmarks
) {
    // Load landmark positions into shared memory for fast lookup
    __shared__ int32_t smem_landmarks[LAW_MAX_LANDMARKS];

    if (threadIdx.x < num_landmarks && threadIdx.x < LAW_MAX_LANDMARKS) {
        smem_landmarks[threadIdx.x] = landmark_positions[threadIdx.x];
    }
    __syncthreads();

    // Compute walker ID and source assignment
    int num_sources = num_landmarks + 1;  // landmarks + query
    int total_walkers = num_sources * walkers_per_source;

    int walker_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (walker_id >= total_walkers) return;

    // Which source does this walker belong to?
    int source_idx = walker_id / walkers_per_source;
    bool is_query_source = (source_idx == num_landmarks);
    int source_landmark_idx = is_query_source ? -1 : source_idx;

    // Get source position and attention row
    int source_pos;
    float visit_weight;
    const float* attention_row;

    if (is_query_source) {
        source_pos = seq_len - 1;
        visit_weight = query_boost;
        attention_row = query_attention;
    } else {
        source_pos = smem_landmarks[source_idx];
        visit_weight = 1.0f;
        attention_row = landmark_attention + source_idx * seq_len;
    }

    // Load PRNG state
    PCGState rng;
    rng.state = rng_states[walker_id * 2];
    rng.inc = rng_states[walker_id * 2 + 1];

    // Sample initial position from source's attention distribution
    // Exclude sink and the source itself
    float total_weight = 0.0f;
    for (int j = LAW_SINK_SIZE; j < source_pos; ++j) {
        total_weight += attention_row[j];
    }

    if (total_weight <= 1e-8f) {
        // No valid starting positions
        rng_states[walker_id * 2] = rng.state;
        rng_states[walker_id * 2 + 1] = rng.inc;
        return;
    }

    // Sample starting position
    float target = pcg_uniform(&rng) * total_weight;
    float cumsum = 0.0f;
    int current_pos = LAW_SINK_SIZE;

    for (int j = LAW_SINK_SIZE; j < source_pos; ++j) {
        cumsum += attention_row[j];
        if (cumsum >= target) {
            current_pos = j;
            break;
        }
    }

    // ==========================================================================
    // Absorbing Random Walk Loop
    // ==========================================================================

    for (int step = 0; step < LAW_MAX_STEPS; ++step) {
        // Record visit at current position
        atomicAdd(&visit_counts[current_pos], (int32_t)visit_weight);

        // ----- ABSORPTION CHECK -----

        // 1. Sink absorption (tokens 0 to SINK_SIZE-1)
        if (current_pos < LAW_SINK_SIZE) {
            break;
        }

        // 2. Landmark absorption (if enabled)
        if (absorb_at_landmarks && num_landmarks > 0) {
            bool absorbed = false;
            for (int lm = 0; lm < num_landmarks && lm < LAW_MAX_LANDMARKS; ++lm) {
                // Absorb if we hit a DIFFERENT landmark than our source
                if (smem_landmarks[lm] == current_pos && lm != source_landmark_idx) {
                    absorbed = true;
                    break;
                }
            }
            if (absorbed) {
                break;
            }
        }

        // ----- TRANSITION -----

        // Determine transition probabilities
        // Priority: cached attention > H2O fallback
        const float* trans_row = nullptr;
        bool use_h2o = true;

        // Check if current position is in query window (last 64 tokens)
        int window_start = seq_len > 64 ? seq_len - 64 : 0;
        if (current_pos >= window_start) {
            trans_row = query_attention;
            use_h2o = false;
        }

        // Check if current position is a landmark (use its cached attention)
        if (use_h2o) {
            for (int lm = 0; lm < num_landmarks && lm < LAW_MAX_LANDMARKS; ++lm) {
                if (smem_landmarks[lm] == current_pos) {
                    trans_row = landmark_attention + lm * seq_len;
                    use_h2o = false;
                    break;
                }
            }
        }

        // Compute transition weights (only to earlier positions)
        total_weight = 0.0f;

        if (use_h2o) {
            for (int j = 0; j < current_pos; ++j) {
                total_weight += h2o_scores[j];
            }
        } else {
            for (int j = 0; j < current_pos; ++j) {
                total_weight += trans_row[j];
            }
        }

        // Handle dead end
        if (total_weight <= 1e-8f) {
            // Heuristic jump toward sink
            int distance = current_pos - LAW_SINK_SIZE;
            int jump = max(1, (int)(distance * (0.02f + pcg_uniform(&rng) * 0.08f)));
            current_pos = max(LAW_SINK_SIZE, current_pos - jump);
            continue;
        }

        // Sample next position
        target = pcg_uniform(&rng) * total_weight;
        cumsum = 0.0f;
        int next_pos = 0;

        if (use_h2o) {
            for (int j = 0; j < current_pos; ++j) {
                cumsum += h2o_scores[j];
                if (cumsum >= target) {
                    next_pos = j;
                    break;
                }
            }
        } else {
            for (int j = 0; j < current_pos; ++j) {
                cumsum += trans_row[j];
                if (cumsum >= target) {
                    next_pos = j;
                    break;
                }
            }
        }

        current_pos = next_pos;
    }

    // Store updated PRNG state
    rng_states[walker_id * 2] = rng.state;
    rng_states[walker_id * 2 + 1] = rng.inc;
}

// =============================================================================
// Normalization Kernel: Segment-Aware (FIXED for Landmark Absorption)
// =============================================================================

/**
 * Normalize visit counts by SEGMENT-AWARE reachable walker count.
 *
 * KEY INSIGHT: With landmark absorption, position p can only be reached by:
 *   1. The NEXT landmark after p (walkers from further landmarks absorb earlier)
 *   2. Query walkers (with probability of reaching based on distance)
 *
 * For position p between landmarks L_i and L_{i+1}:
 *   - Only L_{i+1} walkers can reach p (L_{i+2}, L_{i+3}, ... absorb at L_{i+1})
 *   - Query contribution is reduced by number of landmarks between query and p
 *
 * This fixes the over-estimation bug where we counted ALL landmarks > p.
 */
__global__ void landmark_reachability_normalize_kernel(
    const int32_t* __restrict__ visit_counts,
    float* __restrict__ normalized_scores,
    const int32_t* __restrict__ landmark_positions,
    int num_landmarks,
    int walkers_per_source,
    float query_boost,
    int seq_len,
    int sink_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;

    // Sink positions get zero score (always kept separately)
    if (idx < sink_size) {
        normalized_scores[idx] = 0.0f;
        return;
    }

    // Find the NEXT landmark after this position (first landmark > idx)
    int next_landmark_pos = seq_len;  // Default: query position
    int next_landmark_idx = -1;
    for (int lm = 0; lm < num_landmarks; ++lm) {
        if (landmark_positions[lm] > idx && landmark_positions[lm] < next_landmark_pos) {
            next_landmark_pos = landmark_positions[lm];
            next_landmark_idx = lm;
        }
    }

    // Count reachable walkers (SEGMENT-AWARE)
    float total_reachable = 0.0f;

    // From the NEXT landmark only (others absorb before reaching us)
    if (next_landmark_idx >= 0) {
        total_reachable += (float)walkers_per_source;
    }

    // From query: Query can reach if it's the next source OR if it passes through
    // Query position is always seq_len - 1
    int query_pos = seq_len - 1;
    if (query_pos > idx) {
        if (next_landmark_idx < 0 || query_pos < next_landmark_pos) {
            // Query is closer than any landmark - full contribution
            total_reachable += (float)walkers_per_source * query_boost;
        } else {
            // Query is farther than next landmark
            // Query walkers have probability of reaching based on random walk
            // Use a decay factor based on number of landmarks between query and position
            int landmarks_between = 0;
            for (int lm = 0; lm < num_landmarks; ++lm) {
                if (landmark_positions[lm] > idx && landmark_positions[lm] < query_pos) {
                    landmarks_between++;
                }
            }
            // Each landmark has ~50% chance of absorbing, so decay by 0.5^n
            // But clamp to reasonable minimum
            float query_reach_prob = fmaxf(0.1f, powf(0.5f, (float)landmarks_between));
            total_reachable += (float)walkers_per_source * query_boost * query_reach_prob;
        }
    }

    // Normalize with minimum floor to avoid division issues
    if (total_reachable > 1e-8f) {
        normalized_scores[idx] = (float)visit_counts[idx] / total_reachable;
    } else {
        // Fallback: use raw visit count scaled by total walkers
        float total_walkers = (float)(num_landmarks + 1) * walkers_per_source;
        normalized_scores[idx] = (float)visit_counts[idx] / total_walkers;
    }
}

// =============================================================================
// Host Launcher Functions
// =============================================================================

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
) {
    int num_sources = num_landmarks + 1;
    int total_walkers = num_sources * walkers_per_source;

    if (total_walkers <= 0 || seq_len <= 0) return;

    dim3 block(LAW_BLOCK_SIZE);
    dim3 grid((total_walkers + block.x - 1) / block.x);

    landmark_absorbing_walker_kernel<<<grid, block, 0, stream>>>(
        landmark_attention,
        query_attention,
        h2o_scores,
        visit_counts,
        rng_states,
        landmark_positions,
        num_landmarks,
        walkers_per_source,
        query_boost,
        seq_len,
        absorb_at_landmarks
    );
}

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
) {
    if (seq_len <= 0) return;

    dim3 block(LAW_BLOCK_SIZE);
    dim3 grid((seq_len + block.x - 1) / block.x);

    landmark_reachability_normalize_kernel<<<grid, block, 0, stream>>>(
        visit_counts,
        normalized_scores,
        landmark_positions,
        num_landmarks,
        walkers_per_source,
        query_boost,
        seq_len,
        sink_size
    );
}

}  // namespace circuit_kv
