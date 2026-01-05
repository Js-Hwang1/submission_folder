/**
 * CircuitKV - Common CUDA Definitions
 *
 * This header defines:
 * 1. Fast PRNG (PCG-based for lightweight random walks)
 * 2. AdjacencyList data structure (sparse graph cache)
 * 3. Shared device helper functions
 *
 * Memory Layout:
 * - AdjacencyList uses [N, K] layout where K=32 (neighbors per token)
 * - This allows coalesced memory access when all threads in a warp
 *   read neighbors for consecutive tokens
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cstdint>

namespace circuit_kv {

// =============================================================================
// Configuration Constants
// =============================================================================

constexpr int DEFAULT_TOP_K = 32;         // Neighbors per token (elbow point)
constexpr int DEFAULT_NUM_WALKERS = 1024; // Parallel random walkers
constexpr int DEFAULT_NUM_STEPS = 20;     // Steps per walker (legacy PPR)
constexpr float DEFAULT_ALPHA = 0.85f;    // Teleport probability (legacy PPR)

// CircuitKV (Absorbing Random Walk) Parameters
constexpr int DEFAULT_SINK_SIZE = 4;      // Absorbing boundary (first N tokens)
constexpr int DEFAULT_MAX_STEPS = 100;    // Safety timeout for walks

// =============================================================================
// Fast PRNG: PCG-based Random Number Generator
// =============================================================================

/**
 * Lightweight PCG (Permuted Congruential Generator) state.
 * Much faster to initialize than curand_state_t and produces
 * high-quality random numbers for Monte Carlo walks.
 *
 * Based on PCG-XSH-RR (32-bit output, 64-bit state)
 */
struct PCGState {
    uint64_t state;
    uint64_t inc;  // Must be odd
};

/**
 * Initialize a PCG state with a seed.
 * Each thread should use a unique (seed + thread_id) combination.
 */
__device__ __forceinline__ void pcg_init(PCGState* rng, uint64_t seed, uint64_t seq) {
    rng->state = 0u;
    rng->inc = (seq << 1u) | 1u;  // Ensure inc is odd
    // Advance state once
    rng->state = rng->state * 6364136223846793005ULL + rng->inc;
    rng->state += seed;
    rng->state = rng->state * 6364136223846793005ULL + rng->inc;
}

/**
 * Generate a random 32-bit unsigned integer.
 */
__device__ __forceinline__ uint32_t pcg_next(PCGState* rng) {
    uint64_t oldstate = rng->state;
    // LCG step
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    // PCG output function (XSH-RR)
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

/**
 * Generate a random float in [0, 1).
 */
__device__ __forceinline__ float pcg_uniform(PCGState* rng) {
    // Use upper 24 bits for float mantissa
    return (pcg_next(rng) >> 8) * (1.0f / 16777216.0f);
}

/**
 * Generate a random integer in [0, max_val).
 */
__device__ __forceinline__ int pcg_int(PCGState* rng, int max_val) {
    // Fast modulo for power-of-2 is handled automatically by compiler
    return pcg_next(rng) % max_val;
}

// =============================================================================
// Sparse Graph: Adjacency List Structure
// =============================================================================

/**
 * Sparse Adjacency List for the Attention Graph.
 *
 * Storage Layout: [max_seq_len, top_k] in row-major order
 * - adj_list[i * top_k + k] = k-th neighbor of token i
 * - Value of -1 indicates no neighbor (for tokens < top_k edges)
 *
 * This layout enables:
 * - Coalesced reads when warps process consecutive tokens
 * - Efficient atomic updates during graph construction
 * - O(1) lookup of neighbors for a given token
 */
struct AdjacencyList {
    int32_t* __restrict__ neighbors;  // [max_seq_len, top_k]
    float* __restrict__ weights;      // [max_seq_len, top_k] (optional, for weighted walks)
    int max_seq_len;
    int top_k;

    /**
     * Get neighbor index k of token i.
     */
    __device__ __forceinline__ int get_neighbor(int token_idx, int k) const {
        return neighbors[token_idx * top_k + k];
    }

    /**
     * Set neighbor k of token i.
     */
    __device__ __forceinline__ void set_neighbor(int token_idx, int k, int neighbor) {
        neighbors[token_idx * top_k + k] = neighbor;
    }

    /**
     * Get weight of edge from token_idx to its k-th neighbor.
     */
    __device__ __forceinline__ float get_weight(int token_idx, int k) const {
        return weights[token_idx * top_k + k];
    }

    /**
     * Set weight of edge from token_idx to its k-th neighbor.
     */
    __device__ __forceinline__ void set_weight(int token_idx, int k, float w) {
        weights[token_idx * top_k + k] = w;
    }
};

// =============================================================================
// Visit Counts (PageRank Scores)
// =============================================================================

/**
 * Visit counts from random walks.
 * Higher count = higher PPR score = more important token.
 */
struct VisitCounts {
    int32_t* __restrict__ counts;  // [max_seq_len]
    int max_seq_len;

    /**
     * Atomic increment of visit count for a token.
     */
    __device__ __forceinline__ void increment(int token_idx) {
        atomicAdd(&counts[token_idx], 1);
    }

    /**
     * Get visit count for a token.
     */
    __device__ __forceinline__ int get(int token_idx) const {
        return counts[token_idx];
    }

    /**
     * Reset all counts to zero.
     */
    __host__ void reset(cudaStream_t stream) {
        cudaMemsetAsync(counts, 0, max_seq_len * sizeof(int32_t), stream);
    }
};

// =============================================================================
// Warp-Level Primitives for Top-K
// =============================================================================

/**
 * Warp-level reduction to find maximum value and its index.
 * All threads in the warp participate.
 *
 * Returns the maximum value; index is stored in *max_idx.
 */
__device__ __forceinline__ float warp_reduce_max_with_idx(
    float val,
    int idx,
    int* max_idx
) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    // Broadcast result to all threads in warp
    val = __shfl_sync(0xffffffff, val, 0);
    *max_idx = __shfl_sync(0xffffffff, idx, 0);
    return val;
}

/**
 * Warp-level reduction to find minimum value.
 */
__device__ __forceinline__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fminf(val, other);
    }
    return __shfl_sync(0xffffffff, val, 0);
}

// =============================================================================
// Half-Precision Helpers
// =============================================================================

/**
 * Convert half to float.
 */
__device__ __forceinline__ float half_to_float(__half h) {
    return __half2float(h);
}

/**
 * Dot product of two FP16 vectors using FP32 accumulator.
 * Vectors are accessed with stride for flexibility.
 */
__device__ __forceinline__ float dot_product_fp16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    int dim,
    int stride_a = 1,
    int stride_b = 1
) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += half_to_float(a[i * stride_a]) * half_to_float(b[i * stride_b]);
    }
    return sum;
}

}  // namespace circuit_kv
