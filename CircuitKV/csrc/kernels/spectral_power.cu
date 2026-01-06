/**
 * CircuitKV - Spectral Power Iteration Kernel
 *
 * Purpose:
 *   Compute the dominant eigenvector of the attention graph using power iteration.
 *   This captures "global importance" (eigenvector centrality) - tokens that are
 *   attended to by other important tokens get high scores.
 *
 * Algorithm:
 *   1. Initialize v = uniform (all 1/sqrt(N))
 *   2. For each iteration (default 10):
 *      a. v_new = A @ v  (sparse matrix-vector multiply using adjacency list)
 *      b. v = v_new / ||v_new||_2  (normalize)
 *   3. Return v as importance scores
 *
 * Key Properties:
 *   - GLOBAL STRUCTURE: Unlike H2O (local degree), spectral captures global patterns
 *   - SUMMARIZATION BOOST: Important for tasks where global context matters
 *   - COMPLEMENTARY TO WALKER: Walker captures "bridge" tokens, Spectral captures "hub" tokens
 *
 * Combined with Walker via MAX:
 *   final_score[i] = max(spectral_score[i], walker_score[i])
 *   This ensures both hub tokens AND bridge tokens are preserved.
 */

#include "kernels/common.cuh"
#include "include/kernels.h"

namespace circuit_kv {

// =============================================================================
// Configuration
// =============================================================================

constexpr int SPECTRAL_BLOCK_SIZE = 256;
constexpr int DEFAULT_POWER_ITERATIONS = 10;

// =============================================================================
// Kernel 1: Sparse Matrix-Vector Multiply (SpMV) using Adjacency List
// =============================================================================

/**
 * Compute v_out = A @ v_in using the sparse adjacency list representation.
 * Each thread handles one row (token) of the matrix.
 *
 * @param adj_list     Sparse neighbors [seq_len, top_k]
 * @param adj_weights  Edge weights [seq_len, top_k]
 * @param v_in         Input vector [seq_len]
 * @param v_out        Output vector [seq_len]
 * @param seq_len      Current sequence length
 * @param top_k        Max neighbors per token
 */
__global__ void spmv_kernel(
    const int32_t* __restrict__ adj_list,
    const float* __restrict__ adj_weights,
    const float* __restrict__ v_in,
    float* __restrict__ v_out,
    int seq_len,
    int top_k
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;

    // Compute dot product: sum over neighbors
    float sum = 0.0f;
    const int32_t* neighbors = adj_list + row * top_k;
    const float* weights = adj_weights + row * top_k;

    for (int k = 0; k < top_k; ++k) {
        int neighbor = neighbors[k];
        if (neighbor >= 0 && neighbor < seq_len) {
            sum += weights[k] * v_in[neighbor];
        }
    }

    v_out[row] = sum;
}

// =============================================================================
// Kernel 2: Compute L2 Norm (Reduction)
// =============================================================================

/**
 * Compute squared L2 norm of a vector using parallel reduction.
 * Each block computes partial sum, final reduction on CPU or separate kernel.
 */
__global__ void l2_norm_squared_kernel(
    const float* __restrict__ v,
    float* __restrict__ partial_sums,
    int seq_len
) {
    __shared__ float shared_sum[SPECTRAL_BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes sum of squared elements it handles
    float sum = 0.0f;
    for (int i = idx; i < seq_len; i += blockDim.x * gridDim.x) {
        float val = v[i];
        sum += val * val;
    }

    shared_sum[tid] = sum;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}

/**
 * Final reduction of partial sums to get total norm squared.
 * Single block reduces all partial sums.
 */
__global__ void final_reduce_kernel(
    float* __restrict__ partial_sums,
    float* __restrict__ norm_out,
    int num_blocks
) {
    __shared__ float shared_sum[SPECTRAL_BLOCK_SIZE];

    int tid = threadIdx.x;
    float sum = 0.0f;

    // Each thread handles multiple partial sums
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial_sums[i];
    }

    shared_sum[tid] = sum;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        norm_out[0] = sqrtf(shared_sum[0]);
    }
}

// =============================================================================
// Kernel 3: Normalize Vector
// =============================================================================

/**
 * Normalize vector: v = v / norm
 */
__global__ void normalize_kernel(
    float* __restrict__ v,
    const float* __restrict__ norm,
    int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;

    float n = norm[0];
    if (n > 1e-8f) {
        v[idx] /= n;
    }
}

// =============================================================================
// Kernel 4: Initialize Uniform Vector
// =============================================================================

/**
 * Initialize vector to uniform distribution: v[i] = 1/sqrt(N)
 */
__global__ void init_uniform_kernel(
    float* __restrict__ v,
    int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;

    v[idx] = rsqrtf((float)seq_len);  // 1/sqrt(N)
}

// =============================================================================
// Kernel 5: Convert Visit Counts to Normalized Scores
// =============================================================================

/**
 * Convert integer visit counts from walker to normalized float scores.
 * Also computes L2 norm for normalization.
 */
__global__ void convert_visits_to_scores_kernel(
    const int32_t* __restrict__ visit_counts,
    float* __restrict__ scores,
    int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;

    scores[idx] = (float)visit_counts[idx];
}

// =============================================================================
// Kernel 6: MAX Combine Spectral and Walker Scores
// =============================================================================

/**
 * Combine spectral and walker scores using element-wise MAX.
 * Both inputs should be normalized to [0, 1] range before calling.
 *
 * final_score[i] = max(spectral[i], walker[i])
 *
 * This ensures BOTH hub tokens (high spectral) AND bridge tokens (high walker)
 * are preserved in the final selection.
 */
__global__ void max_combine_kernel(
    const float* __restrict__ spectral_scores,
    const float* __restrict__ walker_scores,
    float* __restrict__ combined_scores,
    int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;

    combined_scores[idx] = fmaxf(spectral_scores[idx], walker_scores[idx]);
}

/**
 * Normalize scores to [0, 1] range by dividing by max value.
 * Finds max in first pass, normalizes in second.
 */
__global__ void find_max_kernel(
    const float* __restrict__ v,
    float* __restrict__ partial_max,
    int seq_len
) {
    __shared__ float shared_max[SPECTRAL_BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float max_val = -1e30f;
    for (int i = idx; i < seq_len; i += blockDim.x * gridDim.x) {
        max_val = fmaxf(max_val, v[i]);
    }

    shared_max[tid] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_max[blockIdx.x] = shared_max[0];
    }
}

__global__ void final_max_reduce_kernel(
    float* __restrict__ partial_max,
    float* __restrict__ max_out,
    int num_blocks
) {
    __shared__ float shared_max[SPECTRAL_BLOCK_SIZE];

    int tid = threadIdx.x;
    float max_val = -1e30f;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        max_val = fmaxf(max_val, partial_max[i]);
    }

    shared_max[tid] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_out[0] = shared_max[0];
    }
}

__global__ void scale_by_max_kernel(
    float* __restrict__ v,
    const float* __restrict__ max_val,
    int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;

    float m = max_val[0];
    if (m > 1e-8f) {
        v[idx] /= m;
    }
}

// =============================================================================
// Host Launcher Functions
// =============================================================================

void launch_spmv_kernel(
    const int32_t* adj_list,
    const float* adj_weights,
    const float* v_in,
    float* v_out,
    int seq_len,
    int top_k,
    cudaStream_t stream
) {
    if (seq_len <= 0) return;

    dim3 block(SPECTRAL_BLOCK_SIZE);
    dim3 grid((seq_len + block.x - 1) / block.x);

    spmv_kernel<<<grid, block, 0, stream>>>(
        adj_list, adj_weights, v_in, v_out, seq_len, top_k
    );
}

void launch_normalize_vector_kernel(
    float* v,
    float* partial_sums,
    float* norm_out,
    int seq_len,
    int num_reduce_blocks,
    cudaStream_t stream
) {
    if (seq_len <= 0) return;

    dim3 block(SPECTRAL_BLOCK_SIZE);
    dim3 grid_norm((seq_len + block.x - 1) / block.x);

    // Step 1: Compute partial squared sums
    l2_norm_squared_kernel<<<num_reduce_blocks, block, 0, stream>>>(
        v, partial_sums, seq_len
    );

    // Step 2: Final reduction to get norm
    final_reduce_kernel<<<1, block, 0, stream>>>(
        partial_sums, norm_out, num_reduce_blocks
    );

    // Step 3: Normalize
    normalize_kernel<<<grid_norm, block, 0, stream>>>(
        v, norm_out, seq_len
    );
}

void launch_init_uniform_kernel(
    float* v,
    int seq_len,
    cudaStream_t stream
) {
    if (seq_len <= 0) return;

    dim3 block(SPECTRAL_BLOCK_SIZE);
    dim3 grid((seq_len + block.x - 1) / block.x);

    init_uniform_kernel<<<grid, block, 0, stream>>>(v, seq_len);
}

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
) {
    if (seq_len <= 0 || num_iterations <= 0) return;

    dim3 block(SPECTRAL_BLOCK_SIZE);
    int num_reduce_blocks = (seq_len + block.x - 1) / block.x;
    if (num_reduce_blocks > 256) num_reduce_blocks = 256;  // Cap for reduction

    // Initialize v to uniform
    launch_init_uniform_kernel(v, seq_len, stream);

    // Power iteration loop
    for (int iter = 0; iter < num_iterations; ++iter) {
        // v_temp = A @ v
        launch_spmv_kernel(adj_list, adj_weights, v, v_temp, seq_len, top_k, stream);

        // v = normalize(v_temp)
        // Copy v_temp to v first, then normalize in-place
        cudaMemcpyAsync(v, v_temp, seq_len * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        launch_normalize_vector_kernel(v, partial_sums, norm_out, seq_len, num_reduce_blocks, stream);
    }
}

void launch_convert_visits_kernel(
    const int32_t* visit_counts,
    float* scores,
    int seq_len,
    cudaStream_t stream
) {
    if (seq_len <= 0) return;

    dim3 block(SPECTRAL_BLOCK_SIZE);
    dim3 grid((seq_len + block.x - 1) / block.x);

    convert_visits_to_scores_kernel<<<grid, block, 0, stream>>>(
        visit_counts, scores, seq_len
    );
}

void launch_normalize_to_unit_max(
    float* v,
    float* partial_max,
    float* max_out,
    int seq_len,
    int num_reduce_blocks,
    cudaStream_t stream
) {
    if (seq_len <= 0) return;

    dim3 block(SPECTRAL_BLOCK_SIZE);
    dim3 grid((seq_len + block.x - 1) / block.x);

    // Find max
    find_max_kernel<<<num_reduce_blocks, block, 0, stream>>>(
        v, partial_max, seq_len
    );

    // Reduce to single max
    final_max_reduce_kernel<<<1, block, 0, stream>>>(
        partial_max, max_out, num_reduce_blocks
    );

    // Scale
    scale_by_max_kernel<<<grid, block, 0, stream>>>(
        v, max_out, seq_len
    );
}

void launch_max_combine_kernel(
    const float* spectral_scores,
    const float* walker_scores,
    float* combined_scores,
    int seq_len,
    cudaStream_t stream
) {
    if (seq_len <= 0) return;

    dim3 block(SPECTRAL_BLOCK_SIZE);
    dim3 grid((seq_len + block.x - 1) / block.x);

    max_combine_kernel<<<grid, block, 0, stream>>>(
        spectral_scores, walker_scores, combined_scores, seq_len
    );
}

}  // namespace circuit_kv
