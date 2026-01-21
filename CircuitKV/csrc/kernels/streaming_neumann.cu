/**
 * Streaming Neumann Series Kernel for CircuitKV v4.5.3
 *
 * Computes Q.t() @ v WITHOUT materializing the full attention matrix.
 * Memory: O(n) instead of O(n²)
 *
 * Key insight: We only need window attention [W, n] and approximate
 * prefix attention on-the-fly using attention-weighted heuristic.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "common.cuh"

namespace circuit_kv {

// Warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
template<int BLOCK_SIZE>
__device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}

/**
 * Fused kernel: Compute BOTH QI and HI Neumann iterations simultaneously
 *
 * For each output position j:
 *   v_out[j] = Σᵢ Q[i,j] × v[i]
 *            = Σᵢ (P[i+sink, j+sink] × v[i])
 *
 * Where P[i,j] = attention[i,j] / row_sum[i]
 *
 * - Window positions: use exact attention from window_attn
 * - Prefix positions: use attention-weighted approximation
 */
template<int BLOCK_SIZE = 256>
__global__ void streaming_dual_neumann_kernel(
    const float* __restrict__ window_attn,    // [window_size, seq_len]
    const float* __restrict__ v_qi,           // [n_transient] - QI input
    const float* __restrict__ v_hi,           // [n_transient] - HI input
    float* __restrict__ v_qi_out,             // [n_transient] - QI output
    float* __restrict__ v_hi_out,             // [n_transient] - HI output
    const float* __restrict__ h2o_scores,     // [seq_len]
    const float* __restrict__ h2o_cumsum,     // [n_prefix]
    const float* __restrict__ window_row_sums,// [window_size]
    int seq_len,
    int window_size,
    int sink_size,
    int n_transient,
    int n_prefix
) {
    int j = blockIdx.x;
    if (j >= n_transient) return;

    int j_full = j + sink_size;

    __shared__ float shared_qi[32];
    __shared__ float shared_hi[32];

    float local_qi = 0.0f;
    float local_hi = 0.0f;

    int window_start = seq_len - window_size;

    for (int i = threadIdx.x; i < n_transient; i += BLOCK_SIZE) {
        float vqi_i = v_qi[i];
        float vhi_i = v_hi[i];

        // Skip if both are zero (or very small)
        if (fabsf(vqi_i) < 1e-10f && fabsf(vhi_i) < 1e-10f) continue;

        int i_full = i + sink_size;

        float attn_ij = 0.0f;
        float row_sum = 1.0f;

        if (i_full >= window_start) {
            // Window region: use exact attention
            int window_row = i_full - window_start;
            attn_ij = window_attn[window_row * seq_len + j_full];
            row_sum = window_row_sums[window_row];
        } else {
            // Prefix region: use attention-weighted approximation
            // P[i_full, j_full] ≈ h2o[j_full] / cumsum[i_full-1] if j_full < i_full
            if (j_full < i_full) {
                attn_ij = h2o_scores[j_full];
                int cumsum_idx = i_full - sink_size - 1;
                if (cumsum_idx >= 0 && cumsum_idx < n_prefix) {
                    row_sum = h2o_cumsum[cumsum_idx] + 1e-8f;
                }
            }
            // else: j_full >= i_full means causal mask blocks it (attn_ij stays 0)
        }

        if (row_sum > 1e-8f && attn_ij > 0.0f) {
            float P_ij = attn_ij / row_sum;
            local_qi += P_ij * vqi_i;
            local_hi += P_ij * vhi_i;
        }
    }

    // Reduction for QI
    float total_qi = block_reduce_sum<BLOCK_SIZE>(local_qi, shared_qi);
    __syncthreads();

    // Reduction for HI
    float total_hi = block_reduce_sum<BLOCK_SIZE>(local_hi, shared_hi);

    if (threadIdx.x == 0) {
        v_qi_out[j] = total_qi;
        v_hi_out[j] = total_hi;
    }
}

/**
 * Host function: Full streaming Neumann series computation
 * Computes both QI and HI without materializing the attention matrix
 */
std::tuple<torch::Tensor, torch::Tensor> streaming_neumann_dual(
    torch::Tensor window_attn,     // [window_size, seq_len]
    int query_idx,
    int sink_size,
    int num_iterations,
    float temperature
) {
    TORCH_CHECK(window_attn.is_cuda(), "window_attn must be a CUDA tensor");
    TORCH_CHECK(window_attn.is_contiguous(), "window_attn must be contiguous");

    auto window_shape = window_attn.sizes();
    int window_size = window_shape[0];
    int seq_len = window_shape[1];
    int n_transient = seq_len - sink_size;
    int n_prefix = seq_len - window_size - sink_size;

    auto device = window_attn.device();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // Convert to float32 if needed
    if (window_attn.dtype() != torch::kFloat32) {
        window_attn = window_attn.to(torch::kFloat32);
    }

    // Apply temperature if needed
    if (temperature != 1.0f && temperature > 0.0f) {
        window_attn = window_attn.pow(1.0f / temperature);
    }

    // Precompute h2o_scores (column sums of window attention)
    torch::Tensor h2o_scores = window_attn.sum(0);  // [seq_len]

    // Precompute h2o_cumsum for prefix region
    torch::Tensor h2o_prefix;
    torch::Tensor h2o_cumsum;
    if (n_prefix > 0) {
        h2o_prefix = h2o_scores.slice(0, sink_size, sink_size + n_prefix).contiguous();
        h2o_prefix = h2o_prefix.clamp_min(1e-8f);
        h2o_cumsum = h2o_prefix.cumsum(0);  // [n_prefix]
    } else {
        h2o_cumsum = torch::zeros({1}, options);
    }

    // Precompute window row sums
    torch::Tensor window_row_sums = window_attn.sum(1);  // [window_size]

    // Initialize QI: one-hot at query position
    int query_transient_idx = query_idx - sink_size;
    torch::Tensor v_qi = torch::zeros({n_transient}, options);
    if (query_transient_idx >= 0 && query_transient_idx < n_transient) {
        v_qi[query_transient_idx] = 1.0f;
    }
    torch::Tensor result_qi = v_qi.clone();

    // Initialize HI: uniform start
    torch::Tensor v_hi = torch::ones({n_transient}, options) / static_cast<float>(n_transient);
    torch::Tensor result_hi = v_hi.clone();

    // Allocate output buffers
    torch::Tensor v_qi_out = torch::zeros({n_transient}, options);
    torch::Tensor v_hi_out = torch::zeros({n_transient}, options);

    // Neumann iterations using CUDA kernel
    const int BLOCK_SIZE = 256;
    dim3 grid(n_transient);
    dim3 block(BLOCK_SIZE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    for (int iter = 0; iter < num_iterations; iter++) {
        streaming_dual_neumann_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(
            window_attn.data_ptr<float>(),
            v_qi.data_ptr<float>(),
            v_hi.data_ptr<float>(),
            v_qi_out.data_ptr<float>(),
            v_hi_out.data_ptr<float>(),
            h2o_scores.data_ptr<float>(),
            h2o_cumsum.data_ptr<float>(),
            window_row_sums.data_ptr<float>(),
            seq_len,
            window_size,
            sink_size,
            n_transient,
            n_prefix
        );

        // Accumulate results
        result_qi.add_(v_qi_out);
        result_hi.add_(v_hi_out);

        // Swap for next iteration
        std::swap(v_qi, v_qi_out);
        std::swap(v_hi, v_hi_out);
    }

    // Map back to full sequence
    torch::Tensor qi_scores = torch::zeros({seq_len}, options);
    qi_scores.slice(0, sink_size, seq_len).copy_(result_qi);
    float qi_sink_val = result_qi.sum().item<float>() * 0.01f;
    qi_scores.slice(0, 0, sink_size).fill_(qi_sink_val);

    torch::Tensor hi_scores = torch::zeros({seq_len}, options);
    hi_scores.slice(0, sink_size, seq_len).copy_(result_hi);
    float hi_sink_val = result_hi.sum().item<float>() * 0.01f;
    hi_scores.slice(0, 0, sink_size).fill_(hi_sink_val);

    // Normalize to [0, 1]
    float qi_max = qi_scores.max().item<float>();
    if (qi_max > 0) qi_scores.div_(qi_max);

    float hi_max = hi_scores.max().item<float>();
    if (hi_max > 0) hi_scores.div_(hi_max);

    return std::make_tuple(qi_scores, hi_scores);
}

}  // namespace circuit_kv
