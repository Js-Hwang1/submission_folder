/**
 * CircuitKV - CUDA Compatibility Helpers
 *
 * This header provides compatibility macros and utilities for working with
 * PyTorch's C++ extension system and CUDA.
 */

#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =============================================================================
// Error Checking Macros
// =============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            TORCH_CHECK(false, "CUDA error at ", __FILE__, ":", __LINE__,      \
                        " - ", cudaGetErrorString(err));                       \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_LAST()                                                      \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            TORCH_CHECK(false, "CUDA kernel error at ", __FILE__, ":",         \
                        __LINE__, " - ", cudaGetErrorString(err));             \
        }                                                                      \
    } while (0)

// =============================================================================
// Tensor Type Checks
// =============================================================================

#define CHECK_CUDA(x)                                                          \
    TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

#define CHECK_DTYPE(x, dtype)                                                  \
    TORCH_CHECK((x).dtype() == dtype, #x " must be ", #dtype)

// =============================================================================
// Block/Thread Configuration Helpers
// =============================================================================

namespace circuit_kv {

constexpr int WARP_SIZE = 32;

/**
 * Round up to the nearest multiple of n.
 */
__host__ __device__ __forceinline__ int round_up(int x, int n) {
    return ((x + n - 1) / n) * n;
}

/**
 * Divide and round up.
 */
__host__ __device__ __forceinline__ int div_ceil(int x, int n) {
    return (x + n - 1) / n;
}

/**
 * Get the current CUDA stream from PyTorch.
 */
inline cudaStream_t get_cuda_stream() {
    return at::cuda::getCurrentCUDAStream();
}

/**
 * Create a new CUDA stream.
 */
inline cudaStream_t create_cuda_stream() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return stream;
}

/**
 * Destroy a CUDA stream.
 */
inline void destroy_cuda_stream(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamDestroy(stream));
}

}  // namespace circuit_kv
