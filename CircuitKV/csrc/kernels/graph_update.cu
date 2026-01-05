/**
 * CircuitKV - Graph Update Kernel
 *
 * Kernel 1: update_graph_kernel
 *
 * Purpose:
 *   When a new token T is generated, compute its attention to all past tokens
 *   and store the Top-K highest-scoring indices in the adjacency list.
 *
 * Algorithm:
 *   1. Each thread computes dot products for a subset of keys
 *   2. Block-level Top-K reduction using shared memory
 *   3. Write final Top-K indices to adj_list[T, :]
 *
 * Block/Thread Mapping:
 *   - 1 block processes one new token's neighbors
 *   - 256 threads per block, each handles ceil(seq_len/256) keys
 *   - Shared memory: stores (score, index) pairs for reduction
 *
 * Memory Access Pattern:
 *   - Query: 1 x head_dim (broadcast to all threads)
 *   - Keys: seq_len x head_dim (coalesced reads per warp)
 *   - Output: 1 x top_k (single write at end)
 */

#include <cuda_fp16.h>
#include "kernels/common.cuh"
#include "include/kernels.h"

namespace circuit_kv {

// =============================================================================
// Configuration
// =============================================================================

constexpr int GRAPH_UPDATE_BLOCK_SIZE = 256;
constexpr int GRAPH_UPDATE_MAX_TOP_K = 64;  // Max K we support in shared mem

// =============================================================================
// Device Helper: Load Query into Registers (Broadcast)
// =============================================================================

/**
 * Cooperative load of query vector into shared memory.
 * All threads in block read the same query, cache in smem.
 */
template <int HEAD_DIM>
__device__ __forceinline__ void load_query_to_smem(
    const __half* __restrict__ query,
    float* __restrict__ smem_query,
    int head_dim
) {
    // Each thread loads a portion
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        smem_query[i] = __half2float(query[i]);
    }
    __syncthreads();
}

// =============================================================================
// Device Helper: Compute Dot Product
// =============================================================================

/**
 * Compute dot product between query (in smem) and key[key_idx].
 */
__device__ __forceinline__ float compute_attention_score(
    const float* __restrict__ smem_query,
    const __half* __restrict__ keys,
    int key_idx,
    int head_dim
) {
    float sum = 0.0f;
    const __half* key = keys + key_idx * head_dim;

    // Unrolled loop for common head_dim values
    #pragma unroll 4
    for (int d = 0; d < head_dim; ++d) {
        sum += smem_query[d] * __half2float(key[d]);
    }

    return sum;
}

__device__ __forceinline__ float compute_attention_score_fp32(
    const float* __restrict__ smem_query,
    const float* __restrict__ keys,
    int key_idx,
    int head_dim
) {
    float sum = 0.0f;
    const float* key = keys + key_idx * head_dim;

    #pragma unroll 4
    for (int d = 0; d < head_dim; ++d) {
        sum += smem_query[d] * key[d];
    }

    return sum;
}

// =============================================================================
// Device Helper: Block-Level Top-K Selection
// =============================================================================

/**
 * Structure to hold score-index pairs for sorting.
 */
struct ScoreIndexPair {
    float score;
    int index;
};

/**
 * Insert a new (score, index) pair into a sorted top-k array.
 * Maintains descending order by score.
 *
 * @param heap      Array of top-k pairs (descending order)
 * @param k         Current size of heap (≤ max_k)
 * @param max_k     Maximum heap size (top_k)
 * @param score     New score to insert
 * @param index     New index to insert
 * @return          New size of heap
 */
__device__ __forceinline__ int heap_insert(
    ScoreIndexPair* heap,
    int k,
    int max_k,
    float score,
    int index
) {
    // If heap not full, find insertion point and shift
    if (k < max_k) {
        int pos = k;
        // Find position (descending order)
        while (pos > 0 && heap[pos - 1].score < score) {
            heap[pos] = heap[pos - 1];
            pos--;
        }
        heap[pos] = {score, index};
        return k + 1;
    }

    // Heap full: check if new score beats minimum (last element)
    if (score > heap[max_k - 1].score) {
        // Find insertion point
        int pos = max_k - 1;
        while (pos > 0 && heap[pos - 1].score < score) {
            heap[pos] = heap[pos - 1];
            pos--;
        }
        heap[pos] = {score, index};
    }

    return max_k;
}

// =============================================================================
// Main Kernel: Graph Update (FP16 version)
// =============================================================================

/**
 * Graph update kernel - computes Top-K neighbors for a new token.
 *
 * Each thread:
 *   1. Maintains a local top-k heap
 *   2. Iterates through assigned keys, inserting high scores
 *   3. Participates in block-level reduction
 *
 * Shared memory layout:
 *   [0, head_dim): query vector (float)
 *   [head_dim, head_dim + BLOCK_SIZE * top_k): per-thread top-k pairs
 */
__global__ void update_graph_kernel(
    const __half* __restrict__ query,
    const __half* __restrict__ keys,
    int32_t* __restrict__ adj_list,
    float* __restrict__ adj_weights,
    int current_idx,
    int seq_len,
    int head_dim,
    int top_k
) {
    // Dynamic shared memory
    extern __shared__ char smem[];
    float* smem_query = reinterpret_cast<float*>(smem);

    // Load query into shared memory
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        smem_query[i] = __half2float(query[i]);
    }
    __syncthreads();

    // Each thread maintains a local top-k heap
    ScoreIndexPair local_heap[GRAPH_UPDATE_MAX_TOP_K];
    int local_k = 0;

    // Initialize heap to avoid undefined behavior on first comparison
    for (int i = 0; i < GRAPH_UPDATE_MAX_TOP_K; ++i) {
        local_heap[i].score = -1e30f;  // Negative infinity
        local_heap[i].index = -1;
    }

    // Each thread processes keys with stride = blockDim.x
    for (int key_idx = threadIdx.x; key_idx < seq_len; key_idx += blockDim.x) {
        float score = compute_attention_score(smem_query, keys, key_idx, head_dim);
        local_k = heap_insert(local_heap, local_k, top_k, score, key_idx);
    }

    // Block-level reduction to find global top-k
    // We use shared memory to collect all local heaps

    // Shared memory for collecting results
    __shared__ ScoreIndexPair block_heap[GRAPH_UPDATE_MAX_TOP_K];
    __shared__ int block_heap_size;

    if (threadIdx.x == 0) {
        block_heap_size = 0;
    }
    __syncthreads();

    // Each thread contributes its local heap elements one at a time
    // This is a simple O(threads * top_k) approach - can be optimized with parallel reduction
    for (int i = 0; i < local_k; ++i) {
        // Serialize writes to block heap (simple but not optimal)
        for (int t = 0; t < blockDim.x; ++t) {
            if (threadIdx.x == t && local_k > i) {
                int old_size = block_heap_size;
                block_heap_size = heap_insert(
                    block_heap, old_size, top_k,
                    local_heap[i].score, local_heap[i].index
                );
            }
            __syncthreads();
        }
    }

    // Thread 0 writes final results
    if (threadIdx.x == 0) {
        int32_t* row = adj_list + current_idx * top_k;
        float* weight_row = adj_weights + current_idx * top_k;

        for (int k = 0; k < top_k; ++k) {
            if (k < block_heap_size) {
                row[k] = block_heap[k].index;
                weight_row[k] = block_heap[k].score;
            } else {
                row[k] = -1;  // No neighbor
                weight_row[k] = 0.0f;
            }
        }
    }
}

/**
 * FP32 version for testing/debugging.
 */
__global__ void update_graph_kernel_fp32(
    const float* __restrict__ query,
    const float* __restrict__ keys,
    int32_t* __restrict__ adj_list,
    float* __restrict__ adj_weights,
    int current_idx,
    int seq_len,
    int head_dim,
    int top_k
) {
    extern __shared__ char smem[];
    float* smem_query = reinterpret_cast<float*>(smem);

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        smem_query[i] = query[i];
    }
    __syncthreads();

    ScoreIndexPair local_heap[GRAPH_UPDATE_MAX_TOP_K];
    int local_k = 0;

    // Initialize heap to avoid undefined behavior on first comparison
    for (int i = 0; i < GRAPH_UPDATE_MAX_TOP_K; ++i) {
        local_heap[i].score = -1e30f;  // Negative infinity
        local_heap[i].index = -1;
    }

    for (int key_idx = threadIdx.x; key_idx < seq_len; key_idx += blockDim.x) {
        float score = compute_attention_score_fp32(smem_query, keys, key_idx, head_dim);
        local_k = heap_insert(local_heap, local_k, top_k, score, key_idx);
    }

    __shared__ ScoreIndexPair block_heap[GRAPH_UPDATE_MAX_TOP_K];
    __shared__ int block_heap_size;

    if (threadIdx.x == 0) {
        block_heap_size = 0;
    }
    __syncthreads();

    for (int i = 0; i < local_k; ++i) {
        for (int t = 0; t < blockDim.x; ++t) {
            if (threadIdx.x == t && local_k > i) {
                int old_size = block_heap_size;
                block_heap_size = heap_insert(
                    block_heap, old_size, top_k,
                    local_heap[i].score, local_heap[i].index
                );
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        int32_t* row = adj_list + current_idx * top_k;
        float* weight_row = adj_weights + current_idx * top_k;

        for (int k = 0; k < top_k; ++k) {
            if (k < block_heap_size) {
                row[k] = block_heap[k].index;
                weight_row[k] = block_heap[k].score;
            } else {
                row[k] = -1;
                weight_row[k] = 0.0f;
            }
        }
    }
}

// =============================================================================
// Host Launcher Functions
// =============================================================================

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
) {
    // Validate inputs
    if (seq_len <= 0 || top_k <= 0 || top_k > GRAPH_UPDATE_MAX_TOP_K) {
        return;
    }

    // Calculate shared memory size
    size_t smem_size = head_dim * sizeof(float);

    // Launch configuration
    dim3 grid(1);
    dim3 block(GRAPH_UPDATE_BLOCK_SIZE);

    update_graph_kernel<<<grid, block, smem_size, stream>>>(
        query, keys, adj_list, adj_weights,
        current_idx, seq_len, head_dim, top_k
    );
}

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
) {
    if (seq_len <= 0 || top_k <= 0 || top_k > GRAPH_UPDATE_MAX_TOP_K) {
        return;
    }

    size_t smem_size = head_dim * sizeof(float);
    dim3 grid(1);
    dim3 block(GRAPH_UPDATE_BLOCK_SIZE);

    update_graph_kernel_fp32<<<grid, block, smem_size, stream>>>(
        query, keys, adj_list, adj_weights,
        current_idx, seq_len, head_dim, top_k
    );
}

// =============================================================================
// Legacy: Build Reverse Graph (kept for potential future use)
// =============================================================================

/**
 * Build reverse adjacency graph from forward graph.
 *
 * For each edge i->j in forward graph, we add edge j->i to reverse graph.
 * This captures "hub centrality" - tokens that many other tokens attend TO.
 *
 * Block/Thread Mapping:
 *   - One block per source token
 *   - 64 threads per block (handles up to 64 neighbors)
 *   - Uses atomicAdd for concurrent edge insertion
 */
__global__ void build_reverse_graph_kernel(
    const int32_t* __restrict__ adj_list,
    int32_t* __restrict__ rev_adj_list,
    int32_t* __restrict__ rev_adj_count,
    int current_idx,
    int seq_len,
    int top_k
) {
    // Each block handles one source token
    int src_token = blockIdx.x;
    if (src_token > current_idx) return;

    // Each thread handles one neighbor slot
    int neighbor_slot = threadIdx.x;
    if (neighbor_slot >= top_k) return;

    // Get the target of this edge: src_token -> target
    int target = adj_list[src_token * top_k + neighbor_slot];

    // Skip invalid edges
    if (target < 0 || target >= seq_len) return;

    // Add reverse edge: target -> src_token
    // Atomically get the next slot in target's reverse adjacency
    int slot = atomicAdd(&rev_adj_count[target], 1);

    // Only insert if we have room (cap at top_k reverse edges)
    if (slot < top_k) {
        rev_adj_list[target * top_k + slot] = src_token;
    }
}

void launch_build_reverse_graph_kernel(
    const int32_t* adj_list,
    int32_t* rev_adj_list,
    int32_t* rev_adj_count,
    int current_idx,
    int seq_len,
    int top_k,
    cudaStream_t stream
) {
    if (current_idx < 0 || seq_len <= 0 || top_k <= 0) {
        return;
    }

    // One block per token (0 to current_idx inclusive)
    dim3 grid(current_idx + 1);
    // 64 threads per block (enough for top_k up to 64)
    dim3 block(64);

    build_reverse_graph_kernel<<<grid, block, 0, stream>>>(
        adj_list, rev_adj_list, rev_adj_count,
        current_idx, seq_len, top_k
    );
}

// =============================================================================
// Query-Biased PPR: Compute Query Affinity Scores
// =============================================================================

/**
 * Compute query affinity for each context position.
 *
 * For position j, computes: affinity[j] = mean_{i in query_window} attn(Q[i], K[j])
 * where attn is softmax-normalized attention.
 *
 * Each thread handles one context position.
 */
__global__ void compute_query_affinity_kernel(
    const __half* __restrict__ queries,  // [query_window, head_dim]
    const __half* __restrict__ keys,     // [seq_len, head_dim]
    float* __restrict__ query_affinity,  // [seq_len]
    int query_window,
    int seq_len,
    int head_dim
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= seq_len) return;

    float scale = 1.0f / sqrtf((float)head_dim);
    float sum_attn = 0.0f;

    // For each query in the window
    for (int q = 0; q < query_window; ++q) {
        // Compute Q[q] · K[j]
        float dot = 0.0f;
        const __half* query_ptr = queries + q * head_dim;
        const __half* key_ptr = keys + j * head_dim;

        for (int d = 0; d < head_dim; ++d) {
            dot += __half2float(query_ptr[d]) * __half2float(key_ptr[d]);
        }
        dot *= scale;

        // We need softmax, but computing full softmax is expensive.
        // Use exp(dot) as unnormalized attention weight.
        // The walker will normalize locally among neighbors.
        sum_attn += expf(fminf(dot, 20.0f));  // Clamp to avoid overflow
    }

    query_affinity[j] = sum_attn / (float)query_window;
}

__global__ void compute_query_affinity_kernel_fp32(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    float* __restrict__ query_affinity,
    int query_window,
    int seq_len,
    int head_dim
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= seq_len) return;

    float scale = 1.0f / sqrtf((float)head_dim);
    float sum_attn = 0.0f;

    for (int q = 0; q < query_window; ++q) {
        float dot = 0.0f;
        const float* query_ptr = queries + q * head_dim;
        const float* key_ptr = keys + j * head_dim;

        for (int d = 0; d < head_dim; ++d) {
            dot += query_ptr[d] * key_ptr[d];
        }
        dot *= scale;
        sum_attn += expf(fminf(dot, 20.0f));
    }

    query_affinity[j] = sum_attn / (float)query_window;
}

void launch_compute_query_affinity_kernel(
    const __half* queries,
    const __half* keys,
    float* query_affinity,
    int query_window,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    if (seq_len <= 0 || query_window <= 0) return;

    dim3 block(256);
    dim3 grid((seq_len + block.x - 1) / block.x);

    compute_query_affinity_kernel<<<grid, block, 0, stream>>>(
        queries, keys, query_affinity,
        query_window, seq_len, head_dim
    );
}

void launch_compute_query_affinity_kernel_fp32(
    const float* queries,
    const float* keys,
    float* query_affinity,
    int query_window,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    if (seq_len <= 0 || query_window <= 0) return;

    dim3 block(256);
    dim3 grid((seq_len + block.x - 1) / block.x);

    compute_query_affinity_kernel_fp32<<<grid, block, 0, stream>>>(
        queries, keys, query_affinity,
        query_window, seq_len, head_dim
    );
}

}  // namespace circuit_kv
