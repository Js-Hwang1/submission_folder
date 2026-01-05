/**
 * CircuitKV - Circuit Manager Implementation
 *
 * This file implements the CircuitGraph class, which manages the sparse
 * attention graph and orchestrates the CUDA kernels for current-flow computation.
 *
 * CircuitKV: Simulates electrical current flowing from Query (Source) to
 * Context Start (Sink) via absorbing random walks.
 */

#include "circuit_manager.h"
#include "compat.h"
#include "include/kernels.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <chrono>

namespace circuit_kv {

// =============================================================================
// Constructor / Destructor
// =============================================================================

CircuitGraph::CircuitGraph(
    int max_seq_len,
    int top_k,
    float alpha,
    int num_walkers,
    int num_steps,
    int query_window
)
    : max_seq_len_(max_seq_len)
    , top_k_(top_k)
    , alpha_(alpha)  // Kept for API compatibility (unused)
    , num_walkers_(num_walkers)
    , num_steps_(num_steps)  // Kept for API compatibility (unused, uses MAX_STEPS)
    , query_window_(query_window)  // Kept for API compatibility (unused)
    , current_seq_len_(0)
    , adj_list_(nullptr)
    , adj_weights_(nullptr)
    , visit_counts_(nullptr)
    , rng_states_(nullptr)
    , main_stream_(nullptr)
    , sidecar_stream_(nullptr)
    , rng_initialized_(false)
{
    TORCH_CHECK(max_seq_len > 0, "max_seq_len must be positive");
    TORCH_CHECK(top_k > 0 && top_k <= 64, "top_k must be in (0, 64]");
    TORCH_CHECK(num_walkers > 0, "num_walkers must be positive");

    allocate_memory();

    // Create sidecar stream for async walker kernel
    CUDA_CHECK(cudaStreamCreate(&sidecar_stream_));

    // Initialize RNG with time-based seed
    auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
    init_rng(static_cast<uint64_t>(seed));
}

CircuitGraph::~CircuitGraph() {
    try {
        if (sidecar_stream_) {
            cudaStreamSynchronize(sidecar_stream_);
            cudaStreamDestroy(sidecar_stream_);
        }
        free_memory();
    } catch (...) {
        // Suppress exceptions in destructor
    }
}

// Move constructor
CircuitGraph::CircuitGraph(CircuitGraph&& other) noexcept
    : max_seq_len_(other.max_seq_len_)
    , top_k_(other.top_k_)
    , alpha_(other.alpha_)
    , num_walkers_(other.num_walkers_)
    , num_steps_(other.num_steps_)
    , query_window_(other.query_window_)
    , current_seq_len_(other.current_seq_len_)
    , adj_list_(other.adj_list_)
    , adj_weights_(other.adj_weights_)
    , visit_counts_(other.visit_counts_)
    , rng_states_(other.rng_states_)
    , main_stream_(other.main_stream_)
    , sidecar_stream_(other.sidecar_stream_)
    , rng_initialized_(other.rng_initialized_)
{
    // Null out other's pointers to prevent double-free
    other.adj_list_ = nullptr;
    other.adj_weights_ = nullptr;
    other.visit_counts_ = nullptr;
    other.rng_states_ = nullptr;
    other.sidecar_stream_ = nullptr;
}

// Move assignment
CircuitGraph& CircuitGraph::operator=(CircuitGraph&& other) noexcept {
    if (this != &other) {
        // Free our resources
        if (sidecar_stream_) {
            cudaStreamSynchronize(sidecar_stream_);
            cudaStreamDestroy(sidecar_stream_);
        }
        free_memory();

        // Take other's resources
        max_seq_len_ = other.max_seq_len_;
        top_k_ = other.top_k_;
        alpha_ = other.alpha_;
        num_walkers_ = other.num_walkers_;
        num_steps_ = other.num_steps_;
        query_window_ = other.query_window_;
        current_seq_len_ = other.current_seq_len_;
        adj_list_ = other.adj_list_;
        adj_weights_ = other.adj_weights_;
        visit_counts_ = other.visit_counts_;
        rng_states_ = other.rng_states_;
        main_stream_ = other.main_stream_;
        sidecar_stream_ = other.sidecar_stream_;
        rng_initialized_ = other.rng_initialized_;

        // Null out other's pointers
        other.adj_list_ = nullptr;
        other.adj_weights_ = nullptr;
        other.visit_counts_ = nullptr;
        other.rng_states_ = nullptr;
        other.sidecar_stream_ = nullptr;
    }
    return *this;
}

// =============================================================================
// Memory Management
// =============================================================================

void CircuitGraph::allocate_memory() {
    size_t adj_list_size = max_seq_len_ * top_k_ * sizeof(int32_t);
    size_t adj_weights_size = max_seq_len_ * top_k_ * sizeof(float);
    size_t visit_counts_size = max_seq_len_ * sizeof(int32_t);
    size_t rng_states_size = num_walkers_ * 2 * sizeof(uint64_t);

    // Adjacency graph
    CUDA_CHECK(cudaMalloc(&adj_list_, adj_list_size));
    CUDA_CHECK(cudaMalloc(&adj_weights_, adj_weights_size));

    // Visit counts and RNG
    CUDA_CHECK(cudaMalloc(&visit_counts_, visit_counts_size));
    CUDA_CHECK(cudaMalloc(&rng_states_, rng_states_size));

    // Initialize all to default values
    CUDA_CHECK(cudaMemset(adj_list_, -1, adj_list_size));
    CUDA_CHECK(cudaMemset(adj_weights_, 0, adj_weights_size));
    CUDA_CHECK(cudaMemset(visit_counts_, 0, visit_counts_size));
}

void CircuitGraph::free_memory() {
    if (adj_list_) {
        cudaFree(adj_list_);
        adj_list_ = nullptr;
    }
    if (adj_weights_) {
        cudaFree(adj_weights_);
        adj_weights_ = nullptr;
    }
    if (visit_counts_) {
        cudaFree(visit_counts_);
        visit_counts_ = nullptr;
    }
    if (rng_states_) {
        cudaFree(rng_states_);
        rng_states_ = nullptr;
    }
}

void CircuitGraph::init_rng(uint64_t seed) {
    launch_init_rng_kernel(
        rng_states_,
        seed,
        num_walkers_,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();
    rng_initialized_ = true;
}

// =============================================================================
// Public Methods
// =============================================================================

void CircuitGraph::update_and_step_circuit(
    torch::Tensor query,
    torch::Tensor keys,
    int current_idx
) {
    CHECK_CUDA(query);
    CHECK_CUDA(keys);
    CHECK_CONTIGUOUS(query);
    CHECK_CONTIGUOUS(keys);

    // Get dimensions
    int seq_len = keys.size(0);
    int head_dim = keys.size(1);

    TORCH_CHECK(current_idx >= 0 && current_idx < max_seq_len_,
                "current_idx out of bounds");
    TORCH_CHECK(seq_len <= max_seq_len_,
                "seq_len exceeds max_seq_len");

    // Update current sequence length
    current_seq_len_ = seq_len;

    // Get current CUDA stream from PyTorch
    main_stream_ = at::cuda::getCurrentCUDAStream();

    // Ensure main stream is done with query/keys before we use them
    CUDA_CHECK(cudaStreamSynchronize(main_stream_));

    // CRITICAL: Clear visit counts before walking (Instantaneous Current measurement)
    // Each query gets a fresh start - we measure current flow for THIS query only.
    launch_reset_counts_kernel(
        visit_counts_,
        max_seq_len_,
        sidecar_stream_
    );

    CUDA_CHECK_LAST();

    // Dispatch based on dtype to build the attention graph
    if (query.dtype() == torch::kFloat16) {
        // FP16 path
        const __half* query_ptr = reinterpret_cast<const __half*>(query.data_ptr<at::Half>());
        const __half* keys_ptr = reinterpret_cast<const __half*>(keys.data_ptr<at::Half>());

        // Launch graph update kernel (on sidecar stream)
        launch_graph_update_kernel(
            query_ptr,
            keys_ptr,
            adj_list_,
            adj_weights_,
            current_idx,
            seq_len,
            head_dim,
            top_k_,
            sidecar_stream_
        );
    } else if (query.dtype() == torch::kFloat32) {
        // FP32 path
        const float* query_ptr = query.data_ptr<float>();
        const float* keys_ptr = keys.data_ptr<float>();

        launch_graph_update_kernel_fp32(
            query_ptr,
            keys_ptr,
            adj_list_,
            adj_weights_,
            current_idx,
            seq_len,
            head_dim,
            top_k_,
            sidecar_stream_
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Use float16 or float32.");
    }

    CUDA_CHECK_LAST();

    // Launch CircuitKV Absorbing Walker kernel
    // Walkers flow from SOURCE (current_idx = query position) toward SINK (first 4 tokens)
    // No restart, no teleport - pure current flow simulation
    launch_absorbing_walker_kernel(
        adj_list_,
        adj_weights_,
        visit_counts_,
        rng_states_,
        current_idx,  // SOURCE: current query position (usually last token)
        seq_len,
        top_k_,
        num_walkers_,
        sidecar_stream_
    );

    CUDA_CHECK_LAST();
}

torch::Tensor CircuitGraph::get_scores() {
    // Synchronize sidecar stream to ensure all walks are complete
    CUDA_CHECK(cudaStreamSynchronize(sidecar_stream_));

    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA);

    torch::Tensor scores = torch::empty({max_seq_len_}, options);

    // Copy visit counts to tensor (converting int32 to float32)
    // Simple approach: copy to CPU, convert, copy back
    auto counts_cpu = torch::empty({max_seq_len_}, torch::kInt32);
    CUDA_CHECK(cudaMemcpy(
        counts_cpu.data_ptr<int32_t>(),
        visit_counts_,
        max_seq_len_ * sizeof(int32_t),
        cudaMemcpyDeviceToHost
    ));

    // Convert to float and copy to GPU
    auto scores_cpu = counts_cpu.to(torch::kFloat32);
    CUDA_CHECK(cudaMemcpy(
        scores.data_ptr<float>(),
        scores_cpu.data_ptr<float>(),
        max_seq_len_ * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    return scores;
}

void CircuitGraph::reset() {
    // Synchronize first
    synchronize();

    // Reset adjacency list
    launch_reset_graph_kernel(
        adj_list_,
        max_seq_len_,
        top_k_,
        sidecar_stream_
    );

    // Reset visit counts
    launch_reset_counts_kernel(
        visit_counts_,
        max_seq_len_,
        sidecar_stream_
    );

    current_seq_len_ = 0;

    CUDA_CHECK_LAST();
}

void CircuitGraph::synchronize() {
    if (sidecar_stream_) {
        CUDA_CHECK(cudaStreamSynchronize(sidecar_stream_));
    }
}

}  // namespace circuit_kv
