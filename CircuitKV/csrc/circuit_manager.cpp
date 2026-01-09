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
    , spectral_v_(nullptr)
    , spectral_v_temp_(nullptr)
    , spectral_partial_(nullptr)
    , spectral_scalar_(nullptr)
    , walker_scores_(nullptr)
    , combined_scores_(nullptr)
    , landmark_attention_(nullptr)
    , query_attention_(nullptr)
    , h2o_scores_(nullptr)
    , landmark_positions_(nullptr)
    , num_landmarks_selected_(nullptr)
    , landmark_normalized_(nullptr)
    , landmark_partial_max_(nullptr)
    , landmark_rng_states_(nullptr)
    , max_landmark_walkers_(0)
    , influence_visits_(nullptr)
    , influence_normalized_(nullptr)
    , influence_partial_max_(nullptr)
    , influence_rng_states_(nullptr)
    , influence_max_walkers_(10000)  // Default validated by PoC5
    , influence_rng_initialized_(false)
    , main_stream_(nullptr)
    , sidecar_stream_(nullptr)
    , rng_initialized_(false)
    , landmark_rng_initialized_(false)
    , num_power_iterations_(10)
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
    , spectral_v_(other.spectral_v_)
    , spectral_v_temp_(other.spectral_v_temp_)
    , spectral_partial_(other.spectral_partial_)
    , spectral_scalar_(other.spectral_scalar_)
    , walker_scores_(other.walker_scores_)
    , combined_scores_(other.combined_scores_)
    , landmark_attention_(other.landmark_attention_)
    , query_attention_(other.query_attention_)
    , h2o_scores_(other.h2o_scores_)
    , landmark_positions_(other.landmark_positions_)
    , num_landmarks_selected_(other.num_landmarks_selected_)
    , landmark_normalized_(other.landmark_normalized_)
    , landmark_partial_max_(other.landmark_partial_max_)
    , landmark_rng_states_(other.landmark_rng_states_)
    , max_landmark_walkers_(other.max_landmark_walkers_)
    , influence_visits_(other.influence_visits_)
    , influence_normalized_(other.influence_normalized_)
    , influence_partial_max_(other.influence_partial_max_)
    , influence_rng_states_(other.influence_rng_states_)
    , influence_max_walkers_(other.influence_max_walkers_)
    , influence_rng_initialized_(other.influence_rng_initialized_)
    , main_stream_(other.main_stream_)
    , sidecar_stream_(other.sidecar_stream_)
    , rng_initialized_(other.rng_initialized_)
    , landmark_rng_initialized_(other.landmark_rng_initialized_)
    , num_power_iterations_(other.num_power_iterations_)
{
    // Null out other's pointers to prevent double-free
    other.adj_list_ = nullptr;
    other.adj_weights_ = nullptr;
    other.visit_counts_ = nullptr;
    other.rng_states_ = nullptr;
    other.spectral_v_ = nullptr;
    other.spectral_v_temp_ = nullptr;
    other.spectral_partial_ = nullptr;
    other.spectral_scalar_ = nullptr;
    other.walker_scores_ = nullptr;
    other.combined_scores_ = nullptr;
    other.landmark_attention_ = nullptr;
    other.query_attention_ = nullptr;
    other.h2o_scores_ = nullptr;
    other.landmark_positions_ = nullptr;
    other.num_landmarks_selected_ = nullptr;
    other.landmark_normalized_ = nullptr;
    other.landmark_partial_max_ = nullptr;
    other.landmark_rng_states_ = nullptr;
    other.influence_visits_ = nullptr;
    other.influence_normalized_ = nullptr;
    other.influence_partial_max_ = nullptr;
    other.influence_rng_states_ = nullptr;
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
        spectral_v_ = other.spectral_v_;
        spectral_v_temp_ = other.spectral_v_temp_;
        spectral_partial_ = other.spectral_partial_;
        spectral_scalar_ = other.spectral_scalar_;
        walker_scores_ = other.walker_scores_;
        combined_scores_ = other.combined_scores_;
        landmark_attention_ = other.landmark_attention_;
        query_attention_ = other.query_attention_;
        h2o_scores_ = other.h2o_scores_;
        landmark_positions_ = other.landmark_positions_;
        num_landmarks_selected_ = other.num_landmarks_selected_;
        landmark_normalized_ = other.landmark_normalized_;
        landmark_partial_max_ = other.landmark_partial_max_;
        landmark_rng_states_ = other.landmark_rng_states_;
        max_landmark_walkers_ = other.max_landmark_walkers_;
        influence_visits_ = other.influence_visits_;
        influence_normalized_ = other.influence_normalized_;
        influence_partial_max_ = other.influence_partial_max_;
        influence_rng_states_ = other.influence_rng_states_;
        influence_max_walkers_ = other.influence_max_walkers_;
        influence_rng_initialized_ = other.influence_rng_initialized_;
        main_stream_ = other.main_stream_;
        sidecar_stream_ = other.sidecar_stream_;
        rng_initialized_ = other.rng_initialized_;
        landmark_rng_initialized_ = other.landmark_rng_initialized_;
        num_power_iterations_ = other.num_power_iterations_;

        // Null out other's pointers
        other.adj_list_ = nullptr;
        other.adj_weights_ = nullptr;
        other.visit_counts_ = nullptr;
        other.rng_states_ = nullptr;
        other.spectral_v_ = nullptr;
        other.spectral_v_temp_ = nullptr;
        other.spectral_partial_ = nullptr;
        other.spectral_scalar_ = nullptr;
        other.walker_scores_ = nullptr;
        other.combined_scores_ = nullptr;
        other.landmark_attention_ = nullptr;
        other.query_attention_ = nullptr;
        other.h2o_scores_ = nullptr;
        other.landmark_positions_ = nullptr;
        other.num_landmarks_selected_ = nullptr;
        other.landmark_normalized_ = nullptr;
        other.landmark_partial_max_ = nullptr;
        other.landmark_rng_states_ = nullptr;
        other.influence_visits_ = nullptr;
        other.influence_normalized_ = nullptr;
        other.influence_partial_max_ = nullptr;
        other.influence_rng_states_ = nullptr;
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

    // Spectral buffers
    size_t spectral_v_size = max_seq_len_ * sizeof(float);
    size_t spectral_partial_size = 256 * sizeof(float);  // For reduction
    size_t spectral_scalar_size = sizeof(float);

    // Adjacency graph
    CUDA_CHECK(cudaMalloc(&adj_list_, adj_list_size));
    CUDA_CHECK(cudaMalloc(&adj_weights_, adj_weights_size));

    // Visit counts and RNG
    CUDA_CHECK(cudaMalloc(&visit_counts_, visit_counts_size));
    CUDA_CHECK(cudaMalloc(&rng_states_, rng_states_size));

    // Spectral power iteration buffers
    CUDA_CHECK(cudaMalloc(&spectral_v_, spectral_v_size));
    CUDA_CHECK(cudaMalloc(&spectral_v_temp_, spectral_v_size));
    CUDA_CHECK(cudaMalloc(&spectral_partial_, spectral_partial_size));
    CUDA_CHECK(cudaMalloc(&spectral_scalar_, spectral_scalar_size));

    // Combined scoring buffers
    CUDA_CHECK(cudaMalloc(&walker_scores_, spectral_v_size));
    CUDA_CHECK(cudaMalloc(&combined_scores_, spectral_v_size));

    // Landmark walker buffers
    // Max walkers = (MAX_LANDMARKS + 1) * walkers_per_source (assume max 500 per source)
    max_landmark_walkers_ = (MAX_LANDMARKS + 1) * 500;
    size_t landmark_attention_size = MAX_LANDMARKS * max_seq_len_ * sizeof(float);
    size_t landmark_rng_size = max_landmark_walkers_ * 2 * sizeof(uint64_t);

    CUDA_CHECK(cudaMalloc(&landmark_attention_, landmark_attention_size));
    CUDA_CHECK(cudaMalloc(&query_attention_, spectral_v_size));
    CUDA_CHECK(cudaMalloc(&h2o_scores_, spectral_v_size));
    CUDA_CHECK(cudaMalloc(&landmark_positions_, MAX_LANDMARKS * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&num_landmarks_selected_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&landmark_normalized_, spectral_v_size));
    CUDA_CHECK(cudaMalloc(&landmark_partial_max_, spectral_partial_size));
    CUDA_CHECK(cudaMalloc(&landmark_rng_states_, landmark_rng_size));

    // Initialize all to default values
    CUDA_CHECK(cudaMemset(adj_list_, -1, adj_list_size));
    CUDA_CHECK(cudaMemset(adj_weights_, 0, adj_weights_size));
    CUDA_CHECK(cudaMemset(visit_counts_, 0, visit_counts_size));
    CUDA_CHECK(cudaMemset(spectral_v_, 0, spectral_v_size));
    CUDA_CHECK(cudaMemset(spectral_v_temp_, 0, spectral_v_size));
    CUDA_CHECK(cudaMemset(walker_scores_, 0, spectral_v_size));
    CUDA_CHECK(cudaMemset(combined_scores_, 0, spectral_v_size));

    // Initialize landmark buffers
    CUDA_CHECK(cudaMemset(landmark_attention_, 0, landmark_attention_size));
    CUDA_CHECK(cudaMemset(query_attention_, 0, spectral_v_size));
    CUDA_CHECK(cudaMemset(h2o_scores_, 0, spectral_v_size));
    CUDA_CHECK(cudaMemset(landmark_positions_, 0, MAX_LANDMARKS * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(num_landmarks_selected_, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(landmark_normalized_, 0, spectral_v_size));

    // Influence walker buffers (v1.0.0 - validated by PoC5)
    influence_max_walkers_ = 10000;  // PoC5 validated default
    size_t influence_rng_size = influence_max_walkers_ * 2 * sizeof(uint64_t);

    CUDA_CHECK(cudaMalloc(&influence_visits_, spectral_v_size));  // Float visits
    CUDA_CHECK(cudaMalloc(&influence_normalized_, spectral_v_size));
    CUDA_CHECK(cudaMalloc(&influence_partial_max_, spectral_partial_size));
    CUDA_CHECK(cudaMalloc(&influence_rng_states_, influence_rng_size));

    // Initialize influence buffers
    CUDA_CHECK(cudaMemset(influence_visits_, 0, spectral_v_size));
    CUDA_CHECK(cudaMemset(influence_normalized_, 0, spectral_v_size));
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
    if (spectral_v_) {
        cudaFree(spectral_v_);
        spectral_v_ = nullptr;
    }
    if (spectral_v_temp_) {
        cudaFree(spectral_v_temp_);
        spectral_v_temp_ = nullptr;
    }
    if (spectral_partial_) {
        cudaFree(spectral_partial_);
        spectral_partial_ = nullptr;
    }
    if (spectral_scalar_) {
        cudaFree(spectral_scalar_);
        spectral_scalar_ = nullptr;
    }
    if (walker_scores_) {
        cudaFree(walker_scores_);
        walker_scores_ = nullptr;
    }
    if (combined_scores_) {
        cudaFree(combined_scores_);
        combined_scores_ = nullptr;
    }

    // Free landmark buffers
    if (landmark_attention_) {
        cudaFree(landmark_attention_);
        landmark_attention_ = nullptr;
    }
    if (query_attention_) {
        cudaFree(query_attention_);
        query_attention_ = nullptr;
    }
    if (h2o_scores_) {
        cudaFree(h2o_scores_);
        h2o_scores_ = nullptr;
    }
    if (landmark_positions_) {
        cudaFree(landmark_positions_);
        landmark_positions_ = nullptr;
    }
    if (num_landmarks_selected_) {
        cudaFree(num_landmarks_selected_);
        num_landmarks_selected_ = nullptr;
    }
    if (landmark_normalized_) {
        cudaFree(landmark_normalized_);
        landmark_normalized_ = nullptr;
    }
    if (landmark_partial_max_) {
        cudaFree(landmark_partial_max_);
        landmark_partial_max_ = nullptr;
    }
    if (landmark_rng_states_) {
        cudaFree(landmark_rng_states_);
        landmark_rng_states_ = nullptr;
    }

    // Free influence walker buffers
    if (influence_visits_) {
        cudaFree(influence_visits_);
        influence_visits_ = nullptr;
    }
    if (influence_normalized_) {
        cudaFree(influence_normalized_);
        influence_normalized_ = nullptr;
    }
    if (influence_partial_max_) {
        cudaFree(influence_partial_max_);
        influence_partial_max_ = nullptr;
    }
    if (influence_rng_states_) {
        cudaFree(influence_rng_states_);
        influence_rng_states_ = nullptr;
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

void CircuitGraph::update_and_step_circuit_combined(
    torch::Tensor query,
    torch::Tensor keys,
    int current_idx,
    int num_iterations
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

    // Update current sequence length and power iterations
    current_seq_len_ = seq_len;
    num_power_iterations_ = num_iterations;

    // Get current CUDA stream from PyTorch
    main_stream_ = at::cuda::getCurrentCUDAStream();

    // Ensure main stream is done with query/keys before we use them
    CUDA_CHECK(cudaStreamSynchronize(main_stream_));

    // STEP 1: Build the attention graph (same as before)
    // Dispatch based on dtype
    if (query.dtype() == torch::kFloat16) {
        const __half* query_ptr = reinterpret_cast<const __half*>(query.data_ptr<at::Half>());
        const __half* keys_ptr = reinterpret_cast<const __half*>(keys.data_ptr<at::Half>());

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

    // STEP 2: Clear visit counts for fresh walker measurement
    launch_reset_counts_kernel(
        visit_counts_,
        max_seq_len_,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();

    // STEP 3: Run Absorbing Walker kernel (BOND scoring)
    launch_absorbing_walker_kernel(
        adj_list_,
        adj_weights_,
        visit_counts_,
        rng_states_,
        current_idx,
        seq_len,
        top_k_,
        num_walkers_,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();

    // STEP 4: Run Spectral Power Iteration (ATOM scoring)
    launch_power_iteration(
        adj_list_,
        adj_weights_,
        spectral_v_,
        spectral_v_temp_,
        spectral_partial_,
        spectral_scalar_,
        seq_len,
        top_k_,
        num_power_iterations_,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();

    // STEP 5: Convert walker visit counts to float scores
    launch_convert_visits_kernel(
        visit_counts_,
        walker_scores_,
        seq_len,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();

    // STEP 6: Normalize both scores to [0, 1] range
    int num_reduce_blocks = (seq_len + 255) / 256;
    if (num_reduce_blocks > 256) num_reduce_blocks = 256;

    // Normalize spectral scores
    launch_normalize_to_unit_max(
        spectral_v_,
        spectral_partial_,
        spectral_scalar_,
        seq_len,
        num_reduce_blocks,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();

    // Normalize walker scores
    launch_normalize_to_unit_max(
        walker_scores_,
        spectral_partial_,
        spectral_scalar_,
        seq_len,
        num_reduce_blocks,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();

    // STEP 7: Combine with element-wise MAX
    launch_max_combine_kernel(
        spectral_v_,
        walker_scores_,
        combined_scores_,
        seq_len,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();
}

torch::Tensor CircuitGraph::get_combined_scores() {
    // Synchronize sidecar stream to ensure all operations are complete
    CUDA_CHECK(cudaStreamSynchronize(sidecar_stream_));

    // Create output tensor on GPU
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA);

    torch::Tensor scores = torch::empty({max_seq_len_}, options);

    // Copy combined scores directly (already float)
    CUDA_CHECK(cudaMemcpy(
        scores.data_ptr<float>(),
        combined_scores_,
        max_seq_len_ * sizeof(float),
        cudaMemcpyDeviceToDevice
    ));

    return scores;
}

// =============================================================================
// Landmark-Diverse Walker Methods
// =============================================================================

void CircuitGraph::update_and_step_landmark_walker(
    torch::Tensor attention_matrix,
    int current_idx,
    int num_landmarks,
    int walkers_per_source,
    float query_boost,
    int min_spacing,
    float position_alpha,
    bool use_reachability  // true = reachability (default), false = positional
) {
    CHECK_CUDA(attention_matrix);
    CHECK_CONTIGUOUS(attention_matrix);
    TORCH_CHECK(attention_matrix.dtype() == torch::kFloat32,
                "attention_matrix must be float32");
    TORCH_CHECK(attention_matrix.dim() == 2,
                "attention_matrix must be 2D [seq_len, seq_len]");

    int seq_len = attention_matrix.size(0);
    TORCH_CHECK(attention_matrix.size(1) == seq_len,
                "attention_matrix must be square");
    TORCH_CHECK(current_idx >= 0 && current_idx < seq_len,
                "current_idx out of bounds");
    TORCH_CHECK(seq_len <= max_seq_len_,
                "seq_len exceeds max_seq_len");
    TORCH_CHECK(num_landmarks <= MAX_LANDMARKS,
                "num_landmarks exceeds MAX_LANDMARKS");

    // Update current sequence length
    current_seq_len_ = seq_len;

    // Get current CUDA stream from PyTorch
    main_stream_ = at::cuda::getCurrentCUDAStream();
    CUDA_CHECK(cudaStreamSynchronize(main_stream_));

    const float* attn_ptr = attention_matrix.data_ptr<float>();

    // Initialize landmark RNG if needed
    int total_walkers = (num_landmarks + 1) * walkers_per_source;
    TORCH_CHECK(total_walkers <= max_landmark_walkers_,
                "total_walkers exceeds max_landmark_walkers");

    if (!landmark_rng_initialized_) {
        auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
        launch_init_rng_kernel(
            landmark_rng_states_,
            static_cast<uint64_t>(seed),
            max_landmark_walkers_,
            sidecar_stream_
        );
        landmark_rng_initialized_ = true;
    }

    // STEP 1: Compute H2O scores (column sums)
    // H2O score for j = sum_i attention[i, j]
    launch_compute_h2o_scores_kernel(
        attn_ptr,
        h2o_scores_,
        seq_len,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();

    // STEP 2: Select diverse landmarks
    // Uses greedy selection with spacing constraint
    int sink_buffer = 20;  // Extra buffer from sink
    int window_size = 64;  // Last window_size tokens excluded

    launch_select_landmarks_kernel(
        h2o_scores_,
        landmark_positions_,
        num_landmarks_selected_,
        seq_len,
        num_landmarks,
        min_spacing,
        sink_buffer,
        window_size,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();

    // Synchronize to get actual landmark count (needed for next steps)
    CUDA_CHECK(cudaStreamSynchronize(sidecar_stream_));

    int actual_landmarks;
    CUDA_CHECK(cudaMemcpy(&actual_landmarks, num_landmarks_selected_,
                          sizeof(int), cudaMemcpyDeviceToHost));

    // STEP 3: Cache landmark attention rows
    if (actual_landmarks > 0) {
        launch_cache_landmark_attention_kernel(
            attn_ptr,
            landmark_attention_,
            landmark_positions_,
            actual_landmarks,
            seq_len,
            sidecar_stream_
        );
        CUDA_CHECK_LAST();
    }

    // STEP 4: Copy query's attention row
    CUDA_CHECK(cudaMemcpyAsync(
        query_attention_,
        attn_ptr + current_idx * seq_len,
        seq_len * sizeof(float),
        cudaMemcpyDeviceToDevice,
        sidecar_stream_
    ));

    // STEP 5: Clear visit counts
    launch_reset_counts_kernel(
        visit_counts_,
        max_seq_len_,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();

    // STEP 6: Launch multi-source landmark walker
    // Pass h2o_scores for on-the-fly transitions in non-landmark/non-window positions
    int actual_walkers = (actual_landmarks + 1) * walkers_per_source;
    launch_landmark_walker_kernel(
        landmark_attention_,
        query_attention_,
        h2o_scores_,  // H2O scores for fallback transitions
        visit_counts_,
        landmark_rng_states_,
        landmark_positions_,
        actual_landmarks,
        walkers_per_source,
        query_boost,
        seq_len,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();

    // STEP 7: Apply normalization (reachability or positional)
    int sink_size = 4;  // Standard sink size

    if (use_reachability) {
        // Reachability normalization: accounts for actual walker distribution
        // normalized[p] = visits[p] / total_walkers_that_could_reach[p]
        launch_reachability_normalize_kernel(
            visit_counts_,
            landmark_normalized_,
            landmark_positions_,
            actual_landmarks,
            walkers_per_source,
            query_boost,
            seq_len,
            sink_size,
            sidecar_stream_
        );
    } else {
        // Positional normalization: assumes uniform walker distribution
        // normalized[p] = visits[p] / (1/distance^alpha)
        launch_positional_normalize_kernel(
            visit_counts_,
            landmark_normalized_,
            landmark_partial_max_,
            seq_len,
            sink_size,
            position_alpha,
            sidecar_stream_
        );
    }
    CUDA_CHECK_LAST();
}

torch::Tensor CircuitGraph::get_landmark_scores() {
    // Synchronize sidecar stream to ensure all operations complete
    CUDA_CHECK(cudaStreamSynchronize(sidecar_stream_));

    // Create output tensor on GPU
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA);

    torch::Tensor scores = torch::empty({current_seq_len_}, options);

    // Copy normalized scores
    CUDA_CHECK(cudaMemcpy(
        scores.data_ptr<float>(),
        landmark_normalized_,
        current_seq_len_ * sizeof(float),
        cudaMemcpyDeviceToDevice
    ));

    return scores;
}

// =============================================================================
// Landmark Absorbing Walker Methods (v0.5.0 - Landmarks as Sources AND Sinks)
// =============================================================================

void CircuitGraph::update_and_step_landmark_absorbing_walker(
    torch::Tensor attention_matrix,
    int current_idx,
    int num_landmarks,
    int walkers_per_source,
    float query_boost,
    int min_spacing,
    bool absorb_at_landmarks
) {
    CHECK_CUDA(attention_matrix);
    CHECK_CONTIGUOUS(attention_matrix);
    TORCH_CHECK(attention_matrix.dtype() == torch::kFloat32,
                "attention_matrix must be float32");
    TORCH_CHECK(attention_matrix.dim() == 2,
                "attention_matrix must be 2D [seq_len, seq_len]");

    int seq_len = attention_matrix.size(0);
    TORCH_CHECK(attention_matrix.size(1) == seq_len,
                "attention_matrix must be square");
    TORCH_CHECK(current_idx >= 0 && current_idx < seq_len,
                "current_idx out of bounds");
    TORCH_CHECK(seq_len <= max_seq_len_,
                "seq_len exceeds max_seq_len");
    TORCH_CHECK(num_landmarks <= MAX_LANDMARKS,
                "num_landmarks exceeds MAX_LANDMARKS");

    // Update current sequence length
    current_seq_len_ = seq_len;

    // Get current CUDA stream from PyTorch
    main_stream_ = at::cuda::getCurrentCUDAStream();
    CUDA_CHECK(cudaStreamSynchronize(main_stream_));

    const float* attn_ptr = attention_matrix.data_ptr<float>();

    // Initialize landmark RNG if needed
    int total_walkers = (num_landmarks + 1) * walkers_per_source;
    TORCH_CHECK(total_walkers <= max_landmark_walkers_,
                "total_walkers exceeds max_landmark_walkers");

    if (!landmark_rng_initialized_) {
        auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
        launch_init_rng_kernel(
            landmark_rng_states_,
            static_cast<uint64_t>(seed),
            max_landmark_walkers_,
            sidecar_stream_
        );
        landmark_rng_initialized_ = true;
    }

    // STEP 1: Compute H2O scores (column sums)
    launch_compute_h2o_scores_kernel(
        attn_ptr,
        h2o_scores_,
        seq_len,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();

    // STEP 2: Select diverse landmarks (stratified)
    int sink_buffer = 20;
    int window_size = 64;

    launch_select_landmarks_kernel(
        h2o_scores_,
        landmark_positions_,
        num_landmarks_selected_,
        seq_len,
        num_landmarks,
        min_spacing,
        sink_buffer,
        window_size,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();

    // Synchronize to get actual landmark count
    CUDA_CHECK(cudaStreamSynchronize(sidecar_stream_));

    int actual_landmarks;
    CUDA_CHECK(cudaMemcpy(&actual_landmarks, num_landmarks_selected_,
                          sizeof(int), cudaMemcpyDeviceToHost));

    // STEP 3: Cache landmark attention rows
    if (actual_landmarks > 0) {
        launch_cache_landmark_attention_kernel(
            attn_ptr,
            landmark_attention_,
            landmark_positions_,
            actual_landmarks,
            seq_len,
            sidecar_stream_
        );
        CUDA_CHECK_LAST();
    }

    // STEP 4: Copy query's attention row
    CUDA_CHECK(cudaMemcpyAsync(
        query_attention_,
        attn_ptr + current_idx * seq_len,
        seq_len * sizeof(float),
        cudaMemcpyDeviceToDevice,
        sidecar_stream_
    ));

    // STEP 5: Clear visit counts
    launch_reset_counts_kernel(
        visit_counts_,
        max_seq_len_,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();

    // STEP 6: Launch landmark ABSORBING walker (NEW - landmarks can absorb)
    launch_landmark_absorbing_walker_kernel(
        landmark_attention_,
        query_attention_,
        h2o_scores_,
        visit_counts_,
        landmark_rng_states_,
        landmark_positions_,
        actual_landmarks,
        walkers_per_source,
        query_boost,
        seq_len,
        absorb_at_landmarks,  // NEW: whether landmarks absorb walkers
        sidecar_stream_
    );
    CUDA_CHECK_LAST();

    // STEP 7: Apply reachability normalization
    int sink_size = 4;
    launch_landmark_reachability_normalize_kernel(
        visit_counts_,
        landmark_normalized_,
        landmark_positions_,
        actual_landmarks,
        walkers_per_source,
        query_boost,
        seq_len,
        sink_size,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();
}

torch::Tensor CircuitGraph::get_landmark_absorbing_scores() {
    // Synchronize sidecar stream to ensure all operations complete
    CUDA_CHECK(cudaStreamSynchronize(sidecar_stream_));

    // Create output tensor on GPU
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA);

    torch::Tensor scores = torch::empty({current_seq_len_}, options);

    // Copy normalized scores (reuses landmark_normalized_ buffer)
    CUDA_CHECK(cudaMemcpy(
        scores.data_ptr<float>(),
        landmark_normalized_,
        current_seq_len_ * sizeof(float),
        cudaMemcpyDeviceToDevice
    ));

    return scores;
}

torch::Tensor CircuitGraph::get_landmark_absorbing_raw_visits() {
    // Synchronize sidecar stream
    CUDA_CHECK(cudaStreamSynchronize(sidecar_stream_));

    // Create output tensor on GPU (int32)
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(torch::kCUDA);

    torch::Tensor visits = torch::empty({current_seq_len_}, options);

    // Copy raw visit counts
    CUDA_CHECK(cudaMemcpy(
        visits.data_ptr<int32_t>(),
        visit_counts_,
        current_seq_len_ * sizeof(int32_t),
        cudaMemcpyDeviceToDevice
    ));

    return visits;
}

torch::Tensor CircuitGraph::get_landmark_positions() {
    // Synchronize sidecar stream
    CUDA_CHECK(cudaStreamSynchronize(sidecar_stream_));

    // Get number of landmarks selected
    int num_landmarks = 0;
    CUDA_CHECK(cudaMemcpy(
        &num_landmarks,
        num_landmarks_selected_,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    if (num_landmarks <= 0) {
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    }

    // Create output tensor on GPU (int32)
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(torch::kCUDA);

    torch::Tensor positions = torch::empty({num_landmarks}, options);

    // Copy landmark positions
    CUDA_CHECK(cudaMemcpy(
        positions.data_ptr<int32_t>(),
        landmark_positions_,
        num_landmarks * sizeof(int32_t),
        cudaMemcpyDeviceToDevice
    ));

    return positions;
}

// =============================================================================
// Causal Influence Propagation Walker (v1.0.0 - VALIDATED BY PoC5)
// =============================================================================

void CircuitGraph::update_and_step_influence_walker(
    torch::Tensor attention_matrix,
    int current_idx,
    int num_walkers,
    int max_steps,
    int sink_size,
    float temperature,
    float explore_temp,
    float explore_ratio
) {
    CHECK_CUDA(attention_matrix);
    CHECK_CONTIGUOUS(attention_matrix);
    TORCH_CHECK(attention_matrix.dtype() == torch::kFloat32,
                "attention_matrix must be float32");
    TORCH_CHECK(attention_matrix.dim() == 2,
                "attention_matrix must be 2D [seq_len, seq_len]");

    int seq_len = attention_matrix.size(0);
    TORCH_CHECK(attention_matrix.size(1) == seq_len,
                "attention_matrix must be square");
    TORCH_CHECK(current_idx >= 0 && current_idx < seq_len,
                "current_idx out of bounds");
    TORCH_CHECK(seq_len <= max_seq_len_,
                "seq_len exceeds max_seq_len");
    TORCH_CHECK(num_walkers <= influence_max_walkers_,
                "num_walkers exceeds influence_max_walkers");
    TORCH_CHECK(max_steps > 0 && max_steps <= 100,
                "max_steps must be in (0, 100]");
    TORCH_CHECK(sink_size >= 0 && sink_size < seq_len,
                "sink_size out of bounds");
    TORCH_CHECK(temperature >= 0.1f && temperature <= 100.0f,
                "temperature must be in [0.1, 100.0]");
    TORCH_CHECK(explore_temp >= 0.1f && explore_temp <= 100.0f,
                "explore_temp must be in [0.1, 100.0]");
    TORCH_CHECK(explore_ratio >= 0.0f && explore_ratio <= 1.0f,
                "explore_ratio must be in [0.0, 1.0]");

    // Update current sequence length
    current_seq_len_ = seq_len;

    // Get current CUDA stream from PyTorch
    main_stream_ = at::cuda::getCurrentCUDAStream();
    CUDA_CHECK(cudaStreamSynchronize(main_stream_));

    const float* attn_ptr = attention_matrix.data_ptr<float>();

    // Initialize influence RNG if needed
    if (!influence_rng_initialized_) {
        auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
        launch_init_rng_kernel(
            influence_rng_states_,
            static_cast<uint64_t>(seed),
            influence_max_walkers_,
            sidecar_stream_
        );
        influence_rng_initialized_ = true;
    }

    // STEP 1: Clear visit counts (float buffer)
    launch_clear_influence_visits_kernel(
        influence_visits_,
        seq_len,
        sidecar_stream_
    );
    CUDA_CHECK_LAST();

    // STEP 2: Launch influence walker kernel
    // v1.0.8: Dual-temperature walkers for balanced exploration
    launch_influence_walker_kernel(
        attn_ptr,
        influence_visits_,
        influence_rng_states_,
        seq_len,
        current_idx,
        num_walkers,
        max_steps,
        sink_size,
        sidecar_stream_,
        temperature,     // v1.0.8: Base temperature for logical walkers (70%)
        explore_temp,    // v1.0.8: High temperature for exploratory walkers (30%)
        explore_ratio    // v1.0.8: Ratio of exploratory walkers
    );
    CUDA_CHECK_LAST();

    // STEP 3: Normalize scores (optional, for numerical stability)
    // Find max and normalize to [0, 1] - this preserves rankings
    int num_reduce_blocks = (seq_len + 255) / 256;
    if (num_reduce_blocks > 256) num_reduce_blocks = 256;

    // Use the max reduction and normalization (v1.0.5: with positional adjustment)
    launch_find_max_and_normalize_kernel(
        influence_visits_,
        influence_normalized_,
        influence_partial_max_,
        seq_len,
        num_reduce_blocks,
        sidecar_stream_,
        sink_size  // v1.0.5: Pass sink_size for positional normalization
    );
    CUDA_CHECK_LAST();
}

torch::Tensor CircuitGraph::get_influence_scores() {
    // Synchronize sidecar stream to ensure all operations complete
    CUDA_CHECK(cudaStreamSynchronize(sidecar_stream_));

    // Create output tensor on GPU
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA);

    torch::Tensor scores = torch::empty({current_seq_len_}, options);

    // Copy normalized scores (or raw visits if normalization not used)
    CUDA_CHECK(cudaMemcpy(
        scores.data_ptr<float>(),
        influence_normalized_,
        current_seq_len_ * sizeof(float),
        cudaMemcpyDeviceToDevice
    ));

    return scores;
}

torch::Tensor CircuitGraph::get_influence_raw_visits() {
    // Synchronize sidecar stream to ensure all operations complete
    CUDA_CHECK(cudaStreamSynchronize(sidecar_stream_));

    // Create output tensor on GPU
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA);

    torch::Tensor visits = torch::empty({current_seq_len_}, options);

    // Copy raw visits (before normalization)
    CUDA_CHECK(cudaMemcpy(
        visits.data_ptr<float>(),
        influence_visits_,
        current_seq_len_ * sizeof(float),
        cudaMemcpyDeviceToDevice
    ));

    return visits;
}

}  // namespace circuit_kv
