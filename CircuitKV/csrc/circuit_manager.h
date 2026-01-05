/**
 * CircuitKV - Circuit Manager Header
 *
 * The CircuitGraph class manages:
 * 1. Sparse adjacency list (GPU memory)
 * 2. Visit counts (Current-Flow Betweenness scores)
 * 3. PRNG states for random walkers
 * 4. Async CUDA stream for walker kernel
 *
 * This is the main interface exposed to Python via pybind11.
 */

#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <memory>
#include <cstdint>

namespace circuit_kv {

/**
 * CircuitGraph: Manages the sparse attention graph for current-flow computation.
 *
 * Physics Analogy:
 * - Query = Battery positive terminal (Source)
 * - Context Start (tokens 0-3) = Battery negative terminal (Sink)
 * - Attention weights = Conductance (1/Resistance)
 * - Visit counts = Current flow through each node
 *
 * Thread Safety:
 *   - NOT thread-safe. Designed to be used from a single Python thread.
 *   - Internal CUDA operations use streams for async execution.
 *
 * Memory Management:
 *   - Allocates GPU memory in constructor, frees in destructor.
 *   - Uses raw CUDA memory (not PyTorch tensors) for internal state.
 *   - Returns PyTorch tensors for scores (wrapping internal buffer).
 */
class CircuitGraph {
public:
    /**
     * Construct a CircuitGraph.
     *
     * @param max_seq_len   Maximum sequence length to support
     * @param top_k         Number of neighbors per token
     * @param alpha         (Unused by CircuitKV, kept for API compatibility)
     * @param num_walkers   Number of parallel random walkers
     * @param num_steps     (Unused by CircuitKV, uses MAX_STEPS=100 internally)
     * @param query_window  (Unused, kept for API compatibility)
     */
    CircuitGraph(
        int max_seq_len,
        int top_k,
        float alpha,
        int num_walkers,
        int num_steps,
        int query_window = 64
    );

    ~CircuitGraph();

    // Disable copy (due to CUDA resources)
    CircuitGraph(const CircuitGraph&) = delete;
    CircuitGraph& operator=(const CircuitGraph&) = delete;

    // Allow move
    CircuitGraph(CircuitGraph&&) noexcept;
    CircuitGraph& operator=(CircuitGraph&&) noexcept;

    /**
     * CircuitKV: Current-Flow Betweenness via Absorbing Random Walks.
     *
     * This is the core CircuitKV method. It simulates electrical current flowing
     * from the Query (Source) to the Context Start (Sink), measuring "through-traffic"
     * for each token.
     *
     * Key Properties:
     * - NO RESTART: Walkers never teleport. They march from Source toward Sink.
     * - ABSORBING BOUNDARY: Walks stop when reaching first SINK_SIZE (4) tokens.
     * - TRANSPORT METRIC: Visit counts measure current flow, not just popularity.
     *
     * This "rescues" bridge tokens: tokens that appear once but connect Question
     * to Context. Unlike PPR, they accumulate current because they're on the
     * path from Source to Sink.
     *
     * @param query         Query vector [1, head_dim] or [head_dim], FP16/FP32
     * @param keys          Key cache [seq_len, head_dim], FP16/FP32
     * @param current_idx   Index of the current token (source node)
     */
    void update_and_step_circuit(
        torch::Tensor query,
        torch::Tensor keys,
        int current_idx
    );

    /**
     * Multi-source CircuitKV: Run walks from W sources in parallel (P3 Optimization).
     *
     * This method builds graphs for ALL sources in the observation window and
     * runs W × walkers_per_source walks in a single kernel launch.
     *
     * @param queries        Query vectors [num_sources, head_dim], FP32
     * @param keys           Key cache [seq_len, head_dim], FP32
     * @param source_indices Source token indices [num_sources], INT64
     */
    void update_and_step_circuit_multi_source(
        torch::Tensor queries,
        torch::Tensor keys,
        torch::Tensor source_indices
    );

    /**
     * Bidirectional CircuitKV (RC+B): Run walks in BOTH directions.
     *
     * This method runs absorbing walks in two directions:
     * - Backward: Query -> Sink (following attention edges)
     * - Forward: Sink -> Query (following transpose attention edges)
     *
     * Tokens visited by BOTH directions are "true bridges" and receive bonus scoring:
     *   bridge_score = min(backward_visits, forward_visits)
     *   final_score = max(backward, forward) + 0.5 * bridge_score
     *
     * This captures true reasoning bridges that connect Question to Context
     * from BOTH directions, significantly improving recall on LongBench.
     *
     * @param queries        Query vectors for observation window [W, head_dim], FP32
     * @param keys           Key cache [seq_len, head_dim], FP32
     * @param source_indices Query token indices [W], INT32
     */
    void update_and_step_circuit_bidirectional(
        torch::Tensor queries,
        torch::Tensor keys,
        torch::Tensor source_indices
    );

    /**
     * Get bidirectional scores (combined backward + forward + bridge).
     *
     * Returns scores computed as:
     *   bridge = min(backward, forward)
     *   score = max(backward, forward) + 0.5 * bridge
     *
     * Synchronizes sidecar stream before returning.
     *
     * @return Tensor of shape [max_seq_len] with combined scores
     */
    torch::Tensor get_bidirectional_scores();

    /**
     * Get the current-flow scores.
     *
     * Synchronizes the sidecar stream before returning.
     * Returns a copy of the visit counts as a float tensor.
     *
     * @return Tensor of shape [max_seq_len] with current-flow scores
     */
    torch::Tensor get_scores();

    /**
     * Reset the graph for a new sequence.
     *
     * Clears the adjacency list and visit counts.
     */
    void reset();

    /**
     * Synchronize all pending async operations.
     */
    void synchronize();

private:
    // Configuration
    int max_seq_len_;
    int top_k_;
    float alpha_;  // Kept for API compatibility (unused)
    int num_walkers_;
    int num_steps_;  // Kept for API compatibility (unused, uses MAX_STEPS)
    int query_window_;  // Kept for API compatibility (unused)
    int current_seq_len_;

    // GPU memory: Forward adjacency list (token i attends to token j)
    int32_t* adj_list_;        // [max_seq_len, top_k]
    float* adj_weights_;       // [max_seq_len, top_k]

    // GPU memory: Transpose adjacency list (who attends TO token k) - for RC+B
    int32_t* rev_adj_list_;    // [max_seq_len, top_k]
    float* rev_adj_weights_;   // [max_seq_len, top_k]

    // GPU memory: Visit counts (current-flow scores)
    int32_t* visit_counts_;    // [max_seq_len] - standard (backward) visits

    // GPU memory: Bidirectional visit counts - for RC+B
    int32_t* backward_visits_; // [max_seq_len] - Query->Sink direction
    int32_t* forward_visits_;  // [max_seq_len] - Sink->Query direction

    // GPU memory: PRNG states
    uint64_t* rng_states_;     // [num_walkers * 2]

    // CUDA streams
    cudaStream_t main_stream_;      // Inherited from PyTorch
    cudaStream_t sidecar_stream_;   // Async walker stream

    // Initialization flag
    bool rng_initialized_;

    // Helper methods
    void allocate_memory();
    void free_memory();
    void init_rng(uint64_t seed);
};

}  // namespace circuit_kv
