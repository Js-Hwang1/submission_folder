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

    /**
     * Spectral + Walker + MAX: Combined scoring for CircuitKV.
     *
     * This method computes importance scores using TWO complementary methods:
     * 1. SPECTRAL (Power Iteration): Captures "hub" tokens with global importance
     * 2. WALKER (Absorbing Random Walk): Captures "bridge" tokens on Q→Sink path
     *
     * Final scores use element-wise MAX to preserve BOTH types of important tokens:
     *   combined_score[i] = max(spectral[i], walker[i])
     *
     * This is particularly effective for summarization tasks where both global
     * structure (spectral) and reasoning paths (walker) matter.
     *
     * @param query         Query vector [1, head_dim] or [head_dim], FP16/FP32
     * @param keys          Key cache [seq_len, head_dim], FP16/FP32
     * @param current_idx   Index of the current token (source node for walker)
     * @param num_iterations Number of power iterations for spectral (default 10)
     */
    void update_and_step_circuit_combined(
        torch::Tensor query,
        torch::Tensor keys,
        int current_idx,
        int num_iterations = 10
    );

    /**
     * Get the combined (Spectral + Walker + MAX) scores.
     *
     * @return Tensor of shape [max_seq_len] with combined scores
     */
    torch::Tensor get_combined_scores();

    /**
     * Landmark-Diverse Walker: Multi-source absorbing random walks.
     *
     * This method implements the Landmark-Diverse walker approach (v0.4.0):
     * 1. STRATIFIED landmark selection: one per segment, best H2O within segment
     * 2. Launch walkers from ALL sources (landmarks + query) in parallel
     * 3. Apply normalization (reachability or positional)
     *
     * Key Insight:
     *   Single-source walks miss important tokens not directly visible from query.
     *   By launching walks from geographically-diverse landmarks, we discover
     *   "bridge tokens" through path convergence.
     *
     * Normalization Modes:
     *   - Reachability (default): normalized[p] = visits[p] / total_walkers_that_could_reach[p]
     *   - Positional: normalized[p] = visits[p] / (1/distance^alpha)
     *
     * @param attention_matrix   Full attention matrix [seq_len, seq_len], FP32
     * @param current_idx        Current token index (query position)
     * @param num_landmarks      Number of landmarks to select (default 32)
     * @param walkers_per_source Walkers per source (default 100)
     * @param query_boost        Weight multiplier for query walkers (default 2.0)
     * @param min_spacing        Minimum segment size for stratified (default 50)
     * @param position_alpha     Positional normalization exponent (default 0.6)
     * @param use_reachability   true = reachability norm, false = positional norm (default)
     */
    void update_and_step_landmark_walker(
        torch::Tensor attention_matrix,
        int current_idx,
        int num_landmarks = 32,
        int walkers_per_source = 100,
        float query_boost = 2.0f,
        int min_spacing = 50,
        float position_alpha = 0.6f,
        bool use_reachability = false
    );

    /**
     * Get the landmark walker scores (positionally normalized).
     *
     * @return Tensor of shape [seq_len] with normalized scores
     */
    torch::Tensor get_landmark_scores();

    /**
     * Landmark Absorbing Walker: Landmarks are BOTH sources AND sinks.
     *
     * NEW APPROACH (v0.5.0):
     *   Instead of all walkers flowing to tokens 0-3, walkers absorb at
     *   ANY landmark (or sink). This creates a "mesh" of current flows
     *   between landmarks, capturing local bridge tokens better.
     *
     * Physics Analogy:
     *   - Old: Single battery (Query+ to Sink-)
     *   - New: Mesh network where each landmark can source or sink current
     *
     * Key Benefit:
     *   Captures "local bridges" - tokens that connect ADJACENT landmarks.
     *   A token between landmarks L1 and L2 gets visits from both directions.
     *
     * @param attention_matrix     Full attention matrix [seq_len, seq_len], FP32
     * @param current_idx          Current token index (query position)
     * @param num_landmarks        Number of landmarks to select (default 32)
     * @param walkers_per_source   Walkers per source (default 100)
     * @param query_boost          Weight multiplier for query walkers (default 2.0)
     * @param min_spacing          Minimum segment size for stratified (default 50)
     * @param absorb_at_landmarks  If true, landmarks absorb walkers (NEW); if false, old behavior
     */
    void update_and_step_landmark_absorbing_walker(
        torch::Tensor attention_matrix,
        int current_idx,
        int num_landmarks = 32,
        int walkers_per_source = 100,
        float query_boost = 2.0f,
        int min_spacing = 50,
        bool absorb_at_landmarks = true
    );

    /**
     * Get the landmark absorbing walker scores.
     *
     * @return Tensor of shape [seq_len] with normalized scores
     */
    torch::Tensor get_landmark_absorbing_scores();

    /**
     * DEBUG: Get raw visit counts before normalization.
     *
     * @return Tensor of shape [seq_len] with raw visit counts (int32)
     */
    torch::Tensor get_landmark_absorbing_raw_visits();

    /**
     * DEBUG: Get selected landmark positions.
     *
     * @return Tensor of shape [num_landmarks] with landmark positions (int32)
     */
    torch::Tensor get_landmark_positions();

private:
    // Configuration
    int max_seq_len_;
    int top_k_;
    float alpha_;  // Kept for API compatibility (unused)
    int num_walkers_;
    int num_steps_;  // Kept for API compatibility (unused, uses MAX_STEPS)
    int query_window_;  // Kept for API compatibility (unused)
    int current_seq_len_;

    // GPU memory: Adjacency list (token i attends to token j)
    int32_t* adj_list_;        // [max_seq_len, top_k]
    float* adj_weights_;       // [max_seq_len, top_k]

    // GPU memory: Visit counts (current-flow scores)
    int32_t* visit_counts_;    // [max_seq_len]

    // GPU memory: PRNG states
    uint64_t* rng_states_;     // [num_walkers * 2]

    // GPU memory: Spectral power iteration buffers
    float* spectral_v_;           // [max_seq_len] - eigenvector
    float* spectral_v_temp_;      // [max_seq_len] - temp for SpMV
    float* spectral_partial_;     // [256] - for reduction
    float* spectral_scalar_;      // [1] - for norm/max

    // GPU memory: Combined scores
    float* walker_scores_;        // [max_seq_len] - normalized walker scores
    float* combined_scores_;      // [max_seq_len] - final combined scores

    // GPU memory: Landmark walker buffers
    static constexpr int MAX_LANDMARKS = 32;
    float* landmark_attention_;      // [MAX_LANDMARKS, max_seq_len] - cached attention rows
    float* query_attention_;         // [max_seq_len] - query's attention row
    float* h2o_scores_;              // [max_seq_len] - H2O column sums
    int32_t* landmark_positions_;    // [MAX_LANDMARKS] - selected landmark positions
    int* num_landmarks_selected_;    // [1] - number of landmarks actually selected
    float* landmark_normalized_;     // [max_seq_len] - positionally normalized scores
    float* landmark_partial_max_;    // [256] - for max reduction
    uint64_t* landmark_rng_states_;  // [max_walkers * 2] - RNG for landmark walkers

    // Landmark walker configuration
    int max_landmark_walkers_;       // Maximum total walkers for landmark method

    // CUDA streams
    cudaStream_t main_stream_;      // Inherited from PyTorch
    cudaStream_t sidecar_stream_;   // Async walker stream

    // Initialization flags
    bool rng_initialized_;
    bool landmark_rng_initialized_;

    // Spectral configuration
    int num_power_iterations_;

    // Helper methods
    void allocate_memory();
    void free_memory();
    void init_rng(uint64_t seed);
};

}  // namespace circuit_kv
