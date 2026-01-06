/**
 * CircuitKV - Python Bindings
 *
 * This file exposes the CircuitGraph class and related functions to Python
 * using pybind11 (via PyTorch's extension system).
 */

#include <torch/extension.h>
#include "circuit_manager.h"

namespace circuit_kv {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CircuitKV: Current-Flow Betweenness for KV Cache Eviction";

    // CircuitGraph class
    py::class_<CircuitGraph>(m, "CircuitGraph",
        R"pbdoc(
        CircuitGraph: Manages the sparse attention graph for current-flow computation.

        This class maintains a sparse adjacency list representing the attention
        graph, and runs absorbing random walks to compute current-flow betweenness
        scores for token importance.

        Physics Analogy:
        - Query = Battery positive terminal (Source)
        - Context Start (tokens 0-3) = Battery negative terminal (Sink)
        - Attention weights = Conductance (1/Resistance)
        - Visit counts = Current flow through each node

        The graph update and walker kernels run asynchronously on a sidecar CUDA
        stream, allowing the main generation stream to proceed unblocked.

        Example:
            >>> graph = CircuitGraph(max_seq_len=8192, top_k=32, alpha=0.85,
            ...                      num_walkers=1024, num_steps=100)
            >>> # During generation:
            >>> graph.update_and_step_circuit(query, keys, current_idx)
            >>> # When you need scores:
            >>> scores = graph.get_scores()
        )pbdoc"
    )
    .def(py::init<int, int, float, int, int, int>(),
        py::arg("max_seq_len"),
        py::arg("top_k"),
        py::arg("alpha"),
        py::arg("num_walkers"),
        py::arg("num_steps"),
        py::arg("query_window") = 64,
        R"pbdoc(
        Initialize a CircuitGraph.

        Args:
            max_seq_len: Maximum sequence length to support.
            top_k: Number of neighbors per token in the sparse graph.
            alpha: (Unused by CircuitKV, kept for API compatibility)
            num_walkers: Number of parallel random walkers.
            num_steps: (Unused by CircuitKV, uses MAX_STEPS=100 internally)
            query_window: (Unused, kept for API compatibility)
        )pbdoc"
    )
    .def("update_and_step_circuit", &CircuitGraph::update_and_step_circuit,
        py::arg("query"),
        py::arg("keys"),
        py::arg("current_idx"),
        R"pbdoc(
        CircuitKV: Current-Flow Betweenness via Absorbing Random Walks.

        This is the core CircuitKV method. It simulates electrical current
        flowing from the Query (Source) to the Context Start (Sink), measuring
        "through-traffic" for each token.

        Key Properties:
        - NO RESTART: Walkers never teleport. They march from Source toward Sink.
        - ABSORBING BOUNDARY: Walks stop when reaching first 4 tokens (sink).
        - TRANSPORT METRIC: Visit counts measure current flow, not just popularity.

        This "rescues" bridge tokens: tokens that appear once but connect Question
        to Context. Unlike PPR, they accumulate current because they're on the
        path from Source to Sink.

        Physics Analogy:
        - Query = Battery positive terminal (Source)
        - Context Start (tokens 0-3) = Battery negative terminal (Sink)
        - Attention weights = Conductance (1/Resistance)
        - Visit counts = Current flow through each node

        CRITICAL: Walks ALONG A (source -> neighbor), NOT A^T!
        In causal attention, A[i,j] > 0 means token i attends to token j.
        The walker follows this direction: from query toward past tokens.

        Args:
            query: Query vector [1, head_dim] or [head_dim], FP16/FP32.
            keys: Key cache [seq_len, head_dim], FP16/FP32.
            current_idx: Index of the current token (source node).
        )pbdoc"
    )
    .def("get_scores", &CircuitGraph::get_scores,
        R"pbdoc(
        Get the current-flow importance scores.

        Synchronizes the sidecar stream before returning.

        Returns:
            Tensor of shape [max_seq_len] with visit counts (current-flow scores).
        )pbdoc"
    )
    .def("reset", &CircuitGraph::reset,
        R"pbdoc(
        Reset the graph for a new sequence.

        Clears the adjacency list and visit counts.
        )pbdoc"
    )
    .def("synchronize", &CircuitGraph::synchronize,
        R"pbdoc(
        Wait for all async operations to complete.
        )pbdoc"
    )
    .def("update_and_step_circuit_combined", &CircuitGraph::update_and_step_circuit_combined,
        py::arg("query"),
        py::arg("keys"),
        py::arg("current_idx"),
        py::arg("num_iterations") = 10,
        R"pbdoc(
        Spectral + Walker + MAX: Combined scoring for CircuitKV.

        This method computes importance scores using TWO complementary methods:
        1. SPECTRAL (Power Iteration): Captures "hub" tokens with global importance
        2. WALKER (Absorbing Random Walk): Captures "bridge" tokens on Q→Sink path

        Final scores use element-wise MAX to preserve BOTH types of important tokens:
            combined_score[i] = max(spectral[i], walker[i])

        This is particularly effective for summarization tasks where both global
        structure (spectral) and reasoning paths (walker) matter.

        Args:
            query: Query vector [1, head_dim] or [head_dim], FP16/FP32.
            keys: Key cache [seq_len, head_dim], FP16/FP32.
            current_idx: Index of the current token (source node for walker).
            num_iterations: Number of power iterations for spectral (default 10).
        )pbdoc"
    )
    .def("get_combined_scores", &CircuitGraph::get_combined_scores,
        R"pbdoc(
        Get the combined (Spectral + Walker + MAX) importance scores.

        Synchronizes the sidecar stream before returning.

        Returns:
            Tensor of shape [max_seq_len] with combined scores (normalized to [0, 1]).
        )pbdoc"
    );

    // Version info
    m.attr("__version__") = "0.2.0";
}

}  // namespace circuit_kv
