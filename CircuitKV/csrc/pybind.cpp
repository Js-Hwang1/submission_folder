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
    .def("update_and_step_circuit_multi_source", &CircuitGraph::update_and_step_circuit_multi_source,
        py::arg("queries"),
        py::arg("keys"),
        py::arg("source_indices"),
        R"pbdoc(
        Multi-source CircuitKV: Run walks from W sources in parallel (P3 Optimization).

        This method runs absorbing random walks from ALL tokens in the observation
        window simultaneously, avoiding Python loop overhead. It's used during
        prefill initialization for efficient multi-source importance scoring.

        Algorithm:
        1. Build sparse graphs for all W sources in parallel (one CUDA block per source)
        2. Run walkers from all sources in a single kernel launch
        3. Aggregate visit counts across all walks (atomicAdd)

        This is ~W times faster than calling update_and_step_circuit in a Python loop.

        Args:
            queries: Query vectors for all sources [W, head_dim], FP32.
            keys: Key cache [seq_len, head_dim], FP32.
            source_indices: Token indices for each source [W], INT32.
        )pbdoc"
    )
    .def("update_and_step_circuit_bidirectional", &CircuitGraph::update_and_step_circuit_bidirectional,
        py::arg("queries"),
        py::arg("keys"),
        py::arg("source_indices"),
        R"pbdoc(
        Bidirectional CircuitKV (RC+B): Run walks in BOTH directions.

        This method runs absorbing walks in two directions simultaneously:
        - Backward: Query -> Sink (following attention edges A[i,j])
        - Forward: Sink -> Query (following transpose edges A^T[k,j])

        Tokens visited by BOTH directions are "true bridges" that connect the
        Question to the Context from both ends. These receive bonus scoring.

        Score Combination:
        - bridge = min(backward_visits, forward_visits)  # both directions visited
        - score = max(backward, forward) + 0.5 * bridge  # bonus for true bridges

        This significantly improves Recall@65 on LongBench benchmarks by capturing
        reasoning bridges that single-direction walks might miss.

        Args:
            queries: Query vectors for observation window [W, head_dim], FP32.
            keys: Key cache [seq_len, head_dim], FP32.
            source_indices: Query token indices [W], INT32.
        )pbdoc"
    )
    .def("get_bidirectional_scores", &CircuitGraph::get_bidirectional_scores,
        R"pbdoc(
        Get bidirectional scores (combined backward + forward + bridge bonus).

        Returns scores computed as:
            bridge = min(backward, forward)
            score = max(backward, forward) + 0.5 * bridge

        Synchronizes the sidecar stream before returning.

        Returns:
            Tensor of shape [max_seq_len] with combined bidirectional scores.
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
    );

    // Version info
    m.attr("__version__") = "0.1.0";
}

}  // namespace circuit_kv
