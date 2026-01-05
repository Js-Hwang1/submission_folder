# Project: CircuitKV

**Current State:** Fully implemented as CircuitKV (Absorbing Random Walks).
**Goal:** Beat H2O and GraphKV on LongBench using **Current-Flow Betweenness**.

---

## Core Business Logic (The "Why")

We are solving the **"Reasoning Bridge"** problem in KV-Cache Eviction.
* **H2O / SnapKV (Baseline):** Evicts based on **Popularity** (Degree). Fails on "Bridge Tokens" (tokens that appear once but connect Question to Context).
* **GraphKV / PageRank (Competitor):** Evicts based on **Diffusion**. Fails because "dye" gets stuck in dense clusters (distractors) or dilutes before reaching the bridge.
* **CircuitKV (Ours):** Evicts based on **Transport** (Current-Flow Betweenness).
    * **Logic:** We simulate electrical current flowing from the **Question (Source)** to the **Sink (Context Start)**.
    * **Result:** Any token that lies on the path from Question to Sink gets a high score, *even if it has low degree*. This "rescues" the Bridge tokens.

---

## Architecture & Implementation

**Directory Structure:**
```
CircuitKV/
├── circuit_kv/             # Python package
│   ├── __init__.py
│   ├── engine.py           # CircuitKVMonitor, CircuitKVConfig
│   └── utils.py
├── csrc/
│   ├── circuit_manager.cpp # Host-side orchestration
│   ├── circuit_manager.h   # CircuitGraph class
│   ├── pybind.cpp          # Python bindings
│   ├── compat.h
│   ├── include/
│   │   └── kernels.h       # Kernel launcher declarations
│   └── kernels/
│       ├── circuit_walker.cu  # Absorbing Random Walk kernel
│       ├── graph_update.cu    # Top-K neighbor selection
│       └── common.cuh         # PRNG and shared utilities
├── setup.py
└── pyproject.toml
```

**The "Causal" Physics:**
* **Attention Matrix ($A$):** Lower Triangular (Causal).
    * $A_{i,j} > 0$ means Token $i$ (Future/Query) attends to Token $j$ (Past/Key).
* **Walker Direction:**
    * We simulate **Current Flow**. Current flows along the wire.
    * Walker moves from **Query** → **Bridge** → **Sink**.
    * **Crucial:** Walk **ALONG** $A$ (from Source to Destination). **NEVER** use $A^T$ (Transpose), or the walker will get stuck at the last token (Dead End).

---

## Kernel Specifications (CUDA)

### 1. `csrc/kernels/circuit_walker.cu` (The Circuit Kernel)
This is an **Absorbing Random Walk** kernel.

**Logic:**
* **No Restart:** No `alpha` (teleport probability). The walker *never* restarts.
* **Absorbing Boundary:** The walk **stops immediately** if:
    * `current_node_index < SINK_SIZE` (default 4).
    * `step_count > MAX_STEPS` (default 100, safety break).
* **Traversal:**
    * Start at `source_node` (Last Token / Query position).
    * Sample neighbor based on attention weights ($A_{source, :}$).
    * Move to neighbor.
    * Repeat until Absorption.
* **Scoring:**
    * Atomic Add `+1` to `visits[node]` for every node visited.

### 2. `csrc/circuit_manager.cpp`
* `update_and_step_circuit()` is the main method.
* Clears `visits` buffer before each walk (Instantaneous Current measurement).
* Launches graph_update kernel then absorbing_walker kernel.

---

## Verification Strategy (The Oracle)

We trust **Oracle Verification** over standard benchmarks during development.
* **The Oracle:** The *actual* attention weights of the Query in the final layer.
* **Success Metric:**
    * If `CircuitKV_TopK` overlaps with `Oracle_TopK` significantly more than `H2O_TopK`, we win.
    * Specifically check for **Rare Tokens** (Low Degree) that have **High Oracle Attention**. CircuitKV *must* catch these.

---

## Anti-Patterns (What NOT to do)
1.  **Do not use $A^T$:** In a causal mask, the last token has in-degree 0. Transpose = Death.
2.  **Do not use Eigen-Solvers:** Too slow. We use Monte Carlo approximation (Random Walk).
3.  **Do not re-implement GraphKV:** We do not compute Cosine Similarity ($K^T K$). We use the Attention Matrix ($Q K^T$) directly. Zero overhead.
