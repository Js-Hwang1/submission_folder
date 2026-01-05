# **Instruction: Initialize "SpectralKV" CUDA System**

**Context:**
We are building **SpectralKV**, a novel KV-cache eviction system for LLMs that uses **Personalized PageRank (PPR)** to identify structurally important tokens.
We have validated the math in Python (PPR beats H2O at 25% budget). Now we must build the **high-performance CUDA implementation** to make it deployable.

**Core System Philosophy:**

1. **Asynchronous by Design:** The PPR calculation must run on a sidecar CUDA stream, never blocking the main `decoding` stream.
2. **Sparse Graph:** We do not store the full Attention Matrix ($N^2$). We store a **Sparse Adjacency List** (Top-K neighbors per token) in VRAM.
3. **Fused Kernels:** We need custom kernels for "Graph Update" (finding neighbors) and "Random Walk" (calculating scores).

---

## **Part 1: The File System Structure**

Please initialize the project with the following directory structure. This follows the standard **PyTorch C++ Extension** pattern.

```text
spectral_kv/
├── CLAUDE.md                  # (Existing guidelines)
├── setup.py                   # Build script for C++/CUDA extensions
├── pyproject.toml             # Build system requirements
├── requirements.txt
│
├── spectral_kv/               # Python package
│   ├── __init__.py            # Exposes the C++ extension
│   ├── engine.py              # High-level Python wrapper (The "Monitor")
│   └── utils.py
│
└── csrc/                      # C++ and CUDA Source Code
    ├── compat.h               # CUDA compatibility helpers
    ├── pybind.cpp             # Python bindings (pybind11)
    │
    ├── graph_manager.h        # C++ Class managing the Graph & Walkers
    ├── graph_manager.cpp
    │
    ├── kernels/
    │   ├── graph_update.cu    # Kernel 1: Updates Adjacency List (Top-K)
    │   ├── ppr_walker.cu      # Kernel 2: Runs Random Walks
    │   └── common.cuh         # Shared CUDA device functions (PRNG, etc.)
    │
    └── include/
        └── kernels.h          # Header exposing kernel launchers

```

---

## **Part 2: Kernel Specifications (The "Business Logic")**

We need to implement two specific kernels. Please write the `.cu` and `.cuh` files based on these specs.

### **Kernel 1: `update_graph_kernel` (The Graph Builder)**

* **Trigger:** Runs once per decoding step (when a new token  is generated).
* **Input:**
* `Query Vector` ($1 \times D$): The query of the new token.
* `Key Cache` ($N\times D$): The keys of all past tokens.


* **Logic:**
1. Compute Dot Product (Attention Score) between Query and all $N$ Keys.
2. Perform a **Fused Top-K Selection** (find the indices of the $K=32$ highest scores). *Hint: Use Warp-level primitives or Block-level reduction for speed.*
3. Write these  indices into the `AdjacencyList` at row $T$.


* **Output:** Updates the `AdjacencyList` in VRAM.

### **Kernel 2: `ppr_walker_kernel` (The Shadow Walker)**

* **Trigger:** Runs immediately after Graph Update (on the side stream).
* **Input:**
* `AdjacencyList` ($N\times K$): The sparse graph.
* `Start_Node`: The current token index $T$ (Personalization target).
* `Alpha` (0.85): Teleport probability.


* **Logic (Monte Carlo):**
1. Launch $W=1024$ threads (Walkers).
2. All walkers start at `Start_Node`.
3. **Loop** for $S=20$ steps:
* Generate random number $r \in [0,1]$.
* **If** $r<\alpha$: Teleport back to `Start_Node`.
* **Else**: Read `AdjacencyList[current_pos]`. Pick a random neighbor (0 to $K-1$). Move there.
* Atomic Add `+1` to `VisitCounts[current_pos]`.




* **Output:** Updates `VisitCounts` array (which represents the PageRank scores).

---

## **Part 3: Implementation Requirements (Strict Constraints)**

1. **Memory Coalescing:** The `AdjacencyList` must be stored in a way that allows coalesced reads. Prefer `[N, K]` layout where $K$ is small (32).
2. **Random Number Generation:** Use `curand_kernel`. Initialize the RNG states *once* during setup, do not re-initialize every step.
3. **Bindings:**
* Expose a class `SpectralGraph` to Python.
* Method: `update_and_step(query, keys, current_idx)` -> Returns nothing (updates internal state).
* Method: `get_scores()` -> Returns Tensor of scores.


4. **Hardware Awareness:** Support FP16 (`half`) for the Query/Key inputs, but use `float` for internal accumulators.

---

## **Step-by-Step Execution Plan**

**Task 1:** Create the file structure and `setup.py`. Ensure it compiles a "Hello World" CUDA extension.
**Task 2:** Implement `common.cuh` with the PRNG setup and `AdjacencyList` data structure definition.
**Task 3:** Implement the `update_graph_kernel` (Dot Product + TopK).
**Task 4:** Implement the `ppr_walker_kernel`.
**Task 5:** Bind it all in `graph_manager.cpp` and `pybind.cpp`.

**Immediate Action:**
Start with **Task 1 & 2**. Generate the project skeleton and the `common.cuh` file defining our Graph structure.