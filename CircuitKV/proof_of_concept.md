## Phase 1: Python Prototype (Proof of Concept)

**Goal:** Prove that PageRank eviction yields lower Perplexity than H2O. **Do not write CUDA yet.**

1. **Setup:**
* Load `Llama-2-7b-hf` or `Mistral-7B` using Hugging Face `transformers`.
* Use a dataset like **WikiText-2** or a sample from **LongBench**.


2. **The "God Mode" Run:**
* Run inference on a long sequence (e.g., 4k tokens).
* **Capture:** Hook into the model to save the full Attention Matrix  (this will be huge, do it on CPU/RAM if needed).


3. **The Analysis Script:**
* Construct a graph using `networkx`: `G = nx.from_numpy_array(A)`.
* Calculate PageRank: `scores = nx.pagerank(G)`.
* Calculate H2O scores: `scores = A.sum(axis=0)`.


4. **The Simulation:**
* Retrospectively "evict" tokens that had low PageRank.
* Compute the **Oracle Perplexity** of the remaining tokens.
* **Success Metric:** If `PPL(PageRank) < PPL(H2O)` at 20% budget, proceed to Phase 2.



## Phase 2: The "Graph Cache" Data Structure

You need a specific data structure to make the CUDA kernel fast. We avoid reading 4096-dim FP16 vectors in the walker.

**Structure: `SparseAttentionGraph**`
Allocated in GPU Global Memory (VRAM).

* **Capacity:** Fixed Budget (e.g., 4096 slots).
* **Format:** "Adjacency List" optimized for coalesced access.
```cpp
struct GraphNode {
    int neighbors[4]; // Indices of the 4 tokens this token attends to most
    float weights[4]; // Normalized probability for those 4 edges
};
// Array of size [Batch_Size, Max_Budget]
GraphNode* graph_cache; 

```



**Integration Point:**

* Inside the standard **FlashAttention/SDPA** kernel (or right after):
* Identify the Top-4 Softmax scores for the current query.
* Write these 4 indices into `graph_cache[current_token_idx]`.
* *Overhead:* Negligible (writing 16 bytes per step).



## Phase 3: The Shadow Walker Kernel (CUDA)

This is the core contribution. It runs asynchronously on a separate CUDA stream.

**Kernel Logic:**

1. **Initialization:** Launch 1024 threads ("Walkers").
2. **Spawn:** Each walker starts at a random node  (weighted by recentness, e.g., start at the last 100 tokens).
3. **The Walk Loop (10-20 steps):**
* **Read:** Thread  reads `graph_cache[u]`.
* **Sample:** Pick neighbor  from `neighbors[0...3]` using `weights` and a fast PRNG (XORWOW or PCG).
* **Update:** `atomicAdd(&visit_counts[v], 1)`.
* **Jump:** .


4. **Teleport:** With probability  (0.15), jump back to a "start node" (Standard PageRank damping factor).

**Code Skeleton (CUDA):**

```cpp
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define NEIGHBORS 4
#define WALK_STEPS 20

// The "Graph Cache" - Compressed representation of Attention
struct Node {
    int neighbors[NEIGHBORS];
    float probs[NEIGHBORS]; // Cumulative probability for sampling
};

__global__ void shadow_walker_kernel(
    const Node* __restrict__ graph, // The Sparse Graph
    int* __restrict__ global_counts, // The "Heatmap"
    int num_nodes,
    unsigned long long seed
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Fast local RNG state
    curandState state;
    curand_init(seed, tid, 0, &state);

    // 1. Initialize Walker
    // Start preferentially at recent tokens (e.g., last 10% of graph)
    int current_node = num_nodes - 1 - (curand(&state) % (num_nodes / 10 + 1));

    // 2. The Random Walk
    for (int step = 0; step < WALK_STEPS; step++) {
        // Fetch Node Data (Coalesced read if possible, but random access inevitable)
        Node n = graph[current_node];

        // 3. Sample Next Hop
        float roll = curand_uniform(&state);
        int next_node = -1;
        
        // Simple inverse transform sampling on pre-computed cumulative probs
        #pragma unroll
        for(int i=0; i<NEIGHBORS; i++) {
            if (roll < n.probs[i]) {
                next_node = n.neighbors[i];
                break;
            }
        }
        
        // Handle dead ends (sink nodes) -> Teleport to recent
        if (next_node == -1 || next_node >= num_nodes) {
            current_node = num_nodes - 1 - (curand(&state) % (num_nodes / 10 + 1));
            continue;
        }

        // 4. Record Visit (The "Vote")
        // Use atomicAdd to update global importance
        atomicAdd(&global_counts[next_node], 1);
        
        // Move
        current_node = next_node;
    }
}

```

## Phase 4: The Monitor & Eviction Policy

You need a host-side (C++ or Python) controller to use the `visit_counts`.

**Logic:**

1. **Frequency:** Run the Shadow Kernel every  generation steps.
2. **Normalization:** Read `visit_counts` from GPU to CPU (or keep on GPU for fast masking).
3. **The "Safety Mask":**
* Calculate Threshold  (e.g., top 20% of counts).
* Mark tokens below  as "Evictable."


4. **Lazy Eviction:**
* When the cache is effectively full, perform the memory compaction based on the Mask.
* *Critical:* Always keep a "Local Window" (last 50 tokens) protected, regardless of walker counts (Standard Sinkhorn/StreamingLLM logic).



---

# Novelty Checklist for ICML 2026

To ensure this gets accepted, your paper must emphasize these points:

1. **Asynchronous by Design:** "Unlike H2O, which pauses generation to sort scores, ShadowKV maintains a live 'importance heatmap' in the background."
2. **Structural vs. Greedy:** "We show cases where H2O fails (The 'Needle' problem) because the needle had low direct attention, but ShadowKV rescues it because it was structurally central."
3. **The "Graph Cache" Abstraction:** "We introduce the *Sparse Attention Graph Cache*, a lightweight () structure that decouples reasoning from heavy VRAM access."

### Next Step for You

Start **Phase 1** immediately.
Do you need a Python script template to extract the Attention Matrix from a huggingface model run? That is the first blocker usually.