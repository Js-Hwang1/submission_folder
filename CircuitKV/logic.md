# Part 1: The Business Logic & Rationale

### 1. The Core Philosophy

**SpectralKV** separates the *Generation* of language from the *Evaluation* of memory.

* **Current State (H2O):** "Eviction happens synchronously during generation. If I need to evict, I look at simple statistics (Accumulated Attention) and delete immediately."
* **SpectralKV State:** "Generation proceeds uninterrupted. A 'Shadow Process' (asynchronous monitor) constantly wanders through the memory graph to find structurally critical nodes, updating a 'Safe List' that the generator respects."

### 2. The Mathematical Formulation (The "PageRank" Hypothesis)

We treat the KV Cache as a **Directed Graph** $G=(V,E)$.

* **Nodes ($V$):** The tokens currently in the cache.
* **Edges ($E$):** The attention weights. If token $j$ (future) attends to token $i$ (past), there is a directed edge $j\rightarrow i$ with weight $w_{ji} \propto \text{Attention}(Q_j, K_i)$ .

**The Objective:**
We want to estimate the **Stationary Distribution** $\pi$ (PageRank) of this graph.

$\pi(i) =$ THe probability a random walker is at token $i$ after $t \to \infty$ steps.

* **H2O's Limit:** H2O calculates , $\Sigma \text{Attention}$ which is roughly equivalent to **Degree Centrality** (1-step connectivity).
* **SpectralKV's Gain:** PageRank captures **Eigenvector Centrality** (Global connectivity). A token might have low direct attention (low degree) but be the *only* path to a massive cluster of information (high eigenvector centrality). H2O kills this; SpectralKV keeps it.

### 3. The "Sparse Graph" Constraint

Calculating exact PageRank requires the full $N \times N$ attention matrix, which we cannot store.
**The SpectralKV Innovation:**

1. **Graph Caching:** When token $t$ is generated, we calculate its attention to the past. We **store only the indices** of the Top-$k$ (e.g.,$k=32$ ) tokens it attended to. This creates a sparse, static graph in memory (cheap: $4 \times \text{int}32$ per token).
2. **Random Walks:** The Shadow Kernel runs random walks on this *sparse index graph*, not the heavy KV vectors.

