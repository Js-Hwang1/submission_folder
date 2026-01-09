# Causal Influence Walker: Mathematical Foundation & Current Issues

## 1. Core Hypothesis

### The Problem: KV Cache Eviction
During LLM generation with long contexts, we must evict tokens from the KV cache due to memory constraints. The challenge: **which tokens to keep?**

### Existing Approaches

| Method | Metric | Limitation |
|--------|--------|------------|
| **H2O** | Column sum of attention (popularity) | Misses "bridge tokens" with low direct attention |
| **GraphKV/PageRank** | Stationary distribution | Diffusion gets stuck in dense clusters |

### Our Hypothesis: Causal Influence Propagation

**Core Idea:** A token $j$ is important if it lies on the *causal influence path* from the query to the context.

**Physics Analogy:**
- Query = Battery positive terminal (Source)
- Context Start (tokens 0-3) = Battery negative terminal (Sink)
- Attention weights = Conductance
- Visit counts = Current flow

**Key Insight:** Bridge tokens (low degree but critical for reasoning) will have HIGH current flow because they're on the path from Source to Sink, even if they have low direct attention.

---

## 2. Mathematical Formulation

### 2.1 Attention Matrix

Let $A \in \mathbb{R}^{n \times n}$ be the causal attention matrix:
$$A_{i,j} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d}}\right) \quad \text{for } j \leq i$$

Key property: $A$ is lower triangular (causal mask).

### 2.2 Random Walk on Attention Graph

**Walker dynamics:** Starting at position $s$ (query), at each step:
$$P(\text{move to } j \mid \text{at } i) = \frac{A_{i,j}^{1/T}}{\sum_{k \in \mathcal{V}} A_{i,k}^{1/T}}$$

where:
- $T$ = temperature (controls exploration vs. exploitation)
- $\mathcal{V}$ = valid positions (excludes sink: $\{4, 5, ..., i\}$)

**v1.0.7 Temperature Scaling:**
$$\tilde{A}_{i,j} = A_{i,j}^{1/T}$$

Effect of temperature:
- $T = 1$: Raw attention (concentrated)
- $T = 2$: Flattened distribution (more exploration)
- $T \to \infty$: Uniform random walk

### 2.3 Visit Count Scoring

For $N$ walkers, each taking $S$ steps:
$$\text{visits}[j] = \sum_{w=1}^{N} \sum_{s=0}^{S-1} \mathbb{1}[\text{walker } w \text{ visits } j \text{ at step } s]$$

### 2.4 Positional Opportunity Normalization

**Problem:** Early positions have inherent advantage (more walkers can reach them).

**Solution:** Normalize by "opportunity":
$$\text{adjusted}[j] = \frac{\text{visits}[j]}{\sqrt{n - j + 1}}$$

where $n$ is sequence length. The $\sqrt{\cdot}$ provides gentler correction.

### 2.5 Final Score

$$\text{score}[j] = \begin{cases}
1.0 & \text{if } j < \text{sink\_size} \quad \text{(always keep sink)} \\
\frac{\text{adjusted}[j]}{\max_k \text{adjusted}[k]} & \text{otherwise}
\end{cases}$$

---

## 3. Algorithm (v1.0.7)

```
INFLUENCE_WALKER(A, query_idx, num_walkers, max_steps, sink_size, temperature):
    visits[0..n-1] = 0

    FOR walker_id = 0 to num_walkers-1:
        # Multi-source initialization (v1.0.6)
        IF walker_id < num_walkers/2:
            pos = query_idx  # Query-start
        ELSE:
            pos = UNIFORM_RANDOM(sink_size, query_idx)  # Random-start

        FOR step = 0 to max_steps-1:
            IF pos < sink_size: BREAK  # Reached sink

            # Temperature-scaled sampling (v1.0.7)
            probs[j] = A[pos, j]^(1/T) for j in [sink_size, pos]
            probs = probs / sum(probs)  # Renormalize

            next_pos = SAMPLE(probs)
            visits[next_pos] += 1
            pos = next_pos

    # Positional normalization (v1.0.5)
    FOR j = sink_size to n-1:
        adjusted[j] = visits[j] / sqrt(n - j + 1)

    # Max normalization + sink protection (v1.0.7)
    scores[0..sink_size-1] = 1.0  # Always keep sink
    scores[sink_size..n-1] = adjusted / max(adjusted)

    RETURN scores
```

---

## 4. Version History & Fixes

| Version | Problem | Solution |
|---------|---------|----------|
| v1.0.0 | Weighted visits caused exponential decay | Unweighted visits |
| v1.0.1 | Path weights decayed to ~1e-10 | Removed path weighting entirely |
| v1.0.2 | Walkers absorbed at BOS (sink) immediately | Removed absorption, only count non-sink |
| v1.0.3 | Even without absorption, walkers spent time at BOS | Exclude sink from sampling entirely |
| v1.0.4 | Step 0 = direct attention (same as H2O) | Skip counting step 0 |
| v1.0.5 | Early positions had inherent advantage | Positional opportunity normalization |
| v1.0.6 | Middle positions unreachable from query | Multi-source walks (50% random start) |
| v1.0.7 | Attention concentration → walkers converge | Temperature-based exploration |
| v1.0.7+ | Sink tokens had score 0 → evicted first | Sink tokens forced to score 1.0 |

---

## 5. Empirical Observations (From Debug Logs)

### 5.1 Visit Distribution Problem

**Raw visit statistics (seq_len ≈ 7000):**
```
Position 4:        454,756 visits (91% of all visits)
Positions 5-14:    ~10,000 visits combined
Middle positions:  ~0 visits
```

**By quartile:**
```
Q1 (0-25%):   98.3% of visits   ← EXTREME concentration
Q2 (25-50%):   0.9% of visits
Q3 (50-75%):   0.4% of visits
Q4 (75-100%):  0.4% of visits
```

### 5.2 The Attention Concentration Problem

**Root cause:** LLM attention is highly concentrated:
- ~80-90% goes to BOS/early tokens (system prompt, template)
- ~10-15% goes to recent tokens (local attention window)
- ~1-5% goes to middle positions (few-shot examples, main content)

Even with multi-source walks, walkers follow attention and **converge to the same high-attention positions**.

### 5.3 Why This Fails for TREC (Few-Shot Classification)

**TREC task structure:**
```
[System Template] [Example 1] [Example 2] ... [Example N] [Query]
     positions        positions 100-2000                  position ~7000
      0-100           (FEW-SHOT EXAMPLES)
```

**The problem:**
1. Walkers start at query or random positions
2. They follow attention backward
3. Attention paths go to: Template (early) OR Recent tokens (late)
4. Middle positions (few-shot examples) are **never visited**
5. Few-shot examples get evicted → 0% accuracy

---

## 6. Fundamental Issues

### Issue 1: Attention ≠ Semantic Importance

**For few-shot tasks:**
- Few-shot examples are semantically critical (model needs them to classify)
- But during generation, attention to them is LOW
- They were important during ENCODING, not GENERATION

**Implication:** Any method based purely on generation-time attention will miss encoding-time important tokens.

### Issue 2: Walker Convergence (Rich Get Richer)

**The stationary distribution problem:**
- Random walks converge to stationary distribution
- Stationary distribution emphasizes "hubs" (high-degree nodes)
- This is fundamentally different from H2O's column sums
- Middle positions with moderate attention are starved

**H2O uses:** $\text{importance}[j] = \sum_i A_{i,j}$ (column sum)

**Walker approximates:** Stationary distribution $\pi$ where $\pi = \pi A$

These can be **very different** for the same attention matrix.

### Issue 3: Temperature Trade-off

**High temperature (T > 2):**
- Pro: Better exploration, walkers visit more positions
- Con: Loses attention signal, may keep irrelevant tokens

**Low temperature (T = 1):**
- Pro: Faithful to attention
- Con: Poor exploration, misses moderate-attention tokens

**No single temperature works for all tasks.**

### Issue 4: Positional Normalization Assumptions

**Current formula:** $\text{adjusted}[j] = \frac{\text{visits}[j]}{\sqrt{n - j + 1}}$

**Assumption:** Opportunity scales as $\sqrt{n - j + 1}$

**Reality:** The actual opportunity depends on:
- Attention distribution (task-dependent)
- Walker starting positions
- Number of steps

The normalization is approximate at best.

---

## 7. What We've Tried vs. What H2O Does

| Aspect | H2O | Influence Walker |
|--------|-----|------------------|
| **Metric** | Column sum (direct popularity) | Multi-hop paths (influence) |
| **Computation** | Sum during encoding | Random walks at generation |
| **Coverage** | All positions that received attention | Only positions on walk paths |
| **Bias** | Favors heavy hitters | Favors early + recent positions |
| **Few-shot** | Works (examples attended during encoding) | Fails (examples not on walk paths) |

**Key insight:** H2O captures **encoding-time** importance. Our walker captures **generation-time** importance. These are fundamentally different.

---

## 8. Potential Solutions (Not Yet Implemented)

### Option A: Hybrid Approach (Rejected)
Combine walker scores with H2O:
$$\text{final}[j] = \max(\text{walker}[j], \text{h2o}[j])$$

**Rejected because:** Degrades novelty of the work.

### Option B: Encoding-Time Walker
Run walker on cumulative attention from encoding phase instead of generation-time attention.

**Challenge:** Requires architectural changes to collect encoding attention.

### Option C: Reverse Walks
Walk from sink to query (using $A^T$) to find positions that "feed into" the query.

**Challenge:** $A^T$ has issues with causal mask (last token has zero in-degree).

### Option D: Adaptive Temperature
Dynamically adjust temperature based on attention distribution.

**Challenge:** No clear heuristic for choosing temperature.

### Option E: Stratified Sampling
Divide sequence into regions, ensure minimum retention from each region.

**Challenge:** Structural/heuristic, not principled.

---

## 9. Current Code Structure

```
CircuitKV/csrc/kernels/influence_walker.cu
├── influence_walker_kernel()           # Main walker kernel
│   ├── Multi-source initialization     # v1.0.6
│   ├── Temperature-scaled sampling     # v1.0.7
│   └── Unweighted visit counting       # v1.0.1
├── influence_positional_adjust_kernel() # v1.0.5
├── influence_find_adjusted_max_kernel() # Excludes sink
└── influence_normalize_by_max_kernel()  # v1.0.7: sink → 1.0
```

**Key parameters:**
- `num_walkers = 10,000`
- `max_steps = 10`
- `sink_size = 4`
- `temperature = 2.0`

---

## 10. Open Questions

1. **Is generation-time attention the right signal for KV cache eviction?**
   - Few-shot examples are important during encoding, not generation
   - Should we use cumulative attention instead?

2. **How to balance exploration vs. exploitation?**
   - High temperature loses attention signal
   - Low temperature causes convergence
   - Is there an optimal temperature?

3. **Is random walk the right algorithm?**
   - Converges to stationary distribution (not what we want)
   - Maybe betweenness centrality is better? (computationally expensive)

4. **Can we do better than positional normalization?**
   - Current sqrt normalization is a heuristic
   - What's the principled correction for opportunity bias?

---

## 11. Summary

**What we're trying to do:** Find tokens on the "causal influence path" from query to context using random walks on the attention graph.

**Why it should work:** Bridge tokens (critical for reasoning) should have high current flow even if they have low direct attention.

**Why it's failing:**
1. Attention is too concentrated → walkers converge to early/late positions
2. Middle positions (few-shot examples) are never visited
3. Generation-time attention ≠ encoding-time importance

**Current state:** v1.0.7 with temperature-based exploration and sink protection. TREC still fails (0% accuracy) because the fundamental issue is that few-shot examples don't receive attention during generation.
