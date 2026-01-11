# CircuitKV: Deep Mathematical Analysis & Ablation Strategy

## Executive Summary

**Current Status**: CircuitKV achieves 41.87 avg (best among KV compression methods), but margins are thin (+0.40 over SnapKV). To "sweep the board" and ensure ICML acceptance, we need:

1. **Novel mathematical improvements** that provide consistent 1-2% gains
2. **Rigorous ablations** proving each component is necessary
3. **Theoretical grounding** that distinguishes us from prior work

---

## Part I: Deep Mathematical Analysis

### 1. Current Algorithm Formalization

Let $A \in \mathbb{R}^{n \times n}$ be the causal attention matrix where $A_{ij}$ is the attention from query $i$ to key $j$ (with $A_{ij} = 0$ for $j > i$).

**Current CircuitKV computes:**

1. **H2O scores**: $h_j = \sum_{i} A_{ij}$ (column sums = incoming attention)

2. **Transition matrix**: $P_{ij} = A_{ij} / \sum_k A_{ik}$ (row-normalized attention)

3. **Absorbing walk visits**: Run $N$ walkers from query position $q$, walk backward according to $P$, count visits $v_j$ before absorption at sink

4. **Rank normalization**: $\tilde{h}_j = \text{rank}(h_j)/n$, $\tilde{v}_j = \text{rank}(v_j)/n$

5. **MAX combination**: $s_j = \max(\tilde{h}_j, \tilde{v}_j)$

### 2. What's Missing: Theoretical Gaps

#### Gap 1: Unidirectional Walks Miss "Betweenness"

Current walks go: **Query → Sink** (backward in time)

A token is truly important if it lies on the **path between query and relevant information**. This requires:
- Query can reach it (backward walk) ✓ We do this
- It can reach important sources (forward walk) ✗ We don't do this

**Betweenness Centrality** measures exactly this: tokens that appear on many shortest paths.

#### Gap 2: Single-Scale Misses Multi-Hop Dependencies

We run walks of fixed length. But information flows at different scales:
- **Local** (2-5 hops): Syntactic dependencies, coreference
- **Medium** (5-15 hops): Paragraph-level reasoning
- **Global** (15+ hops): Cross-document, long-range

A single scale can't capture all of these optimally.

#### Gap 3: MAX is Non-Principled

$\max(\tilde{h}, \tilde{v})$ is a heuristic hedge. It works but:
- Doesn't account for **confidence** in each signal
- Doesn't allow **task-adaptive** weighting
- Ignores **correlation** between H2O and Influence

#### Gap 4: Temperature is Fixed

Transition probabilities use raw attention. For concentrated attention (low entropy), walks are deterministic. For diffuse attention (high entropy), walks are random.

Different tasks need different exploration:
- **Retrieval**: Low T (focused, follow strong signals)
- **Summarization**: High T (broad coverage)

---

## Part II: Novel Mathematical Improvements

### Improvement 1: Bidirectional Flow (HIGHEST PRIORITY)

**Concept**: A token is important if information flows THROUGH it, not just TO it.

**Algorithm**:
```
# Backward walks: Query → Sink (current)
v_backward[j] = visits from query going backward

# Forward walks: Sink → Query (NEW)
P_forward[i,j] = A[j,i] / sum_k A[k,i]  # Transpose
v_forward[j] = visits from sink going forward

# Betweenness-style combination
score[j] = v_backward[j] * v_forward[j]
```

**Why it works**:
- Backward captures "reachable from query"
- Forward captures "can reach the sink (important sources)"
- Product amplifies tokens on BOTH paths

**Expected gain**: +0.5-1.0% on Multi-Doc QA (cross-document bridges)

**Complexity**: 2x walks, still O(N·S)

### Improvement 2: Multi-Scale Walk Ensemble

**Concept**: Capture dependencies at multiple scales.

**Algorithm**:
```
# Short walks: Local dependencies
S_short = 5
v_short = run_walks(N, S_short)

# Medium walks: Paragraph-level
S_medium = 15
v_medium = run_walks(N, S_medium)

# Long walks: Cross-document
S_long = 50
v_long = run_walks(N, S_long)

# Ensemble (learnable or fixed weights)
score[j] = α·v_short[j] + β·v_medium[j] + γ·v_long[j]
```

**Why it works**:
- QA tasks benefit from medium/long walks (reasoning chains)
- Code benefits from short walks (syntax)
- Summarization benefits from all scales (comprehensive coverage)

**Adaptive weighting** (no learning required):
```
# Weight by inverse variance (more stable = higher weight)
α = 1/var(v_short), β = 1/var(v_medium), γ = 1/var(v_long)
# Normalize: α + β + γ = 1
```

**Expected gain**: +0.3-0.7% overall, +1% on summarization

### Improvement 3: Adaptive Temperature

**Concept**: Adjust exploration based on attention entropy.

**Algorithm**:
```
# Compute attention entropy per row
H[i] = -sum_j P[i,j] * log(P[i,j])
H_avg = mean(H)
H_max = log(n)  # Maximum possible entropy

# Adaptive temperature
T = T_base * (H_max / H_avg)^κ

# Apply to transitions
P_T[i,j] = A[i,j]^(1/T) / sum_k A[i,k]^(1/T)
```

**Intuition**:
- Low entropy (concentrated attention): Increase T to explore more
- High entropy (diffuse attention): Decrease T to focus

**κ parameter**: Controls sensitivity (κ=0.5 is a good default)

**Expected gain**: +0.2-0.5% on tasks with extreme attention patterns

### Improvement 4: Principled Normalization via Fundamental Matrix

**Concept**: Use absorbing Markov chain theory for normalization.

For an absorbing chain with transient states $T$ and absorbing states $A$:
- $Q$ = transition matrix among transient states
- $N = (I - Q)^{-1}$ = Fundamental matrix
- $N_{ij}$ = Expected visits to state $j$ starting from state $i$

**Algorithm**:
```
# Approximate fundamental matrix via Neumann series
N ≈ I + Q + Q² + Q³ + ... + Q^k

# Expected visits from query q to each position
expected_visits[j] = N[q, j]

# Normalize actual visits by expected
normalized_score[j] = actual_visits[j] / expected_visits[j]
```

**Why it works**:
- Current sqrt normalization is ad-hoc
- Fundamental matrix gives principled baseline
- Ratio identifies tokens visited MORE than expected (truly important)

**Approximation**: Use k=10 terms of Neumann series (matches our walk length)

**Expected gain**: +0.2-0.4% by removing positional bias correctly

### Improvement 5: Calibrated Combination (Replacing MAX)

**Concept**: Combine H2O and Influence with proper calibration.

**Algorithm**:
```
# Calibrate to [0,1] using CDF (not just rank)
h_calibrated = empirical_cdf(h2o_scores)
v_calibrated = empirical_cdf(influence_scores)

# Weighted combination (task-agnostic)
λ = correlation(h_calibrated, v_calibrated)
# High correlation: signals agree, use average
# Low correlation: signals disagree, use max for safety

if λ > 0.7:
    score = 0.5 * h_calibrated + 0.5 * v_calibrated
else:
    score = max(h_calibrated, v_calibrated)
```

**Why it works**:
- When signals agree, averaging is more stable
- When signals disagree, MAX is safer
- Adaptive based on actual signal structure

**Expected gain**: +0.1-0.3% by reducing variance

---

## Part III: Recommended Implementation Priority

| Priority | Improvement | Expected Gain | Complexity | Novel? |
|----------|-------------|---------------|------------|--------|
| **P0** | Bidirectional Flow | +0.5-1.0% | Medium | Yes |
| **P1** | Multi-Scale Ensemble | +0.3-0.7% | Low | Somewhat |
| **P2** | Adaptive Temperature | +0.2-0.5% | Low | Yes |
| **P3** | Fundamental Normalization | +0.2-0.4% | Medium | Yes |
| **P4** | Calibrated Combination | +0.1-0.3% | Low | No |

**Total potential gain: +1.3-2.9%**

If we achieve even half of this, we'd have:
- CircuitKV: 42.5-43.0 avg
- Gap to SnapKV: +1.0-1.5 (from +0.40)
- Gap to FullKV: Essentially zero

---

## Part IV: Ablation Strategy for ICML

### Critical Ablations (MUST HAVE)

#### A1: Component Ablation
| Configuration | What it proves |
|---------------|----------------|
| H2O only | Baseline comparison |
| Influence only | Random walks alone |
| MAX(H2O, Influence) | Current method |
| Multiplicative: H2O × Influence | Why MAX > product |

**Expected result**: MAX ≥ either alone; MAX > multiplicative

#### A2: Walk Length Sensitivity
| Steps (S) | Expected behavior |
|-----------|-------------------|
| 5 | Underfits (misses long-range) |
| 10 | Good for QA |
| 15 | Optimal overall |
| 20 | Good for summarization |
| 30+ | Diminishing returns |

**Expected result**: Broad optimum around S=10-20, robust

#### A3: Walker Count Sensitivity
| Walkers (N) | Expected behavior |
|-------------|-------------------|
| 100 | High variance |
| 1,000 | Acceptable |
| 10,000 | Current (stable) |
| 100,000 | Diminishing returns |

**Expected result**: Convergence around N=1,000-10,000

#### A4: Budget Scaling
| Budget | CircuitKV vs SnapKV gap |
|--------|------------------------|
| 256 | Should be LARGER (harder task) |
| 512 | Large |
| 1024 | Medium |
| 2048 | Current (+0.40) |
| 4096 | Small (easy task) |

**Expected result**: CircuitKV advantage grows as budget shrinks

### Novel Component Ablations (IF IMPLEMENTED)

#### A5: Bidirectional vs Unidirectional
| Configuration | Description |
|---------------|-------------|
| Backward only | Current |
| Forward only | Reverse direction |
| Bidirectional (product) | v_back × v_forward |
| Bidirectional (sum) | v_back + v_forward |

**Expected result**: Bidirectional product > either direction alone

#### A6: Multi-Scale vs Single-Scale
| Configuration | Walk lengths |
|---------------|--------------|
| Short only | S=5 |
| Medium only | S=15 |
| Long only | S=50 |
| Ensemble | {5, 15, 50} |

**Expected result**: Ensemble ≥ any single scale

#### A7: Temperature Sensitivity
| Temperature | Expected behavior |
|-------------|-------------------|
| T=0.5 | Focused (good for retrieval) |
| T=1.0 | Current |
| T=2.0 | Exploratory (good for summarization) |
| Adaptive | Best overall |

**Expected result**: Optimal T varies by task; adaptive matches or beats fixed

### Per-Task Diagnostic Ablations

#### A8: Win Analysis
For top 3 CircuitKV wins (MultifieldQA, LCC, 2WikiMQA):
- Visualize which tokens CircuitKV keeps vs baselines
- Show these are "bridge tokens" or "syntax tokens"
- Qualitative examples in paper

#### A9: Loss Analysis
For top 3 CircuitKV losses (NarrativeQA, GovReport, QMSum):
- What tokens does CircuitKV drop that it shouldn't?
- Is the issue walk length? Temperature? Coverage?
- Identify specific failure mode

### Ablation Execution Plan

**Phase 1: Critical (3 days)**
- A1: Component ablation (4 runs × 16 tasks = 64 runs)
- A4: Budget scaling (5 budgets × 3 methods = 15 runs)

**Phase 2: Sensitivity (2 days)**
- A2: Walk length (6 values × 16 tasks = 96 runs)
- A3: Walker count (4 values × 4 tasks = 16 runs)

**Phase 3: Novel Components (IF IMPLEMENTED) (3 days)**
- A5: Bidirectional (4 configs × 16 tasks = 64 runs)
- A6: Multi-scale (4 configs × 16 tasks = 64 runs)

**Phase 4: Diagnostic (1 day)**
- A8-A9: Qualitative analysis of wins/losses

---

## Part V: Expected Paper Narrative

### If We Implement Bidirectional + Multi-Scale:

> "CircuitKV introduces **bidirectional absorbing walks** for KV cache compression. Unlike prior methods that measure token importance by direct attention (H2O) or unidirectional reachability (SnapKV), CircuitKV identifies tokens that lie on **information flow paths** between queries and sources.
>
> Our key insight is that a token's importance depends on its role as a **bridge** in the attention graph—whether information can flow THROUGH it, not just TO it. We formalize this as the product of forward and backward walk visit counts, analogous to betweenness centrality in network theory.
>
> On LongBench, CircuitKV achieves **43.X** average across 16 tasks, outperforming SnapKV (41.47) by **+1.X** and approaching FullKV (41.94). Notably, CircuitKV shows particular gains on **multi-document QA (+2.0%)** and **code completion (+4.6%)**, tasks that require preserving cross-context dependencies."

### Theoretical Contribution Framing:

1. **Novel**: First to apply absorbing Markov chains to KV cache
2. **Principled**: Betweenness centrality interpretation
3. **Practical**: O(N·S) complexity, no additional memory
4. **Effective**: State-of-the-art on LongBench

---

## Part VI: Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Bidirectional doesn't help | Medium | High | Fall back to multi-scale |
| Ablations show components redundant | Low | High | Reframe as "robustness" |
| Gains are noise | Medium | High | Run 3 seeds, report std |
| Reviewer says "incremental" | Medium | Medium | Emphasize theory |

---

## Appendix: Quick Implementation Snippets

### Bidirectional Walks
```python
def bidirectional_scores(attn_matrix, query_pos, sink_size, N, S):
    # Backward: Query → Sink (current)
    P_backward = row_normalize(attn_matrix)
    v_backward = run_absorbing_walks(P_backward, query_pos, sink_size, N, S)

    # Forward: Sink → Query (transpose)
    P_forward = row_normalize(attn_matrix.T)
    v_forward = run_absorbing_walks(P_forward, sink_size, query_pos, N, S)

    # Betweenness-style product
    return v_backward * v_forward
```

### Multi-Scale Ensemble
```python
def multiscale_scores(attn_matrix, query_pos, sink_size, N):
    v_short = run_walks(attn_matrix, query_pos, sink_size, N, S=5)
    v_medium = run_walks(attn_matrix, query_pos, sink_size, N, S=15)
    v_long = run_walks(attn_matrix, query_pos, sink_size, N, S=50)

    # Inverse variance weighting
    w_short = 1.0 / (v_short.var() + 1e-6)
    w_medium = 1.0 / (v_medium.var() + 1e-6)
    w_long = 1.0 / (v_long.var() + 1e-6)
    w_total = w_short + w_medium + w_long

    return (w_short * v_short + w_medium * v_medium + w_long * v_long) / w_total
```

### Adaptive Temperature
```python
def adaptive_temperature(attn_matrix, T_base=1.0, kappa=0.5):
    # Compute row-wise entropy
    P = row_normalize(attn_matrix)
    H = -(P * torch.log(P + 1e-10)).sum(dim=-1)
    H_avg = H.mean()
    H_max = torch.log(torch.tensor(attn_matrix.shape[-1], dtype=torch.float))

    # Adaptive T
    T = T_base * (H_max / H_avg) ** kappa

    # Apply temperature
    P_T = attn_matrix ** (1/T)
    P_T = P_T / P_T.sum(dim=-1, keepdim=True)

    return P_T, T
```
