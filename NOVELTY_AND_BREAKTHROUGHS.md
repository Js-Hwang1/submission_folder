# CircuitKV: Novelty Analysis & Proposed Breakthroughs

**Document Version**: 1.0.0
**Last Updated**: 2026-01-11
**Status**: Ready for Implementation
**Target Venue**: ICML 2026

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Prior Work Analysis](#prior-work-analysis)
3. [CircuitKV's Novel Contribution](#circuitkvs-novel-contribution)
4. [Current Performance Issues](#current-performance-issues)
5. [Proposed Breakthroughs](#proposed-breakthroughs)
6. [Implementation Priority](#implementation-priority)
7. [Paper Framing Recommendations](#paper-framing-recommendations)

---

## Executive Summary

### The Core Novelty Claim

**CircuitKV uses ABSORBING random walks (transient visit counts with termination at sink) rather than STEADY-STATE distributions (TokenRank/PageRank). This measures current-flow betweenness—how many times a walker passes THROUGH a token on its path from query to context-start—which is fundamentally different from equilibrium importance.**

### Key Differentiation

| Approach | What It Measures | CircuitKV Difference |
|----------|------------------|---------------------|
| **TokenRank** | Steady-state probability (eigenvalue) | We use **transient** visits before absorption |
| **PageRank** | Equilibrium with teleport (restart) | We have **no restart**, walkers terminate at sink |
| **H2O** | Cumulative attention (1-hop) | We capture **multi-hop paths** |
| **Attention Flow** | Max-flow for interpretation | We optimize for **KV cache compression** |

### Current Performance (20 samples per task)

| Task Category | Working | Struggling | Broken |
|--------------|---------|------------|--------|
| Document QA | qasper (48.2), multifieldqa (39.0) | narrativeqa (29.8) | - |
| Multi-Doc QA | - | hotpotqa (31.1), 2wikimqa (36.8) | musique (17.8) |
| Summarization | - | gov_report (26.9), multi_news (26.3) | - |
| Few-shot | triviaqa (54.2) | - | **TREC (0.0)**, samsum (5.5) |
| Synthetic | - | - | passage_count (1.7), passage_retrieval (5.0) |
| Code | - | lcc (16.7), repobench-p (24.9) | - |

---

## Prior Work Analysis

### Critical Papers Reviewed

#### 1. TokenRank (arXiv:2507.17657, 2025)
- **URL**: https://arxiv.org/abs/2507.17657
- **Method**: Steady-state vector of attention Markov chain
- **Key Quote**: "TokenRank is defined as the steady state vector of the Markov chain"
- **Difference from CircuitKV**: TokenRank uses eigenvalue (equilibrium), we use transient visits with absorption
- **Threat Level**: MEDIUM - Must directly compare and differentiate

#### 2. Attention Flow (ACL 2020)
- **URL**: https://aclanthology.org/2020.acl-main.385/
- **Method**: Max-flow algorithm on attention DAG for interpretability
- **Difference from CircuitKV**: For post-hoc interpretation, not real-time compression
- **Threat Level**: LOW - Different use case

#### 3. PyramidKV (arXiv:2406.02069, 2024)
- **URL**: https://arxiv.org/abs/2406.02069
- **Method**: Layer-adaptive KV cache budget allocation
- **Difference from CircuitKV**: Budget allocation (orthogonal), not token scoring
- **Threat Level**: LOW - Orthogonal contribution, can combine

#### 4. Attention with Markov (arXiv:2402.04161, 2024)
- **URL**: https://arxiv.org/abs/2402.04161
- **Method**: Framework connecting self-attention to Markov chains
- **Difference from CircuitKV**: Theoretical framework, doesn't address KV cache
- **Threat Level**: LOW - Should cite as theoretical foundation

#### 5. FastGAT (arXiv:2006.08796, 2020)
- **URL**: https://arxiv.org/abs/2006.08796
- **Method**: Uses effective resistance for graph sparsification
- **Difference from CircuitKV**: For GNN acceleration, not transformer KV cache
- **Threat Level**: LOW - Different domain

### Novelty Verification Matrix

| CircuitKV Element | Prior Work? | Novel? | Evidence |
|-------------------|-------------|--------|----------|
| Absorbing walks for KV cache | None found | **YES** | Searched "absorbing Markov chain KV cache" - no results |
| Sink = attention sinks formalization | StreamingLLM (implicit) | **YES** | We formalize why sinks matter mathematically |
| Current-flow betweenness for tokens | Graph theory (not NLP) | **YES** | Novel application to transformer attention |
| Multi-source landmark walks | Multi-source PPR exists | **PARTIAL** | Soft-absorption at landmarks may be novel |
| MAX(H2O, Influence) hedging | Ensemble methods exist | **NO** | Engineering choice, not contribution |
| sqrt(n-p) normalization | None | **NO** | Heuristic without theoretical justification |

---

## CircuitKV's Novel Contribution

### Mathematical Distinction: Absorbing vs Ergodic Chains

**Ergodic (TokenRank, PageRank):**
```
π = lim_{t→∞} P^t · start
Measures: "Where does a walker END UP at equilibrium?"
Property: Unique stationary distribution exists
```

**Absorbing (CircuitKV):**
```
visits[j] = Σ_{t=0}^{T_absorb} 1[walker at j at time t]
Measures: "How many times does walker PASS THROUGH j before stopping?"
Property: Transient states have finite expected visits
```

### The Fundamental Matrix Connection

For an absorbing Markov chain:
```
P = | Q  R |    where Q = transient→transient, R = transient→absorbing
    | 0  I |

Fundamental Matrix: N = (I - Q)^{-1}
N[i,j] = Expected number of visits to transient state j, starting from i
```

**This is what CircuitKV approximates via Monte Carlo walks.**

### Physics Analogy: Electrical Networks

```
Source (Battery +) = Query token (last position)
Sink (Battery -)   = Attention sinks (first 4 tokens)
Edge conductance   = Attention weights
Visit counts       = Current flow through each node

Key Insight: "Bridge tokens" have high current flow even if they have
low direct attention, because they're on the path from source to sink.
```

---

## Current Performance Issues

### Issue 1: Walker Sink Collapse (97% Q1 Concentration)

**Observation** (from debug logs):
```
Visit distribution by quartile:
  Q1 (early):    ~489,000 visits (97.8%)
  Q2-Q4:         ~11,000 visits  (2.2%)
```

**Root Cause**: Early positions accumulate visits because every walker must pass through them to reach the sink.

**Current Mitigation**: `sqrt(seq_len - p)` normalization - but this is heuristically motivated, not principled.

### Issue 2: 15-31% Tokens Never Visited

**Observation**: Many tokens receive zero visits, meaning only H2O scores matter for them.

**Impact**: The random walk component adds nothing for these tokens.

### Issue 3: TREC = 0.0 (Classification Failure)

**Task**: 50-class question type classification with few-shot examples.

**Observation**:
- Top-10 attended tokens: ALWAYS kept (10/10)
- Budget generous: 2048 of ~5000 tokens kept
- Query attention entropy: moderate (42-65%)

**Root Cause Hypothesis**: Few-shot instruction patterns ("Question: X\nType: Y") are being evicted despite having critical structural importance. The model loses the classification template.

### Issue 4: Synthetic Tasks Fail (passage_count, passage_retrieval)

**Observation**: These tasks require global context scanning, not local retrieval.

**Hypothesis**: The query→sink flow direction biases toward early tokens, missing tokens that need to be compared globally.

---

## Proposed Breakthroughs

### Breakthrough 1: Fundamental Matrix Normalization (Priority: CRITICAL)

**Problem**: Current `sqrt(n-p)` normalization is ad-hoc. Reviewers will ask "why sqrt?"

**Solution**: Use proper absorbing chain theory.

**Mathematical Formulation**:
```
Expected visits: E[visits[j] | start at i] = N[i,j] where N = (I - Q)^{-1}

Proper normalization:
  normalized[j] = observed_visits[j] / Σ_i start[i] · N[i,j]
```

**Efficient Approximation** (avoid O(n³)):
```
N ≈ I + Q + Q² + ... + Q^k    (truncated Neumann series)

Estimation via k-step walks:
  (Q^k)[i,j] ≈ (1/M) · Σ_m 1[walk from i reaches j in exactly k steps]
```

**Implementation**:
1. Run short probe walks (k=1,2,3,4,5) to estimate Q^k terms
2. Sum to get N̂ ≈ I + Q̂ + Q̂² + ... + Q̂^5
3. Normalize: score[j] = visits[j] / N̂[start,j]

**Expected Impact**: Fixes 97% Q1 concentration by accounting for expected visits.

**Novelty Status**: ✅ NOVEL - Fundamental matrix applied to KV cache is new.

---

### Breakthrough 2: Instruction-Anchored Absorbing Boundaries (Priority: CRITICAL)

**Problem**: TREC = 0.0 because few-shot instruction patterns are evicted.

**Solution**: Detect and protect instruction-anchoring tokens by expanding the absorbing boundary.

**Algorithm**:
```python
def detect_instruction_anchors(tokens, attention, threshold=0.1):
    """Detect tokens that anchor instruction patterns."""
    anchors = set()

    # Pattern tokens to protect
    pattern_tokens = {"Type", ":", "Question", "Answer", "\n", "Example"}

    for j, token in enumerate(tokens):
        if token in pattern_tokens:
            # Check if high self-attention (structural importance)
            if attention[j, j] > threshold:
                anchors.add(j)

    return anchors

# Extended absorbing boundary
sink = {0, 1, 2, 3}  # Original
anchors = detect_instruction_anchors(tokens, attention)
absorbing_states = sink.union(anchors)
```

**Mathematical Framing**:
```
Standard: Absorbing states = {context-start tokens}
Extended: Absorbing states = {context-start} ∪ {instruction anchors}
```

**Expected Impact**: TREC 0.0 → 30-50% (comparable to H2O baseline).

**Novelty Status**: ✅ NOVEL - Adaptive absorbing boundary based on task structure.

---

### Breakthrough 3: Multi-Horizon Walk Ensemble (Priority: HIGH)

**Problem**: Single walk length is a hyperparameter that doesn't generalize across tasks.

**Observation**:
- Short contexts (TREC ~5k tokens): need short walks
- Long contexts (narrativeqa ~8k tokens): need longer walks
- Multi-hop (hotpotqa): need medium walks for reasoning chains

**Solution**: Run walkers with different horizons and combine:

```python
def multi_horizon_scores(attention, query_pos, horizons=[10, 50, 200]):
    """Combine walk scores from multiple time horizons."""

    scores_by_horizon = []
    for max_steps in horizons:
        visits = run_absorbing_walks(attention, query_pos, max_steps)
        normalized = normalize_visits(visits)
        scores_by_horizon.append(normalized)

    # Adaptive weights based on attention entropy
    entropy = compute_attention_entropy(attention[query_pos])
    focus_ratio = 1 - entropy / log(seq_len)

    if focus_ratio > 0.7:  # Focused attention
        weights = [0.6, 0.3, 0.1]  # Favor short
    elif focus_ratio < 0.3:  # Diffuse attention
        weights = [0.1, 0.3, 0.6]  # Favor long
    else:
        weights = [0.33, 0.34, 0.33]  # Balanced

    return sum(w * s for w, s in zip(weights, scores_by_horizon))
```

**Mathematical Justification**: Different walk lengths approximate different powers of Q:
- Short walks ≈ I + Q + Q²
- Long walks ≈ (I - Q)^{-1} (full fundamental matrix)

The ensemble approximates a weighted fundamental matrix.

**Expected Impact**: +3-8% across all tasks by adapting to task structure.

**Novelty Status**: ✅ NOVEL - Multi-horizon absorbing walks with adaptive weighting.

---

### Breakthrough 4: Query-Conditioned Transition Reweighting (Priority: MEDIUM)

**Problem**: Walkers follow raw attention, which pulls them toward sink regardless of query relevance.

**Solution**: Reweight transitions to favor query-relevant tokens:

```
P_query(j|i) = P(j|i) × exp(β · sim(K_j, Q)) / Z_i

Where:
- P(j|i) = original attention-based transition
- sim(K_j, Q) = cosine similarity between key j and query Q
- β = temperature parameter
- Z_i = normalizing constant
```

**Implementation**:
```python
def query_conditioned_transition(attention_row, keys, query, beta=1.0):
    """Reweight transitions by query relevance."""
    # Original transition probabilities
    p_orig = attention_row / attention_row.sum()

    # Query relevance weights
    similarities = F.cosine_similarity(keys, query.unsqueeze(0), dim=-1)
    relevance = torch.exp(beta * similarities)

    # Combined probability
    p_query = p_orig * relevance
    p_query = p_query / p_query.sum()

    return p_query
```

**Expected Impact**: +5-15% on retrieval tasks (qasper, triviaqa).

**Novelty Status**: ✅ NOVEL - Tilted random walks with query conditioning for KV cache.

---

### Breakthrough 5: Bidirectional Current Flow (Priority: MEDIUM)

**Problem**: Current walks only flow query→sink. Tokens important in reverse direction are missed.

**Solution**: Run absorbing walks in BOTH directions:

```
Forward Flow (existing):
  Source: query (position n-1)
  Sink: context-start (positions 0-3)
  F[j] = visits in forward direction

Backward Flow (new):
  Source: context-start (positions 0-3)
  Sink: query (position n-1)
  Walk on A^T (transpose attention)
  R[j] = visits in reverse direction

Combined Score:
  score[j] = sqrt(F[j] · R[j])  # Geometric mean
```

**Mathematical Justification**: In electrical networks:
```
current_flow_betweenness(j) ∝ |current through j|
```

For directed graphs, we need both directions explicitly.

**Expected Impact**: +5-10% on multi-hop tasks (hotpotqa, musique, 2wikimqa).

**Novelty Status**: ⚠️ PARTIAL - Bidirectional PPR is known, but absorbing version may be novel.

---

## Implementation Priority

| Rank | Breakthrough | Novelty | Rigor | Impact | Effort | Priority |
|------|-------------|---------|-------|--------|--------|----------|
| **1** | Fundamental Matrix Norm | ✅ | ✅ Strong | High | Medium | **CRITICAL** |
| **2** | Instruction Anchors | ✅ | ⚠️ Heuristic | Critical (TREC) | Low | **CRITICAL** |
| **3** | Multi-Horizon Ensemble | ✅ | ✅ | Medium | Low | **HIGH** |
| **4** | Query-Conditioned Trans. | ✅ | ✅ | Medium | Medium | **MEDIUM** |
| **5** | Bidirectional Flow | ⚠️ | ✅ | Medium | Medium | **MEDIUM** |

### Recommended Implementation Order

1. **Breakthrough 2** (Instruction Anchors) - Quick fix for TREC, low effort
2. **Breakthrough 1** (Fundamental Matrix) - Core theoretical improvement
3. **Breakthrough 3** (Multi-Horizon) - Easy to implement, general improvement
4. **Breakthrough 4 & 5** - If time permits before deadline

---

## Paper Framing Recommendations

### Suggested Title

**"CircuitKV: Current-Flow Betweenness for KV Cache Compression via Absorbing Random Walks"**

### Abstract Key Points

1. Frame as **absorbing Markov chain** (distinct from ergodic TokenRank/PageRank)
2. Emphasize **current-flow betweenness** captures "bridge tokens"
3. Highlight **fundamental matrix normalization** as principled approach
4. Show improvement over H2O, SnapKV, PyramidKV on LongBench

### Related Work Must Include

1. **TokenRank** - Direct comparison (steady-state vs transient)
2. **Attention Flow** - Acknowledge prior max-flow work, differentiate use case
3. **H2O, SnapKV, PyramidKV** - Experimental baselines
4. **StreamingLLM** - We formalize their attention sink intuition

### Key Experiments

1. **Direct TokenRank comparison** - Show different token selections
2. **Ablation: normalization methods** - sqrt vs fundamental matrix
3. **Bridge token analysis** - Visualize tokens we keep that H2O misses
4. **Multi-hop performance** - hotpotqa, 2wikimqa, musique improvements

---

## References

1. Erel et al. "Attention (as Discrete-Time Markov) Chains" arXiv:2507.17657 (2025)
2. Abnar & Zuidema. "Quantifying Attention Flow in Transformers" ACL 2020
3. Cai et al. "PyramidKV: Dynamic KV Cache Compression" arXiv:2406.02069 (2024)
4. Zhang et al. "H2O: Heavy-Hitter Oracle for Efficient Generative Inference" (2023)
5. Xiao et al. "StreamingLLM: Efficient Streaming Language Models with Attention Sinks" (2023)
6. Lovász. "Random Walks on Graphs: A Survey" (1993) - Theoretical foundation

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-11 | Initial novelty analysis and breakthrough proposals |

---

*This document was generated by the Novelty Checker Agent with contributions from Math Reviewer and Analyst Agents.*
