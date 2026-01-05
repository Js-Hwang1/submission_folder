# CircuitKV Improvement Ideas

## Current Status (Baseline: τ=1.0 fixed, no 1/√d scaling)

| Task | CircuitKV | SnapKV | H2O | Gap vs Best |
|------|-----------|--------|-----|-------------|
| NarrativeQA | 25.10 | 25.70 | 24.82 | -0.60 |
| Qasper | 30.85 | 29.79 | 29.35 | **+1.06** |
| HotpotQA | 44.34 | 43.90 | 43.26 | **+0.44** |
| PassageCount | 4.83 | 5.61 | 5.88 | -1.05 |
| TREC | 72.00 | 73.50 | 73.00 | -1.50 |

**Core Insight:** CircuitKV excels at **needle tasks** (path-finding) but struggles with **haystack tasks** (coverage).

---

## Idea 1: Entropy-Adaptive Temperature (RECOMMENDED FIRST)

### Concept
Let the attention distribution's entropy automatically adjust walker sharpness.

- **Concentrated attention** (low entropy) → low τ → sharp walks (needle mode)
- **Diffuse attention** (high entropy) → high τ → uniform walks (haystack mode)

### Physics Motivation
In thermodynamics, entropy measures disorder. High entropy = thermal noise dominates = system "heats up". This is self-regulating based on the signal itself.

### Implementation (graph_update.cu)

```cpp
// After collecting top-k scores in block_heap
if (block_heap_size > 0) {
    // Step 1: Compute entropy of raw score distribution
    float sum_scores = 0.0f;
    for (int k = 0; k < block_heap_size; ++k) {
        sum_scores += expf(block_heap[k].score - block_heap[0].score);  // Numerical stability
    }

    float entropy = 0.0f;
    for (int k = 0; k < block_heap_size; ++k) {
        float p = expf(block_heap[k].score - block_heap[0].score) / sum_scores;
        if (p > 1e-10f) {
            entropy -= p * logf(p);
        }
    }

    // Step 2: Normalize entropy to [0, 1]
    float max_entropy = logf((float)block_heap_size);  // Uniform distribution
    float normalized_entropy = entropy / max_entropy;

    // Step 3: Compute adaptive temperature
    constexpr float TAU_BASE = 1.0f;
    constexpr float ALPHA = 2.0f;  // Entropy sensitivity
    float tau_effective = TAU_BASE * (1.0f + ALPHA * normalized_entropy);
    float inv_temp = 1.0f / tau_effective;

    // Step 4: Apply softmax with adaptive temperature
    float max_score = block_heap[0].score;
    float sum_exp = 0.0f;
    for (int k = 0; k < block_heap_size; ++k) {
        float normalized_score = (block_heap[k].score - max_score) * inv_temp;
        sum_exp += expf(fminf(normalized_score, 20.0f));
    }
    // ... rest of softmax
}
```

### Expected Behavior

| Attention Pattern | Entropy | τ_effective | Walk Mode |
|-------------------|---------|-------------|-----------|
| [0.9, 0.05, 0.05, ...] | ~0.15 | ~1.3 | Sharp (needle) |
| [0.4, 0.3, 0.2, 0.1] | ~0.85 | ~2.7 | Diffuse (haystack) |

### Tunable Parameters
- `TAU_BASE = 1.0`: Minimum temperature (for concentrated attention)
- `ALPHA = 2.0`: How much entropy increases temperature (range: 1.0-4.0)

### Expected Impact
- Needle tasks: Preserved (low entropy → sharp walks unchanged)
- Haystack tasks: +1-2 points (high entropy → more coverage)

---

## Idea 2: Dual-Population Walker Ensemble

### Concept
Run two independent walker populations with different temperatures, combine results.

### Implementation

```cpp
// In circuit_manager.cpp
void update_and_step_circuit_ensemble(...) {
    // Population A: Sharp walks for needle detection
    launch_absorbing_walker_kernel(tau=0.5, num_walkers=512, visits_sharp);

    // Population B: Diffuse walks for coverage
    launch_absorbing_walker_kernel(tau=3.0, num_walkers=512, visits_diffuse);

    // Combine: max preserves tokens important to EITHER mode
    for (int i = 0; i < seq_len; i++) {
        visits_combined[i] = max(visits_sharp[i], visits_diffuse[i]);
    }
}
```

### Physics Motivation
Two parallel circuits at different temperatures:
- Superconducting circuit (τ=0.5): Current flows through lowest-resistance path
- Resistive circuit (τ=3.0): Current distributes across many paths

### Pros
- Clean separation between modes
- No interference between needle/haystack detection
- Mathematically simple combination

### Cons
- 2x walker computation cost
- May need tuning of population sizes

### Expected Impact
- Universal improvement across task types
- +0.5-1.0 on needle tasks, +1.5-2.0 on haystack tasks

---

## Idea 3: Layer-Wise Temperature Gradient

### Concept
Different transformer layers have different attention patterns. Apply temperature gradient across layers.

### Implementation

```cpp
// Temperature decreases with layer depth
float compute_layer_temperature(int layer_idx, int num_layers) {
    constexpr float TAU_EARLY = 3.0f;   // Layers 0-10: diffuse
    constexpr float TAU_LATE = 0.5f;    // Layers 20-32: sharp

    float progress = (float)layer_idx / (float)(num_layers - 1);
    return TAU_EARLY + progress * (TAU_LATE - TAU_EARLY);
}
```

### Physics Motivation
Information flows through a "temperature gradient":
- Hot (diffuse) at input: Broad context capture
- Cold (focused) at output: Task-specific selection

### Layer Semantics
| Layer Range | Temperature | Role |
|-------------|-------------|------|
| 0-10 | τ = 2.5-3.0 | Syntax, local patterns |
| 10-20 | τ = 1.5-2.0 | Semantic grouping |
| 20-32 | τ = 0.5-1.0 | Task-specific focus |

### Pros
- No per-token computation overhead
- Matches known transformer layer behavior
- Single-pass (no ensemble)

### Cons
- Requires passing layer_idx through API
- Less adaptive to specific inputs

### Expected Impact
- Moderate improvement across all tasks (+0.5-1.0)

---

## Idea 4: Multi-Sink Absorption (For Few-Shot Tasks)

### Concept
Current implementation uses fixed SINK_SIZE=4 (first 4 tokens). For few-shot tasks, the "ground truth" is in the examples, not the system prompt.

### Implementation

```cpp
// Detect sink candidates based on attention patterns or token content
struct SinkConfig {
    int* sink_positions;  // Array of sink token indices
    int num_sinks;
};

// Walker terminates when hitting ANY sink
__device__ bool is_sink(int node_idx, SinkConfig* config) {
    for (int i = 0; i < config->num_sinks; i++) {
        if (node_idx == config->sink_positions[i]) return true;
    }
    return false;
}
```

### Sink Detection Strategies
1. **Structural**: Detect newlines, "Example:", "[/INST]" tokens
2. **Attention-based**: Tokens with high aggregate attention from all queries
3. **Fixed patterns**: First 4 + every N tokens (uniform coverage)

### Expected Impact
- TREC: +1.5-2.0 (few-shot examples become reachable)
- PassageCount: +1.0 (passage boundaries become sinks)

---

## Idea 5: Coverage Regularization

### Concept
After computing betweenness scores, add a penalty for "coverage gaps" — regions with no high-score tokens.

### Implementation (Python, in engine.py)

```python
def apply_coverage_regularization(scores, seq_len, window=64, boost=0.5):
    """Ensure no region is completely evicted."""
    for start in range(0, seq_len - window, window):
        end = min(start + window, seq_len)
        region_max = scores[start:end].max()

        if region_max < threshold:
            # Boost the highest-scoring token in this region
            local_argmax = scores[start:end].argmax()
            scores[start + local_argmax] += boost

    return scores
```

### Physics Motivation
In electrical networks, you can't have a region with zero current if it's connected to the circuit. Coverage regularization ensures connectivity.

### Expected Impact
- PassageCount: +1.5-2.0 (ensures all passages have representatives)
- Summarization: +0.5-1.0 (ensures document coverage)

---

## Idea 6: Hybrid Scoring (Betweenness + Degree)

### Concept
Combine CircuitKV (transport/betweenness) with H2O (degree/popularity).

```python
final_score = alpha * betweenness + (1 - alpha) * degree
```

### Implementation

Already partially implemented in `_initialize_charge_from_attention`:
```python
combined = torch.maximum(h2o, circuit)  # Current: Union strategy
```

Could extend to weighted combination:
```python
combined = alpha * circuit + (1 - alpha) * h2o  # Linear blend
```

### Tunable Parameters
- `alpha = 1.0`: Pure CircuitKV (current)
- `alpha = 0.7`: Mostly CircuitKV with H2O backup
- `alpha = 0.5`: Equal weight

### Expected Impact
- Moderate improvement on haystack tasks
- Risk: May dilute CircuitKV's bridge-finding advantage

---

## Priority Ranking

| Priority | Idea | Effort | Expected Gain | Risk |
|----------|------|--------|---------------|------|
| 1 | Entropy-Adaptive Temperature | Medium | +1.0 avg | Low |
| 2 | Dual-Population Ensemble | Medium | +1.5 avg | Low |
| 3 | Coverage Regularization | Low | +0.5 on coverage tasks | Low |
| 4 | Layer-Wise Temperature | Medium | +0.5 avg | Medium |
| 5 | Multi-Sink Absorption | High | +1.5 on few-shot | Medium |
| 6 | Hybrid Scoring | Low | +0.3 avg | Medium |

---

## Experiment Log

### Experiment 1: Boltzmann Conductance (τ=1.0, with 1/√d scaling)
- **Date:** [Current]
- **Change:** Added softmax to edge weights with 1/√d scaling
- **Result:** NarrativeQA 25.53 (+0.43), Qasper 27.73 (-3.12)
- **Analysis:** 1/√d made τ_effective ≈ 11, way too diffuse
- **Action:** Remove 1/√d scaling

### Experiment 2: Boltzmann Conductance (τ=1.0, NO 1/√d scaling)
- **Date:** [Current]
- **Change:** Removed 1/√d scaling from softmax
- **Result:** [PENDING]
- **Expected:** Qasper recovery to ~30+, NarrativeQA may drop slightly

### Experiment 3: Entropy-Adaptive Temperature
- **Date:** [PENDING]
- **Change:** Implement entropy-based τ adjustment
- **Parameters:** TAU_BASE=1.0, ALPHA=2.0
- **Expected:** Needle preserved, haystack +1-2 points

---

## Notes

- Always test on both needle (Qasper, HotpotQA) AND haystack (NarrativeQA, PassageCount) tasks
- Single-task improvements that hurt other tasks are not acceptable
- The goal is SoTA on ALL LongBench categories, not just average
