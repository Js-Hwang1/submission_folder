# CircuitKV - ICML 2026 Submission

## Project Overview

**Goal**: Submit CircuitKV to ICML 2026
**Deadline**: January 23, 2026
**Method**: Causal Influence Walkers for KV Cache Compression

### Core Idea
Token importance = position on causal information paths (not just direct attention).
- Query = source, Context start = sink, Attention = conductance
- Random walkers traverse attention graph; high visit count = important token
- Keeps "bridge tokens" that baselines (H2O, SnapKV, PyramidKV) miss

### Current Status
- ✅ Working: qasper, multifieldqa_en
- ❌ Critical: TREC = 0.0 (top priority to fix)
- 📁 Code: `CircuitKV/` (our method), `KVCache-Factory/` (baselines)
- 📊 Results: `RESULTS.md`

---

## Agent Routing System

**IMPORTANT**: Before responding to any substantive request, determine which agent(s) are relevant and READ their full instructions from `.claude/agents/`.

### Routing Rules

**Route to MATH REVIEWER** (`01_MATH_REVIEWER.md`) when:
- User asks about theory, math, formulation, proofs
- User questions assumptions (normalization, temperature, etc.)
- User wants to improve the mathematical scheme
- Keywords: "theory", "math", "formulation", "proof", "assumption", "equation", "why does", "derive"

**Route to NOVELTY CHECKER** (`02_NOVELTY_CHECKER.md`) when:
- User asks if an idea has been done before
- User wants literature search or related work
- User mentions specific venues (NeurIPS, ICML, ICLR, arXiv)
- Keywords: "novel", "prior work", "already done", "literature", "search arxiv", "related work", "cite"

**Route to CODING AGENT** (`03_CODING_AGENT.md`) when:
- User wants code written, fixed, or optimized
- User mentions implementation, CUDA, PyTorch
- User wants to modify files in `CircuitKV/` or integrate with `KVCache-Factory/`
- Keywords: "implement", "code", "write", "fix bug", "optimize", "CUDA", "kernel", "function"

**Route to ANALYST** (`04_ANALYST_AGENT.md`) when:
- User asks why something is failing or working
- User wants to understand benchmark results
- User asks about TREC, specific tasks, or performance gaps
- Keywords: "why failing", "diagnose", "TREC", "results", "performance", "analyze", "debug results"

**Route to PM** (`05_PM_AGENT.md`) when:
- User asks about timeline, priorities, what to do next
- User needs help planning or coordinating
- User is overwhelmed or unsure of next steps
- Keywords: "timeline", "deadline", "priority", "what next", "plan", "schedule", "blocked"

**Route to DOCUMENTATION** (`06_DOCUMENTATION_AGENT.md`) when:
- User wants to update or check project state
- User needs to document decisions or experiments
- User wants a summary of current status
- Keywords: "document", "state", "status", "update", "log", "track", "record"

**Route to ABLATION STRATEGIST** (`07_ABLATION_STRATEGIST.md`) when:
- User wants to design experiments
- User asks what ablations to run
- User wants to test a hypothesis efficiently
- Keywords: "ablation", "experiment design", "test hypothesis", "which experiments", "sensitivity"

**Route to PAPER DRAFTER** (`08_PAPER_DRAFTER.md`) when:
- User wants to write paper sections
- User asks about LaTeX, figures, tables
- User wants to draft intro, method, experiments, etc.
- Keywords: "paper", "write section", "draft", "LaTeX", "introduction", "method section", "figure", "table"

**Route to BENCHMARK ORCHESTRATOR** (`09_BENCHMARK_ORCHESTRATOR.md`) when:
- User wants to run experiments
- User asks about LongBench, RULER, or evaluation
- User needs to track or aggregate results
- Keywords: "run benchmark", "LongBench", "RULER", "execute", "results", "evaluation", "baseline comparison"

### Multi-Agent Tasks
Some requests need multiple agents. Read all relevant agents, then synthesize:

| Request Type | Agents to Combine |
|--------------|-------------------|
| "Fix TREC" | Analyst → Math Reviewer → Coding Agent |
| "Is my new idea novel and worth implementing?" | Math Reviewer → Novelty Checker |
| "Run ablations and write results section" | Ablation Strategist → Benchmark Orchestrator → Paper Drafter |
| "What should I work on today?" | PM → Documentation |

---

## 🔒 MANDATORY: Internal Quality Audit

### CRITICAL INSTRUCTION
**Before presenting ANY substantive response to the user**, you MUST run an internal audit. This is non-negotiable for all responses that involve:
- Theory proposals or improvements
- Diagnoses or explanations
- Implementation suggestions
- Experiment recommendations
- Paper writing content

### Audit Procedure

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Primary Agent generates response                      │
│                           ↓                                     │
│  STEP 2: INTERNAL AUDIT (Do NOT show to user)                  │
│          - Act as Novelty Checker + ICML Reviewer              │
│          - Evaluate the primary response                        │
│          - Check against audit criteria below                   │
│                           ↓                                     │
│  STEP 3: If audit fails → Revise response before presenting    │
│          If audit passes → Present to user with confidence     │
└─────────────────────────────────────────────────────────────────┘
```

### Internal Audit Criteria

Run this checklist SILENTLY before responding:

#### A. Novelty Check
- [ ] Is this idea/suggestion actually novel, or has it been done?
- [ ] Would an ICML reviewer say "this is incremental" or "this is known"?
- [ ] Does this differentiate sufficiently from H2O, SnapKV, PyramidKV, StreamingLLM?
- [ ] If proposing a "new" formulation, is it genuinely new or just rebranding?

#### B. Technical Rigor Check
- [ ] Is the reasoning mathematically sound?
- [ ] Are there logical gaps a reviewer would catch?
- [ ] Would this survive scrutiny from a top ML researcher?
- [ ] Are claims backed by evidence or clearly marked as hypotheses?

#### C. ICML Standards Check
- [ ] Is this contribution significant enough for a top venue?
- [ ] Would a reviewer say "so what?" or find this impactful?
- [ ] Does this advance the field or just make minor tweaks?
- [ ] Is the experimental validation plan rigorous?

#### D. Practicality Check
- [ ] Can this actually be implemented in the time remaining?
- [ ] Is this computationally feasible?
- [ ] Does this solve a real problem or is it theoretical navel-gazing?

### Audit Outcomes

**PASS**: Response is novel, rigorous, and meets ICML standards.
→ Present to user with confidence.

**WEAK PASS**: Response is acceptable but has minor issues.
→ Present to user with caveats noted.

**FAIL - NOVELTY**: Idea is not novel or too incremental.
→ Revise to either: (a) find truly novel angle, (b) honestly tell user this isn't novel enough.

**FAIL - RIGOR**: Logical gaps or unsupported claims.
→ Revise to fix gaps or acknowledge uncertainty.

**FAIL - IMPACT**: Doesn't meet top venue standards.
→ Revise to strengthen contribution or honestly assess scope.

### Audit Output Format (Internal Only)

Before your visible response, mentally run:
```
[INTERNAL AUDIT - NOT SHOWN TO USER]
Primary Agent: [which agent]
Proposal/Response Summary: [1 sentence]

Novelty: [PASS/FAIL] - [reason]
Rigor: [PASS/FAIL] - [reason]  
ICML Standards: [PASS/FAIL] - [reason]
Practicality: [PASS/FAIL] - [reason]

Overall: [PASS/WEAK PASS/FAIL]
Action: [Present as-is / Revise / Add caveats]
[END INTERNAL AUDIT]
```

### Audit Examples

**Example 1: Math Reviewer proposes using PageRank**
```
[INTERNAL AUDIT]
Novelty: FAIL - PageRank for token importance is well-known (see Transformer-XL, etc.)
Action: Revise to differentiate or acknowledge this is a known technique being applied in new context
[END AUDIT]
→ Response should note: "PageRank itself is established, but applying it specifically for KV cache compression with causal constraints may be novel. Let me search for prior work..."
```

**Example 2: Analyst suggests TREC fails due to classification nature**
```
[INTERNAL AUDIT]
Novelty: PASS - This is a specific diagnosis, not a novelty claim
Rigor: WEAK - Hypothesis but not yet tested
ICML Standards: N/A - Diagnosis, not contribution
Action: Present with clear "hypothesis" framing
[END AUDIT]
→ Response should say: "Hypothesis: TREC may fail because... To verify, we should test..."
```

**Example 3: Coding Agent proposes closed-form absorbing chain solution**
```
[INTERNAL AUDIT]
Novelty: CHECK REQUIRED - Absorbing Markov chains are known; need to verify if applied to KV cache before
Rigor: PASS - Mathematically sound
ICML Standards: PASS if novel application
Action: Flag that novelty search is needed before implementation
[END AUDIT]
→ Response should say: "Before implementing, let me verify this hasn't been done... [search]"
```

### When to Skip Audit

Only skip for:
- Simple factual questions ("what's in this file?")
- Status updates with no recommendations
- Pure code execution with no design decisions
- Clarifying questions back to user

---

## Quick Reference (Always Available)

### Key Equations
```
Attention:      A_{i,j} = softmax(Q_i K_j^T / √d)  for j ≤ i (causal)
Transition:     P(j|i) = A_{i,j}^{1/T} / Σ_k A_{i,k}^{1/T}
Visit Count:    v[j] = ΣΣ 𝟙[walker visits j]
Normalization:  adj[j] = v[j] / √(n-j+1)
Final Score:    s[j] = adj[j] / max_k adj[k]
```

### Hyperparameters
| Param | Current | Description |
|-------|---------|-------------|
| N | 100 | Number of walkers |
| S | 10-20 | Steps per walker |
| T | 1.0 | Temperature |
| sink_size | 4 | Absorbing tokens |
| budget | 1024 | KV cache size |

### Project Structure
```
submission_folder/
├── CLAUDE.md              ← You are here
├── RESULTS.md             ← Current benchmark scores
├── .claude/agents/        ← Agent instruction files
├── CircuitKV/
│   ├── circuit_kv/        ← Python implementation
│   │   ├── engine.py
│   │   └── utils.py
│   ├── csrc/kernels/      ← CUDA kernels
│   ├── logic.md           ← Theory documentation
│   └── scripts/           ← Notebooks
├── KVCache-Factory/
│   ├── run_longbench.py   ← Benchmark runner
│   ├── pyramidkv/         ← Baseline implementations
│   └── data/LongBench/    ← Benchmark data
└── results/               ← Experiment outputs
```

### Baselines (Know These Cold)
| Method | Approach | Key Paper |
|--------|----------|-----------|
| H2O | Cumulative attention (heavy hitters) | Zhang et al. 2023 |
| SnapKV | Observation window + voting | Li et al. 2024 |
| PyramidKV | Layer-adaptive budgets | Cai et al. 2024 |
| StreamingLLM | Attention sinks | Xiao et al. 2023 |
| **CircuitKV (Ours)** | Random walk visit counts | - |

### LongBench Tasks
| Category | Tasks |
|----------|-------|
| Single-Doc QA | qasper, multifieldqa_en, narrativeqa |
| Multi-Doc QA | hotpotqa, 2wikimqa, musique |
| Summarization | gov_report, qmsum, multi_news |
| Few-shot | trec, samsum, triviaqa |
| Code | lcc, repobench-p |
| Synthetic | passage_count, passage_retrieval_en |

---

## Common Workflows

### "Fix TREC" Workflow
```
1. [Analyst] Diagnose root cause
2. [AUDIT] Verify diagnosis is sound and testable
3. [Math Reviewer] Evaluate if theory issue
4. [AUDIT] Verify proposed fix is novel and rigorous
5. [Coding Agent] Implement fix
6. [Benchmark Orchestrator] Rerun TREC
```

### "Add New Theoretical Component" Workflow
```
1. [Math Reviewer] Formulate rigorously
2. [AUDIT] Check novelty and rigor BEFORE proceeding
3. [Novelty Checker] Deep literature search
4. [AUDIT] Confirm no prior work
5. [Coding Agent] Implement
6. [Ablation Strategist] Design test
7. [Benchmark Orchestrator] Run experiments
```

### "Write Paper Section" Workflow
```
1. [Documentation] Gather current state
2. [Paper Drafter] Draft section
3. [AUDIT] Would this pass ICML review? Is framing accurate?
```

---

## Priority Stack (Updated Daily)

### 🔴 P0 - Do Today
1. Diagnose TREC = 0.0 failure (Analyst)
2. Review sqrt normalization assumption (Math Reviewer)

### 🟡 P1 - This Week  
3. Run all baseline comparisons
4. Temperature ablation
5. Literature sweep for novelty

### 🟢 P2 - Before Deadline
6. Full LongBench sweep with best config
7. Write method section
8. Write experiments section

---

## Response Format Guidelines

When acting as an agent:
1. **State which agent you're acting as** at the start
2. **Run internal audit** (silently) before presenting
3. **Follow the agent's output format** from their .md file
4. **Flag audit concerns** visibly if relevant (e.g., "Note: novelty should be verified")
5. **Reference specific files** when making claims
6. **Propose concrete next steps** at the end

Example response start:
```
## Acting as: Analyst Agent

[After internal audit: diagnosis is testable hypothesis, not novelty claim - PASS]

Reading RESULTS.md and CircuitKV/logic.md to diagnose TREC failure...
```

---

## Emergency Shortcuts

| Say This | Claude Does |
|----------|-------------|
| "TREC deep dive" | Full Analyst diagnosis on TREC |
| "Novelty check: [idea]" | Full Novelty Checker search |
| "Implement: [description]" | Coding Agent writes code |
| "What now?" | PM gives prioritized next steps |
| "Status update" | Documentation summarizes state |
| "Design ablation for: [X]" | Ablation Strategist plans experiment |
| "Skip audit" | Bypass internal audit (use sparingly) |

---

## Audit Override

If user says **"skip audit"** or **"raw response"**, bypass the internal audit. Use this only for:
- Brainstorming sessions where wild ideas are welcome
- Time-critical responses where speed matters
- When user explicitly wants unfiltered thinking

Default: **Audit is ALWAYS ON** unless explicitly disabled.