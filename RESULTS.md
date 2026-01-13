# CircuitKV Benchmark Results

## LongBench Evaluation

**Model:** Meta-Llama-3-8B-Instruct
**KV Budget:** 1024 tokens
**Attention:** flash_attention_2
**Hardware:** NVIDIA GH200 Grace Hopper Superchip

### Results Table

| Method | NarrativeQA | Qasper | MultifieldQA | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum | MultiNews | TREC | TriviaQA | SAMSum | PassageCount | PassageRetrieval | LCC | RepoBench-P | **Avg** |
|--------|-------------|--------|--------------|----------|----------|---------|-----------|-------|-----------|------|----------|--------|--------------|------------------|-----|-------------|---------|
| FullKV | 25.56 | 32.27 | 39.71 | 43.56 | 35.29 | 21.18 | 28.74 | 23.20 | 26.73 | 74.00 | 90.48 | 42.53 | 4.80 | 69.75 | 59.27 | 53.92 | **41.94** |
| SnapKV | 25.76 | 26.74 | 37.74 | 43.39 | 34.64 | 19.94 | 24.87 | 22.87 | 25.67 | 73.00 | 90.56 | 41.25 | 5.13 | 69.25 | 60.26 | 56.25 | **41.08** |
| PyramidKV | 25.38 | 26.07 | 37.50 | 43.84 | 34.26 | 21.59 | 24.58 | 22.56 | 25.40 | 72.50 | 90.56 | 41.66 | 5.58 | 69.25 | 59.54 | 55.40 | **41.04** |
| H2O | 25.27 | 26.94 | 35.59 | 42.66 | 30.65 | 18.58 | 26.37 | 22.27 | 26.05 | 73.00 | 91.20 | 41.05 | 5.55 | 69.03 | 59.66 | 56.60 | **40.65** |
| StreamingLLM | 21.48 | 15.47 | 25.87 | 36.64 | 27.96 | 15.50 | 23.28 | 20.98 | 25.63 | 66.50 | 86.02 | 40.42 | 4.58 | 69.25 | 60.25 | 56.77 | **37.29** |
| OURS |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV | 32.51 | 33.34 | 26.22 | 69.00 | 37.28 | 56.60 | **41.94** |
| SnapKV | 30.08 | 32.66 | 24.47 | 68.27 | 37.19 | 58.26 | **41.08** |
| PyramidKV | 29.65 | 33.23 | 24.18 | 68.24 | 37.42 | 57.47 | **41.04** |
| H2O | 29.27 | 30.63 | 24.90 | 68.42 | 37.29 | 58.13 | **40.65** |
| StreamingLLM | 20.94 | 26.70 | 23.30 | 64.31 | 36.92 | 58.51 | **37.29** |
| OURS |


**Category Mapping:**
- Single-Doc QA: NarrativeQA , Qasper, MultifieldQA
- Multi-Doc QA: HotpotQA, 2WikiMQA, Musique
- Summarization: GovReport, QMSum, MultiNews
- Few-shot: TREC, TriviaQA, SAMSum
- Synthetic: PassageCount, PassageRetrieval
- Code: LCC, RepoBench-P

---

**Model:** Meta-Llama-3-8B-Instruct
**KV Budget:** 2048 tokens
**Attention:** flash_attention_2
**Hardware:** NVIDIA GH200 Grace Hopper Superchip

### Results Table

| Method | NarrativeQA | Qasper | MultifieldQA | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum | MultiNews | TREC | TriviaQA | SAMSum | PassageCount | PassageRetrieval | LCC | RepoBench-P | **Avg** |
|--------|-------------|--------|--------------|----------|----------|---------|-----------|-------|-----------|------|----------|--------|--------------|------------------|-----|-------------|---------|
| FullKV | 25.56 | 32.27 | 39.71 | 43.56 | 35.29 | 21.18 | 28.74 | 23.20 | 26.73 | 74.00 | 90.48 | 42.53 | 4.80 | 69.75 | 59.27 | 53.92 | **41.94** |
| SnapKV | 25.70 | 29.79 | 38.97 | 43.90 | 35.04 | 20.92 | 26.95 | 23.46 | 26.11 | 73.50 | 90.56 | 41.83 | 5.61 | 69.25 | 57.70 | 54.23 | **41.47** |
| PyramidKV | 25.07 | 29.61 | 38.25 | 43.69 | 35.35 | 21.43 | 27.02 | 23.25 | 26.14 | 73.50 | 90.56 | 42.08 | 5.30 | 69.25 | 57.86 | 53.26 | **41.35** |
| H2O    | 26.06 | 29.07 | 36.61 | 42.30 | 33.33 | 19.72 | 27.45 | 22.66 | 26.69 | 73.00 | 90.93 | 42.02 | 5.88 | 69.50 | 57.47 | 54.83 | **41.10** |
| StreamingLLM | 24.09 | 24.09 | 30.28 | 39.33 | 31.26 | 17.48 | 24.98 | 21.55 | 26.37 | 70.00 | 89.78 | 41.40 | 5.58 | 68.92 | 60.67 | 57.53 | **39.58** |
| OURS | 24.71 | 30.57 | 40.15 | 44.03 | 36.09 | 21.77 | 26.37 | 22.91 | 26.22 | 73.50 | 90.56 | 41.58 | 5.08 | 69.25 | 60.47 | 56.66 | **41.87** |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV | 32.51 | 33.34 | 26.22 | 69.00 | 37.28 | 56.60 | **41.94** |
| SnapKV | 31.49 | 33.29 | 25.51 | 68.63 | 37.43 | 55.97 | **41.47** |
| PyramidKV | 30.98 | 33.49 | 25.47 | 68.71 | 37.28 | 55.56 | **41.35** |
| H2O    | 30.58 | 31.78 | 25.60 | 68.65 | 37.69 | 56.15 | **41.10** |
| StreamingLLM | 26.15 | 29.36 | 24.30 | 67.06 | 37.25 | 59.10 | **39.58** |
| OURS | 31.81 | 33.96 | 25.17 | 68.55 | 37.17 | 58.57 | **41.87** |


**Category Mapping:**
- Single-Doc QA: NarrativeQA , Qasper, MultifieldQA
- Multi-Doc QA: HotpotQA, 2WikiMQA, Musique
- Summarization: GovReport, QMSum, MultiNews
- Few-shot: TREC, TriviaQA, SAMSum
- Synthetic: PassageCount, PassageRetrieval
- Code: LCC, RepoBench-P  

---

**Model:** Meta-Llama-3.1-8B-Instruct
**KV Budget:** 1024 tokens
**Attention:** flash_attention_2
**Hardware:** NVIDIA GH200 Grace Hopper Superchip

### Results Table

| Method | NarrativeQA | Qasper | MultifieldQA | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum | MultiNews | TREC | TriviaQA | SAMSum | PassageCount | PassageRetrieval | LCC | RepoBench-P | **Avg** |
|--------|-------------|--------|--------------|----------|----------|---------|-----------|-------|-----------|------|----------|--------|--------------|------------------|-----|-------------|---------|
| FullKV | 
| SnapKV |
| PyramidKV |
| H2O |
| StreamingLLM |
| CircuitKV |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV | 
| SnapKV |
| PyramidKV |
| H2O |
| StreamingLLM |
| CircuitKV |


**Category Mapping:**
- Single-Doc QA: NarrativeQA , Qasper, MultifieldQA
- Multi-Doc QA: HotpotQA, 2WikiMQA, Musique
- Summarization: GovReport, QMSum, MultiNews
- Few-shot: TREC, TriviaQA, SAMSum
- Synthetic: PassageCount, PassageRetrieval
- Code: LCC, RepoBench-P  

---


**Model:** Meta-Llama-3.1-8B-Instruct
**KV Budget:** 2048 tokens
**Attention:** flash_attention_2
**Hardware:** NVIDIA GH200 Grace Hopper Superchip

### Results Table

| Method | NarrativeQA | Qasper | MultifieldQA | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum | MultiNews | TREC | TriviaQA | SAMSum | PassageCount | PassageRetrieval | LCC | RepoBench-P | **Avg** |
|--------|-------------|--------|--------------|----------|----------|---------|-----------|-------|-----------|------|----------|--------|--------------|------------------|-----|-------------|---------|
| FullKV | 
| SnapKV |
| PyramidKV |
| H2O |
| StreamingLLM |
| CircuitKV |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV | 
| SnapKV |
| PyramidKV |
| H2O |
| StreamingLLM |
| CircuitKV |


**Category Mapping:**
- Single-Doc QA: NarrativeQA , Qasper, MultifieldQA
- Multi-Doc QA: HotpotQA, 2WikiMQA, Musique
- Summarization: GovReport, QMSum, MultiNews
- Few-shot: TREC, TriviaQA, SAMSum
- Synthetic: PassageCount, PassageRetrieval
- Code: LCC, RepoBench-P  

---


**Model:** Meta-Llama-3.3-70B-Instruct
**KV Budget:** 1024 tokens
**Attention:** flash_attention_2
**Hardware:** NVIDIA GH200 Grace Hopper Superchip

### Results Table

| Method | NarrativeQA | Qasper | MultifieldQA | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum | MultiNews | TREC | TriviaQA | SAMSum | PassageCount | PassageRetrieval | LCC | RepoBench-P | **Avg** |
|--------|-------------|--------|--------------|----------|----------|---------|-----------|-------|-----------|------|----------|--------|--------------|------------------|-----|-------------|---------|
| FullKV | 
| SnapKV |
| PyramidKV |
| H2O |
| StreamingLLM |
| CircuitKV |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV | 
| SnapKV |
| PyramidKV |
| H2O |
| StreamingLLM |
| CircuitKV |


**Category Mapping:**
- Single-Doc QA: NarrativeQA , Qasper, MultifieldQA
- Multi-Doc QA: HotpotQA, 2WikiMQA, Musique
- Summarization: GovReport, QMSum, MultiNews
- Few-shot: TREC, TriviaQA, SAMSum
- Synthetic: PassageCount, PassageRetrieval
- Code: LCC, RepoBench-P  

---


**Model:** Meta-Llama-3.3-70B-Instruct
**KV Budget:** 2048 tokens
**Attention:** flash_attention_2
**Hardware:** NVIDIA GH200 Grace Hopper Superchip

### Results Table

| Method | NarrativeQA | Qasper | MultifieldQA | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum | MultiNews | TREC | TriviaQA | SAMSum | PassageCount | PassageRetrieval | LCC | RepoBench-P | **Avg** |
|--------|-------------|--------|--------------|----------|----------|---------|-----------|-------|-----------|------|----------|--------|--------------|------------------|-----|-------------|---------|
| FullKV | 
| SnapKV |
| PyramidKV |
| H2O |
| StreamingLLM |
| CircuitKV |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV | 
| SnapKV |
| PyramidKV |
| H2O |
| StreamingLLM |
| CircuitKV |


**Category Mapping:**
- Single-Doc QA: NarrativeQA , Qasper, MultifieldQA
- Multi-Doc QA: HotpotQA, 2WikiMQA, Musique
- Summarization: GovReport, QMSum, MultiNews
- Few-shot: TREC, TriviaQA, SAMSum
- Synthetic: PassageCount, PassageRetrieval
- Code: LCC, RepoBench-P  

---