# CircuitKV Benchmark Results

## LongBench Evaluation

**Model:** Meta-Llama-3-8B-Instruct
**KV Budget:** 256 tokens
**Attention:** flash_attention_2
**Hardware:** NVIDIA GH200 Grace Hopper Superchip

### Results Table

| Method | NarrativeQA | Qasper | MultifieldQA | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum | MultiNews | TREC | TriviaQA | SAMSum | PassageCount | PassageRetrieval | LCC | RepoBench-P | **Avg** |
|--------|-------------|--------|--------------|----------|----------|---------|-----------|-------|-----------|------|----------|--------|--------------|------------------|-----|-------------|---------|
| FullKV | 25.56 | 32.27 | 39.71 | 43.56 | 35.29 | 21.18 | 28.74 | 23.20 | 26.73 | 74.00 | 90.48 | 42.53 | 4.80 | 69.75 | 59.27 | 53.92 | **41.94** |
| SnapKV | 23.30 | 19.91 | 37.35 | 42.59 | 33.02 | 19.90 | 21.76 | 22.27 | 22.90 | 71.50 | 90.86 | 40.07 | 5.58 | 69.50 | 59.96 | 55.68 | **39.76** |
| PyramidKV | 23.81 | 19.96 | 35.67 | 42.52 | 31.76 | 20.12 | 21.26 | 22.84 | 22.61 | 71.50 | 90.48 | 39.97 | 5.83 | 69.50 | 58.39 | 53.78 | **39.38** |
| H2O | 23.47 | 18.47 | 29.11 | 35.90 | 28.29 | 15.62 | 23.20 | 21.95 | 24.73 | 60.50 | 88.20 | 38.64 | 5.37 | 67.74 | 58.67 | 52.52 | **37.02** |
| StreamingLLM | 17.78 | 11.04 | 20.68 | 33.22 | 25.81 | 15.99 | 19.17 | 20.51 | 20.65 | 53.00 | 79.51 | 39.40 | 5.83 | 68.37 | 58.35 | 54.51 | **33.99** |
| OURS |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV | 32.51 | 33.34 | 26.22 | 69.00 | 37.28 | 56.60 | **41.94** |
| SnapKV | 26.85 | 31.84 | 22.31 | 67.48 | 37.54 | 57.82 | **39.76** |
| PyramidKV | 26.48 | 31.47 | 22.24 | 67.32 | 37.67 | 56.09 | **39.38** |
| H2O | 23.68 | 26.60 | 23.29 | 62.45 | 36.56 | 55.60 | **37.02** |
| StreamingLLM | 16.50 | 25.01 | 20.11 | 57.30 | 37.10 | 56.43 | **33.99** |
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
**KV Budget:** 512 tokens
**Attention:** flash_attention_2
**Hardware:** NVIDIA GH200 Grace Hopper Superchip

### Results Table

| Method | NarrativeQA | Qasper | MultifieldQA | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum | MultiNews | TREC | TriviaQA | SAMSum | PassageCount | PassageRetrieval | LCC | RepoBench-P | **Avg** |
|--------|-------------|--------|--------------|----------|----------|---------|-----------|-------|-----------|------|----------|--------|--------------|------------------|-----|-------------|---------|
| FullKV | 25.56 | 32.27 | 39.71 | 43.56 | 35.29 | 21.18 | 28.74 | 23.20 | 26.73 | 74.00 | 90.48 | 42.53 | 4.80 | 69.75 | 59.27 | 53.92 | **41.94** |
| SnapKV | 25.55 | 23.80 | 38.46 | 43.78 | 33.42 | 19.92 | 23.28 | 22.46 | 24.23 | 71.50 | 90.57 | 40.34 | 5.43 | 69.50 | 61.10 | 57.41 | **40.67** |
| PyramidKV | 24.79 | 23.44 | 34.85 | 43.29 | 31.63 | 20.09 | 23.47 | 22.92 | 24.26 | 72.00 | 90.61 | 40.82 | 5.83 | 69.50 | 59.30 | 54.62 | **40.09** |
| H2O | 23.56 | 21.45 | 31.57 | 41.00 | 30.73 | 17.97 | 25.09 | 22.41 | 25.58 | 69.00 | 90.67 | 39.98 | 5.65 | 67.62 | 60.86 | 55.98 | **39.32** |
| StreamingLLM | 20.70 | 12.14 | 22.00 | 35.48 | 26.80 | 15.79 | 21.02 | 20.59 | 23.85 | 62.50 | 83.38 | 40.15 | 5.35 | 67.81 | 60.15 | 55.03 | **35.80** |
| OURS |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV | 32.51 | 33.34 | 26.22 | 69.00 | 37.28 | 56.60 | **41.94** |
| SnapKV | 29.27 | 32.37 | 23.32 | 67.47 | 37.47 | 59.26 | **40.67** |
| PyramidKV | 27.69 | 31.67 | 23.55 | 67.81 | 37.67 | 56.96 | **40.09** |
| H2O | 25.53 | 29.90 | 24.36 | 66.55 | 36.64 | 58.42 | **39.32** |
| StreamingLLM | 18.28 | 26.02 | 21.82 | 62.01 | 36.58 | 57.59 | **35.80** |
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