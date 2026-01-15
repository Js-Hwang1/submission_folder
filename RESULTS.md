# CircuitKV Benchmark Results

## LongBench Evaluation

**Model:** Meta-Llama-3-8B-Instruct
**KV Budget:** 128 tokens
**Attention:** flash_attention_2
**Hardware:** NVIDIA H200 HGX Node (8 H200 GPUS per node)

### Results Table

| Method | NarrativeQA | Qasper | MultifieldQA | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum | MultiNews | TREC | TriviaQA | SAMSum | PassageCount | PassageRetrieval | LCC | RepoBench-P | **Avg** |
|--------|-------------|--------|--------------|----------|----------|---------|-----------|-------|-----------|------|----------|--------|--------------|------------------|-----|-------------|---------|
| FullKV | 25.56 | 32.27 | 39.71 | 43.56 | 35.29 | 21.18 | 28.74 | 23.20 | 26.73 | 74.00 | 90.48 | 42.53 | 4.80 | 69.75 | 59.27 | 53.92 | **41.94** |
| SnapKV | 22.18 | 16.07 | 31.43 | 40.33 | 28.97 | 19.15 | 19.78 | 22.40 | 21.46 | 66.00 | 89.72 | 38.89 | 5.75 | 69.00 | 59.06 | 54.76 | **37.81** |
| PyramidKV | 21.88 | 17.08 | 31.13 | 38.54 | 28.66 | 18.67 | 20.05 | 22.51 | 20.90 | 67.00 | 89.35 | 38.67 | 5.92 | 69.00 | 57.75 | 51.84 | **37.43** |
| H2O | 22.95 | 13.72 | 24.50 | 30.42 | 21.74 | 15.54 | 21.81 | 21.19 | 24.51 | 47.00 | 87.76 | 35.35 | 5.45 | 68.19 | 54.51 | 48.69 | **33.96** |
| StreamingLLM | 18.13 | 8.45 | 21.25 | 32.86 | 25.85 | 15.51 | 16.82 | 20.51 | 18.08 | 45.00 | 74.60 | 36.26 | 5.75 | 68.50 | 55.86 | 53.08 | **32.28** |
| OURS | 23.38 | 16.98 | 34.56 | 43.29 | 34.31 | 18.35 | 20.23 | 22.38 | 21.07 | 59.00 | 90.36 | 36.99 | 5.92 | 69.50 | 53.38 | 50.51 | **37.51** |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV | 32.51 | 33.34 | 26.22 | 69.00 | 37.28 | 56.60 | **41.94** |
| SnapKV | 23.23 | 29.48 | 21.21 | 64.87 | 37.38 | 56.91 | **37.81** |
| PyramidKV | 23.36 | 28.62 | 21.15 | 65.01 | 37.46 | 54.80 | **37.43** |
| H2O | 20.39 | 22.57 | 22.50 | 56.70 | 36.82 | 51.60 | **33.96** |
| StreamingLLM | 15.94 | 24.74 | 18.47 | 51.95 | 37.13 | 54.47 | **32.28** |
| OURS | 24.97 | 31.98 | 21.23 | 62.12 | 37.71 | 51.95 | **37.51** |


**Category Mapping:**
- Single-Doc QA: NarrativeQA , Qasper, MultifieldQA
- Multi-Doc QA: HotpotQA, 2WikiMQA, Musique
- Summarization: GovReport, QMSum, MultiNews
- Few-shot: TREC, TriviaQA, SAMSum
- Synthetic: PassageCount, PassageRetrieval
- Code: LCC, RepoBench-P

---

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
| OURS | 24.07 | 22.65 | 38.34 | 43.41 | 32.48 | 19.44 | 21.98 | 22.69 | 23.01 | 67.50 | 90.36 | 39.37 | 5.83 | 69.70 | 56.59 | 53.91 | **39.46** |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV | 32.51 | 33.34 | 26.22 | 69.00 | 37.28 | 56.60 | **41.94** |
| SnapKV | 26.85 | 31.84 | 22.31 | 67.48 | 37.54 | 57.82 | **39.76** |
| PyramidKV | 26.48 | 31.47 | 22.24 | 67.32 | 37.67 | 56.09 | **39.38** |
| H2O | 23.68 | 26.60 | 23.29 | 62.45 | 36.56 | 55.60 | **37.02** |
| StreamingLLM | 16.50 | 25.01 | 20.11 | 57.30 | 37.10 | 56.43 | **33.99** |
| OURS | 28.35 | 31.78 | 22.56 | 65.74 | 37.77 | 55.25 | **39.46** |


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
| OURS | 23.76 | 26.54 | 38.51 | 43.73 | 34.08 | 22.01 | 23.24 | 22.90 | 24.41 | 71.50 | 90.39 | 40.97 | 5.46 | 69.50 | 60.54 | 56.77 | **40.89** |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV | 32.51 | 33.34 | 26.22 | 69.00 | 37.28 | 56.60 | **41.94** |
| SnapKV | 29.27 | 32.37 | 23.32 | 67.47 | 37.47 | 59.26 | **40.67** |
| PyramidKV | 27.69 | 31.67 | 23.55 | 67.81 | 37.67 | 56.96 | **40.09** |
| H2O | 25.53 | 29.90 | 24.36 | 66.55 | 36.64 | 58.42 | **39.32** |
| StreamingLLM | 18.28 | 26.02 | 21.82 | 62.01 | 36.58 | 57.59 | **35.80** |
| OURS | 29.60 | 33.27 | 23.52 | 67.62 | 37.48 | 58.66 | **40.89** |


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
| OURS | 25.08 | 29.45 | 38.70 | 43.07 | 35.70 | 21.90 | 25.21 | 23.54 | 25.60 | 72.50 | 90.56 | 41.62 | 5.38 | 69.25 | 60.00 | 57.44 | **41.56** |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV | 32.51 | 33.34 | 26.22 | 69.00 | 37.28 | 56.60 | **41.94** |
| SnapKV | 30.08 | 32.66 | 24.47 | 68.27 | 37.19 | 58.26 | **41.08** |
| PyramidKV | 29.65 | 33.23 | 24.18 | 68.24 | 37.42 | 57.47 | **41.04** |
| H2O | 29.27 | 30.63 | 24.90 | 68.42 | 37.29 | 58.13 | **40.65** |
| StreamingLLM | 20.94 | 26.70 | 23.30 | 64.31 | 36.92 | 58.51 | **37.29** |
| OURS | 31.08 | 33.56 | 24.78 | 68.23 | 37.32 | 58.72 | **41.56** |


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
| OURS | 25.05 | 29.97 | 40.41 | 44.66 | 36.19 | 21.21 | 25.97 | 23.59 | 26.26 | 74.00 | 90.64 | 42.68 | 5.20 | 69.25 | 59.91 | 56.29 | **41.96** |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV | 32.51 | 33.34 | 26.22 | 69.00 | 37.28 | 56.60 | **41.94** |
| SnapKV | 31.49 | 33.29 | 25.51 | 68.63 | 37.43 | 55.97 | **41.47** |
| PyramidKV | 30.98 | 33.49 | 25.47 | 68.71 | 37.28 | 55.56 | **41.35** |
| H2O    | 30.58 | 31.78 | 25.60 | 68.65 | 37.69 | 56.15 | **41.10** |
| StreamingLLM | 26.15 | 29.36 | 24.30 | 67.06 | 37.25 | 59.10 | **39.58** |
| OURS | 31.81 | 34.02 | 25.27 | 69.11 | 37.23 | 58.10 | **41.96** |


**Category Mapping:**
- Single-Doc QA: NarrativeQA , Qasper, MultifieldQA
- Multi-Doc QA: HotpotQA, 2WikiMQA, Musique
- Summarization: GovReport, QMSum, MultiNews
- Few-shot: TREC, TriviaQA, SAMSum
- Synthetic: PassageCount, PassageRetrieval
- Code: LCC, RepoBench-P

---

**Model:** Meta-Llama-3.1-8B-Instruct
**KV Budget:** 128 tokens
**Attention:** flash_attention_2
**Hardware:** NVIDIA GH200 Grace Hopper Superchip

### Results Table

| Method | NarrativeQA | Qasper | MultifieldQA | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum | MultiNews | TREC | TriviaQA | SAMSum | PassageCount | PassageRetrieval | LCC | RepoBench-P | **Avg** |
|--------|-------------|--------|--------------|----------|----------|---------|-----------|-------|-----------|------|----------|--------|--------------|------------------|-----|-------------|---------|
| FullKV |
| SnapKV | 23.40 | 8.69 | 20.65 | 14.66 | 14.43 | 8.19 | 22.24 | 22.19 | 21.73 | 62.00 | 90.02 | 39.85 | 8.49 | 92.77 | 60.44 | 50.57 | **35.02** |
| PyramidKV | 22.27 | 8.12 | 20.77 | 13.27 | 12.99 | 8.94 | 22.22 | 22.16 | 21.37 | 64.50 | 88.65 | 39.71 | 7.80 | 93.13 | 56.97 | 49.58 | **34.53** |
| H2O |
| StreamingLLM | 16.22 | 5.35 | 14.89 | 10.95 | 12.43 | 6.51 | 17.97 | 18.96 | 18.67 | 40.50 | 85.60 | 38.34 | 9.57 | 94.03 | 58.85 | 49.47 | **31.14** |
| CircuitKV |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV |
| SnapKV | 17.58 | 12.43 | 22.05 | 63.96 | 50.63 | 55.51 | **35.02** |
| PyramidKV | 17.05 | 11.73 | 21.92 | 64.29 | 50.47 | 53.28 | **34.53** |
| H2O |
| StreamingLLM | 12.15 | 9.96 | 18.53 | 54.81 | 51.80 | 54.16 | **31.14** |
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
**KV Budget:** 256 tokens
**Attention:** flash_attention_2
**Hardware:** NVIDIA GH200 Grace Hopper Superchip

### Results Table

| Method | NarrativeQA | Qasper | MultifieldQA | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum | MultiNews | TREC | TriviaQA | SAMSum | PassageCount | PassageRetrieval | LCC | RepoBench-P | **Avg** |
|--------|-------------|--------|--------------|----------|----------|---------|-----------|-------|-----------|------|----------|--------|--------------|------------------|-----|-------------|---------|
| FullKV |
| SnapKV | 27.18 | 9.47 | 23.03 | 15.07 | 15.09 | 9.82 | 24.11 | 22.78 | 23.44 | 70.00 | 91.44 | 41.26 | 7.18 | 95.59 | 61.99 | 53.03 | **36.91** |
| PyramidKV | 25.48 | 8.90 | 22.31 | 14.11 | 14.49 | 9.14 | 23.89 | 22.79 | 23.05 | 69.50 | 90.22 | 40.88 | 7.81 | 96.51 | 59.88 | 52.15 | **36.32** |
| H2O |
| StreamingLLM | 15.90 | 5.57 | 14.89 | 10.42 | 12.30 | 7.02 | 20.42 | 19.54 | 20.59 | 46.00 | 87.50 | 41.02 | 9.57 | 90.86 | 61.03 | 51.31 | **32.12** |
| CircuitKV |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV |
| SnapKV | 19.89 | 13.33 | 23.44 | 67.57 | 51.39 | 57.51 | **36.91** |
| PyramidKV | 18.90 | 12.58 | 23.24 | 66.87 | 52.16 | 56.02 | **36.32** |
| H2O |
| StreamingLLM | 12.12 | 9.91 | 20.18 | 58.17 | 50.22 | 56.17 | **32.12** |
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
**KV Budget:** 512 tokens
**Attention:** flash_attention_2
**Hardware:** NVIDIA GH200 Grace Hopper Superchip

### Results Table

| Method | NarrativeQA | Qasper | MultifieldQA | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum | MultiNews | TREC | TriviaQA | SAMSum | PassageCount | PassageRetrieval | LCC | RepoBench-P | **Avg** |
|--------|-------------|--------|--------------|----------|----------|---------|-----------|-------|-----------|------|----------|--------|--------------|------------------|-----|-------------|---------|
| FullKV |
| SnapKV | 29.46 | 11.15 | 24.97 | 15.36 | 15.72 | 9.47 | 26.35 | 23.43 | 24.66 | 70.50 | 91.73 | 41.29 | 7.64 | 96.39 | 63.68 | 55.48 | **37.96** |
| PyramidKV | 29.58 | 10.44 | 24.66 | 14.42 | 15.78 | 9.83 | 25.72 | 23.58 | 24.51 | 70.00 | 91.88 | 41.38 | 7.58 | 97.13 | 62.72 | 54.15 | **37.71** |
| H2O |
| StreamingLLM | 19.15 | 6.51 | 15.05 | 10.94 | 12.52 | 6.19 | 23.59 | 19.95 | 23.33 | 57.50 | 87.68 | 41.76 | 10.25 | 90.72 | 62.44 | 53.45 | **33.81** |
| CircuitKV |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV |
| SnapKV | 21.86 | 13.52 | 24.81 | 67.84 | 52.02 | 59.58 | **37.96** |
| PyramidKV | 21.56 | 13.34 | 24.60 | 67.75 | 52.36 | 58.44 | **37.71** |
| H2O |
| StreamingLLM | 13.57 | 9.88 | 22.29 | 62.31 | 50.49 | 57.95 | **33.81** |
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
**KV Budget:** 1024 tokens
**Attention:** flash_attention_2
**Hardware:** NVIDIA GH200 Grace Hopper Superchip

### Results Table

| Method | NarrativeQA | Qasper | MultifieldQA | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum | MultiNews | TREC | TriviaQA | SAMSum | PassageCount | PassageRetrieval | LCC | RepoBench-P | **Avg** |
|--------|-------------|--------|--------------|----------|----------|---------|-----------|-------|-----------|------|----------|--------|--------------|------------------|-----|-------------|---------|
| FullKV |
| SnapKV | 32.15 | 11.99 | 27.35 | 15.98 | 15.67 | 10.54 | 28.57 | 23.15 | 25.84 | 70.00 | 91.73 | 43.12 | 7.47 | 97.81 | 64.18 | 56.47 | **38.88** |
| PyramidKV | 31.99 | 11.68 | 27.20 | 15.99 | 15.83 | 10.57 | 27.89 | 23.74 | 26.14 | 70.00 | 92.09 | 42.62 | 8.06 | 97.78 | 64.10 | 55.87 | **38.85** |
| H2O |
| StreamingLLM | 20.50 | 8.07 | 15.51 | 11.62 | 12.39 | 6.72 | 25.77 | 20.11 | 25.44 | 63.50 | 88.84 | 42.65 | 10.12 | 92.10 | 63.10 | 55.66 | **35.13** |
| CircuitKV |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV |
| SnapKV | 23.83 | 14.06 | 25.85 | 68.28 | 52.64 | 60.33 | **38.88** |
| PyramidKV | 23.62 | 14.13 | 25.92 | 68.24 | 52.92 | 59.99 | **38.85** |
| H2O |
| StreamingLLM | 14.69 | 10.24 | 23.77 | 65.00 | 51.11 | 59.38 | **35.13** |
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
| SnapKV | 31.57 | 12.42 | 27.00 | 16.65 | 16.46 | 11.37 | 30.55 | 23.58 | 26.44 | 71.00 | 91.48 | 42.81 | 7.45 | 97.48 | 64.86 | 58.13 | **39.33** |
| PyramidKV | 32.66 | 12.63 | 26.41 | 15.80 | 16.49 | 11.81 | 30.92 | 23.46 | 26.44 | 70.50 | 91.65 | 43.50 | 8.03 | 97.46 | 64.92 | 57.71 | **39.40** |
| H2O |
| StreamingLLM | 22.61 | 10.40 | 17.53 | 12.31 | 13.60 | 7.66 | 28.72 | 20.63 | 26.66 | 67.50 | 90.98 | 42.45 | 7.97 | 95.12 | 64.92 | 57.41 | **36.65** |
| CircuitKV |


### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV |
| SnapKV | 23.66 | 14.83 | 26.86 | 68.43 | 52.47 | 61.50 | **39.33** |
| PyramidKV | 23.90 | 14.70 | 26.94 | 68.55 | 52.75 | 61.32 | **39.40** |
| H2O |
| StreamingLLM | 16.85 | 11.19 | 25.34 | 66.98 | 51.55 | 61.17 | **36.65** |
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