# CircuitKV Benchmark Results

## LongBench Evaluation

**Model:** Meta-Llama-3-8B-Instruct
**KV Budget:** 2048 tokens
**Attention:** flash_attention_2
**Hardware:** NVIDIA GH200 Grace Hopper Superchip

### Results Table

| Method | NarrativeQA | Qasper | MultifieldQA | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum | MultiNews | TREC | TriviaQA | SAMSum | PassageCount | PassageRetrieval | LCC | RepoBench-P | **Avg** |
|--------|-------------|--------|--------------|----------|----------|---------|-----------|-------|-----------|------|----------|--------|--------------|------------------|-----|-------------|---------|
| FullKV | 25.56 | 32.27 | 39.71 | 43.56 | 35.29 | 21.18 | 28.74 | 23.20 | 26.73 | 74.00 | 90.48 | 42.53 | 4.80 | 69.75 | 59.27 | 53.92 | **41.94** |
| SnapKV | 25.70 | 29.79 | 38.97 | 43.90 | 35.04 | 20.92 | 26.95 | 23.46 | 26.11 | 73.50 | 90.56 | 41.83 | 5.61 | 69.25 | 57.70 | 54.23 | **41.47** |
| PyramidKV | 25.07 | 29.61 | 38.25 | 43.69 | 35.35 | 21.43 | 27.02 | 23.25 | 26.05 |      |          |        |              |                  |     |             |         |
| H2O    | 26.06 | 29.07 | 36.61 | 42.30 | 33.33 | 19.72 | 27.45 | 22.66 | 26.69 | 73.00 | 90.93 | 42.02 | 5.88 | 69.50 | 57.47 | 54.83 | **41.10** |
| StreamingLLM |       |        |              |          |          |         |           |       |           |      |          |        |              |                  |     |             |         |
| CircuitKV | 25.10 | 30.85 | 39.99 | 44.34 | 35.05 | 21.36 | 26.16 | 23.21 | 26.39 | 72.00 | 90.56 | 42.14 | 4.83 | 69.25 | 57.57 | 54.58 | **41.46** |



### Task Categories

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Overall** |
|--------|---------------|--------------|---------------|----------|-----------|------|-------------|
| FullKV | 32.51 | 33.34 | 26.22 | 69.00 | 37.28 | 56.60 | **41.94** |
| SnapKV | 31.49 | 33.29 | 25.51 | 68.63 | 37.43 | 55.97 | **41.47** |
| PyramidKV |            |              |               |          |           |      |             |
| H2O    | 30.58 | 31.78 | 25.60 | 68.65 | 37.69 | 56.15 | **41.10** |
| StreamingLLM |         |              |               |          |           |      |             |
| CircuitKV | 31.98 | 33.58 | 25.25 | 68.23 | 37.04 | 56.08 | **41.46** |


**Category Mapping:**
- Single-Doc QA: NarrativeQA , Qasper (25.76), MultifieldQA (37.16) → **Avg: 28.34**
- Multi-Doc QA: HotpotQA, 2WikiMQA, Musique
- Summarization: GovReport, QMSum, MultiNews
- Few-shot: TREC, TriviaQA, SAMSum
- Synthetic: PassageCount, PassageRetrieval
- Code: LCC, RepoBench-P  

---