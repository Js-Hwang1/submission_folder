import os
import json
import sys
sys.path.insert(0, '/Users/j/Desktop/submission_folder/KVCache-Factory')
from metrics import qa_f1_score, rouge_score, classification_score, retrieval_score, count_score, code_sim_score

dataset2metric = {
    "narrativeqa": qa_f1_score, "qasper": qa_f1_score, "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score, "musique": qa_f1_score,
    "gov_report": rouge_score, "qmsum": rouge_score, "multi_news": rouge_score,
    "trec": classification_score, "triviaqa": qa_f1_score, "samsum": rouge_score,
    "passage_count": count_score, "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score, "repobench-p": code_sim_score,
}

def evaluate_file(filepath, dataset):
    predictions, answers = [], []
    all_classes = None
    with open(filepath) as f:
        for line in f:
            data = json.loads(line)
            predictions.append(data["pred"])
            answers.append(data["answers"])
            all_classes = data.get("all_classes")
    
    total = 0
    for pred, gts in zip(predictions, answers):
        if dataset in ["trec", "triviaqa", "samsum"]:
            pred = pred.lstrip('\n').split('\n')[0]
        score = max(dataset2metric[dataset](pred, gt, all_classes=all_classes) for gt in gts)
        total += score
    return round(100 * total / len(predictions), 2)

# Adjust this path to match your results directory
results_dir = "/Users/j/Desktop/results/Llama3-8B-Instruct/Llama3-8B-Instruct-256/meta-llama-3-8b-instruct_SKV_256"

print("CircuitKV Results:")
print("-" * 40)
for dataset in sorted(os.listdir(results_dir)):
    filepath = os.path.join(results_dir, dataset, "snapkv.json")
    if os.path.exists(filepath):
        score = evaluate_file(filepath, dataset)
        print(f"{dataset:25} {score:6.2f}")