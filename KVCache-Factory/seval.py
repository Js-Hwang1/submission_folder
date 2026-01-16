import os
import json
import sys
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

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
        content = f.read().strip()
        if not content:
            return None  # Empty file
        for line in content.split('\n'):
            if not line.strip():
                continue
            data = json.loads(line)
            predictions.append(data["pred"])
            answers.append(data["answers"])
            all_classes = data.get("all_classes")

    if not predictions:
        return None  # No valid predictions

    total = 0
    for pred, gts in zip(predictions, answers):
        if dataset in ["trec", "triviaqa", "samsum"]:
            pred = pred.lstrip('\n').split('\n')[0]
        score = max(dataset2metric[dataset](pred, gt, all_classes=all_classes) for gt in gts)
        total += score
    return round(100 * total / len(predictions), 2)

def evaluate_single_task(args):
    """Worker function for parallel evaluation."""
    method_path, dataset, filepath = args
    try:
        score = evaluate_file(filepath, dataset)
        return (method_path, dataset, score)
    except Exception as e:
        return (method_path, dataset, f"Error: {e}")

def find_all_tasks(root_path):
    """Find all (method_path, dataset, filepath) tasks under root_path."""
    tasks = []
    # Try glob first, but also check if path exists literally (for paths with special chars)
    root_dirs = glob.glob(root_path)
    if not root_dirs and os.path.isdir(root_path):
        root_dirs = [root_path]

    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            continue

        subdirs = [d for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]

        for subdir in subdirs:
            subdir_path = os.path.join(root_dir, subdir)
            has_datasets = any(d in dataset2metric for d in os.listdir(subdir_path)
                              if os.path.isdir(os.path.join(subdir_path, d)))

            if has_datasets:
                tasks.extend(find_tasks_in_method_dir(subdir_path))
            else:
                method_dirs = [d for d in os.listdir(subdir_path)
                              if os.path.isdir(os.path.join(subdir_path, d)) and not d.startswith('.')]
                for method_dir in method_dirs:
                    method_path = os.path.join(subdir_path, method_dir)
                    tasks.extend(find_tasks_in_method_dir(method_path))
    return tasks

def find_tasks_in_method_dir(method_dir):
    """Find all evaluation tasks in a method directory."""
    tasks = []
    for dataset in dataset2metric.keys():
        dataset_dir = os.path.join(method_dir, dataset)
        if os.path.isdir(dataset_dir):
            json_files = glob.glob(os.path.join(dataset_dir, "*.json"))
            if json_files:
                tasks.append((method_dir, dataset, json_files[0]))
    return tasks

def extract_method_name(method_dir):
    """Extract method name from directory like 'meta-llama-3-8b-instruct_H2O_128'."""
    parts = method_dir.split('_')
    return parts[-2] if len(parts) >= 2 else method_dir

def evaluate_root_dir(root_path, num_workers=None):
    """Evaluate a root directory with parallel processing."""
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    # Find all tasks
    print("Scanning for evaluation tasks...")
    tasks = find_all_tasks(root_path)

    if not tasks:
        print(f"No evaluation tasks found matching: {root_path}")
        return

    print(f"Found {len(tasks)} tasks. Running with {num_workers} workers...\n")

    # Run evaluations in parallel
    results = {}  # method_path -> {dataset -> score}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(evaluate_single_task, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            method_path, dataset, score = future.result()
            if method_path not in results:
                results[method_path] = {}
            results[method_path][dataset] = score
            if completed % 20 == 0 or completed == len(tasks):
                print(f"\rProgress: {completed}/{len(tasks)}", end="", flush=True)

    print("\n")
    print_results(root_path, results)

def print_results(root_path, results):
    """Print results organized by root/budget/method."""
    root_dirs = sorted(glob.glob(root_path))
    if not root_dirs and os.path.isdir(root_path):
        root_dirs = [root_path]

    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            continue

        print(f"\n{'='*70}")
        print(f"Root: {root_dir}")
        print('='*70)

        subdirs = sorted([d for d in os.listdir(root_dir)
                         if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])

        for subdir in subdirs:
            subdir_path = os.path.join(root_dir, subdir)
            has_datasets = any(d in dataset2metric for d in os.listdir(subdir_path)
                              if os.path.isdir(os.path.join(subdir_path, d)))

            if has_datasets:
                if subdir_path in results:
                    print(f"\n  {subdir}")
                    print(f"  {'-'*60}")
                    print_method_results(results[subdir_path], indent=4)
            else:
                print(f"\n  [{subdir}]")
                print(f"  {'-'*60}")

                method_dirs = sorted([d for d in os.listdir(subdir_path)
                                     if os.path.isdir(os.path.join(subdir_path, d)) and not d.startswith('.')])

                for method_dir in method_dirs:
                    method_path = os.path.join(subdir_path, method_dir)
                    if method_path in results:
                        method_name = extract_method_name(method_dir)
                        print(f"\n    {method_name}")
                        print_method_results(results[method_path], indent=6)

def print_method_results(dataset_results, indent=4):
    """Print results for a single method with average."""
    prefix = ' ' * indent
    scores = [s for s in dataset_results.values() if isinstance(s, (int, float))]
    avg = sum(scores) / len(scores) if scores else 0

    for dataset in sorted(dataset_results.keys()):
        score = dataset_results[dataset]
        if score is None:
            print(f"{prefix}{dataset:25}     --")
        elif isinstance(score, (int, float)):
            print(f"{prefix}{dataset:25} {score:6.2f}")
        else:
            print(f"{prefix}{dataset:25} {score}")
    print(f"{prefix}{'-'*32}")
    if scores:
        print(f"{prefix}{'AVERAGE':25} {avg:6.2f}")
    else:
        print(f"{prefix}{'AVERAGE':25}     --")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python seval.py <root_directory> [num_workers]")
        print("Example: python seval.py '/Users/j/Desktop/results*/Llama3-8B-Instruct'")
        sys.exit(1)

    root_path = sys.argv[1]
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    evaluate_root_dir(root_path, num_workers)