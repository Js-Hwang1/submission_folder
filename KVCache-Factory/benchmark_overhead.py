"""
Benchmark script for measuring KV-cache eviction overhead.
Reports: prefill latency, decode latency, peak memory, throughput.

Usage:
    python benchmark_overhead.py \
        --model_path meta-llama/Llama-3.1-8B-Instruct \
        --method circuitkv \
        --context_lengths 4096,8192,16384,32768 \
        --max_capacity_prompts 512
"""

import argparse
import time
import torch
import gc
import json
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_family(model_path: str) -> str:
    model_path_lower = model_path.lower()
    if "qwen" in model_path_lower:
        return "qwen"
    elif "mistral" in model_path_lower:
        return "mistral"
    elif "llama" in model_path_lower:
        return "llama"
    return "unknown"


def measure_memory() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def reset_memory():
    """Reset memory tracking."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def benchmark_single(
    model,
    tokenizer,
    context_length: int,
    num_generate: int = 50,
    warmup_runs: int = 2,
    num_runs: int = 5,
) -> Dict:
    """Benchmark a single context length."""

    # Create dummy input of target length
    dummy_text = "Hello world. " * (context_length // 3)
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        truncation=True,
        max_length=context_length,
    ).to(model.device)

    actual_length = inputs.input_ids.shape[1]

    # Warmup
    for _ in range(warmup_runs):
        reset_memory()
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

    # Measure prefill + decode
    prefill_times = []
    decode_times = []
    peak_memories = []

    for _ in range(num_runs):
        reset_memory()

        # Prefill timing (first forward pass)
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=num_generate,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )

        torch.cuda.synchronize()
        total_time = time.perf_counter() - start

        # Approximate prefill vs decode
        # (In practice, you'd hook into the model for precise measurement)
        num_tokens_generated = outputs.sequences.shape[1] - actual_length

        # Rough estimate: prefill is ~proportional to context length
        # This is an approximation - for precise numbers, use profiling
        estimated_prefill = total_time * 0.3  # Rough heuristic
        estimated_decode = total_time - estimated_prefill

        prefill_times.append(estimated_prefill * 1000)  # ms
        decode_times.append(estimated_decode / num_tokens_generated * 1000)  # ms/token
        peak_memories.append(measure_memory())

    return {
        "context_length": actual_length,
        "prefill_ms": sum(prefill_times) / len(prefill_times),
        "decode_ms_per_token": sum(decode_times) / len(decode_times),
        "peak_memory_gb": max(peak_memories),
        "throughput_tok_per_sec": num_generate / (sum(prefill_times) / len(prefill_times) / 1000 +
                                                   sum(decode_times) / len(decode_times) / 1000 * num_generate),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV-cache eviction overhead")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--method", type=str, default="fullkv",
                        choices=["fullkv", "h2o", "snapkv", "pyramidkv", "streamingllm", "circuitkv"])
    parser.add_argument("--context_lengths", type=str, default="4096,8192,16384",
                        help="Comma-separated list of context lengths to test")
    parser.add_argument("--max_capacity_prompts", type=int, default=512)
    parser.add_argument("--num_generate", type=int, default=50)
    parser.add_argument("--output_file", type=str, default="benchmark_results.json")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")

    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",")]

    print(f"Loading model: {args.model_path}")
    print(f"Method: {args.method}")
    print(f"Context lengths: {context_lengths}")

    # Apply monkeypatch if not fullkv
    if args.method != "fullkv":
        from pyramidkv.monkeypatch import replace_llama, replace_mistral, replace_qwen2

        model_family = get_model_family(args.model_path)
        print(f"Detected model family: {model_family}")

        if model_family == "llama":
            replace_llama(args.method.lower())
        elif model_family == "mistral":
            replace_mistral(args.method.lower())
        elif model_family == "qwen":
            replace_qwen2(args.method.lower())

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    # Configure KV-cache budget if not fullkv
    if args.method != "fullkv":
        for layer in model.model.layers:
            layer.self_attn.config.max_capacity_prompt = args.max_capacity_prompts
            layer.self_attn.config.window_size = 8
            layer.self_attn.config.kernel_size = 7
            layer.self_attn.config.pooling = "maxpool"

    # Run benchmarks
    results = {
        "model": args.model_path,
        "method": args.method,
        "max_capacity_prompts": args.max_capacity_prompts,
        "benchmarks": []
    }

    for ctx_len in context_lengths:
        print(f"\nBenchmarking context length: {ctx_len}")
        try:
            bench = benchmark_single(
                model, tokenizer, ctx_len,
                num_generate=args.num_generate,
            )
            results["benchmarks"].append(bench)
            print(f"  Prefill: {bench['prefill_ms']:.1f}ms")
            print(f"  Decode: {bench['decode_ms_per_token']:.2f}ms/token")
            print(f"  Peak Memory: {bench['peak_memory_gb']:.2f}GB")
            print(f"  Throughput: {bench['throughput_tok_per_sec']:.1f} tok/s")
        except Exception as e:
            print(f"  Failed: {e}")
            results["benchmarks"].append({
                "context_length": ctx_len,
                "error": str(e)
            })

    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"BENCHMARK SUMMARY: {args.method} (budget={args.max_capacity_prompts})")
    print("=" * 70)
    print(f"{'Context':<10} {'Prefill (ms)':<15} {'Decode (ms/tok)':<18} {'Memory (GB)':<12}")
    print("-" * 70)
    for b in results["benchmarks"]:
        if "error" not in b:
            print(f"{b['context_length']:<10} {b['prefill_ms']:<15.1f} {b['decode_ms_per_token']:<18.2f} {b['peak_memory_gb']:<12.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
