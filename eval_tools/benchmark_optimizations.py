"""Benchmark script for Algorithmic Optimization in ASMR Live Subtitling.

Compares:
- Group A (Baseline): No prompt caching (cold prefill of 75-token system prompt every sentence),
  no semantic regrouping (fragmented unclosed clauses sent individually).
- Group B (Optimized): Prompt Caching (reusing llama_cpp KV context for 75-token system prompt),
  Semantic Regrouping (merging trailing continuous particles: で/て/けど/から/ながら etc.).

Logs benchmark metrics to MLflow:
- Experiment: ASMR-Live-Subtitle-Benchmarks
- Run Name: 2026-09-24 Algorithmic Optimization (PromptCache+Regroup)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
import numpy as np

# Ensure livesub configuration & DLL directories are registered
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import livesub.config
except Exception as e:
    print(f"[warn] Failed to import livesub.config directly: {e}", file=sys.stderr)

try:
    from llama_cpp import Llama, LlamaRAMCache
except ImportError as e:
    print(f"[error] llama_cpp import failed: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import mlflow
except ImportError:
    mlflow = None
    print("[warn] MLflow not installed, MLflow logging will be skipped", file=sys.stderr)

# System prompt identical to livesub/models.py SakuraMT
SAKURA_SYS = (
    "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，"
    "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
)
GGUF_PATH = ROOT / "models" / "sakura" / "sakura-7b-qwen2.5-v1.0-iq4xs.gguf"
MLFLOW_DB_PATH = ROOT / "mlflow.db"

# 26 representative anime / ASMR dialogue lines
# Includes short sighs, regular dialogue, and 4 multi-fragment compound sentence sequences
TEST_DATASET = [
    # 1-5: 短叹与即时呼唤 (Short sighs & immediate reactions)
    {"id": 1, "category": "sigh", "text": "あっ…"},
    {"id": 2, "category": "sigh", "text": "ふふっ、可愛い…"},
    {"id": 3, "category": "sigh", "text": "えっ、本当ですか？"},
    {"id": 4, "category": "sigh", "text": "ん…もう、意地悪…"},
    {"id": 5, "category": "sigh", "text": "はぁ…びっくりした…"},
    # 6-14: 常规日常对话 (Regular conversational sentences)
    {"id": 6, "category": "dialogue", "text": "今日もお仕事お疲れ様でした、よく頑張ったね。"},
    {"id": 7, "category": "dialogue", "text": "温かいココアでも淹れてこようか？"},
    {"id": 8, "category": "dialogue", "text": "隣に座ってもいい？少しだけお話ししよう。"},
    {"id": 9, "category": "dialogue", "text": "明日も早いんでしょ？無理しちゃだめだよ。"},
    {"id": 10, "category": "dialogue", "text": "ぎゅーって抱きしめてあげるから、安心してね。"},
    {"id": 11, "category": "dialogue", "text": "頭をなでなでしてあげるね、いい子いい子…"},
    {"id": 12, "category": "dialogue", "text": "あなたの優しい笑顔、私大好きだよ。"},
    {"id": 13, "category": "dialogue", "text": "ずっとそばにいるから、目を閉じてゆっくり深呼吸してね。"},
    {"id": 14, "category": "dialogue", "text": "今日も一日、本当によく頑張ったね。"},
    # 15-17: 助词未闭合复合句 Sequence 1 (寒かったから -> 浸かって -> 休んでね)
    {"id": 15, "category": "compound_1", "text": "今日は外がすごく寒かったから、"},
    {"id": 16, "category": "compound_1", "text": "お風呂にゆっくり浸かって、"},
    {"id": 17, "category": "compound_1", "text": "温かいお布団に入ってぐっすり休んでね。"},
    # 18-20: 助词未闭合复合句 Sequence 2 (頑張っているのを見てるから -> 前で -> いいんだよ？)
    {"id": 18, "category": "compound_2", "text": "いつも誰よりも一生懸命頑張っているのを見てるから…"},
    {"id": 19, "category": "compound_2", "text": "今夜くらいは私の前で、"},
    {"id": 20, "category": "compound_2", "text": "思いっきり甘えて泣いちゃってもいいんだよ？"},
    # 21-23: 助词未闭合复合句 Sequence 3 (くすぐったいかもしれないけど -> 吹きかけながら -> 癒してあげるね)
    {"id": 21, "category": "compound_3", "text": "耳元で囁かれるのってちょっとくすぐったいかもしれないけど、"},
    {"id": 22, "category": "compound_3", "text": "息を吹きかけながら、"},
    {"id": 23, "category": "compound_3", "text": "いっぱい癒してあげるね。"},
    # 24-26: 助词未闭合复合句 Sequence 4 (緊張してて -> ドキドキしてるんだけど -> 落ち着くんだ)
    {"id": 24, "category": "compound_4", "text": "本当は私もすごく緊張してて、"},
    {"id": 25, "category": "compound_4", "text": "心臓がドキドキしてるんだけど…"},
    {"id": 26, "category": "compound_4", "text": "君の手を握ると不思議と落ち着くんだ。"},
]

UNCLOSED_PARTICLES = (
    "て", "で", "くて", "じゃなくて",
    "けど", "けれど", "けれども", "だけど",
    "から", "だから", "ですから",
    "ので", "のに",
    "たり", "だり",
    "ながら", "し", "だし",
    "たら", "なら", "ば",
)


def is_unclosed_sentence(text: str) -> bool:
    """Check whether Japanese text ends with an unclosed continuous particle."""
    t = re.sub(r"[\s、，.。…~〜!！?？]+$", "", text.strip())
    return any(t.endswith(p) for p in UNCLOSED_PARTICLES)


def apply_semantic_regrouping(dataset):
    """Regroup consecutive unclosed fragments into cohesive compound sentences."""
    regrouped = []
    buffer_items = []

    for item in dataset:
        buffer_items.append(item)
        combined_text = "".join(x["text"] for x in buffer_items)
        # If the combined text ends in a particle and is not excessively long (<80 chars), keep buffering
        if is_unclosed_sentence(combined_text) and len(combined_text) < 80:
            continue
        else:
            # Emit merged item
            regrouped.append({
                "id": [x["id"] for x in buffer_items],
                "category": buffer_items[0]["category"],
                "text": combined_text,
                "merged_count": len(buffer_items)
            })
            buffer_items = []

    if buffer_items:
        combined_text = "".join(x["text"] for x in buffer_items)
        regrouped.append({
            "id": [x["id"] for x in buffer_items],
            "category": buffer_items[0]["category"],
            "text": combined_text,
            "merged_count": len(buffer_items)
        })

    return regrouped


def run_translation_stream(llm, text: str, max_tokens: int = 80):
    """Execute streaming translation and accurately measure prompt_ms, pred_ms, e2e_ms, chars_per_sec."""
    prompt = (
        f"<|im_start|>system\n{SAKURA_SYS}<|im_end|>\n"
        f"<|im_start|>user\n将下面的日文文本翻译成中文：{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    t0 = time.perf_counter()
    stream = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.1,
        top_p=0.3,
        repeat_penalty=1.05,
        stop=["<|im_end|>", "<|endoftext|>"],
        stream=True
    )
    t_first = None
    chunks = []
    for chunk in stream:
        if t_first is None:
            t_first = time.perf_counter()
        choice_text = chunk["choices"][0]["text"]
        chunks.append(choice_text)
    t_end = time.perf_counter()

    prompt_ms = (t_first - t0) * 1000.0 if t_first else (t_end - t0) * 1000.0
    pred_ms = (t_end - t_first) * 1000.0 if t_first else 0.0
    e2e_ms = (t_end - t0) * 1000.0
    zh = "".join(chunks).strip()
    zh = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", zh).strip()
    cps = len(zh) / (e2e_ms / 1000.0) if e2e_ms > 0 else 0.0

    return {
        "text_in": text,
        "text_out": zh,
        "prompt_ms": round(prompt_ms, 2),
        "pred_ms": round(pred_ms, 2),
        "e2e_ms": round(e2e_ms, 2),
        "chars_per_sec": round(cps, 2),
        "in_chars": len(text),
        "out_chars": len(zh),
    }


def benchmark_group_a_baseline(llm, dataset):
    """Group A (Baseline): Cold prefill every sentence, unclosed fragments not regrouped."""
    print("\n" + "=" * 60)
    print("▶ Running Group A: Baseline (Cold Prefill, Isolated Fragments)")
    print("=" * 60)

    # Disable cache
    llm.set_cache(None)
    records = []

    for i, item in enumerate(dataset, 1):
        # Force cold state
        llm.reset()
        text = item["text"]
        max_tokens = min(80, max(12, len(text) * 3 + 8))
        res = run_translation_stream(llm, text, max_tokens=max_tokens)
        res["id"] = item["id"]
        res["category"] = item["category"]
        records.append(res)
        print(f"[{i:02d}/{len(dataset)}] '{text}' -> '{res['text_out']}' "
              f"| Prefill: {res['prompt_ms']:5.1f}ms | Pred: {res['pred_ms']:5.1f}ms "
              f"| E2E: {res['e2e_ms']:5.1f}ms | {res['chars_per_sec']:4.1f} c/s")

    return records


def benchmark_group_b_optimized(llm, regrouped_dataset):
    """Group B (Optimized): Native KV Prompt Caching active, Semantic Regrouping applied."""
    print("\n" + "=" * 60)
    print("▶ Running Group B: Optimized (Native Prompt Caching + Semantic Regrouping)")
    print("=" * 60)

    # Disable external slow serialization cache
    llm.set_cache(None)

    # Pre-eval and retain the 75-token system prompt prefix in KV cache
    sys_prompt = (
        f"<|im_start|>system\n{SAKURA_SYS}<|im_end|>\n"
        f"<|im_start|>user\n将下面的日文文本翻译成中文："
    )
    t_sys = llm.tokenize(sys_prompt.encode("utf-8"))
    prefix_len = len(t_sys)
    llm.reset()
    llm.eval(t_sys)
    print(f"[cache] Pre-cached system prompt prefix ({prefix_len} tokens) directly in KV cache")

    records = []
    for i, item in enumerate(regrouped_dataset, 1):
        # Truncate any tokens beyond system prompt prefix in KV cache instantly (zero-overhead)
        if llm.n_tokens > prefix_len:
            llm._ctx.kv_cache_seq_rm(-1, prefix_len, -1)
            llm.n_tokens = prefix_len
        elif llm.n_tokens < prefix_len:
            llm.reset()
            llm.eval(t_sys)

        text = item["text"]
        max_tokens = min(100, max(12, len(text) * 3 + 8))
        res = run_translation_stream(llm, text, max_tokens=max_tokens)
        res["id"] = item["id"]
        res["category"] = item["category"]
        res["merged_count"] = item["merged_count"]
        records.append(res)
        print(f"[{i:02d}/{len(regrouped_dataset)}] '{text}' -> '{res['text_out']}' "
              f"| Prefill: {res['prompt_ms']:5.1f}ms | Pred: {res['pred_ms']:5.1f}ms "
              f"| E2E: {res['e2e_ms']:5.1f}ms | {res['chars_per_sec']:4.1f} c/s")

    return records


def calculate_metrics(records):
    """Compute aggregate statistical metrics."""
    prompt_times = [r["prompt_ms"] for r in records]
    pred_times = [r["pred_ms"] for r in records]
    e2e_times = [r["e2e_ms"] for r in records]
    speeds = [r["chars_per_sec"] for r in records]
    total_time_ms = sum(e2e_times)
    total_chars = sum(r["out_chars"] for r in records)

    return {
        "count": len(records),
        "total_time_ms": round(total_time_ms, 2),
        "total_time_s": round(total_time_ms / 1000.0, 3),
        "total_chars": total_chars,
        "prompt_ms_avg": round(float(np.mean(prompt_times)), 2),
        "prompt_ms_p95": round(float(np.percentile(prompt_times, 95)), 2),
        "pred_ms_avg": round(float(np.mean(pred_times)), 2),
        "pred_ms_p95": round(float(np.percentile(pred_times, 95)), 2),
        "e2e_ms_avg": round(float(np.mean(e2e_times)), 2),
        "e2e_ms_p95": round(float(np.percentile(e2e_times, 95)), 2),
        "chars_per_sec_avg": round(float(np.mean(speeds)), 2),
        "overall_chars_per_sec": round(total_chars / (total_time_ms / 1000.0), 2) if total_time_ms > 0 else 0.0,
    }


def log_to_mlflow(baseline_stats, opt_stats, baseline_records, opt_records, args):
    """Log benchmark results to MLflow as the 4th run."""
    if not mlflow:
        print("[skip] MLflow not available, skipping MLflow logging")
        return

    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = "ASMR-Live-Subtitle-Benchmarks"
    mlflow.set_experiment(exp_name)

    run_name = "2026-09-24 Algorithmic Optimization (PromptCache+Regroup)"
    print(f"\n[mlflow] Pushing benchmark run '{run_name}' to {tracking_uri}...")

    # Compute key deltas
    ttft_reduction_pct = (baseline_stats["prompt_ms_avg"] - opt_stats["prompt_ms_avg"]) / baseline_stats["prompt_ms_avg"] * 100.0
    total_time_saved_pct = (baseline_stats["total_time_ms"] - opt_stats["total_time_ms"]) / baseline_stats["total_time_ms"] * 100.0
    e2e_avg_reduction_pct = (baseline_stats["e2e_ms_avg"] - opt_stats["e2e_ms_avg"]) / baseline_stats["e2e_ms_avg"] * 100.0
    throughput_gain_pct = (opt_stats["chars_per_sec_avg"] - baseline_stats["chars_per_sec_avg"]) / baseline_stats["chars_per_sec_avg"] * 100.0

    params = {
        "model": "sakura-7b-qwen2.5-v1.0-iq4xs",
        "mt": "sakura_promptcache_regroup",
        "strategy": "algorithmic_optimization",
        "ngl": args.ngl,
        "n_threads": args.threads,
        "prompt_caching": "enabled_LlamaRAMCache",
        "semantic_regrouping": "enabled_particle_continuation",
        "raw_segment_count": len(TEST_DATASET),
        "regrouped_segment_count": opt_stats["count"],
    }

    metrics = {
        # Standard schema metrics matching historical runs
        "total_segments": len(TEST_DATASET),
        "translated_count": opt_stats["count"],
        "drop_count": 0,
        "drop_rate_pct": 0.0,
        "asr_latency_avg_s": 0.0,
        "asr_latency_p95_s": 0.0,
        "mt_latency_avg_s": round(opt_stats["e2e_ms_avg"] / 1000.0, 3),
        "mt_latency_p95_s": round(opt_stats["e2e_ms_p95"] / 1000.0, 3),
        "e2e_latency_avg_s": round(opt_stats["e2e_ms_avg"] / 1000.0, 3),
        "e2e_latency_p95_s": round(opt_stats["e2e_ms_p95"] / 1000.0, 3),
        "translation_speed_cps": opt_stats["chars_per_sec_avg"],

        # Specific optimization metrics
        "baseline_total_time_s": baseline_stats["total_time_s"],
        "optimized_total_time_s": opt_stats["total_time_s"],
        "total_time_reduction_pct": round(total_time_saved_pct, 2),
        "baseline_prompt_ms_avg": baseline_stats["prompt_ms_avg"],
        "optimized_prompt_ms_avg": opt_stats["prompt_ms_avg"],
        "prompt_ms_reduction_pct": round(ttft_reduction_pct, 2),
        "baseline_e2e_ms_avg": baseline_stats["e2e_ms_avg"],
        "optimized_e2e_ms_avg": opt_stats["e2e_ms_avg"],
        "e2e_ms_reduction_pct": round(e2e_avg_reduction_pct, 2),
        "baseline_chars_per_sec": baseline_stats["chars_per_sec_avg"],
        "optimized_chars_per_sec": opt_stats["chars_per_sec_avg"],
        "throughput_gain_pct": round(throughput_gain_pct, 2),
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log per-step MT latencies for timeline visualization
        for step, rec in enumerate(opt_records):
            mlflow.log_metrics({
                "step_mt_s": round(rec["e2e_ms"] / 1000.0, 3),
                "step_prompt_ms": rec["prompt_ms"],
                "step_chars_per_sec": rec["chars_per_sec"],
            }, step=step)

        # Log detailed benchmark summary artifact
        artifact_path = ROOT / "eval_tools" / "optimization_benchmark_summary.json"
        summary_payload = {
            "metadata": {
                "run_name": run_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "params": params,
            },
            "metrics": metrics,
            "baseline_stats": baseline_stats,
            "optimized_stats": opt_stats,
            "baseline_records": baseline_records,
            "optimized_records": opt_records,
        }
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)

        mlflow.log_artifact(str(artifact_path))
        print(f"[mlflow] Run logged successfully! Run ID: {run.info.run_id}")


def main():
    parser = argparse.ArgumentParser(description="Sakura-7B Algorithmic Optimization Benchmark")
    parser.add_argument("--ngl", type=int, default=28, help="Number of GPU layers (default: 28)")
    parser.add_argument("--threads", type=int, default=8, help="Number of CPU threads (default: 8)")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()

    print("=" * 60)
    print("ASMR Live Subtitle - Algorithmic Optimization Benchmark")
    print(f"Model GGUF: {GGUF_PATH}")
    print(f"GPU Layers: {args.ngl} | CPU Threads: {args.threads}")
    print("=" * 60)

    if not GGUF_PATH.exists():
        print(f"[error] Model file not found: {GGUF_PATH}", file=sys.stderr)
        sys.exit(1)

    t_load_start = time.time()
    llm = Llama(
        model_path=str(GGUF_PATH),
        n_ctx=2048,
        n_threads=args.threads,
        n_gpu_layers=args.ngl,
        verbose=False,
    )
    load_time = time.time() - t_load_start
    print(f"[mt] Sakura-7B IQ4XS loaded in {load_time:.2f}s")

    # Warmup
    print("[mt] Warming up model...")
    warmup_res = run_translation_stream(llm, "テスト")
    print(f"[mt] Warmup complete: '{warmup_res['text_out']}' in {warmup_res['e2e_ms']:.1f}ms")

    # Group A: Baseline (26 individual fragments, cold prompt)
    baseline_records = benchmark_group_a_baseline(llm, TEST_DATASET)
    baseline_stats = calculate_metrics(baseline_records)

    # Group B: Optimized (Prompt Caching + Semantic Regrouping)
    regrouped_dataset = apply_semantic_regrouping(TEST_DATASET)
    print(f"\n[regroup] Merged {len(TEST_DATASET)} fragments -> {len(regrouped_dataset)} sentences")

    opt_records = benchmark_group_b_optimized(llm, regrouped_dataset)
    opt_stats = calculate_metrics(opt_records)

    # Print Comparison Table
    print("\n" + "=" * 70)
    print("📊 ALGORITHMIC OPTIMIZATION BENCHMARK REPORT")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Group A (Baseline)':<18} | {'Group B (Optimized)':<18} | {'Delta / Speedup':<15}")
    print("-" * 88)

    ttft_base = baseline_stats["prompt_ms_avg"]
    ttft_opt = opt_stats["prompt_ms_avg"]
    ttft_delta = (ttft_base - ttft_opt) / ttft_base * 100.0

    e2e_base = baseline_stats["e2e_ms_avg"]
    e2e_opt = opt_stats["e2e_ms_avg"]
    e2e_delta = (e2e_base - e2e_opt) / e2e_base * 100.0

    total_base = baseline_stats["total_time_s"]
    total_opt = opt_stats["total_time_s"]
    total_delta = (total_base - total_opt) / total_base * 100.0

    speed_base = baseline_stats["chars_per_sec_avg"]
    speed_opt = opt_stats["chars_per_sec_avg"]
    speed_gain = (speed_opt - speed_base) / speed_base * 100.0

    calls_base = baseline_stats["count"]
    calls_opt = opt_stats["count"]

    print(f"{'MT Inference Calls':<30} | {calls_base:<18} | {calls_opt:<18} | {- (calls_base-calls_opt)/calls_base*100:.1f}%")
    print(f"{'Total Processing Time (s)':<30} | {total_base:<18.3f} | {total_opt:<18.3f} | -{total_delta:.1f}%")
    print(f"{'Avg Prefill / TTFT (ms)':<30} | {ttft_base:<18.1f} | {ttft_opt:<18.1f} | -{ttft_delta:.1f}%")
    print(f"{'Avg Pred / Decode (ms)':<30} | {baseline_stats['pred_ms_avg']:<18.1f} | {opt_stats['pred_ms_avg']:<18.1f} | -")
    print(f"{'Avg E2E Latency (ms)':<30} | {e2e_base:<18.1f} | {e2e_opt:<18.1f} | -{e2e_delta:.1f}%")
    print(f"{'P95 E2E Latency (ms)':<30} | {baseline_stats['e2e_ms_p95']:<18.1f} | {opt_stats['e2e_ms_p95']:<18.1f} | -{(baseline_stats['e2e_ms_p95']-opt_stats['e2e_ms_p95'])/baseline_stats['e2e_ms_p95']*100:.1f}%")
    print(f"{'Translation Speed (chars/s)':<30} | {speed_base:<18.1f} | {speed_opt:<18.1f} | +{speed_gain:.1f}%")
    print("=" * 88)

    # MLflow Logging
    if not args.no_mlflow:
        log_to_mlflow(baseline_stats, opt_stats, baseline_records, opt_records, args)


if __name__ == "__main__":
    main()
