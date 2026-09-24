"""Benchmark script for Speculative Decoding (7B Target + 0.5B Draft) in ASMR Live Subtitling.

Compares:
- Group A (Baseline): Sakura 7B solitary decoding (no draft model).
- Group B (Speculative): Sakura 7B + Qwen2.5-0.5B draft model collaborative speculative decoding.

Core physical metrics:
- pred_latency_avg_ms: Average token generation latency (ms)
- tokens_per_second: Generation throughput (tokens/s)
- acceptance_rate_pct: Draft tokens acceptance / hit rate (%)
- vram_overhead_mb: Extra GPU memory consumed by the 0.5B draft model (MB)

MLflow:
- Tracking URI: sqlite:///D:/ADLINK/asmr-live-sub/mlflow.db
- Experiment: ASMR-Live-Subtitle-Benchmarks
- Run Name: 2026-09-24 Speculative Decoding (7B + 0.5B Draft)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure livesub configuration & DLL directories are registered
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import livesub.config
except Exception as e:
    print(f"[warn] Failed to import livesub.config directly: {e}", file=sys.stderr)

try:
    from llama_cpp import Llama, LlamaDraftModel
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
TARGET_7B_PATH = ROOT / "models" / "sakura" / "sakura-7b-qwen2.5-v1.0-iq4xs.gguf"
DRAFT_GENERIC_PATH = ROOT / "models" / "draft" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
DRAFT_ANIME_PATH = ROOT / "models" / "draft" / "qwen2.5-0.5b-anime-draft-q4.gguf"
DRAFT_05B_PATH = DRAFT_ANIME_PATH if DRAFT_ANIME_PATH.exists() else DRAFT_GENERIC_PATH
MLFLOW_DB_PATH = ROOT / "mlflow.db"

# 20 representative anime / ASMR dialogue lines
BENCHMARK_SAMPLES = [
    # 1-5: 短叹与即时反应 (Short sighs & immediate reactions)
    {"id": 1, "category": "sigh", "text": "あっ…"},
    {"id": 2, "category": "sigh", "text": "ふふっ、可愛い…"},
    {"id": 3, "category": "sigh", "text": "えっ、本当ですか？"},
    {"id": 4, "category": "sigh", "text": "ん…もう、意地悪…"},
    {"id": 5, "category": "sigh", "text": "はぁ…びっくりした…"},
    # 6-15: 日常温柔陪伴与催眠台词 (Gentle companionship & soothing dialogue)
    {"id": 6, "category": "dialogue", "text": "今日もお仕事お疲れ様でした、よく頑張ったね。"},
    {"id": 7, "category": "dialogue", "text": "温かいココアでも淹れてこようか？"},
    {"id": 8, "category": "dialogue", "text": "隣に座ってもいい？少しだけお話ししよう。"},
    {"id": 9, "category": "dialogue", "text": "明日も早いんでしょ？無理しちゃだめだよ。"},
    {"id": 10, "category": "dialogue", "text": "ぎゅーって抱きしめてあげるから、安心してね。"},
    {"id": 11, "category": "dialogue", "text": "頭をなでなでしてあげるね、いい子いい子…"},
    {"id": 12, "category": "dialogue", "text": "あなたの優しい笑顔、私大好きだよ。"},
    {"id": 13, "category": "dialogue", "text": "ずっとそばにいるから、目を閉じてゆっくり深呼吸してね。"},
    {"id": 14, "category": "dialogue", "text": "今日も一日、本当によく頑張ったね。"},
    {"id": 15, "category": "dialogue", "text": "大丈夫、私がずっとついてるからね。"},
    # 16-20: 复合句与情感抒发长句 (Long compound sentences & emotional lines)
    {"id": 16, "category": "compound", "text": "今日は外がすごく寒かったから、お風呂にゆっくり浸かって、温かいお布団に入ってぐっすり休んでね。"},
    {"id": 17, "category": "compound", "text": "いつも誰よりも一生懸命頑張っているのを見てるから、今夜くらいは思いっきり甘えて泣いちゃってもいいんだよ？"},
    {"id": 18, "category": "compound", "text": "耳元で囁かれるのってちょっとくすぐったいかもしれないけど、息を吹きかけながらいっぱい癒してあげるね。"},
    {"id": 19, "category": "compound", "text": "本当は私もすごく緊張しててドキドキしてるんだけど、君の手を握ると不思議と落ち着くんだ。"},
    {"id": 20, "category": "compound", "text": "眠れないなら朝までずっとお話ししていようか、君が眠くなるまで何時間でも付き合うからね。"},
]


def get_gpu_memory_used_mb() -> float:
    """Query current GPU VRAM usage in MB using nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return 0.0


class IncrementalDraftModel(LlamaDraftModel):
    """High-performance Draft Model with incremental KV caching and precise acceptance tracking."""

    def __init__(self, model_path: str, k: int = 3, n_gpu_layers: int = 99, n_ctx: int = 512):
        self.k = k
        self.draft_llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.cached_tokens: List[int] = []
        self.total_proposed = 0
        self.total_accepted = 0
        self.step_proposed = 0
        self.step_accepted = 0
        self.last_draft: List[int] = []
        self.last_prefix_len = 0

    def start_sentence(self):
        """Prepare draft state for a new generation turn."""
        self.step_proposed = 0
        self.step_accepted = 0
        self.last_draft = []
        self.last_prefix_len = 0
        # Reset KV cache cleanly for each independent sentence
        self.draft_llm.reset()
        self.cached_tokens = []

    def finish_sentence(self, final_input_ids: List[int]) -> Tuple[int, int]:
        """Tally any remaining accepted tokens from the last draft round."""
        if self.last_draft:
            matched = 0
            for i, tok in enumerate(self.last_draft):
                pos = self.last_prefix_len + i
                if pos < len(final_input_ids) and final_input_ids[pos] == tok:
                    matched += 1
                else:
                    break
            self.total_accepted += matched
            self.step_accepted += matched
            self.last_draft = []
        return self.step_proposed, self.step_accepted

    def __call__(self, input_ids: np.ndarray, /, **kwargs: Any) -> np.ndarray:
        cur_ids = [int(t) for t in input_ids]

        # 1. Update acceptance tracking from previous round
        if self.last_draft:
            matched = 0
            for i, tok in enumerate(self.last_draft):
                pos = self.last_prefix_len + i
                if pos < len(cur_ids) and cur_ids[pos] == tok:
                    matched += 1
                else:
                    break
            self.total_accepted += matched
            self.step_accepted += matched
            self.last_draft = []

        # 2. Incremental KV synchronization with target model context
        common_len = 0
        max_common = min(len(self.cached_tokens), len(cur_ids))
        while common_len < max_common and self.cached_tokens[common_len] == cur_ids[common_len]:
            common_len += 1

        if common_len < len(self.cached_tokens):
            self.draft_llm._ctx.kv_cache_seq_rm(-1, common_len, -1)
            self.draft_llm.n_tokens = common_len
            self.cached_tokens = self.cached_tokens[:common_len]

        tokens_to_eval = cur_ids[common_len:]
        if tokens_to_eval:
            self.draft_llm.eval(tokens_to_eval)
            self.cached_tokens.extend(tokens_to_eval)

        # 3. Speculatively generate k draft tokens greedily
        drafts: List[int] = []
        for _ in range(self.k):
            tok = self.draft_llm.sample(temp=0.0)
            drafts.append(tok)
            self.draft_llm.eval([tok])
            self.cached_tokens.append(tok)

        self.total_proposed += len(drafts)
        self.step_proposed += len(drafts)
        self.last_draft = drafts
        self.last_prefix_len = len(cur_ids)
        return np.array(drafts, dtype=np.intc)


def run_translation_stream(llm: Llama, text: str, max_tokens: int = 100) -> Dict[str, Any]:
    """Execute streaming translation and measure prompt_ms, pred_ms, e2e_ms, and tokens_per_second."""
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
        stream=True,
    )
    t_first = None
    chunks = []
    token_count = 0
    for chunk in stream:
        if t_first is None:
            t_first = time.perf_counter()
        choice_text = chunk["choices"][0]["text"]
        chunks.append(choice_text)
        token_count += 1
    t_end = time.perf_counter()

    prompt_ms = (t_first - t0) * 1000.0 if t_first else (t_end - t0) * 1000.0
    pred_ms = (t_end - t_first) * 1000.0 if t_first else 0.0
    e2e_ms = (t_end - t0) * 1000.0

    zh = "".join(chunks).strip()
    zh = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", zh).strip()

    tps = token_count / (pred_ms / 1000.0) if pred_ms > 0 else 0.0
    cps = len(zh) / (e2e_ms / 1000.0) if e2e_ms > 0 else 0.0

    return {
        "text_in": text,
        "text_out": zh,
        "token_count": token_count,
        "prompt_ms": round(prompt_ms, 2),
        "pred_ms": round(pred_ms, 2),
        "e2e_ms": round(e2e_ms, 2),
        "tokens_per_second": round(tps, 2),
        "chars_per_sec": round(cps, 2),
        "in_chars": len(text),
        "out_chars": len(zh),
    }


def benchmark_group_a_baseline(target_path: Path, dataset: List[Dict[str, Any]], ngl: int) -> Tuple[List[Dict[str, Any]], float]:
    """Group A (Baseline): 7B Solitary Decoding without Draft Model."""
    print("\n" + "=" * 70)
    print("▶ Running Group A: Baseline (Sakura 7B Solitary Decoding)")
    print("=" * 70)

    vram_before = get_gpu_memory_used_mb()
    print(f"[vram] GPU VRAM before loading 7B model: {vram_before:.1f} MB")

    llm = Llama(
        model_path=str(target_path),
        n_ctx=512,
        n_gpu_layers=ngl,
        verbose=False,
    )
    vram_after = get_gpu_memory_used_mb()
    vram_7b = max(0.0, vram_after - vram_before)
    print(f"[vram] GPU VRAM with 7B model loaded: {vram_after:.1f} MB (Delta: +{vram_7b:.1f} MB)")

    records = []
    for i, item in enumerate(dataset, 1):
        llm.reset()
        text = item["text"]
        max_tokens = min(100, max(16, len(text) * 3 + 10))
        res = run_translation_stream(llm, text, max_tokens=max_tokens)
        res["id"] = item["id"]
        res["category"] = item["category"]
        records.append(res)
        print(
            f"[{i:02d}/{len(dataset)}] '{text}' -> '{res['text_out']}' "
            f"| Pred: {res['pred_ms']:5.1f}ms | TPS: {res['tokens_per_second']:5.1f} tok/s "
            f"| E2E: {res['e2e_ms']:5.1f}ms | Tokens: {res['token_count']}"
        )

    # Release baseline LLM to free VRAM for clean group B comparison
    del llm
    time.sleep(1.0)
    return records, vram_7b


def benchmark_group_b_speculative(
    target_path: Path, draft_path: Path, dataset: List[Dict[str, Any]], ngl: int, draft_k: int
) -> Tuple[List[Dict[str, Any]], float, float]:
    """Group B (Speculative): Sakura 7B + Qwen2.5-0.5B Speculative Decoding."""
    print("\n" + "=" * 70)
    print(f"▶ Running Group B: Speculative Decoding (Sakura 7B + 0.5B Draft, K={draft_k})")
    print("=" * 70)

    vram_before = get_gpu_memory_used_mb()
    print(f"[vram] GPU VRAM before speculative setup: {vram_before:.1f} MB")

    # 1. Initialize draft model
    draft = IncrementalDraftModel(
        model_path=str(draft_path),
        k=draft_k,
        n_gpu_layers=ngl,
        n_ctx=512,
    )
    vram_after_draft = get_gpu_memory_used_mb()
    vram_draft_alone = max(0.0, vram_after_draft - vram_before)
    print(f"[vram] GPU VRAM with 0.5B draft loaded: {vram_after_draft:.1f} MB (Draft VRAM: +{vram_draft_alone:.1f} MB)")

    # 2. Initialize target model with draft_model bound
    llm = Llama(
        model_path=str(target_path),
        n_ctx=512,
        n_gpu_layers=ngl,
        draft_model=draft,
        verbose=False,
    )
    vram_after_both = get_gpu_memory_used_mb()
    total_spec_vram = max(0.0, vram_after_both - vram_before)
    print(f"[vram] GPU VRAM with 7B + 0.5B loaded: {vram_after_both:.1f} MB (Total VRAM: +{total_spec_vram:.1f} MB)")

    records = []
    for i, item in enumerate(dataset, 1):
        llm.reset()
        draft.start_sentence()
        text = item["text"]
        max_tokens = min(100, max(16, len(text) * 3 + 10))
        res = run_translation_stream(llm, text, max_tokens=max_tokens)
        step_prop, step_acc = draft.finish_sentence(list(llm._input_ids[: llm.n_tokens]))

        step_acc_rate = (step_acc / step_prop * 100.0) if step_prop > 0 else 0.0
        res["id"] = item["id"]
        res["category"] = item["category"]
        res["step_proposed"] = step_prop
        res["step_accepted"] = step_acc
        res["step_acceptance_rate_pct"] = round(step_acc_rate, 2)
        records.append(res)

        print(
            f"[{i:02d}/{len(dataset)}] '{text}' -> '{res['text_out']}' "
            f"| Pred: {res['pred_ms']:5.1f}ms | TPS: {res['tokens_per_second']:5.1f} tok/s "
            f"| AcceptRate: {step_acc_rate:4.1f}% ({step_acc}/{step_prop})"
        )

    del llm
    del draft
    time.sleep(1.0)
    return records, vram_draft_alone, total_spec_vram


def calculate_group_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistical metrics for a benchmark group."""
    prompt_times = [r["prompt_ms"] for r in records]
    pred_times = [r["pred_ms"] for r in records]
    e2e_times = [r["e2e_ms"] for r in records]
    tps_list = [r["tokens_per_second"] for r in records]
    cps_list = [r["chars_per_sec"] for r in records]
    total_tokens = sum(r["token_count"] for r in records)
    total_pred_time_ms = sum(pred_times)
    total_time_ms = sum(e2e_times)
    total_chars = sum(r["out_chars"] for r in records)

    overall_tps = (total_tokens / (total_pred_time_ms / 1000.0)) if total_pred_time_ms > 0 else 0.0
    overall_cps = (total_chars / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

    stats = {
        "count": len(records),
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "total_pred_time_ms": round(total_pred_time_ms, 2),
        "total_time_ms": round(total_time_ms, 2),
        "total_time_s": round(total_time_ms / 1000.0, 3),
        "pred_latency_avg_ms": round(float(np.mean(pred_times)), 2),
        "pred_latency_p95_ms": round(float(np.percentile(pred_times, 95)), 2),
        "prompt_latency_avg_ms": round(float(np.mean(prompt_times)), 2),
        "e2e_latency_avg_ms": round(float(np.mean(e2e_times)), 2),
        "e2e_latency_p95_ms": round(float(np.percentile(e2e_times, 95)), 2),
        "tokens_per_second_avg": round(float(np.mean(tps_list)), 2),
        "overall_tokens_per_second": round(overall_tps, 2),
        "chars_per_sec_avg": round(float(np.mean(cps_list)), 2),
        "overall_chars_per_sec": round(overall_cps, 2),
    }

    # Speculative specific stats if present
    if "step_proposed" in records[0]:
        tot_prop = sum(r["step_proposed"] for r in records)
        tot_acc = sum(r["step_accepted"] for r in records)
        acc_rate = (tot_acc / tot_prop * 100.0) if tot_prop > 0 else 0.0
        stats["total_proposed"] = tot_prop
        stats["total_accepted"] = tot_acc
        stats["acceptance_rate_pct"] = round(acc_rate, 2)

    return stats


def log_to_mlflow(
    baseline_stats: Dict[str, Any],
    spec_stats: Dict[str, Any],
    baseline_records: List[Dict[str, Any]],
    spec_records: List[Dict[str, Any]],
    vram_7b: float,
    vram_overhead_mb: float,
    args: Any,
) -> None:
    """Log speculative decoding benchmark results to MLflow as the 5th run."""
    if not mlflow:
        print("[skip] MLflow not available, skipping MLflow logging")
        return

    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = "ASMR-Live-Subtitle-Benchmarks"
    mlflow.set_experiment(exp_name)

    run_name = getattr(args, "run_name", None)
    if not run_name:
        if "anime" in args.draft_path.name.lower():
            run_name = "2026-09-24 Speculative (Fine-tuned 0.5B Draft)"
        else:
            run_name = "2026-09-24 Speculative Decoding (7B + 0.5B Draft)"

    print(f"\n[mlflow] Pushing benchmark run '{run_name}' to {tracking_uri}...")

    # Deltas
    # Speedup: positive if spec TPS > baseline TPS
    tps_gain_pct = (
        (spec_stats["overall_tokens_per_second"] - baseline_stats["overall_tokens_per_second"])
        / baseline_stats["overall_tokens_per_second"]
        * 100.0
        if baseline_stats["overall_tokens_per_second"] > 0
        else 0.0
    )
    # Latency reduction: positive if spec latency is lower than baseline
    pred_latency_reduction_pct = (
        (baseline_stats["pred_latency_avg_ms"] - spec_stats["pred_latency_avg_ms"])
        / baseline_stats["pred_latency_avg_ms"]
        * 100.0
        if baseline_stats["pred_latency_avg_ms"] > 0
        else 0.0
    )

    params = {
        "target_model": args.target_path.stem,
        "draft_model": args.draft_path.stem,
        "decoding_strategy": "speculative_decoding",
        "draft_k": args.draft_k,
        "ngl": args.ngl,
        "sample_count": len(BENCHMARK_SAMPLES),
        "target_ctx": 512,
        "draft_ctx": 512,
    }

    metrics = {
        # Core physical metrics requested
        "pred_latency_avg_ms": spec_stats["pred_latency_avg_ms"],
        "tokens_per_second": spec_stats["overall_tokens_per_second"],
        "acceptance_rate_pct": spec_stats.get("acceptance_rate_pct", 0.0),
        "vram_overhead_mb": round(vram_overhead_mb, 1),

        # Baseline comparison metrics
        "baseline_pred_latency_avg_ms": baseline_stats["pred_latency_avg_ms"],
        "baseline_tokens_per_second": baseline_stats["overall_tokens_per_second"],
        "baseline_vram_7b_mb": round(vram_7b, 1),
        "speedup_pct": round(tps_gain_pct, 2),
        "pred_latency_reduction_pct": round(pred_latency_reduction_pct, 2),

        # Standard schema metrics matching historical runs
        "total_segments": len(BENCHMARK_SAMPLES),
        "translated_count": spec_stats["count"],
        "drop_count": 0,
        "drop_rate_pct": 0.0,
        "asr_latency_avg_s": 0.0,
        "asr_latency_p95_s": 0.0,
        "mt_latency_avg_s": round(spec_stats["e2e_latency_avg_ms"] / 1000.0, 3),
        "mt_latency_p95_s": round(spec_stats["e2e_latency_p95_ms"] / 1000.0, 3),
        "e2e_latency_avg_s": round(spec_stats["e2e_latency_avg_ms"] / 1000.0, 3),
        "e2e_latency_p95_s": round(spec_stats["e2e_latency_p95_ms"] / 1000.0, 3),
        "translation_speed_cps": spec_stats["overall_chars_per_sec"],
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log per-step metrics
        for step, rec in enumerate(spec_records):
            mlflow.log_metrics(
                {
                    "step_pred_ms": rec["pred_ms"],
                    "step_tokens_per_second": rec["tokens_per_second"],
                    "step_acceptance_rate_pct": rec["step_acceptance_rate_pct"],
                },
                step=step,
            )

        # Write summary JSON artifact
        artifact_path = ROOT / "eval_tools" / "speculative_benchmark_summary.json"
        summary_payload = {
            "metadata": {
                "run_name": run_name,
                "run_id": run.info.run_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "params": params,
            },
            "metrics": metrics,
            "baseline_stats": baseline_stats,
            "speculative_stats": spec_stats,
            "baseline_records": baseline_records,
            "speculative_records": spec_records,
        }
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)

        mlflow.log_artifact(str(artifact_path))
        print(f"[mlflow] Successfully logged Run ID: {run.info.run_id}")


def main():
    parser = argparse.ArgumentParser(description="ASMR Live Subtitle Speculative Decoding Benchmark")
    parser.add_argument("--ngl", type=int, default=99, help="Number of GPU layers to offload (default: 99)")
    parser.add_argument("--draft-k", type=int, default=3, help="Draft tokens lookahead size (default: 3)")
    parser.add_argument("--target-path", type=Path, default=TARGET_7B_PATH, help="Path to 7B target GGUF")
    parser.add_argument("--draft-path", type=Path, default=DRAFT_05B_PATH, help="Path to 0.5B draft GGUF")
    parser.add_argument("--run-name", type=str, default=None, help="Custom MLflow run name")
    args = parser.parse_args()

    print("=" * 70)
    print("ASMR Live Subtitle: Speculative Decoding Benchmark (7B + 0.5B)")
    print(f"Target Model: {args.target_path.name}")
    print(f"Draft Model:  {args.draft_path.name}")
    print(f"Draft K:      {args.draft_k}")
    print(f"GPU Offload:  ngl={args.ngl}")
    print(f"Sample Count: {len(BENCHMARK_SAMPLES)}")
    print("=" * 70)

    # 1. Run Group A (Baseline)
    baseline_records, vram_7b = benchmark_group_a_baseline(
        args.target_path, BENCHMARK_SAMPLES, args.ngl
    )
    baseline_stats = calculate_group_metrics(baseline_records)

    # 2. Run Group B (Speculative)
    spec_records, vram_draft_alone, total_spec_vram = benchmark_group_b_speculative(
        args.target_path, args.draft_path, BENCHMARK_SAMPLES, args.ngl, args.draft_k
    )
    spec_stats = calculate_group_metrics(spec_records)
    vram_overhead_mb = vram_draft_alone

    # 3. Print Comparison Table
    print("\n" + "=" * 70)
    print("SPECULATIVE DECODING BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Metric':<32} | {'Group A (Baseline 7B)':<20} | {'Group B (Speculative 7B+0.5B)':<25}")
    print("-" * 83)
    print(f"{'Avg Pred Latency (ms)':<32} | {baseline_stats['pred_latency_avg_ms']:<20.1f} | {spec_stats['pred_latency_avg_ms']:<25.1f}")
    print(f"{'Tokens Per Second (tok/s)':<32} | {baseline_stats['overall_tokens_per_second']:<20.1f} | {spec_stats['overall_tokens_per_second']:<25.1f}")
    print(f"{'Draft Acceptance Rate (%)':<32} | {'N/A':<20} | {spec_stats.get('acceptance_rate_pct', 0.0):<25.1f}%")
    print(f"{'GPU VRAM Overhead (MB)':<32} | {vram_7b:<20.1f} | +{vram_overhead_mb:<24.1f}")
    print(f"{'Total Pred Time (ms)':<32} | {baseline_stats['total_pred_time_ms']:<20.1f} | {spec_stats['total_pred_time_ms']:<25.1f}")
    print(f"{'Overall Chars/Sec (c/s)':<32} | {baseline_stats['overall_chars_per_sec']:<20.1f} | {spec_stats['overall_chars_per_sec']:<25.1f}")
    print("-" * 83)

    speedup = (
        (spec_stats["overall_tokens_per_second"] - baseline_stats["overall_tokens_per_second"])
        / baseline_stats["overall_tokens_per_second"]
        * 100.0
        if baseline_stats["overall_tokens_per_second"] > 0
        else 0.0
    )
    if speedup > 0:
        print(f"==> VERDICT: POSITIVE SPEEDUP (+{speedup:.1f}%) via Speculative Decoding.")
    else:
        print(f"==> VERDICT: NEGATIVE OVERHEAD ({speedup:.1f}%) due to low acceptance rate or draft overhead.")

    # 4. Log to MLflow
    log_to_mlflow(
        baseline_stats,
        spec_stats,
        baseline_records,
        spec_records,
        vram_7b,
        vram_overhead_mb,
        args,
    )


if __name__ == "__main__":
    main()
