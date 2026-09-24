"""End-to-End Benchmark for Streaming Typewriter TTFT and Pipelined Double Buffering.

Evaluates:
1. Module 1: Streaming Typewriter First Token Latency (TTFT)
   - 20 representative anime / ASMR dialogue samples.
   - Measures Time to First Token (TTFT) vs. Full Completion Latency.
   - Computes perceived latency reduction percentage.
2. Module 2: Pipeline Overlapping (Sequential vs. Double Buffering)
   - 10 consecutive dialogue clips from real benchmark audio (benchmarks/scene_a_dialogue.wav).
   - Mode A (Sequential): Segment 1 ASR -> Segment 1 MT -> Segment 2 ASR -> Segment 2 MT...
   - Mode B (Pipelined Double Buffering): Thread 1 runs ASR(i+1) concurrently while Thread 2 runs MT(i).
   - Measures total pipeline batch time, time saved by eliminating idle bubbles, and overlapping efficiency.

Logs the benchmark as the 7th run in MLflow:
- Tracking URI: sqlite:///D:/ADLINK/asmr-live-sub/mlflow.db
- Experiment: ASMR-Live-Subtitle-Benchmarks
- Run Name: 2026-09-24 End-to-End Pipeline (Streaming + Overlapping)
"""

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import av
import numpy as np

# Ensure livesub configuration & CUDA DLL directories are initialized
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import livesub.config
except Exception as e:
    print(f"[warn] Failed to import livesub.config: {e}", file=sys.stderr)

try:
    from faster_whisper import WhisperModel
except ImportError as e:
    print(f"[error] faster_whisper import failed: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from llama_cpp import Llama
except ImportError as e:
    print(f"[error] llama_cpp import failed: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import mlflow
except ImportError:
    mlflow = None
    print("[warn] MLflow not installed, skipping MLflow logging", file=sys.stderr)

from livesub.audio import Segmenter

# Models & Paths
SAKURA_GGUF = ROOT / "models" / "sakura" / "sakura-7b-qwen2.5-v1.0-iq4xs.gguf"
ASR_MODEL_DIR = ROOT / "models" / "anime-whisper-ct2"
TEST_AUDIO_PATH = ROOT / "benchmarks" / "scene_a_dialogue.wav"
MLFLOW_DB_PATH = ROOT / "mlflow.db"

SAKURA_SYS = (
    "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，"
    "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
)

# 20 representative anime / ASMR dialogue samples
BENCHMARK_SAMPLES_20 = [
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


# =========================================================================
# Module 1: Streaming Typewriter TTFT Benchmark
# =========================================================================

def benchmark_streaming_ttft(llm: Llama, dataset: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Measure TTFT (Time to First Token) vs Full Completion Latency across 20 samples."""
    print("\n" + "=" * 76)
    print("▶ EXPERIMENT MODULE 1: Streaming Typewriter TTFT Benchmark (20 Samples)")
    print("=" * 76)

    records = []
    # Warmup 1 sample
    warmup_prompt = (
        f"<|im_start|>system\n{SAKURA_SYS}<|im_end|>\n"
        f"<|im_start|>user\n将下面的日文文本翻译成中文：おはよう。<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    for _ in llm(warmup_prompt, max_tokens=16, stream=True):
        pass

    for i, item in enumerate(dataset, 1):
        text = item["text"]
        max_tokens = min(100, max(16, len(text) * 3 + 8))
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
        first_token = ""
        chunks = []
        token_count = 0

        for chunk in stream:
            piece = chunk["choices"][0]["text"]
            token_count += 1
            chunks.append(piece)
            if t_first is None and piece and piece.strip():
                t_first = time.perf_counter()
                first_token = piece.strip()

        t_end = time.perf_counter()

        ttft_ms = (t_first - t0) * 1000.0 if t_first else (t_end - t0) * 1000.0
        full_ms = (t_end - t0) * 1000.0
        pred_ms = (t_end - t_first) * 1000.0 if t_first else 0.0
        reduction_pct = (full_ms - ttft_ms) / full_ms * 100.0 if full_ms > 0 else 0.0

        zh = "".join(chunks).strip()
        zh = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", zh).strip()

        tps = (token_count - 1) / (pred_ms / 1000.0) if pred_ms > 0 and token_count > 1 else (token_count / (full_ms / 1000.0))

        rec = {
            "id": item["id"],
            "category": item["category"],
            "ja": text,
            "zh": zh,
            "first_token": first_token,
            "token_count": token_count,
            "time_to_first_token_ms": round(ttft_ms, 2),
            "full_completion_latency_ms": round(full_ms, 2),
            "pred_decode_ms": round(pred_ms, 2),
            "ttft_reduction_pct": round(reduction_pct, 2),
            "tokens_per_second": round(tps, 2),
        }
        records.append(rec)

        print(
            f"[{i:02d}/20] TTFT: {ttft_ms:5.1f} ms | Full: {full_ms:6.1f} ms | "
            f"Reduction: {reduction_pct:5.1f}% | 1stTok: '{first_token}' | Zh: '{zh[:18]}...'"
        )

    # Compute aggregate statistics
    ttft_list = [r["time_to_first_token_ms"] for r in records]
    full_list = [r["full_completion_latency_ms"] for r in records]
    red_list = [r["ttft_reduction_pct"] for r in records]
    tps_list = [r["tokens_per_second"] for r in records]

    summary = {
        "sample_count": len(records),
        "ttft_avg_ms": round(float(np.mean(ttft_list)), 2),
        "ttft_p50_ms": round(float(np.median(ttft_list)), 2),
        "ttft_p95_ms": round(float(np.percentile(ttft_list, 95)), 2),
        "ttft_min_ms": round(float(np.min(ttft_list)), 2),
        "ttft_max_ms": round(float(np.max(ttft_list)), 2),
        "full_completion_avg_ms": round(float(np.mean(full_list)), 2),
        "full_completion_p50_ms": round(float(np.median(full_list)), 2),
        "full_completion_p95_ms": round(float(np.percentile(full_list, 95)), 2),
        "full_completion_min_ms": round(float(np.min(full_list)), 2),
        "full_completion_max_ms": round(float(np.max(full_list)), 2),
        "ttft_reduction_pct_avg": round(float(np.mean(red_list)), 2),
        "tokens_per_second_avg": round(float(np.mean(tps_list)), 2),
    }

    print("-" * 76)
    print(f"Module 1 Summary:")
    print(f"  • Average TTFT (首字上屏耗时):        {summary['ttft_avg_ms']:.1f} ms (P95: {summary['ttft_p95_ms']:.1f} ms)")
    print(f"  • Average Full Completion (整句完成): {summary['full_completion_avg_ms']:.1f} ms (P95: {summary['full_completion_p95_ms']:.1f} ms)")
    print(f"  • Perceived Latency Reduction (降幅): {summary['ttft_reduction_pct_avg']:.1f}%")
    print("=" * 76)

    return records, summary


# =========================================================================
# Module 2: Pipeline Overlapping Benchmark (Sequential vs Pipelined)
# =========================================================================

def load_real_dialogue_clips(audio_path: Path, count: int = 10) -> List[Dict[str, Any]]:
    """Extract consecutive realistic dialogue clips from real benchmark audio using livesub Segmenter."""
    print(f"\n[audio] Segmenting '{audio_path.name}' with Segmenter(adaptive, hang_s=2.0)...")
    q = queue.Queue()
    seg = Segmenter(q, rate=16000, strategy="adaptive", hang_s=2.0, max_s=5.0)

    with av.open(str(audio_path)) as c:
        for frame in c.decode(audio=0):
            x = frame.to_ndarray().astype(np.float32)
            scale = 32768.0 if any(s in frame.format.name for s in ("s16", "s32")) else 1.0
            if x.ndim > 1:
                x = x.mean(axis=0)
            seg.feed(x / scale)
    seg.flush()

    raw_items = []
    while not q.empty():
        raw_items.append(q.get())

    # Filter out empty or negligible noise clips (< 1.5s)
    valid_clips = []
    for idx, item in enumerate(raw_items):
        audio, t0, t1 = item
        dur_s = len(audio) / 16000.0
        if dur_s >= 1.5:
            valid_clips.append({
                "clip_id": len(valid_clips) + 1,
                "audio": audio,
                "t_start": t0,
                "t_end": t1,
                "dur_s": round(dur_s, 2),
            })
        if len(valid_clips) >= count:
            break

    print(f"[audio] Extracted {len(valid_clips)} valid consecutive dialogue clips for pipeline benchmarking.")
    for c in valid_clips:
        print(f"  Clip {c['clip_id']:02d}: duration={c['dur_s']}s, range=[{c['t_start']:.2f}s - {c['t_end']:.2f}s]")

    return valid_clips


def run_single_asr(asr_model: WhisperModel, audio: np.ndarray) -> Tuple[str, float]:
    """Execute ASR on single audio array, returning (ja_text, latency_ms)."""
    t0 = time.perf_counter()
    segs, info = asr_model.transcribe(
        audio,
        language="ja",
        task="transcribe",
        beam_size=1,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        patience=2,
    )
    ja = " ".join(s.text.strip() for s in segs).strip()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return ja, latency_ms


def run_single_mt(llm: Llama, ja_text: str) -> Tuple[str, float]:
    """Execute MT translation on text, returning (zh_text, latency_ms)."""
    if not ja_text:
        return "", 0.0
    prompt = (
        f"<|im_start|>system\n{SAKURA_SYS}<|im_end|>\n"
        f"<|im_start|>user\n将下面的日文文本翻译成中文：{ja_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    max_tokens = min(100, max(16, len(ja_text) * 3 + 8))
    t0 = time.perf_counter()
    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.1,
        top_p=0.3,
        repeat_penalty=1.05,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    zh = (out["choices"][0]["text"] or "").strip()
    zh = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", zh).strip()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return zh, latency_ms


def benchmark_mode_a_sequential(
    asr_model: WhisperModel, llm: Llama, clips: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], float]:
    """Mode A: Strictly sequential processing (Segment 1 ASR -> MT -> Segment 2 ASR -> MT...)."""
    print("\n" + "=" * 76)
    print("▶ EXPERIMENT MODULE 2: Mode A (Sequential ASR -> MT Processing)")
    print("=" * 76)

    records = []
    t_start_total = time.perf_counter()

    for idx, c in enumerate(clips):
        t_seg_start = time.perf_counter()
        ja, asr_ms = run_single_asr(asr_model, c["audio"])
        zh, mt_ms = run_single_mt(llm, ja)
        total_seg_ms = (time.perf_counter() - t_seg_start) * 1000.0

        rec = {
            "clip_id": c["clip_id"],
            "dur_s": c["dur_s"],
            "ja": ja,
            "zh": zh,
            "asr_ms": round(asr_ms, 2),
            "mt_ms": round(mt_ms, 2),
            "seg_total_ms": round(total_seg_ms, 2),
        }
        records.append(rec)
        print(
            f"[Seq {c['clip_id']:02d}/10] ASR: {asr_ms:6.1f} ms | MT: {mt_ms:6.1f} ms | "
            f"SegTotal: {total_seg_ms:6.1f} ms | JA: '{ja[:15]}' -> ZH: '{zh[:15]}'"
        )

    total_seq_ms = (time.perf_counter() - t_start_total) * 1000.0
    print("-" * 76)
    print(f"Mode A (Sequential) Total Pipeline Time: {total_seq_ms:7.1f} ms")
    print("=" * 76)

    return records, total_seq_ms


def benchmark_mode_b_pipelined(
    asr_model: WhisperModel, llm: Llama, clips: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], float]:
    """Mode B: Pipelined double buffering (Thread 1 runs ASR(i+1) concurrently with Thread 2 running MT(i))."""
    print("\n" + "=" * 76)
    print("▶ EXPERIMENT MODULE 2: Mode B (Pipelined Double Buffering Overlapping)")
    print("=" * 76)

    mt_q: queue.Queue = queue.Queue(maxsize=4)
    results: List[Dict[str, Any]] = []
    results_lock = threading.Lock()

    def asr_worker():
        for c in clips:
            ja, asr_ms = run_single_asr(asr_model, c["audio"])
            mt_q.put((c["clip_id"], c["dur_s"], ja, asr_ms))
        mt_q.put(None)  # Sentinel to terminate MT worker

    def mt_worker():
        while True:
            item = mt_q.get()
            if item is None:
                break
            clip_id, dur_s, ja, asr_ms = item
            zh, mt_ms = run_single_mt(llm, ja)
            with results_lock:
                results.append({
                    "clip_id": clip_id,
                    "dur_s": dur_s,
                    "ja": ja,
                    "zh": zh,
                    "asr_ms": round(asr_ms, 2),
                    "mt_ms": round(mt_ms, 2),
                })
                print(
                    f"[Pipe {clip_id:02d}/10] ASR: {asr_ms:6.1f} ms | MT: {mt_ms:6.1f} ms | "
                    f"JA: '{ja[:15]}' -> ZH: '{zh[:15]}'"
                )

    t_start_total = time.perf_counter()

    t_asr = threading.Thread(target=asr_worker, name="ASR-Worker")
    t_mt = threading.Thread(target=mt_worker, name="MT-Worker")

    t_asr.start()
    t_mt.start()

    t_asr.join()
    t_mt.join()

    total_pipe_ms = (time.perf_counter() - t_start_total) * 1000.0

    # Ensure results sorted by clip_id
    results.sort(key=lambda x: x["clip_id"])

    print("-" * 76)
    print(f"Mode B (Pipelined) Total Pipeline Time: {total_pipe_ms:7.1f} ms")
    print("=" * 76)

    return results, total_pipe_ms


def compute_pipeline_metrics(
    seq_records: List[Dict[str, Any]],
    pipe_records: List[Dict[str, Any]],
    seq_total_ms: float,
    pipe_total_ms: float,
) -> Dict[str, Any]:
    """Compute throughput, latency, savings, and overlapping efficiency."""
    sum_asr_ms = sum(r["asr_ms"] for r in seq_records)
    sum_mt_ms = sum(r["mt_ms"] for r in seq_records)
    avg_asr_ms = float(np.mean([r["asr_ms"] for r in seq_records]))
    avg_mt_ms = float(np.mean([r["mt_ms"] for r in seq_records]))

    saved_time_ms = seq_total_ms - pipe_total_ms
    time_reduction_pct = (saved_time_ms / seq_total_ms * 100.0) if seq_total_ms > 0 else 0.0
    speedup_ratio = (seq_total_ms / pipe_total_ms) if pipe_total_ms > 0 else 1.0

    # Theoretical maximum overlapping time in 2-stage pipeline
    # Total work = sum_asr + sum_mt. The smaller of the two stages can theoretically be hidden completely.
    max_overlap_possible_ms = min(sum_asr_ms, sum_mt_ms)
    actual_overlap_ms = (sum_asr_ms + sum_mt_ms) - pipe_total_ms
    overlap_efficiency_pct = (
        (actual_overlap_ms / max_overlap_possible_ms * 100.0)
        if max_overlap_possible_ms > 0
        else 0.0
    )
    overlap_efficiency_pct = max(0.0, min(100.0, overlap_efficiency_pct))

    return {
        "clip_count": len(seq_records),
        "sequential_total_pipeline_time_ms": round(seq_total_ms, 2),
        "pipelined_total_pipeline_time_ms": round(pipe_total_ms, 2),
        "pipeline_time_saved_ms": round(saved_time_ms, 2),
        "pipeline_throughput_time_reduction_pct": round(time_reduction_pct, 2),
        "pipeline_speedup_ratio": round(speedup_ratio, 2),
        "pipeline_avg_asr_latency_ms": round(avg_asr_ms, 2),
        "pipeline_avg_mt_latency_ms": round(avg_mt_ms, 2),
        "sum_asr_time_ms": round(sum_asr_ms, 2),
        "sum_mt_time_ms": round(sum_mt_ms, 2),
        "actual_overlap_ms": round(actual_overlap_ms, 2),
        "pipeline_overlapping_efficiency_pct": round(overlap_efficiency_pct, 2),
    }


# =========================================================================
# MLflow Logging
# =========================================================================

def log_to_mlflow(
    streaming_summary: Dict[str, Any],
    streaming_records: List[Dict[str, Any]],
    pipeline_summary: Dict[str, Any],
    seq_records: List[Dict[str, Any]],
    pipe_records: List[Dict[str, Any]],
    args: argparse.Namespace,
):
    """Push benchmark results as the 7th run to local MLflow database."""
    if not mlflow:
        print("[skip] MLflow not available, skipping MLflow logging")
        return

    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = "ASMR-Live-Subtitle-Benchmarks"
    mlflow.set_experiment(exp_name)

    run_name = args.run_name or "2026-09-24 End-to-End Pipeline (Streaming + Overlapping)"
    print(f"\n[mlflow] Pushing 7th Run '{run_name}' to {tracking_uri}...")

    params = {
        "streaming_typewriter_enabled": "True",
        "pipeline_double_buffering_enabled": "True",
        "target_mt_model": SAKURA_GGUF.name,
        "asr_model": ASR_MODEL_DIR.name,
        "streaming_sample_count": streaming_summary["sample_count"],
        "pipeline_clip_count": pipeline_summary["clip_count"],
        "gpu_offload_ngl": args.ngl,
        "asr_device": "cuda",
        "mt_device": "cuda",
        "pipeline_strategy": "adaptive_double_buffering",
    }

    metrics = {
        "streaming_ttft_avg_ms": streaming_summary["ttft_avg_ms"],
        "streaming_ttft_p50_ms": streaming_summary["ttft_p50_ms"],
        "streaming_ttft_p95_ms": streaming_summary["ttft_p95_ms"],
        "streaming_full_completion_avg_ms": streaming_summary["full_completion_avg_ms"],
        "streaming_full_completion_p95_ms": streaming_summary["full_completion_p95_ms"],
        "streaming_ttft_reduction_pct": streaming_summary["ttft_reduction_pct_avg"],
        "streaming_tokens_per_second": streaming_summary["tokens_per_second_avg"],
        "sequential_total_pipeline_time_ms": pipeline_summary["sequential_total_pipeline_time_ms"],
        "pipelined_total_pipeline_time_ms": pipeline_summary["pipelined_total_pipeline_time_ms"],
        "pipeline_time_saved_ms": pipeline_summary["pipeline_time_saved_ms"],
        "pipeline_throughput_time_reduction_pct": pipeline_summary["pipeline_throughput_time_reduction_pct"],
        "pipeline_overlapping_efficiency_pct": pipeline_summary["pipeline_overlapping_efficiency_pct"],
        "pipeline_speedup_ratio": pipeline_summary["pipeline_speedup_ratio"],
        "pipeline_avg_asr_latency_ms": pipeline_summary["pipeline_avg_asr_latency_ms"],
        "pipeline_avg_mt_latency_ms": pipeline_summary["pipeline_avg_mt_latency_ms"],
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log per-step metrics for Module 1
        for step, rec in enumerate(streaming_records):
            mlflow.log_metrics(
                {
                    "step_ttft_ms": rec["time_to_first_token_ms"],
                    "step_full_completion_ms": rec["full_completion_latency_ms"],
                    "step_ttft_reduction_pct": rec["ttft_reduction_pct"],
                },
                step=step,
            )

        # Log per-step metrics for Module 2
        for step, (s_rec, p_rec) in enumerate(zip(seq_records, pipe_records)):
            mlflow.log_metrics(
                {
                    "step_seq_asr_ms": s_rec["asr_ms"],
                    "step_seq_mt_ms": s_rec["mt_ms"],
                    "step_pipe_asr_ms": p_rec["asr_ms"],
                    "step_pipe_mt_ms": p_rec["mt_ms"],
                },
                step=step,
            )

        # Write summary JSON artifact
        artifact_path = ROOT / "eval_tools" / "pipeline_streaming_benchmark_summary.json"
        summary_payload = {
            "metadata": {
                "run_name": run_name,
                "run_id": run.info.run_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "params": params,
            },
            "streaming_summary": streaming_summary,
            "pipeline_summary": pipeline_summary,
            "streaming_records": streaming_records,
            "sequential_records": seq_records,
            "pipelined_records": pipe_records,
        }

        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)

        mlflow.log_artifact(str(artifact_path))
        print(f"[mlflow] Successfully logged Run ID: {run.info.run_id}")


# =========================================================================
# Main Entry Point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="ASMR Live Subtitle Streaming TTFT & Pipeline Overlapping Benchmark")
    parser.add_argument("--ngl", type=int, default=99, help="Number of GPU layers for Sakura-7B (default: 99)")
    parser.add_argument("--target-path", type=Path, default=SAKURA_GGUF, help="Path to Sakura 7B GGUF")
    parser.add_argument("--asr-dir", type=Path, default=ASR_MODEL_DIR, help="Path to Anime Whisper CT2 model")
    parser.add_argument("--audio-path", type=Path, default=TEST_AUDIO_PATH, help="Path to benchmark dialogue audio")
    parser.add_argument("--run-name", type=str, default=None, help="Custom MLflow run name")
    args = parser.parse_args()

    print("=" * 76)
    print("ASMR Live Subtitle: Streaming TTFT & Pipeline Overlapping Benchmark")
    print(f"Target MT Model: {args.target_path.name} (ngl={args.ngl})")
    print(f"ASR Model:       {args.asr_dir.name} (cuda fp16)")
    print(f"Benchmark Audio: {args.audio_path.name}")
    print("=" * 76)

    # 1. Load Models
    print("\n[load] Initializing ASR (anime-whisper-ct2) on CUDA...")
    asr_model = WhisperModel(str(args.asr_dir), device="cuda", compute_type="float16", local_files_only=True)

    print("[load] Initializing MT (Sakura-7B IQ4XS) on CUDA...")
    llm = Llama(
        model_path=str(args.target_path),
        n_ctx=2048,
        n_gpu_layers=args.ngl,
        verbose=False,
    )
    print("[load] Both models loaded successfully.")

    # 2. Experiment Module 1: Streaming Typewriter TTFT
    streaming_records, streaming_summary = benchmark_streaming_ttft(llm, BENCHMARK_SAMPLES_20)

    # 3. Prepare 10 real dialogue clips for Module 2
    clips_10 = load_real_dialogue_clips(args.audio_path, count=10)

    # Warmup pipeline execution
    warmup_ja, _ = run_single_asr(asr_model, clips_10[0]["audio"])
    run_single_mt(llm, warmup_ja)

    # 4. Experiment Module 2: Mode A (Sequential)
    seq_records, seq_total_ms = benchmark_mode_a_sequential(asr_model, llm, clips_10)

    # 5. Experiment Module 2: Mode B (Pipelined Double Buffering)
    pipe_records, pipe_total_ms = benchmark_mode_b_pipelined(asr_model, llm, clips_10)

    # 6. Compute Pipeline Comparative Metrics
    pipeline_summary = compute_pipeline_metrics(seq_records, pipe_records, seq_total_ms, pipe_total_ms)

    # 7. Print Final Comprehensive Report
    print("\n" + "=" * 76)
    print("FINAL BENCHMARK COMPARISON REPORT")
    print("=" * 76)
    print("【实验模块一：流式打字机首字延迟 (Streaming TTFT)】")
    print(f"  • 首字吐出平均延迟 (TTFT Avg):      {streaming_summary['ttft_avg_ms']:>8.1f} ms  (P95: {streaming_summary['ttft_p95_ms']:.1f} ms)")
    print(f"  • 整句完成平均耗时 (Full Comp Avg):  {streaming_summary['full_completion_avg_ms']:>8.1f} ms  (P95: {streaming_summary['full_completion_p95_ms']:.1f} ms)")
    print(f"  • 首字体感延迟缩减 (TTFT Reduction): {streaming_summary['ttft_reduction_pct_avg']:>8.1f} %")
    print(f"  • 流式生成平均吞吐:                  {streaming_summary['tokens_per_second_avg']:>8.1f} tokens/s")
    print("-" * 76)
    print("【实验模块二：流水线双缓冲重叠并发 (Pipeline Overlapping)】")
    print(f"  • 模式 A 串行总耗时 (Sequential):   {pipeline_summary['sequential_total_pipeline_time_ms']:>8.1f} ms")
    print(f"  • 模式 B 并行总耗时 (Pipelined):    {pipeline_summary['pipelined_total_pipeline_time_ms']:>8.1f} ms")
    print(f"  • 消除交替气泡节省时间 (Saved Time):{pipeline_summary['pipeline_time_saved_ms']:>8.1f} ms")
    print(f"  • 连续流水线总吞吐耗时缩减率:        {pipeline_summary['pipeline_throughput_time_reduction_pct']:>8.1f} %")
    print(f"  • 流水线重叠效率 (Overlapping Eff): {pipeline_summary['pipeline_overlapping_efficiency_pct']:>8.1f} %")
    print(f"  • 端到端并发加速比 (Speedup):        {pipeline_summary['pipeline_speedup_ratio']:>8.2f} x")
    print("=" * 76)

    # 8. Log to MLflow (7th Run)
    log_to_mlflow(
        streaming_summary,
        streaming_records,
        pipeline_summary,
        seq_records,
        pipe_records,
        args,
    )


if __name__ == "__main__":
    main()
