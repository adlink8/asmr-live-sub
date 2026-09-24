"""Import real ASMR live subtitle trace logs into MLflow for cross-run benchmark comparison."""
import json
import os
import sys
from pathlib import Path
import numpy as np

try:
    import mlflow
except ImportError:
    print("[error] MLflow is not installed in current environment", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"

RUN_FILES = [
    ("2026-09-20 Baseline (Fixed Chunk)", LOGS_DIR / "work-20260920.jsonl"),
    ("2026-09-22 Adaptive VAD (Sakura 7B)", LOGS_DIR / "work-20260922.jsonl"),
    ("2026-09-23 Production Fast (Tuned)", LOGS_DIR / "work-20260923.jsonl"),
]

def parse_and_log_run(run_name, jsonl_path):
    if not jsonl_path.exists():
        print(f"[skip] {jsonl_path} does not exist")
        return

    print(f"\n[processing] Parsing {run_name} from {jsonl_path.name}...")
    
    start_params = {
        "model": "anime",
        "mt": "sakura",
        "strategy": "adaptive",
        "max_s": 5.0,
        "hang_s": 2.0,
        "ngl": 99,
    }
    
    asr_latencies = []
    mt_latencies = []
    e2e_latencies = []
    char_speeds = []
    drop_count = 0
    total_segments = 0
    translated_count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            
            kind = data.get("kind")
            if kind == "start":
                for k in ["model", "mt", "strategy", "max_s", "hang_s", "ngl"]:
                    if k in data:
                        start_params[k] = data[k]
            elif kind == "asr":
                total_segments += 1
                if data.get("drop"):
                    drop_count += 1
                asr_s = data.get("asr_s")
                if asr_s and asr_s > 0:
                    asr_latencies.append(asr_s)
            elif kind == "mt":
                translated_count += 1
                mt_s = data.get("mt_s") or data.get("gen_s")
                if mt_s and mt_s > 0:
                    mt_latencies.append(mt_s)
                asr_s = data.get("asr_s", 0)
                if asr_s and mt_s:
                    e2e_latencies.append(asr_s + mt_s)
                out_chars = data.get("out_chars", 0)
                if out_chars and mt_s and mt_s > 0:
                    char_speeds.append(out_chars / mt_s)

    if not asr_latencies and not mt_latencies:
        print(f"[skip] No valid ASR/MT metrics in {jsonl_path.name}")
        return

    # Calculate statistics
    drop_rate = (drop_count / total_segments * 100) if total_segments else 0.0
    avg_asr = float(np.mean(asr_latencies)) if asr_latencies else 0.0
    p95_asr = float(np.percentile(asr_latencies, 95)) if asr_latencies else 0.0
    avg_mt = float(np.mean(mt_latencies)) if mt_latencies else 0.0
    p95_mt = float(np.percentile(mt_latencies, 95)) if mt_latencies else 0.0
    avg_e2e = float(np.mean(e2e_latencies)) if e2e_latencies else (avg_asr + avg_mt)
    p95_e2e = float(np.percentile(e2e_latencies, 95)) if e2e_latencies else (p95_asr + p95_mt)
    avg_speed = float(np.mean(char_speeds)) if char_speeds else 0.0

    print(f" -> Segments: {total_segments}, Translated: {translated_count}, DropRate: {drop_rate:.1f}%")
    print(f" -> ASR Avg: {avg_asr:.3f}s (P95: {p95_asr:.3f}s) | MT Avg: {avg_mt:.3f}s (P95: {p95_mt:.3f}s)")
    print(f" -> E2E Avg: {avg_e2e:.3f}s (P95: {p95_e2e:.3f}s) | Speed: {avg_speed:.1f} chars/s")

    with mlflow.start_run(run_name=run_name):
        # Log Params
        mlflow.log_params(start_params)
        mlflow.log_param("log_file", jsonl_path.name)
        
        # Log Metrics
        mlflow.log_metrics({
            "total_segments": total_segments,
            "translated_count": translated_count,
            "drop_count": drop_count,
            "drop_rate_pct": drop_rate,
            "asr_latency_avg_s": avg_asr,
            "asr_latency_p95_s": p95_asr,
            "mt_latency_avg_s": avg_mt,
            "mt_latency_p95_s": p95_mt,
            "e2e_latency_avg_s": avg_e2e,
            "e2e_latency_p95_s": p95_e2e,
            "translation_speed_cps": avg_speed,
        })

        # Log timeline steps (first 60 data points) to visualize latency fluctuation
        sample_len = min(60, len(asr_latencies), len(mt_latencies))
        for step in range(sample_len):
            mlflow.log_metrics({
                "step_asr_s": asr_latencies[step],
                "step_mt_s": mt_latencies[step],
            }, step=step)

def main():
    mlflow.set_experiment("ASMR-Live-Subtitle-Benchmarks")
    print("[mlflow] Tracking experiment: ASMR-Live-Subtitle-Benchmarks")
    for name, path in RUN_FILES:
        parse_and_log_run(name, path)
    print("\n[mlflow] All historical benchmark runs imported successfully!")

if __name__ == "__main__":
    main()
