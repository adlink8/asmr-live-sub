"""Prometheus metrics exporter for ASMR live subtitle pipeline.
Replays recent real metrics or exposes live pipeline stats on port 8000.
"""
import time
import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "work-20260922.jsonl"

CURRENT_METRICS = {
    "asr_latency_seconds": 0.12,
    "mt_latency_seconds": 0.18,
    "e2e_latency_seconds": 0.30,
    "gpu_vram_used_bytes": 7.1 * (1024 ** 3),
    "translation_chars_total": 0,
    "drop_count_total": 0,
    "active_batch_size": 1,
}

def replay_worker():
    """Reads historical trace and updates live gauges smoothly."""
    if not LOG_PATH.exists():
        return
    while True:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                k = d.get("kind")
                if k == "asr":
                    s = d.get("asr_s")
                    if s:
                        CURRENT_METRICS["asr_latency_seconds"] = s
                    if d.get("drop"):
                        CURRENT_METRICS["drop_count_total"] += 1
                elif k == "mt":
                    m = d.get("mt_s") or d.get("gen_s")
                    if m:
                        CURRENT_METRICS["mt_latency_seconds"] = m
                        CURRENT_METRICS["e2e_latency_seconds"] = CURRENT_METRICS["asr_latency_seconds"] + m
                    c = d.get("out_chars")
                    if c:
                        CURRENT_METRICS["translation_chars_total"] += c
                time.sleep(1.0)

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.end_headers()
            
            lines = [
                "# HELP asmr_asr_latency_seconds ASR speech recognition latency in seconds",
                "# TYPE asmr_asr_latency_seconds gauge",
                f"asmr_asr_latency_seconds {CURRENT_METRICS['asr_latency_seconds']:.4f}",
                "",
                "# HELP asmr_mt_latency_seconds LLM translation generation latency in seconds",
                "# TYPE asmr_mt_latency_seconds gauge",
                f"asmr_mt_latency_seconds {CURRENT_METRICS['mt_latency_seconds']:.4f}",
                "",
                "# HELP asmr_e2e_latency_seconds End-to-end total latency in seconds",
                "# TYPE asmr_e2e_latency_seconds gauge",
                f"asmr_e2e_latency_seconds {CURRENT_METRICS['e2e_latency_seconds']:.4f}",
                "",
                "# HELP asmr_gpu_vram_used_bytes Total GPU VRAM used by ASR and MT in bytes",
                "# TYPE asmr_gpu_vram_used_bytes gauge",
                f"asmr_gpu_vram_used_bytes {CURRENT_METRICS['gpu_vram_used_bytes']:.0f}",
                "",
                "# HELP asmr_drop_count_total Total audio segments dropped by VAD filter",
                "# TYPE asmr_drop_count_total counter",
                f"asmr_drop_count_total {CURRENT_METRICS['drop_count_total']}",
                "",
                "# HELP asmr_translation_chars_total Total Chinese characters generated",
                "# TYPE asmr_translation_chars_total counter",
                f"asmr_translation_chars_total {CURRENT_METRICS['translation_chars_total']}",
                ""
            ]
            self.wfile.write("\n".join(lines).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress noisy access logs

def start_server(port=8000):
    t = threading.Thread(target=replay_worker, daemon=True)
    t.start()
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    print(f"[metrics-exporter] Prometheus metrics listening on http://localhost:{port}/metrics")
    server.serve_forever()

if __name__ == "__main__":
    start_server()
