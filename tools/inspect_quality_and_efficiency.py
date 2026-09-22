import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOAK_JSONL = ROOT / "logs" / "soak_runs" / "adaptive_15m.jsonl"
M30_JSONL = ROOT / "logs" / "marathon_runs" / "30m_adaptive.jsonl"
BENCH_JSONL = ROOT / "logs" / "benchmark_runs" / "adaptive_5s_scene_a.jsonl"

def analyze_detail(jsonl_path: Path):
    if not jsonl_path.exists():
        return {}
    events = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try: events.append(json.loads(line))
                except: pass

    asr_evs = [e for e in events if e.get("kind") == "asr"]
    mt_evs = [e for e in events if e.get("kind") == "mt"]

    asr_times = [e["asr_s"] for e in asr_evs if "asr_s" in e]
    mt_times = [e["mt_s"] for e in mt_evs if "mt_s" in e]
    audio_lens = [e["audio_s"] for e in asr_evs if "audio_s" in e]
    pred_per_s = [e["pred_per_s"] for e in mt_evs if e.get("pred_per_s")]

    def p(vals, pct):
        if not vals: return 0.0
        s = sorted(vals)
        k = (len(s) - 1) * pct
        f = int(k)
        c = min(f + 1, len(s) - 1)
        return round(s[f] + (k - f) * (s[c] - s[f]), 3)

    return {
        "audio_p50": p(audio_lens, 0.5),
        "asr_p50": p(asr_times, 0.5),
        "asr_p90": p(asr_times, 0.9),
        "mt_p50": p(mt_times, 0.5),
        "mt_p90": p(mt_times, 0.9),
        "tokens_per_s": p(pred_per_s, 0.5) if pred_per_s else "N/A",
        "qa_pairs": [(m.get("ja", ""), m.get("zh", ""), m.get("asr_s"), m.get("mt_s")) for m in mt_evs if m.get("zh")]
    }

def print_summary(label, jsonl_path):
    d = analyze_detail(jsonl_path)
    if not d: return
    print(f"\n=======================================================")
    print(f"【{label}】")
    print(f"  各环节延迟分解 (p50 / p90):")
    print(f"    1. 音频段长 (蓄水等待) : p50={d['audio_p50']}s")
    print(f"    2. ASR识别 (GPU fp16)  : p50={d['asr_p50']}s | p90={d['asr_p90']}s")
    print(f"    3. 大模型翻译 (CPU 7B)  : p50={d['mt_p50']}s | p90={d['mt_p90']}s")
    print(f"  真实字幕抽检 (JA -> ZH):")
    for ja, zh, a_s, m_s in d["qa_pairs"]:
        print(f"    日文原文: {ja}")
        print(f"    中文译文: {zh}")
        print(f"    (ASR耗时: {a_s}s, 翻译耗时: {m_s}s)")
        print()

print_summary("15分钟长音频 (Soak 15m)", SOAK_JSONL)
print_summary("30分钟大样本 (Marathon 30m)", M30_JSONL)
print_summary("Scene A 连续对话 (Scene A 3m)", BENCH_JSONL)
