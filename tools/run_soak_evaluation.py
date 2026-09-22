import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOGS_DIR = ROOT / "logs" / "soak_runs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SOAK_WAV = ROOT / "benchmarks" / "soak_15min.wav"

STRATS = [
    {
        "id": "baseline_15m",
        "name": "策略 0: Baseline (原有逻辑, 15分钟)",
        "args": ["--strategy", "baseline", "--max-s", "8.0", "--hang-s", "2.0"]
    },
    {
        "id": "adaptive_15m",
        "name": "策略 2: 自适应阶梯衰减 (优选方案, 15分钟)",
        "args": ["--strategy", "adaptive", "--max-s", "5.0", "--hang-s", "2.0"]
    }
]

def analyze_soak(jsonl_path: Path):
    events = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass

    asr_evs = [e for e in events if e.get("kind") == "asr"]
    mt_evs = [e for e in events if e.get("kind") == "mt"]
    drop_evs = [e for e in events if e.get("kind") == "seg-drop"]
    batch_evs = [e for e in events if e.get("kind") == "asr-batch"]

    delays = []
    durations = []
    asr_map = {e.get("seg_id"): e for e in asr_evs if e.get("seg_id") is not None}

    for mt in mt_evs:
        sid = mt.get("seg_id")
        asr = asr_map.get(sid)
        if asr:
            audio_s = asr.get("audio_s") or 0.0
            asr_s = asr.get("asr_s") or 0.0
            mt_s = mt.get("mt_s") or 0.0
            delays.append(round(audio_s + asr_s + mt_s, 2))

    for asr in asr_evs:
        d = asr.get("audio_s")
        if d is not None:
            durations.append(d)

    def p(vals, pct):
        if not vals:
            return 0.0
        s = sorted(vals)
        k = (len(s) - 1) * pct
        f = int(k)
        c = min(f + 1, len(s) - 1)
        return round(s[f] + (k - f) * (s[c] - s[f]), 2)

    total_subtitles = len([m for m in mt_evs if m.get("zh") and not m.get("zh_refusal")])
    over_7s = len([d for d in durations if d >= 7.0])
    over_5s = len([d for d in durations if d >= 5.0])
    total_olds = sum(b.get("olds", 0) for b in batch_evs)

    return {
        "total_asr": len(asr_evs),
        "total_drop": len(drop_evs),
        "subtitles": total_subtitles,
        "dur_p50": p(durations, 0.50),
        "dur_p90": p(durations, 0.90),
        "dur_p99": p(durations, 0.99),
        "dur_max": max(durations) if durations else 0.0,
        "over_5s_count": over_5s,
        "over_5s_pct": round(over_5s / len(durations) * 100, 1) if durations else 0.0,
        "over_7s_count": over_7s,
        "over_7s_pct": round(over_7s / len(durations) * 100, 1) if durations else 0.0,
        "delay_p50": p(delays, 0.50),
        "delay_p90": p(delays, 0.90),
        "delay_p99": p(delays, 0.99),
        "delay_max": max(delays) if delays else 0.0,
        "queue_olds": total_olds,
        "sample_subs": [(m.get("ja", ""), m.get("zh", "")) for m in mt_evs[:6] if m.get("zh")],
    }

def run_test(strat):
    run_name = strat["id"]
    log_file = LOGS_DIR / f"{run_name}.txt"
    work_file = LOGS_DIR / f"{run_name}.jsonl"

    if log_file.exists(): log_file.unlink()
    if work_file.exists(): work_file.unlink()

    cmd = [
        str(PYTHON), "live_sub.py",
        "--model", "anime",
        "--mt", "sakura",
        "--layer", "1",
        "--source-audio", str(SOAK_WAV),
        "--replay",
        "--fast",
        "--log", str(log_file),
    ] + strat["args"]

    print(f"\n" + "="*70)
    print(f"[*] 启动 15 分钟耐力压测: {strat['name']}")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    cost_s = time.time() - t0

    if not work_file.exists():
        print(f"[!] 错误: 未生成 {work_file}")
        print(p.stderr[-300:] if p.stderr else "")
        return {}

    print(f"    15 分钟长音频吞吐处理完成，总耗时: {cost_s:.1f}s")
    metrics = analyze_soak(work_file)
    metrics["process_time_s"] = round(cost_s, 1)
    return metrics

def main():
    print("=== 开始 ASMR 15 分钟（900 秒）终局真实耐力测试 ===")
    results = {}
    for s in STRATS:
        results[s["id"]] = run_test(s)

    with open(LOGS_DIR / "soak_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "="*80)
    print("=== 15 分钟长时压测对比总榜 (15-Minute Soak Results) ===")
    print("="*80)
    print(f"{'指标项目':<25} | {'策略 0: Baseline (原版)':<25} | {'策略 2: 自适应衰减 (新版)':<25}")
    print("-" * 80)
    
    b = results["baseline_15m"]
    a = results["adaptive_15m"]

    rows = [
        ("ASR 分段识别总数", f"{b['total_asr']} 段", f"{a['total_asr']} 段"),
        ("有效中文字幕产出", f"{b['subtitles']} 条", f"{a['subtitles']} 条"),
        ("段长 p50 中位数", f"{b['dur_p50']} 秒", f"{a['dur_p50']} 秒"),
        ("段长 p90 分位数", f"{b['dur_p90']} 秒", f"{a['dur_p90']} 秒"),
        ("段长 Max 最大值", f"{b['dur_max']} 秒", f"{a['dur_max']} 秒"),
        ("≥ 5.0s 长段积压率", f"{b['over_5s_pct']}% ({b['over_5s_count']}段)", f"{a['over_5s_pct']}% ({a['over_5s_count']}段)"),
        ("≥ 7.0s 恶性拖延率", f"{b['over_7s_pct']}% ({b['over_7s_count']}段)", f"{a['over_7s_pct']}% ({a['over_7s_count']}段)"),
        ("端到端延迟 p50", f"{b['delay_p50']} 秒", f"{a['delay_p50']} 秒"),
        ("端到端延迟 p90", f"{b['delay_p90']} 秒", f"{a['delay_p90']} 秒"),
        ("端到端延迟 Max", f"{b['delay_max']} 秒", f"{a['delay_max']} 秒"),
        ("队列积压降级数 (olds)", f"{b['queue_olds']} (健康)", f"{a['queue_olds']} (健康)"),
        ("900s 处理总耗时", f"{b['process_time_s']} 秒", f"{a['process_time_s']} 秒"),
    ]

    for name, b_val, a_val in rows:
        print(f"{name:<25} | {b_val:<25} | {a_val:<25}")

if __name__ == "__main__":
    main()
