import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOGS_DIR = ROOT / "logs" / "marathon_runs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_jsonl(jsonl_path: Path):
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
        "sample_subs": [(m.get("ja", ""), m.get("zh", "")) for m in mt_evs[:8] if m.get("zh")],
    }

def run_session(audio_path: Path, strategy_args: list, run_name: str, label: str):
    log_file = LOGS_DIR / f"{run_name}.txt"
    work_file = LOGS_DIR / f"{run_name}.jsonl"

    if log_file.exists(): log_file.unlink()
    if work_file.exists(): work_file.unlink()

    cmd = [
        str(PYTHON), "live_sub.py",
        "--model", "anime",
        "--mt", "sakura",
        "--mt-ngl", "99",
        "--layer", "1",
        "--source-audio", str(audio_path),
        "--replay",
        "--fast",
        "--log", str(log_file),
    ] + strategy_args

    print(f"\n" + "="*75)
    print(f"[*] 启动压测: {label}")
    print(f"    音频: {audio_path.name} | 模式: fast 推流")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    cost_s = time.time() - t0

    if not work_file.exists():
        print(f"[!] 错误: 未生成 {work_file}")
        print(p.stderr[-300:] if p.stderr else "")
        return {}

    print(f"    处理完成！物理音频时长: {audio_path.stat().st_size/32000:.0f}s, 推理计算耗时: {cost_s:.1f}s")
    metrics = analyze_jsonl(work_file)
    metrics["process_time_s"] = round(cost_s, 1)
    return metrics

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "30m"
    if mode == "30m":
        print("=== 第一批：30 分钟真实长音频对决测试 ===")
        res_base = run_session(ROOT / "benchmarks" / "soak_30min.wav",
                               ["--strategy", "baseline", "--max-s", "8.0", "--hang-s", "2.0"],
                               "30m_baseline", "30分钟 Baseline (原方案)")
        res_adapt = run_session(ROOT / "benchmarks" / "soak_30min.wav",
                                ["--strategy", "adaptive", "--max-s", "5.0", "--hang-s", "2.0"],
                                "30m_adaptive", "30分钟 自适应衰减 (新方案)")
        out_json = LOGS_DIR / "report_30m.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"baseline": res_base, "adaptive": res_adapt}, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 30 分钟评测完成，数据已落盘: {out_json}")
    elif mode == "60m":
        print("=== 第二批：1 小时超长马拉松耐力测试 ===")
        res_60m = run_session(ROOT / "benchmarks" / "soak_60min.wav",
                              ["--strategy", "adaptive", "--max-s", "5.0", "--hang-s", "2.0"],
                              "60m_adaptive", "1小时 自适应衰减耐力长跑")
        out_json = LOGS_DIR / "report_60m.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"adaptive_60m": res_60m}, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 1 小时评测完成，数据已落盘: {out_json}")
