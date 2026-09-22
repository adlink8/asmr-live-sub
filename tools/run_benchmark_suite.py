import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOGS_DIR = ROOT / "logs" / "benchmark_runs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SCENES = [
    ("scene_a", ROOT / "benchmarks" / "scene_a_dialogue.wav", "Scene A: 连续长句念白"),
    ("scene_b", ROOT / "benchmarks" / "scene_b_interactive.wav", "Scene B: 间歇互动相槌"),
    ("scene_c", ROOT / "benchmarks" / "scene_c_whisper_soft.wav", "Scene C: 弱音耳语伴睡"),
]

STRATEGIES = [
    {
        "id": "baseline",
        "name": "策略 0: Baseline (当前基准)",
        "args": ["--strategy", "baseline", "--max-s", "8.0", "--hang-s", "2.0"]
    },
    {
        "id": "hardcap_5s",
        "name": "策略 1: 严格物理硬切 (Max 5.0s)",
        "args": ["--strategy", "hardcap", "--max-s", "5.0", "--hang-s", "2.0"]
    },
    {
        "id": "adaptive_5s",
        "name": "策略 2: 自适应阶梯衰减 (3s衰减, Max 5.0s)",
        "args": ["--strategy", "adaptive", "--max-s", "5.0", "--hang-s", "2.0"]
    },
]

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

    # 配对 seg_id 算延迟 (Audio Dur + ASR + MT)
    delays = []
    durations = []
    
    # 建立 seg_id -> asr mapping
    asr_map = {e.get("seg_id"): e for e in asr_evs if e.get("seg_id") is not None}
    
    for mt in mt_evs:
        sid = mt.get("seg_id")
        asr = asr_map.get(sid)
        if asr:
            audio_s = asr.get("audio_s") or 0.0
            asr_s = asr.get("asr_s") or 0.0
            mt_s = mt.get("mt_s") or 0.0
            total_delay = round(audio_s + asr_s + mt_s, 2)
            delays.append(total_delay)

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

    return {
        "total_asr": len(asr_evs),
        "total_drop": len(drop_evs),
        "subtitles": total_subtitles,
        "dur_p50": p(durations, 0.50),
        "dur_p90": p(durations, 0.90),
        "dur_max": max(durations) if durations else 0.0,
        "over_5s_pct": round(over_5s / len(durations) * 100, 1) if durations else 0.0,
        "over_7s_pct": round(over_7s / len(durations) * 100, 1) if durations else 0.0,
        "delay_p50": p(delays, 0.50),
        "delay_p90": p(delays, 0.90),
        "delay_max": max(delays) if delays else 0.0,
        "sample_subs": [(m.get("ja", ""), m.get("zh", "")) for m in mt_evs[:4] if m.get("zh")],
    }

def run_single(strat, scene):
    s_id, s_path, s_name = scene
    run_name = f"{strat['id']}_{s_id}"
    log_file = LOGS_DIR / f"{run_name}.txt"
    work_file = LOGS_DIR / f"{run_name}.jsonl"

    if log_file.exists(): log_file.unlink()
    if work_file.exists(): work_file.unlink()

    cmd = [
        str(PYTHON), "live_sub.py",
        "--model", "anime",
        "--mt", "sakura",
        "--layer", "1",
        "--source-audio", str(s_path),
        "--replay",
        "--fast",
        "--log", str(log_file),
    ] + strat["args"]

    print(f"\n=======================================================")
    print(f"[*] 运行评测: {strat['name']} | {s_name}")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    cost_s = time.time() - t0

    if not work_file.exists():
        print(f"[!] 错误: 未生成 {work_file}")
        print("STDOUT:", p.stdout[-300:] if p.stdout else "")
        print("STDERR:", p.stderr[-300:] if p.stderr else "")
        return {}

    print(f"    完成: 推流与处理耗时 {cost_s:.1f}s")
    metrics = analyze_jsonl(work_file)
    return metrics

def main():
    results = {}
    for strat in STRATEGIES:
        results[strat["id"]] = {}
        for scene in SCENES:
            m = run_single(strat, scene)
            results[strat["id"]][scene[0]] = m

    # 生成汇报报告
    print("\n\n" + "="*80)
    print("=== 全场景切块策略真实对比评测报告 (Benchmark Results) ===")
    print("="*80 + "\n")

    for s_id, _, s_name in SCENES:
        print(f"\n### 【{s_name}】")
        print(f"| 策略方案 | 段长 p50 | 段长 max | ≥5s 占比 | 端到端延迟 p50 | 端到端延迟 p90 | 产出字幕数 |")
        print(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for strat in STRATEGIES:
            m = results[strat["id"]][s_id]
            print(f"| {strat['name']} | {m['dur_p50']}s | {m['dur_max']}s | {m['over_5s_pct']}% | **{m['delay_p50']}s** | {m['delay_p90']}s | {m['subtitles']} 条 |")

    # 保存 JSON
    with open(LOGS_DIR / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n数据已保存到: {LOGS_DIR / 'final_report.json'}")

if __name__ == "__main__":
    main()
