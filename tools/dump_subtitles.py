import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def export_subs(jsonl_path: Path, title: str):
    if not jsonl_path.exists():
        return
    events = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try: events.append(json.loads(line))
                except: pass

    asr_map = {e.get("seg_id"): e for e in events if e.get("kind") == "asr" and e.get("seg_id") is not None}
    mt_evs = [e for e in events if e.get("kind") == "mt" and e.get("zh")]

    print(f"\n{'='*80}")
    print(f"【{title}】完整中日双语字幕清单 (共 {len(mt_evs)} 条)")
    print(f"{'='*80}\n")

    for i, m in enumerate(mt_evs, 1):
        sid = m.get("seg_id")
        asr = asr_map.get(sid, {})
        t_start = asr.get("t_start")
        t_end = asr.get("t_end")
        audio_s = asr.get("audio_s", 0.0)
        asr_s = m.get("asr_s", 0.0)
        mt_s = m.get("mt_s", 0.0)
        total_delay = round(audio_s + asr_s + mt_s, 2)

        time_str = f"[{t_start:.1f}s -> {t_end:.1f}s]" if t_start is not None and t_end is not None else "[--:--]"
        
        print(f"[{i:02d}] 时间轴: {time_str} | 段长: {audio_s:.1f}s | 端到端延迟: {total_delay}s (ASR:{asr_s:.2f}s + MT:{mt_s:.2f}s)")
        print(f"     [日文原文]: {m.get('ja')}")
        print(f"     [中文翻译]: {m.get('zh')}")
        print("-" * 80)

def main():
    # 打印 15 分钟长时压测完整字幕
    export_subs(ROOT / "logs" / "soak_runs" / "adaptive_15m.jsonl", "15 分钟真实 ASMR 完整流 (蔚蓝档案 Miyo 剧情轨)")
    # 打印 Scene A 连续剧情对话完整字幕
    export_subs(ROOT / "logs" / "benchmark_runs" / "adaptive_5s_scene_a.jsonl", "Scene A 连续剧情对话 (蔚蓝档案 剧情台词轨)")
    # 打印 Scene B 互动相槌完整字幕
    export_subs(ROOT / "logs" / "benchmark_runs" / "adaptive_5s_scene_b.jsonl", "Scene B 间歇互动相槌 (Hololive 天使耳语轨)")

if __name__ == "__main__":
    main()
