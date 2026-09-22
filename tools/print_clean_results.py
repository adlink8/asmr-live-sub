import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / "logs" / "benchmark_runs" / "custom_scenes_full_result.json"
with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    print("=" * 80)
    print(item["title"])
    print(f"有效字幕数: {item['count']} 条 (180秒音频)")
    print(f"切块段长 p50: {item['audio_p50']}s")
    print(f"ASR 耗时 (p50/p90): {item['asr_p50']}s / {item['asr_p90']}s")
    print(f"MT 耗时  (p50/p90): {item['mt_p50']}s / {item['mt_p90']}s")
    print(f"端到端总延迟 (p50/p90/max): {item['delay_p50']}s / {item['delay_p90']}s / {item['delay_max']}s")
    print("\n所有字幕 (完整时序):")
    for i, s in enumerate(item["subtitles"], 1):
        print(f"[{i:02d}] 段长: {s['audio_s']:.1f}s | 延迟: {s['delay']:.1f}s (ASR {s['asr_s']:.2f}s + MT {s['mt_s']:.2f}s)")
        print(f"     JA: {s['ja']}")
        print(f"     ZH: {s['zh']}")
    print()
