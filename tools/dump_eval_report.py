import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
json_path = ROOT / "logs" / "benchmark_runs" / "custom_scenes_full_result.json"
out_md = ROOT / "logs" / "benchmark_runs" / "report.md"

with open(json_path, "r", encoding="utf-8") as f:
    scenes = json.load(f)

lines = []
lines.append("# 全流程评测报告：高频说话与随机说话场景测试\n")

for item in scenes:
    lines.append(f"## {item['title']}\n")
    lines.append("### 1. 效率总览 (Efficiency Breakdown)")
    lines.append(f"- **有效产出字幕**: {item['count']} 条 (180秒音频)")
    lines.append(f"- **音频切块时长 (audio_s)**: p50 = **{item['audio_p50']}s**")
    lines.append(f"- **ASR 识别耗时 (asr_s)**: p50 = **{item['asr_p50']}s** | p90 = **{item['asr_p90']}s** (anime-whisper, GPU fp16)")
    lines.append(f"- **大模型翻译耗时 (mt_s)**: p50 = **{item['mt_p50']}s** | p90 = **{item['mt_p90']}s** (Sakura-7B, GPU Tensor Cores ngl=99)")
    lines.append(f"- **端到端总延迟 (Total Delay)**: p50 = **{item['delay_p50']}s** | p90 = **{item['delay_p90']}s** | max = **{item['delay_max']}s**\n")
    
    lines.append("### 2. 完整时序中日双语字幕清单 (Subtitles)\n")
    lines.append("| 序号 | 切块段长 | ASR耗时 | 翻译耗时 | 总延迟 | 日文原文 (ASR) | 中文译文 (Sakura-7B) |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for i, s in enumerate(item["subtitles"], 1):
        ja = s["ja"].replace("|", "\\|")
        zh = s["zh"].replace("|", "\\|")
        lines.append(f"| {i:02d} | {s['audio_s']:.1f}s | {s['asr_s']:.2f}s | {s['mt_s']:.2f}s | **{s['delay']:.1f}s** | {ja} | {zh} |")
    lines.append("\n---\n")

with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Report written to {out_md} successfully.")
