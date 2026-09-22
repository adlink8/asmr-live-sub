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

CUSTOM_SCENES = [
    ("dense_talk", ROOT / "benchmarks" / "scene_dense_talk.wav", "【场景一：高频极速对话】Hololive 特典杂谈闲聊（3人轮流连珠炮，人声密度 99%）"),
    ("random_talk", ROOT / "benchmarks" / "scene_random_talk.wav", "【场景二：随机节奏说话】蔚蓝档案 圣园未夜（忽快忽慢、随机停顿掏耳与互动，人声密度 66%）"),
]

def run_scene(scene_id, scene_path, scene_title):
    log_file = LOGS_DIR / f"custom_{scene_id}.txt"
    work_file = LOGS_DIR / f"custom_{scene_id}.jsonl"

    if log_file.exists(): log_file.unlink()
    if work_file.exists(): work_file.unlink()

    cmd = [
        str(PYTHON), "live_sub.py",
        "--model", "anime",
        "--mt", "sakura",
        "--mt-ngl", "99",
        "--layer", "1",
        "--strategy", "adaptive",
        "--max-s", "5.0",
        "--hang-s", "2.0",
        "--source-audio", str(scene_path),
        "--replay",
        "--fast",
        "--log", str(log_file),
    ]

    print(f"\n{'='*75}")
    print(f"[*] 开始运行【纯 GPU 双模型加速】评测: {scene_title}")
    print(f"    音频输入: {scene_path.name} (时长 180s)")
    print(f"    模型配置: anime-whisper (GPU fp16) + Sakura-7B (GPU Tensor Cores ngl=99)")
    print(f"    切块配置: 策略 2 自适应阶梯衰减 (3s衰减, Max 5.0s)")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    cost_s = time.time() - t0
    print(f"    运行完成! 整体推理与处理耗时: {cost_s:.1f}s")

    if not work_file.exists():
        print(f"[!] 错误: 未生成日志 {work_file}")
        print("STDOUT:", p.stdout[-500:] if p.stdout else "")
        print("STDERR:", p.stderr[-500:] if p.stderr else "")
        return None

    # 解析日志
    events = []
    with work_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: events.append(json.loads(line))
                except: pass

    asr_evs = [e for e in events if e.get("kind") == "asr"]
    mt_evs = [e for e in events if e.get("kind") == "mt"]
    asr_map = {e.get("seg_id"): e for e in asr_evs if e.get("seg_id") is not None}

    subtitles = []
    for mt in mt_evs:
        sid = mt.get("seg_id")
        asr = asr_map.get(sid, {})
        ja = mt.get("ja", "").strip()
        zh = mt.get("zh", "").strip()
        if zh and not mt.get("zh_refusal"):
            audio_s = asr.get("audio_s") or 0.0
            asr_s = asr.get("asr_s") or 0.0
            mt_s = mt.get("mt_s") or 0.0
            t0_s = asr.get("t0") or 0.0
            t1_s = asr.get("t1") or 0.0
            total_delay = round(audio_s + asr_s + mt_s, 2)
            tok_s = mt.get("pred_per_s") or 0.0
            subtitles.append({
                "seg_id": sid,
                "t0": t0_s,
                "t1": t1_s,
                "audio_s": audio_s,
                "asr_s": asr_s,
                "mt_s": mt_s,
                "delay": total_delay,
                "tok_s": tok_s,
                "ja": ja,
                "zh": zh
            })

    def p(vals, pct):
        if not vals: return 0.0
        s = sorted(vals)
        k = (len(s) - 1) * pct
        f = int(k)
        c = min(f + 1, len(s) - 1)
        return round(s[f] + (k - f) * (s[c] - s[f]), 2)

    audio_lens = [s["audio_s"] for s in subtitles]
    asr_times = [s["asr_s"] for s in subtitles]
    mt_times = [s["mt_s"] for s in subtitles]
    delays = [s["delay"] for s in subtitles]
    tokens = [s["tok_s"] for s in subtitles if s.get("tok_s") is not None and s["tok_s"] > 0]

    stats = {
        "title": scene_title,
        "count": len(subtitles),
        "audio_p50": p(audio_lens, 0.5),
        "asr_p50": p(asr_times, 0.5),
        "asr_p90": p(asr_times, 0.9),
        "mt_p50": p(mt_times, 0.5),
        "mt_p90": p(mt_times, 0.9),
        "delay_p50": p(delays, 0.5),
        "delay_p90": p(delays, 0.9),
        "delay_max": max(delays) if delays else 0.0,
        "tokens_p50": p(tokens, 0.5) if tokens else 0.0,
        "subtitles": subtitles
    }
    return stats

def main():
    all_stats = []
    for s_id, s_path, s_title in CUSTOM_SCENES:
        st = run_scene(s_id, s_path, s_title)
        if st:
            all_stats.append(st)

    # 汇总输出
    print("\n\n" + "="*80)
    print("=== 全流程评测结果与双语字幕完整清单 ===")
    print("="*80)

    for st in all_stats:
        print(f"\n\n######################################################################")
        print(f"{st['title']}")
        print(f"######################################################################")
        print(f"【效率总览】")
        print(f"  * 产出有效字幕总数: {st['count']} 条 (180秒音频)")
        print(f"  * 音频切块时长 p50 : {st['audio_p50']}s")
        print(f"  * ASR 耗时 (p50/p90): {st['asr_p50']}s / {st['asr_p90']}s")
        print(f"  * MT 耗时  (p50/p90): {st['mt_p50']}s / {st['mt_p90']}s (Token速度 ~ {st['tokens_p50']} tok/s)")
        print(f"  * 端到端总延迟 (p50/p90/max): {st['delay_p50']}s / {st['delay_p90']}s / {st['delay_max']}s")
        print(f"\n【完整流程中日双语字幕清单】(按时序输出):")
        print(f"{'-'*75}")
        for i, sub in enumerate(st["subtitles"], 1):
            print(f"[{i:02d}] 时间: {sub['t0']:.1f}s ~ {sub['t1']:.1f}s | 切块: {sub['audio_s']:.1f}s | 延迟: {sub['delay']:.1f}s (ASR {sub['asr_s']:.2f}s + MT {sub['mt_s']:.2f}s)")
            print(f"     JA: {sub['ja']}")
            print(f"     ZH: {sub['zh']}")
            print()

    # 保存最终详细结果为 json
    out_json = LOGS_DIR / "custom_scenes_full_result.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\n完整评测数据已持久化至: {out_json}")

if __name__ == "__main__":
    main()
