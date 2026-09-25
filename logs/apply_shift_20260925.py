"""应用人工字幕时间轴漂移修正（2026-09-25）。

检测结论：受影响窗的缺陷不是"整行错位"，而是**每窗一个恒定的时间偏移**
（VTT 时间轴相对音频存在随音轨位置累积的慢漂移；窗内近似恒定）。
LLM 逐窗语义比对（中日双语抽样 5~16 对）确认各窗偏移量后，
对本表所列窗把 segments 的 t0/t1 整体平移 δ 秒。

k=0 验证窗只追加 note，不动时间戳。324799_w2/w3（ASR 零产出）完全不动。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"

# scene -> (seconds, confidence)
SHIFTS = {
    "scene_asmr1449384_t03_w2": (6.5, "high"),
    "scene_asmr1449384_t03_w3": (14.5, "high"),
    "scene_asmr1463510_t05_w2": (14.0, "high"),
    "scene_asmr1463510_t05_w3": (18.0, "high"),
    "scene_asmr1497366_t07_w2": (7.0, "high"),
    "scene_asmr1497366_t07_w3": (14.0, "medium"),
    "scene_asmr1521586_t01_w2": (4.3, "high"),
    "scene_asmr1521586_t01_w3": (11.0, "high"),
    "scene_asmr1527130_t04_w2": (9.0, "high"),
    "scene_asmr1527130_t04_w3": (12.5, "high"),
    "scene_asmr299717_t02_w2": (7.0, "medium"),
    "scene_asmr416809_t01_w2": (19.5, "high"),
    "scene_asmr416809_t01_w3": (38.5, "high"),
    "scene_asmr416816_t01_w3": (13.0, "high"),
}

# k=0 语义验证窗：只追加 note
VERIFIED = [
    "scene_asmr1449384_t03", "scene_asmr1463510_t05", "scene_asmr1497366_t07",
    "scene_asmr1521586_t01", "scene_asmr1527130_t04", "scene_asmr324799_t03",
]

METHOD = "pair-overlap + LLM semantic sampling (per-window constant time offset; VTT timebase drift is continuous, not line-quantized)"

for scene, (sec, conf) in SHIFTS.items():
    p = DATASET / f"{scene}.human_zh.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    n = len(d["segments"])
    for seg in d["segments"]:
        seg["t0"] = round(float(seg["t0"]) + sec, 2)
        seg["t1"] = round(float(seg["t1"]) + sec, 2)
    d["timeline_shift"] = {
        "lines": None,
        "seconds": sec,
        "applied_at": "2026-09-25",
        "method": METHOD,
        "confidence": conf,
    }
    d["note"] = (d.get("note") or "").rstrip() + (
        f"\n2026-09-25 时间轴修正：全窗 cue 平移 +{sec}s（VTT 时间轴相对音频系统性滞后，"
        f"逐对语义比对确认；{conf} 置信）。原始时间戳见 git 591558b。")
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SHIFT] {scene}: +{sec}s ({conf}), {n} cues")

for scene in VERIFIED:
    p = DATASET / f"{scene}.human_zh.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["note"] = (d.get("note") or "").rstrip() + (
        "\n2026-09-25 时间轴核验：verified aligned, shift=0"
        "（pair-overlap + LLM 语义抽样；窗内残余漂移 ≤ 约5s，未跨 ASR 段边界，配对不受影响）。")
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK   ] {scene}: verified aligned, shift=0")
