"""导出每窗 cue 表 + ASR 段表（只读），供 LLM 逐窗语义比对漂移方向与幅度。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
RUNS = ROOT / "logs" / "benchmark_runs"

SCENES = [
    "scene_asmr1449384_t03", "scene_asmr1449384_t03_w2", "scene_asmr1449384_t03_w3",
    "scene_asmr1463510_t05", "scene_asmr1463510_t05_w2", "scene_asmr1463510_t05_w3",
    "scene_asmr1497366_t07", "scene_asmr1497366_t07_w2", "scene_asmr1497366_t07_w3",
    "scene_asmr1521586_t01", "scene_asmr1521586_t01_w2", "scene_asmr1521586_t01_w3",
    "scene_asmr1527130_t04", "scene_asmr1527130_t04_w2", "scene_asmr1527130_t04_w3",
    "scene_asmr299717_t02_w2",
    "scene_asmr324799_t03", "scene_asmr324799_t03_w2", "scene_asmr324799_t03_w3",
    "scene_asmr416809_t01_w2", "scene_asmr416809_t01_w3",
    "scene_asmr416816_t01_w3",
]

for scene in (sys.argv[1:] or SCENES):
    human = json.loads((DATASET / f"{scene}.human_zh.json").read_text(encoding="utf-8"))
    segs = []
    for line in (RUNS / f"{scene}.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("kind") == "asr" and (e.get("ja") or "").strip():
            segs.append((float(e["t_start"]), float(e["t_end"]), e["ja"].strip()))
    segs.sort()
    print(f"===== {scene}  cues={len(human['segments'])} asr={len(segs)} =====")
    print("-- CUES (idx, t0, zh) --")
    for i, c in enumerate(human["segments"]):
        print(f"  {i:2d} {float(c['t0']):7.2f} {c['zh']}")
    print("-- ASR (t0, t1, ja) --")
    for t0, t1, ja in segs:
        print(f"  {t0:7.2f} {t1:7.2f} {ja}")
    print()
