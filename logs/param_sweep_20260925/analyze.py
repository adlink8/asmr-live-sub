# -*- coding: utf-8 -*-
"""Phase 1 汇总：每配置 段长p50/p90 | 段数 | missing% | 空识别%。

口径：
- 段长 = WorkLog asr 事件 audio_s（音频域秒）；p50/p90 用 ceil 索引（同 realtime_compare.pct）。
- 段数 = asr 事件数；空识别 = asr 事件 ja 为空的比例；seg-drop 单列（segmenter 吐出但被丢弃）。
- missing% = cmp/<cfg>_<scene>.json 的 missing / cues_in_range（realtime_compare 既有口径）。
- 字幕条数 = mt 事件数。
- cfg_prod 行直接读既有 logs/benchmark_runs 产物（adaptive max5 hang2），不重跑。
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(r"D:/ADLINK/asmr-live-sub")
SWEEP = ROOT / "logs" / "param_sweep_20260925"
RUNS = ROOT / "logs" / "benchmark_runs"

SCENES5 = [
    "scene_asmr299717_t02",
    "scene_asmr416809_t01",
    "scene_asmr416816_t01",
    "scene_asmr1521586_t01",
    "scene_asmr1463510_t05",
]
CONFIGS = ["cfg_base", "cfg_h15", "cfg_h10", "cfg_h07", "cfg_h10_m6", "cfg_adapt"]


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    return sorted_vals[max(0, math.ceil(q * len(sorted_vals)) - 1)]


def analyze_jsonl(path):
    lens, n_asr, n_empty, n_drop, n_mt, n_mt_empty = [], 0, 0, 0, 0, 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        k = e.get("kind")
        if k == "asr":
            n_asr += 1
            if not (e.get("ja") or "").strip():
                n_empty += 1
            lens.append(float(e.get("audio_s") or (e.get("t_end", 0) - e.get("t_start", 0))))
        elif k == "seg-drop":
            n_drop += 1
        elif k == "mt":
            n_mt += 1
            if e.get("zh_empty"):
                n_mt_empty += 1
    lens.sort()
    return {
        "n_asr": n_asr, "n_empty": n_empty, "n_drop": n_drop,
        "n_mt": n_mt, "n_mt_empty": n_mt_empty,
        "p50": pct(lens, 0.5), "p90": pct(lens, 0.9), "lens": lens,
    }


def cmp_meta(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["meta"]


def fmt_row(name, wins):
    """wins: {scene: dict with analyze + cmp meta}"""
    all_lens = sorted(v for w in wins.values() for v in w["lens"])
    n_asr = sum(w["n_asr"] for w in wins.values())
    n_empty = sum(w["n_empty"] for w in wins.values())
    n_drop = sum(w["n_drop"] for w in wins.values())
    n_mt = sum(w["n_mt"] for w in wins.values())
    n_mt_empty = sum(w["n_mt_empty"] for w in wins.values())
    missing = sum((w["meta"]["missing"] if w["meta"] else 0) for w in wins.values())
    in_range = sum((w["meta"]["cues_in_range"] if w["meta"] else 0) for w in wins.values())
    p50, p90 = pct(all_lens, 0.5), pct(all_lens, 0.9)
    miss_pct = 100.0 * missing / in_range if in_range else float("nan")
    empty_pct = 100.0 * n_empty / n_asr if n_asr else float("nan")
    return {
        "cfg": name, "n_win": len(wins), "p50": p50, "p90": p90,
        "n_asr": n_asr, "n_drop": n_drop, "empty_pct": round(empty_pct, 1),
        "n_mt": n_mt, "n_mt_empty": n_mt_empty,
        "missing": missing, "in_range": in_range, "miss_pct": round(miss_pct, 1),
        "per_win": {s: {"p50": w["p50"], "p90": w["p90"], "n_asr": w["n_asr"],
                        "miss_pct": (round(100.0 * w["meta"]["missing"] / w["meta"]["cues_in_range"], 1)
                                     if w["meta"] and w["meta"]["cues_in_range"] else None)}
                    for s, w in wins.items()},
    }


def main():
    scenes = sys.argv[1].split(",") if len(sys.argv) > 1 else SCENES5
    configs = sys.argv[2].split(",") if len(sys.argv) > 2 else CONFIGS
    rows = []
    for cfg in configs:
        wins = {}
        for s in scenes:
            if cfg == "cfg_prod":  # 既有 26/25 窗产物（adaptive max5 hang2），不重跑
                jl, cm = RUNS / f"{s}.jsonl", None
            else:
                jl, cm = SWEEP / f"{cfg}_{s}.jsonl", SWEEP / "cmp" / f"{cfg}_{s}.json"
            if not jl.exists():
                continue
            a = analyze_jsonl(jl)
            a["meta"] = cmp_meta(cm)
            wins[s] = a
        if wins:
            rows.append(fmt_row(cfg, wins))

    out = SWEEP / ("_phase1_summary.json" if scenes == SCENES5 else "_phase2_summary.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    hdr = f"{'cfg':<12}{'n_win':>5}{'seg_p50':>9}{'seg_p90':>9}{'段数':>6}{'drop':>6}{'空识别%':>8}{'mt条':>6}{'mt空':>5}{'miss/范围':>10}{'missing%':>9}"
    print(hdr)
    for r in rows:
        print(f"{r['cfg']:<12}{r['n_win']:>5}{r['p50']:>9.2f}{r['p90']:>9.2f}{r['n_asr']:>6}"
              f"{r['n_drop']:>6}{r['empty_pct']:>8}{r['n_mt']:>6}{r['n_mt_empty']:>5}"
              f"{str(r['missing'])+'/'+str(r['in_range']):>10}{r['miss_pct']:>9}")
    print("\n逐窗 p50/missing%:")
    for r in rows:
        parts = [f"{s.replace('scene_asmr',''):>16}: {w['p50']:.2f}s/{w['miss_pct']}%"
                 for s, w in r["per_win"].items()]
        print(f"{r['cfg']:<12} " + " ".join(parts))


if __name__ == "__main__":
    main()
