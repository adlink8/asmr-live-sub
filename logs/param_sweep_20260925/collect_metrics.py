# -*- coding: utf-8 -*-
"""参数扫描指标汇总（只读分析，不改 livesub/tools/dataset）。

口径（对齐既有产物）：
- 段数 = WorkLog jsonl 中 asr 事件数；空识别 = ja 为空的 asr 事件。
- 段长 = t_end - t_start（音频域秒）。分位数用 ceil 索引（同
  tools/realtime_compare.py 的 pct），n 小时不插值。
- missing（同 tools/build_anchor_scene.py）：人工 cue 时间窗与任何
  ja 非空 asr 段零重叠 → missing；missing% = missing / 总 cue 数
  （锚点窗 human_zh 时间轴即本地音频时间轴，无偏移）。
- 字幕条数 = mt 事件中 zh 非空且非拒答的数量。
- 质量对齐检查：同窗已跑过的 22 窗批量产物（logs/benchmark_runs，adaptive/
  max5/hang2）不参与本扫描统计，仅 Phase 2 对照时另行引用。

用法：python collect_metrics.py [--phase2]
输出：metrics.json + metrics.md（每 cfg 一行）。
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(r"D:/ADLINK/asmr-live-sub")
OUT = ROOT / "logs" / "param_sweep_20260925"
DATASET = ROOT / "dataset"

SUBSET5 = [
    "scene_asmr299717_t02",
    "scene_asmr416809_t01",
    "scene_asmr416816_t01",
    "scene_asmr1521586_t01",
    "scene_asmr1463510_t05",
]
WINDOWS26 = [
    "scene_asmr1449384_t03", "scene_asmr1449384_t03_w2", "scene_asmr1449384_t03_w3",
    "scene_asmr1463510_t05", "scene_asmr1463510_t05_w2", "scene_asmr1463510_t05_w3",
    "scene_asmr1497366_t07", "scene_asmr1497366_t07_w2", "scene_asmr1497366_t07_w3",
    "scene_asmr1521586_t01", "scene_asmr1521586_t01_w2", "scene_asmr1521586_t01_w3",
    "scene_asmr1527130_t04", "scene_asmr1527130_t04_w2", "scene_asmr1527130_t04_w3",
    "scene_asmr299717_t02", "scene_asmr299717_t02_w2",
    "scene_asmr324799_t03", "scene_asmr324799_t03_w2", "scene_asmr324799_t03_w3",
    "scene_asmr401391_t01",
    "scene_asmr416809_t01", "scene_asmr416809_t01_w2", "scene_asmr416809_t01_w3",
    "scene_asmr416816_t01", "scene_asmr416816_t01_w3",
]
CFGS = ["cfg_base", "cfg_h15", "cfg_h10", "cfg_h07", "cfg_h10_m6", "cfg_adapt"]


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    return sorted_vals[max(0, math.ceil(q * len(sorted_vals)) - 1)]


def load_run(jsonl_path):
    """→ (segs, n_subs)：segs=[(t0,t1,ja)]，n_subs=有效 zh 翻译条数。"""
    segs, n_subs = [], 0
    for line in jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("kind") == "asr":
            segs.append((float(e.get("t_start") or 0.0), float(e.get("t_end") or 0.0),
                         (e.get("ja") or "").strip()))
        elif e.get("kind") == "mt" and (e.get("zh") or "").strip() and not e.get("zh_refusal"):
            n_subs += 1
    return segs, n_subs


def load_cues(scene):
    p = DATASET / f"{scene}.human_zh.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return [(float(s["t0"]), float(s["t1"]), (s.get("zh") or "").strip())
            for s in data["segments"]]


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def window_metrics(cfg, scene):
    jl = OUT / f"{cfg}_{scene}.jsonl"
    if not jl.exists():
        return None
    segs, n_subs = load_run(jl)
    durs = sorted(t1 - t0 for t0, t1, _ in segs)
    n_empty = sum(1 for _, _, ja in segs if not ja)
    voiced = [(t0, t1) for t0, t1, ja in segs if ja]
    cues = load_cues(scene)
    n_missing = 0
    for c0, c1, _ in cues:
        if not any(overlap(t0, t1, c0, c1) > 0 for t0, t1 in voiced):
            n_missing += 1
    return {
        "n_seg": len(segs), "n_empty": n_empty, "n_subs": n_subs,
        "dur_p50": round(pct(durs, 0.5), 2), "dur_p90": round(pct(durs, 0.9), 2),
        "n_cues": len(cues), "n_missing": n_missing,
        "missing_pct": round(100.0 * n_missing / len(cues), 1) if cues else None,
        "empty_pct": round(100.0 * n_empty / len(segs), 1) if segs else None,
    }


def aggregate(per_win):
    """pooled：段长分位对全段合并算，missing 对 cue 合并算。"""
    all_durs, segs_n, empty_n, subs_n, cues_n, miss_n = [], 0, 0, 0, 0, 0
    for m in per_win:
        if m is None:
            continue
        all_durs.append(m["_durs"])
        segs_n += m["n_seg"]
        empty_n += m["n_empty"]
        subs_n += m["n_subs"]
        cues_n += m["n_cues"]
        miss_n += m["n_missing"]
    flat = sorted(x for d in all_durs for x in d)
    return {
        "windows": len(all_durs),
        "n_seg": segs_n, "n_empty": empty_n, "n_subs": subs_n,
        "dur_p50": round(pct(flat, 0.5), 2) if flat else None,
        "dur_p90": round(pct(flat, 0.9), 2) if flat else None,
        "n_cues": cues_n, "n_missing": miss_n,
        "missing_pct": round(100.0 * miss_n / cues_n, 1) if cues_n else None,
        "empty_pct": round(100.0 * empty_n / segs_n, 1) if segs_n else None,
    }


def main():
    phase2 = "--phase2" in sys.argv
    if phase2:
        groups = {"cfg_base": WINDOWS26, "cfg_adapt_full": WINDOWS26}
    else:
        groups = {c: SUBSET5 for c in CFGS}
    report = {}
    for cfg, scenes in groups.items():
        per_win = {}
        for s in scenes:
            m = window_metrics(cfg, s)
            if m is None:
                print(f"[WARN] 缺产物: {cfg}_{s}")
                continue
            durs = []
            jl = OUT / f"{cfg}_{s}.jsonl"
            segs, _ = load_run(jl)
            durs = sorted(t1 - t0 for t0, t1, _ in segs)
            m["_durs"] = durs
            per_win[s] = m
        agg = aggregate(list(per_win.values()))
        report[cfg] = {"per_window": {k: {kk: vv for kk, vv in v.items() if kk != "_durs"}
                                      for k, v in per_win.items()},
                       "pooled": agg}
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    lines = ["| 配置 | 窗数 | 段数 | 空识别% | 段长p50 | 段长p90 | 字幕条数 | cue数 | missing | missing% |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for cfg, r in report.items():
        a = r["pooled"]
        lines.append(f"| {cfg} | {a['windows']} | {a['n_seg']} | {a['empty_pct']}% | "
                     f"{a['dur_p50']}s | {a['dur_p90']}s | {a['n_subs']} | "
                     f"{a['n_cues']} | {a['n_missing']} | {a['missing_pct']}% |")
    (OUT / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
