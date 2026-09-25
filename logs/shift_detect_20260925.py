"""逐窗检测人工字幕时间轴漂移（只读，不改任何文件）。

对每个窗，尝试 k ∈ {-2,-1,0,+1,+2} 行的 cue 序列平移：
  shift k 的定义：文本 i 取 cue[i+k] 的时间戳（k>0 = 文本整体后移，
  修正"字幕早于语音"；k<0 = 文本整体前移）。
  越界（i+k 不在窗内）的 cue 不参与该 k 的配对统计。
判据（几何）：配对数最大、总重叠秒数最大。
同时导出 k=0 与最优 k 的抽样配对（cue 中文 ↔ 匹配到的 ASR 日文），
供 LLM 语义抽样比对。

用法：python logs/shift_detect_20260925.py [--samples 6] [--out logs/shift_detect_report.json]
"""
import argparse
import json
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


def load_asr(scene):
    segs = []
    p = RUNS / f"{scene}.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("kind") == "asr" and (e.get("ja") or "").strip():
            segs.append({"seg_id": e.get("seg_id"),
                         "t0": float(e.get("t_start") or 0.0),
                         "t1": float(e.get("t_end") or 0.0),
                         "ja": e["ja"].strip()})
    segs.sort(key=lambda s: s["t0"])
    return segs


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def best_hits(c0, c1, segs):
    """返回该 cue 时间窗内重叠>0 的 ASR 段，按段时间排序。"""
    hits = [s for s in segs if overlap(s["t0"], s["t1"], c0, c1) > 0]
    hits.sort(key=lambda s: s["t0"])
    return hits


def shifted_cues(cues, k):
    """文本 i 取 cue[i+k] 的时间；越界返回 None。"""
    out = []
    for i, c in enumerate(cues):
        j = i + k
        if 0 <= j < len(cues):
            out.append({"seg_id": c["seg_id"], "zh": c["zh"],
                        "t0": float(cues[j]["t0"]), "t1": float(cues[j]["t1"]),
                        "src_idx": j})
        else:
            out.append(None)
    return out


def eval_k(cues, segs, k):
    sc = shifted_cues(cues, k)
    n_valid = sum(1 for c in sc if c)
    paired = 0
    total_ov = 0.0
    details = []
    for c in sc:
        if not c:
            details.append(None)
            continue
        hits = best_hits(c["t0"], c["t1"], segs)
        if hits:
            paired += 1
            ov = sum(overlap(s["t0"], s["t1"], c["t0"], c["t1"]) for s in hits)
            total_ov += ov
            details.append({"seg_id": c["seg_id"], "zh": c["zh"],
                            "t0": round(c["t0"], 2), "t1": round(c["t1"], 2),
                            "ov": round(ov, 2),
                            "ja": " / ".join(s["ja"] for s in hits)})
        else:
            details.append({"seg_id": c["seg_id"], "zh": c["zh"], "ja": None})
    return {"k": k, "n_valid": n_valid, "paired": paired,
            "missing": n_valid - paired, "total_ov": round(total_ov, 2),
            "details": details}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "logs" / "shift_detect_report.json"))
    args = ap.parse_args()

    report = {}
    for scene in SCENES:
        human = json.loads((DATASET / f"{scene}.human_zh.json").read_text(encoding="utf-8"))
        cues = human["segments"]
        segs = load_asr(scene)
        evals = [eval_k(cues, segs, k) for k in (-2, -1, 0, 1, 2)]
        # 最优 k：配对数最大，其次总重叠
        best = max(evals, key=lambda e: (e["paired"], e["total_ov"]))
        entry = {
            "n_cues": len(cues), "n_asr": len(segs),
            "k_table": [{k: e[k] for k in ("k", "paired", "missing", "total_ov")}
                        for e in evals],
            "best_k": best["k"],
        }
        # 抽样：取前 N 个 k=0 时有 ja 的对话型 cue，对比 k=0 与 best_k 的配对
        picks = 0
        samples = []
        for d0 in evals[2]["details"]:  # k=0 位于 evals 索引 2
            if d0 is None or d0.get("ja") is None:
                continue
            i = next(idx for idx, dd in enumerate(evals[2]["details"]) if dd is d0)
            db = best["details"][i]
            samples.append({"k0": d0, "kbest": db})
            picks += 1
            if picks >= args.samples:
                break
        entry["samples_k0_vs_kbest"] = samples
        report[scene] = entry
        print(f"[{scene}] cues={len(cues)} asr={len(segs)} "
              f"k_table={[(e['k'], e['paired'], e['total_ov']) for e in evals]} "
              f"best_k={best['k']}")

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print("report ->", args.out)


if __name__ == "__main__":
    main()
