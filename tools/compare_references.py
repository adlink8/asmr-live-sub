"""多参考译文对比：实时译文 vs 多个独立参考（Sakura 精译 / 云端模型 / 人工译文）。

evaluate_accuracy 一次只吃一个 --mt-ref；参考层变多后（gold gt 的 Sakura 精译、
step-3.7-flash 云端译、双语字幕人工译文），需要一张矩阵看全貌：
  实时 vs 各参考（端到端差距）
  各参考之间（参考层内部差距——人工译文是锚点，其他参考离锚点多远）
剔除逻辑与 evaluate_accuracy 一致（gt excluded 段双侧不参评），保证口径可比。

用法：
  python tools/compare_references.py --live <jsonl> --gt <gt.json> \
    --ref sakura=dataset/scene_random_talk.ref_zh.json \
    --ref stepflash=dataset/scene_random_talk.ref_zh_stepflash.json \
    [--ref human=dataset/scene_random_talk.human_zh.json] [--out <md>]
"""
import argparse
import json
import sys
from pathlib import Path
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_accuracy import normalize_zh  # noqa: E402


def load_live_zh(live_p: Path, gt_segments):
    events = [json.loads(l) for l in live_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    asr = [e for e in events if e.get("kind") == "asr" and e.get("ja")]
    mt = {e.get("seg_id"): e for e in events if e.get("kind") == "mt"}

    def best_overlap_gt(t0, t1):
        bo, bg = 0.0, None
        for g in gt_segments:
            ov = max(0.0, min(t1, g.get("t1", 0.0)) - max(t0, g.get("t0", 0.0)))
            if ov > bo:
                bo, bg = ov, g
        return bo, bg

    zh, dropped = [], 0
    for a in asr:
        t0, t1 = a.get("t_start") or 0.0, a.get("t_end") or 0.0
        ov, g = best_overlap_gt(t0, t1)
        if g is not None and ov > 0.3 and g.get("excluded"):
            dropped += 1
            continue
        zh.append((mt.get(a.get("seg_id")) or {}).get("zh") or "")
    return normalize_zh("".join(zh)), dropped


def load_ref_zh(ref_p: Path):
    d = json.loads(ref_p.read_text(encoding="utf-8"))
    segs = d.get("segments", [])
    return normalize_zh("".join(s.get("zh") or "" for s in segs if not s.get("excluded")))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", required=True)
    ap.add_argument("--gt", required=True, help="gt json（用于双侧剔除，口径同 evaluate）")
    ap.add_argument("--ref", action="append", required=True,
                    help="name=path，可多次；human_zh.json 亦可")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    gt_segments = gt.get("segments", [])
    live_zh, dropped = load_live_zh(Path(args.live), gt_segments)

    refs = {}
    for item in args.ref:
        name, _, path = item.partition("=")
        refs[name] = load_ref_zh(Path(path))

    lines = ["# 多参考译文对比", "",
             f"- live: `{args.live}`（剔除联动丢 {dropped} 段）",
             f"- gt: `{args.gt}`", "",
             "## 实时 vs 各参考（端到端差距）", "",
             "| 参考 | 相似度 |", "| :--- | :--- |"]
    for name, zh in refs.items():
        lines.append(f"| {name} | {round(fuzz.ratio(zh, live_zh), 2)}% |")

    lines += ["", "## 参考层内部（相互距离；人工译文是锚点）", "",
              "| A \\ B | " + " | ".join(refs) + " |", "| :--- |" + " :--- |" * len(refs)]
    names = list(refs)
    for a in names:
        row = [f"| {a} "]
        for b in names:
            row.append(f"| {round(fuzz.ratio(refs[a], refs[b]), 2)}% " if a != b else "| — ")
        lines.append("".join(row) + "|")

    md = "\n".join(lines)
    print(md)
    if args.out:
        Path(args.out).write_text(md + "\n", encoding="utf-8")
        print(f"\n[OK] 已写入 {args.out}")


if __name__ == "__main__":
    main()
