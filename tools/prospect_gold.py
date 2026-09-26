"""翻译社团金级探矿：抽样验证"汉化组"作品的双语字幕成色。

背景：registry_full.json 全池登记发现 10 家翻译系社团共 1155 部（官方中文版
标记 100%），但"官方中文版"≠"有现成双语字幕文件"。本脚本分层抽样，对每部
复用 classify_work 做内容级语言判定，再做 ja×zh 同轴配对（同 stem 或 asmr.one
嵌套名 startswith），产出真金级率。

    .venv\\Scripts\\python.exe tools/prospect_gold.py \
        --registry D:/Downloads/asmr-collect-staging/registry_full.json \
        --out D:/Downloads/asmr-collect-staging/prospect_gold.json
"""
import argparse
import json
import random
import time
from pathlib import Path

from asmrone_collect import (THROTTLE, TRANSLATION_CIRCLES, api_json,
                             classify_work)


def norm_stem(path_str):
    return Path(path_str).stem.lower()


def zh_ja_pairs(zh_paths, ja_paths):
    """ja×zh 同轴配对：同 stem 直配；否则 asmr.one 嵌套名（字幕文件名
    含音频全名）startswith 配。返回配对路径列表。"""
    pairs = []
    used_ja = set()
    for zp in zh_paths:
        zs = norm_stem(zp)
        hit = None
        for i, jp in enumerate(ja_paths):
            if i in used_ja:
                continue
            js = norm_stem(jp)
            if js == zs or js.startswith(zp) or zs.startswith(jp):
                hit = i
                break
        if hit is not None:
            used_ja.add(hit)
            pairs.append(zp)
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry", required=True)
    ap.add_argument("--per-circle", type=int, default=3,
                    help="每个社团抽样部数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reg = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    by_circle = {}
    for w in reg["works"]:
        name = (w.get("circle") or {}).get("name") or ""
        if name in TRANSLATION_CIRCLES:
            by_circle.setdefault(name, []).append(w)
    print("社团部数:", {k[:12]: len(v) for k, v in sorted(by_circle.items())})

    rng = random.Random(args.seed)
    sample = []
    for name, ws in sorted(by_circle.items()):
        rng.shuffle(ws)
        sample.extend(ws[:args.per_circle])
    print(f"抽样 {len(sample)} 部，开始逐部检测（每部约 10~30s）...\n")

    rows, n_pair_works = [], 0
    for i, w in enumerate(sample):
        wid, title = w["id"], w.get("title", "")
        circle = (w.get("circle") or {}).get("name", "")
        try:
            info = classify_work(wid, title)
        except Exception as e:  # noqa: BLE001
            print(f"[{i+1}/{len(sample)}] RJ{wid} 检测失败: {e}")
            time.sleep(THROTTLE)
            continue
        if info is None:
            row = {"id": wid, "circle": circle, "zh": 0, "ja": 0,
                   "pairs": 0, "coverage": None, "gold_ready": False}
        else:
            zh_files = info.get("zh_files") or []
            ja_files = info.get("ja_files") or []
            pairs = zh_ja_pairs(zh_files, ja_files)
            n_pair_works += bool(pairs)
            row = {"id": wid, "circle": circle, "title": title[:40],
                   "zh": len(zh_files), "ja": len(ja_files), "pairs": len(pairs),
                   "coverage": info.get("sub_coverage"),
                   "gold_ready": bool(info.get("gold_ready"))}
        rows.append(row)
        cov = row["coverage"]
        cov_s = "?" if cov is None else f"{cov:.0%}"
        print(f"[{i+1}/{len(sample)}] RJ{row['id']} zh×{row['zh']} "
              f"ja×{row['ja']} 同轴配对×{row['pairs']} "
              f"覆盖率={cov_s} {row['circle'][:10]}")
        time.sleep(THROTTLE)

    n = len(rows)
    n_zh = sum(1 for r in rows if r["zh"])
    n_ja = sum(1 for r in rows if r["ja"])
    summary = {
        "sampled": n, "with_zh_sub": n_zh, "with_ja_sub": n_ja,
        "with_gold_pairs": n_pair_works,
        "gold_rate": n_pair_works / n if n else 0,
    }
    Path(args.out).write_text(json.dumps(
        {"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n=== 探矿结论 ===")
    print(f"抽样 {n} 部 | 有中文字幕 {n_zh} | 有日文字幕 {n_ja} | "
          f"ja×zh 可同轴配对 {n_pair_works}（金级率 "
          f"{summary['gold_rate']:.0%}）")
    print(f"明细: {args.out}")


if __name__ == "__main__":
    main()
