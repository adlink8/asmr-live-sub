"""真值 curation：对 teacher 级 gt 应用机械剔除规则。

背景：teacher 全篇解码（beam=5、无 VAD）在 ASMR 的非言语区/多人重叠区会产出
复读机退化或乱码段，拿它当真值会得出误导性坏数字。两条机械规则（与实时管线
LOWCONF_LOGPROB=-1.15 同源，实测干净段 lp∈[-0.92,-0.12]、退化段 lp∈[-3.3,-1.4]，
阈值落在空隙）：

  1. 含 U+FFFD 替换符 = CT2 明确吐出的解码乱码 → decode_garbage
  2. avg_logprob < -1.15 = 低置信（复读机退化/非言语区幻觉）→ low_confidence

干净段不动。金级 gt（官方台本对齐）不走本脚本——它的 excluded 由对齐器判。
幂等：重复运行结果一致。teacher 解码本身非确定，本脚本只 curation 不重生成，
数据集里钉死的是具体文件。
"""
import argparse
import json
import sys
import time
from pathlib import Path

LOWCONF_LOGPROB = -1.15


def curate(gt_path: Path):
    d = json.loads(gt_path.read_text(encoding="utf-8"))
    n_dec = n_low = 0
    for s in d.get("segments", []):
        ja = s.get("ja") or s.get("official_ja") or ""
        lp = s.get("avg_logprob")
        if "\ufffd" in ja:
            s["excluded"], s["exclude_reason"] = True, "decode_garbage"
            n_dec += 1
        elif lp is not None and lp < LOWCONF_LOGPROB:
            s["excluded"], s["exclude_reason"] = True, "low_confidence"
            n_low += 1
        elif not s.get("excluded"):
            s.setdefault("exclude_reason", "")
    kept = sum(1 for s in d.get("segments", []) if not s.get("excluded"))
    d["curation"] = {
        "rules": [f"U+FFFD -> decode_garbage",
                  f"avg_logprob < {LOWCONF_LOGPROB} -> low_confidence"],
        "excluded_decode_garbage": n_dec,
        "excluded_low_confidence": n_low,
        "segments_total": len(d.get("segments", [])),
        "segments_kept": kept,
        "curated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    gt_path.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"[OK] {gt_path.name}: 总 {d['curation']['segments_total']} 段，"
          f"剔除 乱码 {n_dec} + 低置信 {n_low}，保留 {kept}")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gt", nargs="+", help="teacher 级 gt json（可多个）")
    args = ap.parse_args()
    for p in args.gt:
        curate(Path(p))


if __name__ == "__main__":
    main()
