#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""clean_pairs.py — 裁判池：清洗 collect_finetune 产出的 MT 训练对（对齐有效性判定）。

与语义判分不同，这里只回答一个问题：**human 是否是 ja 同一话语的中文表达**。
- valid   = 同一句话（允许意译）→ 进训练集
- partial = 部分重叠/相邻句 → 剔除（宁缺毋脏）
- invalid = 完全无关 → 剔除

双裁判合成：
  --api        step-5-preview（SEMANTIC_JUDGE_* 环境变量）判一遍
  --verdicts   会话内 GLM（cron 晨间代理）判好的清单再判一遍
  两票都 available 时取交集（都 valid 才保留）；只有一票时按该票。

用法：
  python tools/clean_pairs.py --work dataset_finetune/<dir> --api
  python tools/clean_pairs.py --work dataset_finetune/<dir> --verdicts glm.json
"""
import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RUBRIC = """你是字幕训练对清洗裁判。对每对判断人工中文字幕(human)是否是日文原文(ja)同一话语的中文表达。
- valid：human 传达了 ja 的核心意思（允许意译、语气润色）。
- partial：部分重叠、相邻句、或疑似时间错位。
- invalid：完全无关。
只输出 JSON 数组：[{"id":<原样>,"verdict":"valid|partial|invalid"}]
待判数据：
"""


def api_judge(items, base, key, model, batch=10):
    out = {}
    for i in range(0, len(items), batch):
        chunk = [{"id": it["id"], "ja": it["ja"], "human": it["human"]} for it in items[i:i + batch]]
        payload = json.dumps({"model": model, "temperature": 0,
                              "messages": [{"role": "user", "content": RUBRIC + json.dumps(chunk, ensure_ascii=False)}]}).encode("utf-8")
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=payload,
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    text = json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
                for v in json.loads(re.search(r"\[.*\]", text, re.S).group(0)):
                    out[v["id"]] = v["verdict"]
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="dataset_finetune/<作品目录>")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--api", action="store_true")
    g.add_argument("--verdicts", help="GLM 判好的 {id:verdict} JSON（键=文件名:seg_id）")
    args = ap.parse_args()
    work = Path(args.work)
    mt_files = sorted((work / "mt").glob("*.json"))
    if not mt_files:
        raise SystemExit("[NG] 无 mt/*.json 可清洗")

    items, index = [], {}
    for f in mt_files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for p in d["pairs"]:
            iid = f"{f.stem}:{p['seg_id']}"
            index[iid] = (f, p)
            items.append({"id": iid, "ja": p.get("ja", ""), "human": p.get("human", "")})
    if not items:
        raise SystemExit("[NG] mt 对为空")

    votes = []
    if args.api:
        base, key, model = (os.environ.get(k, "") for k in
                            ("SEMANTIC_JUDGE_BASE_URL", "SEMANTIC_JUDGE_API_KEY", "SEMANTIC_JUDGE_MODEL"))
        if not (base and key and model):
            raise SystemExit("[NG] 缺 SEMANTIC_JUDGE_* 环境变量")
        votes.append(api_judge(items, base, key, model))
    if args.verdicts:
        votes.append(json.loads(Path(args.verdicts).read_text(encoding="utf-8")))

    (work / "mt_clean").mkdir(exist_ok=True)
    kept_by_file, stats = {}, {"total": len(items), "kept": 0, "dropped": 0}
    for it in items:
        vs = [v[it["id"]] for v in votes if it["id"] in v]
        ok = bool(vs) and all(v == "valid" for v in vs)
        f, p = index[it["id"]]
        if ok:
            kept_by_file.setdefault(f, []).append(p)
            stats["kept"] += 1
        else:
            stats["dropped"] += 1
    for f, pairs in kept_by_file.items():
        (work / "mt_clean" / f.name).write_text(
            json.dumps({"pairs": pairs}, ensure_ascii=False, indent=1), encoding="utf-8")
    (work / "clean_report.json").write_text(
        json.dumps({"votes": len(votes), **stats}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {work.name}: 总 {stats['total']} 保留 {stats['kept']} 剔除 {stats['dropped']}（{len(votes)} 票取交集）")


if __name__ == "__main__":
    main()
