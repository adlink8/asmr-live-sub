#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""step5_pair_exam.py — 翻译对语义质检考试（step-5 盲判 vs 人工判定对答案）。

用途：
  1. 校准考试：对 pair_gold_reviewed.jsonl 里带 human_verdict 的复核桶逐条盲判
     （只给 ja+human，不给 Sakura 回译、不给人工判定，防锚定），与人工 1/0 对答案。
  2. 上岗后预筛：--rescueCSV 对救援清单判好/坏，人工只裁分歧。

口径沿用项目 meaning-only：correct=核心意思忠实（同义改写/语序调整算对）；
partial=部分传达/代词错/意思偏移但大意在；wrong=意思错误/反转/答非所问。
拟声/纯计每对另标 kind，单独统计不移出结果。

用法：
  python tools/step5_pair_exam.py --limit 10          # 试跑 10 条
  python tools/step5_pair_exam.py                     # 全量考卷（925）
  python tools/step5_pair_exam.py --rescueCSV <path>  # 救援清单预筛
密钥：C:/Users/li/.zcode/stepfun.key（不入仓库不入日志）；
      可用 SEMANTIC_JUDGE_BASE_URL / SEMANTIC_JUDGE_MODEL 覆盖端点与模型。
"""
import argparse
import csv
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HUNT = Path(r"D:/Downloads/asmr-gold-hunt/pair_gold")
KEY_FILE = Path(r"C:/Users/li/.zcode/stepfun.key")

COUNT_RE = re.compile(r"^[\d〇零一二两三四五六七八九十百千万半]+[つ个個只枚度回歳才]*$")
SOUND_RE = re.compile(r"^[\s～~—…・。．、！!？?\-—♪♫ぁぃぅぇぉっゃゅょァィゥェォッャュョ"
                      r"啊嗚呜嗯呣唔哩呐呢啦呀哇哦噢喔哎欸诶嘿呵哈呼哼えおあいうんのねよな"
                      r"はぁ-ぉんヴ]+$")


def guess_kind(text: str) -> str:
    t = re.sub(r"\s+", "", text or "")
    if not t or SOUND_RE.fullmatch(t):
        return "sound"
    if COUNT_RE.fullmatch(t):
        return "countdown"
    return "dialogue"


RUBRICS = {
    "lenient": [
        "你是日译中字幕的语义质检裁判。对每一对判断：中文是否忠实传达了日文的意思。",
        "输入字段：seg_id、ja（日文原文）、human（中文译文，待检）。",
        "判分口径（meaning-only）：",
        "- correct：译文传达了日文句的核心意思（不抠措辞/语气/译风；同义改写、语序调整算对）。",
        "- partial：只传达了部分意思，或有明显偏差（漏译半句、代词指代错、多/少一个词导致",
        "  意思偏移但大意还在、语气明显错但大意在）。",
        "- wrong：意思错误、方向反转、答非所问，或与日文完全无关。",
    ],
    "strict": [
        "你是日译中训练数据的高标准质检裁判。这些中文将作为翻译模型的训练目标，必须忠实。",
        "对每一对判断：中文是否完整、忠实地传达了日文的意思。",
        "输入字段：seg_id、ja（日文原文）、human（中文译文，待检）。",
        "判分口径（faithfulness-first）：",
        "- correct：意思完整等价——所有实义内容（动作/对象/程度/否定）都译出；允许同义改写、语序调整、语气词取舍。",
        "- partial：大意在但有实质偏移——漏译个别实义词、代词指代与日文不符、多/少词导致意思偏移、引语/呼喊结构被改写。",
        "- wrong：意思错误、方向反转、漏译半句以上、答非所问，或与日文无关。",
        "注意：日文省略的主语按上下文合理补出不算错；但日文明确说了 A 而中文只译出 B，算 partial 或 wrong。",
    ],
}


def build_rubric(chunk, mode="lenient"):
    lines = RUBRICS[mode] + [
        "每对另标 kind：dialogue=对白/叙述；sound=拟声/纯语气；countdown=纯计数。",
        "sound/countdown 也要判 verdict，但统计时单独归类。",
        "note：一两句话说明理由，正确且无疑的可留空。",
        "只输出 JSON 数组：[{\"seg_id\":..,\"verdict\":\"correct|partial|wrong\",\"kind\":\"dialogue|sound|countdown\",\"note\":\"..\"}]",
        "待判数据：",
        json.dumps(chunk, ensure_ascii=False),
    ]
    return "\n".join(lines)


class CensoredError(RuntimeError):
    """内容合规门拒绝（HTTP 451 censorship_blocked）：确定性失败，重试无意义。"""


class AuthError(RuntimeError):
    """认证失败（401/403）：key 失效/被封，继续跑必然全挂——立即终止任务。"""


AUTH_CODES = (401, 403)


def call_api(chunk, base_url, api_key, model, timeout=180, rubric="lenient", protocol="chat"):
    rubric_text = build_rubric(chunk, rubric)
    if protocol == "responses":
        # xAI Responses 协议（部分中转站的 grok-4.x 仅支持此协议）
        payload = json.dumps({"model": model, "input": rubric_text, "temperature": 0}).encode("utf-8")
        path = "/responses"
    else:
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": rubric_text}],
            "temperature": 0,
        }).encode("utf-8")
        path = "/chat/completions"
    req = urllib.request.Request(
        base_url.rstrip("/") + path, data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if protocol == "responses":
                texts = [c.get("text", "") for it in data.get("output", []) if it.get("type") == "message"
                         for c in it.get("content", []) if c.get("type") == "output_text"]
                text = "\n".join(texts)
            else:
                text = data["choices"][0]["message"]["content"]
            m = re.search(r"\[.*\]", text, re.S)
            if not m:
                # 有应答但无 JSON 数组：该批大概率被软拒答（纯文字拒绝）→ 按内容门二分
                raise CensoredError("no-json-soft-refusal") from None
            return json.loads(m.group(0))
        except urllib.error.HTTPError as e:
            if e.code == 451:
                raise CensoredError("censorship_blocked") from e
            if e.code in AUTH_CODES:
                raise AuthError(f"HTTP {e.code}: key 认证失败，任务终止") from e
            last_err = e
            time.sleep(8 * (attempt + 1))  # 429 限流：长退避
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, IndexError) as e:
            last_err = e
            time.sleep(8 * (attempt + 1))
    raise RuntimeError(f"API 3 次重试均失败: {last_err}")


def load_exam_pairs():
    pairs, meta = [], {}
    for line in open(HUNT / "pair_gold_reviewed.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d.get("human_verdict") is None:
            continue
        # seg_id 每条音轨各自从 s0000 编号，跨轨同名——键必须带 rj+track
        key = f"{d['rj']}:{d.get('track','')}:{d['seg_id']}"
        seg = {"seg_id": key, "ja": d["ja"], "human": d["human_simp"]}
        pairs.append(seg)
        meta[key] = d
    return pairs, meta


def load_rescue_pairs(path):
    pairs, meta = [], {}
    for row in csv.DictReader(open(path, encoding="utf-8-sig")):
        seg = {"seg_id": f"{row['RJ']}:{row['track']}:{row['起点秒']}",
               "ja": row["日文(原)"], "human": row["中文(归一·简)"]}
        pairs.append(seg)
        meta[seg["seg_id"]] = row
    return pairs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescueCSV", default="", help="救援清单 CSV（默认考模式=考卷 925）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（试跑）")
    ap.add_argument("--workers", type=int, default=4, help="并发批改线程数")
    ap.add_argument("--rubric", choices=("lenient", "strict"), default="lenient",
                    help="判分口径：lenient=意思传达即可；strict=训练数据忠实度标准")
    ap.add_argument("--protocol", choices=("chat", "responses"), default="chat",
                    help="API 协议：chat=OpenAI chat completions；responses=xAI Responses（部分 grok-4.x 中转仅支持此协议）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.rescueCSV:
        pairs, meta = load_rescue_pairs(args.rescueCSV)
        out_path = Path(args.out) if args.out else HUNT / f"step5_rescue_verdicts_{args.rubric}.jsonl"
    else:
        pairs, meta = load_exam_pairs()
        out_path = Path(args.out) if args.out else HUNT / f"step5_exam_925_{args.rubric}.jsonl"
    if args.limit:
        pairs = pairs[:args.limit]

    api_key = os.environ.get("SEMANTIC_JUDGE_API_KEY") or KEY_FILE.read_text(encoding="utf-8").strip()
    base_url = os.environ.get("SEMANTIC_JUDGE_BASE_URL", "https://api.stepfun.com/step_plan/v1")
    model = os.environ.get("SEMANTIC_JUDGE_MODEL", "step-5-preview")

    done = {}
    if out_path.exists():  # 断点续跑
        for line in open(out_path, encoding="utf-8"):
            d = json.loads(line)
            done[d["seg_id"]] = d
        if Path(str(out_path) + ".blocked").exists():
            for sid in open(str(out_path) + ".blocked", encoding="utf-8"):
                done[sid.strip()] = {"seg_id": sid.strip(), "verdict": "censored"}
        print(f"[resume] 已有 {len(done)} 条（含 blocked），跳过")

    batch = 10
    lock = threading.Lock()
    progress = {"done": len(done)}
    chunks = [[p for p in pairs[i:i + batch] if p["seg_id"] not in done]
              for i in range(0, len(pairs), batch)]
    chunks = [c for c in chunks if c]

    def work(chunk):
        # 451=内容合规门：确定性拒绝 → 二分下探，良性条目放行，单条被拒记 blocked
        def run(c):
            try:
                return call_api(c, base_url, api_key, model, rubric=args.rubric, protocol=args.protocol), []
            except CensoredError:
                if len(c) == 1:
                    return [], [c[0]["seg_id"]]
                mid = len(c) // 2
                v1, b1 = run(c[:mid])
                v2, b2 = run(c[mid:])
                return v1 + v2, b1 + b2
        vs, blocked_ids = run(chunk)
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                for v in vs:
                    v["seg_id"] = str(v.get("seg_id", ""))
                    f.write(json.dumps(v, ensure_ascii=False) + "\n")
            if blocked_ids:
                with open(str(out_path) + ".blocked", "a", encoding="utf-8") as bf:
                    for sid in blocked_ids:
                        bf.write(sid + "\n")
            progress["done"] += len(chunk) - len(blocked_ids)
        if vs:
            print(f"[{progress['done']}/{len(pairs)}] 本批 {len(vs)} 条", flush=True)
        if blocked_ids:
            print(f"[censored] {len(blocked_ids)} 条触发内容门已记 blocked", flush=True)

    fails = 0
    auth_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, c): c for c in chunks}
        for fut in as_completed(futs):
            try:
                fut.result()
            except AuthError as e:
                auth_fail += 1
                print(f"[AUTH-FAIL] {e}——已连续认证失败，剩余批次取消", flush=True)
                for f2 in futs:
                    f2.cancel()
            except Exception as e:
                fails += 1
                print(f"[fail] 一批 {len(futs[fut])} 条失败: {e}", flush=True)
    if auth_fail:
        print(f"[HALT] 认证失败终止：已判 {progress['done']}/{len(pairs)}，重跑前先检查 key/中转站状态", flush=True)
        return 2
    if fails:
        print(f"[warn] {fails} 批 3 次重试后仍失败——重跑本命令自动续批", flush=True)

    # ---- 汇总 ----
    results = {}
    for line in open(out_path, encoding="utf-8"):
        d = json.loads(line)
        results[d["seg_id"]] = d
    if not args.rescueCSV:
        agree_len = agree_str = n = 0
        by_kind = Counter()
        for p in pairs:
            r = results.get(p["seg_id"])
            if not r:
                continue
            hv = meta[p["seg_id"]]["human_verdict"]
            jv = r.get("verdict")
            kind = r.get("kind", "dialogue")
            by_kind[(kind, jv)] += 1
            n += 1
            if jv in ("correct", "partial") and hv == 1: agree_len += 1
            if jv == "wrong" and hv == 0: agree_len += 1
            if jv == "correct" and hv == 1: agree_str += 1
            if jv in ("partial", "wrong") and hv == 0: agree_str += 1
        print(f"\n=== 考试结果（人工 {n} 条）===")
        print(f"宽松口径（correct+partial=好）一致率: {agree_len}/{n} = {agree_len*100/n:.1f}%")
        print(f"严格口径（correct=好）一致率:       {agree_str}/{n} = {agree_str*100/n:.1f}%")
        print("裁判判定分布:", dict(Counter(r.get("verdict") for r in results.values())))
        print("按 kind×verdict:", dict(by_kind))
    else:
        good = [p for p in pairs if results.get(p["seg_id"], {}).get("verdict") in ("correct",)]
        bad = [p for p in pairs if results.get(p["seg_id"], {}).get("verdict") == "wrong"]
        mid = [p for p in pairs if results.get(p["seg_id"], {}).get("verdict") == "partial"]
        print(f"\n=== 救援预筛（{len(pairs)} 条）===")
        print(f"step5 判好(correct): {len(good)} | 拿不准(partial): {len(mid)} | 判坏(wrong): {len(bad)}")
        print(f"→ 建议你只细读 partial 与 wrong 中的可疑项，{len(good)} 条 correct 可快速放行")


if __name__ == "__main__":
    sys.exit(main())
