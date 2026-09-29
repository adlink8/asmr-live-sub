#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""local_zh2ja_pilot.py — 本地无审查模型中文→日文翻译试点。

目的：验证"人工中文 → 机器日文"回译路线的本地可行性。
  1. 从金级对抽样（真实日文原文 + 人工中文都齐，天然带标准答案）
  2. 喂人工中文给本地模型 → 产出机器日文
  3. 机器日文 vs 真实日文：表记归一化（片假名→平假名、去标点）后字符相似度
  4. 拒答检测：输出含拒绝措辞/空输出 = 拒答（无审查模型应≈0）

用法：
  python tools/local_zh2ja_pilot.py --model qwen3:4b-instruct --out logs/pilot_4b.jsonl
  python tools/local_zh2ja_pilot.py --model hf.co/DevQuasar/huihui-ai.Qwen3-8B-abliterated-GGUF:Q4_K_M --out logs/pilot_8b.jsonl
依赖：ollama 已起（localhost:11434）；模型已 pull。
"""
import argparse
import difflib
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

HUNT = Path(r"D:/Downloads/asmr-gold-hunt/pair_gold")
OLLAMA = "http://localhost:11434/api/chat"

PUNCT = re.compile(r"[\s、。！？：；「」『』（）(),.!?;:\"'…―ー～·，♪♡×○…\-\[\]]+")
REFUSAL_RE = re.compile(
    r"无法|不能协助|不能帮助|不能提|抱歉|对不起|不能输出|不能翻|无法提供|"
    r"cannot|can't|sorry|i'm unable|unable to|として応じられ|お答えでき|できません", re.I)
SOUND_RE = re.compile(r"^[\s～~—…・。．、！!？?\-—♪♫ぁぃぅぇぉっゃゅょァィゥェォッャュョ"
                      r"啊嗚呜嗯呣唔哩呐呢啦呀哇哦噢喔哎欸诶嘿呵哈呼哼えおあいうんのねよな"
                      r"はぁ-ぉんヴ]+$")
COUNT_RE = re.compile(r"^[\d〇零一二两三四五六七八九十百千万半]+[つ个個只枚度回歳才]*$")


def guess_kind(text):
    t = re.sub(r"\s+", "", text or "")
    if not t or SOUND_RE.fullmatch(t):
        return "sound"
    if COUNT_RE.fullmatch(t):
        return "countdown"
    return "dialogue"


def norm_ja(t: str) -> str:
    """片假名→平假名 + 去标点，消表记噪声（项目定规）。"""
    out = []
    for c in (t or "").strip():
        code = ord(c)
        if 0x30A1 <= code <= 0x30F6:
            c = chr(code - 0x60)
        out.append(c)
    return PUNCT.sub("", "".join(out))


def load_sample(n_dialogue=70, n_sound=15, seed=42):
    rng = random.Random(seed)
    dial, snd = [], []
    for line in open(HUNT / "pair_gold_reviewed.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d.get("verdict_final") != "gold" or not d.get("human_verdict"):
            continue  # 只抽人工判好+合议定案的金级
        ja, zh = (d.get("ja") or "").strip(), (d.get("human_simp") or d.get("human") or "").strip()
        if not ja or not zh:
            continue
        item = {"rj": d["rj"], "track": d.get("track", ""), "seg_id": d["seg_id"], "ja": ja, "zh": zh}
        k = guess_kind(zh)
        if k == "sound":
            snd.append(item)
        elif k == "dialogue":
            dial.append(item)
    rng.shuffle(dial); rng.shuffle(snd)
    sample = dial[:n_dialogue] + snd[:n_sound]
    rng.shuffle(sample)
    return sample


def translate(model, zh_text, timeout=180):
    """调用 ollama：中文→日文。/no_think 压掉 Qwen3 思考模式。"""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "你是专业的日文译者。把用户给出的中文翻译成自然的日文口语。只输出日文译文，不要解释、不要罗马字。/no_think"},
            {"role": "user", "content": zh_text},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 2048},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    dt = time.time() - t0
    content = (data.get("message") or {}).get("content", "").strip()
    # 若模型先思考后作答（<think> 标签），取思考块之外的部分
    m = re.search(r"</think>\s*(.+)", content, re.S)
    if m:
        content = m.group(1).strip()
    return content, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-dialogue", type=int, default=70)
    ap.add_argument("--n-sound", type=int, default=15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sample = load_sample(args.n_dialogue, args.n_sound, args.seed)
    print(f"抽样 {len(sample)} 条（对白 {args.n_dialogue}+拟声 {args.n_sound}，seed={args.seed}）→ {args.model}")

    out_f = open(args.out, "w", encoding="utf-8")
    sims, lats = [], []
    stats = Counter()
    for i, item in enumerate(sample, 1):
        try:
            out, dt = translate(args.model, item["zh"])
        except Exception as e:
            out, dt = f"__ERROR__: {e}", 0.0
        refused = bool(REFUSAL_RE.search(out)) or not out or out.startswith("__ERROR__")
        sim = difflib.SequenceMatcher(None, norm_ja(out), norm_ja(item["ja"])).ratio() if not refused else 0.0
        stats["refused" if refused else "ok"] += 1
        if not refused:
            sims.append(sim); lats.append(dt)
        rec = {**item, "model_ja": out, "sim": round(sim, 3), "latency": round(dt, 1),
               "refused": refused, "kind": guess_kind(item["zh"])}
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        mark = "REFUSED" if refused else f"sim={sim:.2f}"
        print(f"[{i}/{len(sample)}] {mark} {dt:.1f}s", flush=True)

    # 汇总
    n_ok = len(sims)
    sims_sorted = sorted(sims)
    med = sims_sorted[n_ok // 2] if n_ok else 0
    p25 = sims_sorted[n_ok // 4] if n_ok else 0
    print(f"\n=== {args.model} 试点结果 ===")
    print(f"有效翻译 {n_ok} | 拒答/空/错 {stats['refused']} ({stats['refused']*100//max(len(sample),1)}%)")
    print(f"与真实日文相似度: 均值 {sum(sims)/n_ok:.3f} | 中位 {med:.3f} | p25 {p25:.3f} (n={n_ok})")
    if lats:
        print(f"延迟: 均值 {sum(lats)/len(lats):.1f}s | 总耗时 {sum(lats)/60:.1f}min")
    dial_s = [r["sim"] for r in map(json.loads, open(args.out, encoding="utf-8")) if r["kind"] == "dialogue" and not r["refused"]]
    snd_s = [r["sim"] for r in map(json.loads, open(args.out, encoding="utf-8")) if r["kind"] == "sound" and not r["refused"]]
    if dial_s: print(f"对白相似度均值 {sum(dial_s)/len(dial_s):.3f} (n={len(dial_s)})")
    if snd_s: print(f"拟声相似度均值 {sum(snd_s)/len(snd_s):.3f} (n={len(snd_s)})")


if __name__ == "__main__":
    sys.exit(main())
