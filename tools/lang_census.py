#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lang_census.py — 全站字幕语言构成普查。

回答一个问题：每部作品的字幕是 只有中文 / 只有日文 / 双语？
范围= text_inventory.jsonl 里所有带字幕类文本的作品
     （vtt/srt/lrc/ass/ssa 扩展名，或 b_sniff 判定的 txt 皮带时间戳字幕）。
每部抽至多 --samples 个不同轨的字幕文件下载判语言（轨名去重后首/中/尾各取一，
防"前几轨全是一种语言"的集数错位误判）。

文件级：逐行 detect_lang 计数，两种语言各 >=5 行判双语文件。
作品级：任一文件双语，或同时有中文文件和日文文件 => bilingual；
        只有中文文件 => zh_only；只有日文文件 => ja_only；全失败 => unknown。

产出 lang_census.jsonl 每行一部 {id, verdict, files:[{title, kind, ja_lines, zh_lines}]}，
断点续扫。用法：
  python tools/lang_census.py --limit 20        # 冒烟
  python tools/lang_census.py                   # 全量（约 2.5h，6 并发）
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from asmrone_collect import api_get, detect_lang, parse_sub_text  # noqa: E402

TIMED = re.compile(r"\.(vtt|srt|lrc|ass|ssa)$", re.I)
MEDIA = re.compile(r"\.(mp3|wav|flac|m4a)$", re.I)
FORMAT = re.compile(r"\.(vtt|srt|lrc|ass|ssa|txt)$", re.I)


def track_key(title):
    """轨名去重键：去格式后缀/媒体后缀/空白，供'不同轨'抽样。"""
    s = FORMAT.sub("", title)
    s = MEDIA.sub("", s)
    return re.sub(r"\s+", "", s).lower()


def node_folder(url):
    """字幕所在文件夹（URL 倒数第二段）——同作品的日文/中文字幕几乎总在
    不同文件夹，按文件夹分散取样能最大化命中两种语言。"""
    try:
        segs = url.rstrip("/").split("/")
        import urllib.parse
        return urllib.parse.unquote(segs[-2]) if len(segs) >= 2 else ""
    except Exception:  # noqa: BLE001
        return ""


def pick_nodes(texts, n, allow_txt=False):
    """修正版采样（2026-09-27 二次返工）：v1 按轨名去重+抽3个，把同轨名
    另一种语言的字幕丢了（RJ1187282 十轨日文+十轨中文被判'只有日文'）。
    现在：按 (文件夹, 轨名) 去重，先每文件夹取一个（最多 n-1 个文件夹），
    剩余名额在节点里首/中/尾补——文件夹分散 + 位置分散双保险。"""
    has_timed = any(TIMED.search(t.get("title", "")) for t in texts)
    seen, picks = set(), []
    for t in texts:
        title = t.get("title", "")
        ok = TIMED.search(title) is not None or (allow_txt and not has_timed
                                                 and title.lower().endswith(".txt"))
        if not ok:
            continue
        k = (node_folder(t.get("url", "")), track_key(title), title)
        if k in seen:
            continue
        seen.add(k)
        picks.append(t)
    if len(picks) <= n:
        return picks
    by_folder = {}
    for t in picks:
        by_folder.setdefault(node_folder(t.get("url", "")), []).append(t)
    folders = sorted(by_folder)
    out = [fs[0] for fs in (by_folder[f] for f in folders)][: n - 1]
    rest = [t for t in picks if t not in out]
    for i in {0, len(rest) // 2, len(rest) - 1}:
        if rest[i] not in out:
            out.append(rest[i])
        if len(out) >= n:
            break
    return out[:n]


TS_LINE = re.compile(r"(\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{3}")
KANA_RATIO_OK = 0.4   # 行内假名占比 >=0.4 视为日文行
HAN_RATIO_OK = 0.1    # 行内假名占比 <=0.1 视为中文行（中间地带跳过，宁缺勿错）


def _line_kind(ln):
    from asmrone_collect import KANA_RE, HAN_RE
    kana, han = len(KANA_RE.findall(ln)), len(HAN_RE.findall(ln))
    tot = kana + han
    if tot < 2:
        return None
    r = kana / tot
    if r >= KANA_RATIO_OK:
        return "ja"
    if r <= HAN_RATIO_OK:
        return "zh"
    return None


def classify_file(raw):
    """-> (kind, ja_lines, zh_lines)。
    双语真判据：同一 cue 的多行里同时有日文行和中文行（两行字幕惯例），
    >=10 个这样的 cue 才判双语。单语文件用全文本假名占比（detect_lang 设计用途，
    对整文件稳健；逐行判短句会把日文汉字行误判成中文，已废弃）。"""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text)
    bi_cues = ja_lines = zh_lines = 0
    for b in blocks:
        lines = [l.strip() for l in b.strip().splitlines() if l.strip()]
        if not any(TS_LINE.search(l) or "-->" in l for l in lines[:3]):
            continue  # 非字幕块（readme 头、LRC 标签等）
        body = lines[1:] if "-->" in lines[0] or TS_LINE.search(lines[0]) else lines[2:]
        kinds = [_line_kind(l) for l in body if len(l) >= 2]
        if "ja" in kinds and "zh" in kinds:
            bi_cues += 1
        ja_lines += sum(1 for k in kinds if k == "ja")
        zh_lines += sum(1 for k in kinds if k == "zh")
    if bi_cues >= 10:
        return "bilingual", ja_lines, zh_lines
    mono = detect_lang(parse_sub_text(raw))
    return mono, ja_lines, zh_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv", default=r"D:/Downloads/asmr-collect-staging/text_inventory.jsonl")
    ap.add_argument("--sniff", default=r"D:/Downloads/asmr-collect-staging/b_sniff.jsonl")
    ap.add_argument("--out", default=r"D:/Downloads/asmr-collect-staging")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--req-gap", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # txt 皮带时间戳字幕的作品清单（这些的正文是 vtt，必须查 txt 节点）
    subtxt_ids = set()
    sniff_path = Path(args.sniff)
    if sniff_path.exists():
        for ln in open(sniff_path, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if r.get("verdict") == "sub_txt":
                subtxt_ids.add(r["id"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cens_path = out / "lang_census.jsonl"
    done = set()
    if cens_path.exists():
        for ln in open(cens_path, encoding="utf-8"):
            try:
                done.add(json.loads(ln)["id"])
            except Exception:  # noqa: BLE001
                continue
        print(f"[resume] 已普查 {len(done)} 部", flush=True)

    works = []
    seen_row = set()
    for ln in open(args.inv, encoding="utf-8"):
        try:
            r = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        if "error" in r or r["id"] in done or r["id"] in seen_row:
            continue
        seen_row.add(r["id"])
        texts = r.get("texts", [])
        has_sub = any(TIMED.search(t.get("title", "")) for t in texts) \
            or r["id"] in subtxt_ids
        if not has_sub:
            continue
        nodes = pick_nodes(texts, args.samples, allow_txt=(r["id"] in subtxt_ids))
        if nodes:
            works.append((r["id"], nodes))
    if args.limit:
        works = works[:args.limit]
    print(f"[list] 带字幕作品待普查 {len(works)} 部", flush=True)

    PRIO = {"bilingual": 3, "zh": 2, "ja": 2, "?": 0}

    def probe(job):
        wid, nodes = job
        files, ok_n = [], 0
        for n in nodes:
            try:
                raw = api_get(n["url"])
            except Exception as e:  # noqa: BLE001
                files.append({"title": n.get("title", "")[:40], "err": str(e)[:50]})
                continue
            ok_n += 1
            kind, ja, zh = classify_file(raw)
            files.append({"title": n.get("title", "")[:40], "kind": kind,
                          "ja": ja, "zh": zh})
            time.sleep(args.req_gap)
        kinds = [f["kind"] for f in files if "kind" in f]
        if not kinds:
            verdict = "unknown"
        elif "bilingual" in kinds:
            verdict = "bilingual"
        elif "zh" in kinds and "ja" in kinds:
            verdict = "bilingual"
        elif "zh" in kinds:
            verdict = "zh_only"
        else:
            verdict = "ja_only"
        return {"id": wid, "verdict": verdict if ok_n else "unknown",
                "files": files}

    t0 = time.time()
    n = 0
    with open(cens_path, "a", encoding="utf-8") as fj:
        with ThreadPoolExecutor(args.workers) as ex:
            for row in ex.map(probe, works):
                fj.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                if n % 200 == 0:
                    fj.flush()
                    rate = n / max(time.time() - t0, 1)
                    print(f"[{n}/{len(works)}] {rate:.1f}部/s "
                          f"ETA {(len(works)-n)/max(rate,0.1)/60:.0f}min", flush=True)
    print(f"\n[OK] 字幕语言普查 {n} 部 -> {cens_path}", flush=True)


if __name__ == "__main__":
    main()
