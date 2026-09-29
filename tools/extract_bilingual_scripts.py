#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""extract_bilingual_scripts.py — 双语台本文本级提取管线（日文台本×中文台本 成对发售作品）。

与 ingest_gold_pairs（字幕时间轴配对）互补：台本没有时间轴，改走
"编码回退读取 → 语言指纹分类 → 轨号/章节号配对 → 句级单调DP对齐 → Sakura×人工中文验钞"。
产出与金级管线同构：RJ*/mt/*.json（直通对）+ pair_bilingual.jsonl（全量留痕）。

用法：
  python tools/extract_bilingual_scripts.py                 # 提取+配对+对齐（不验钞）
  python tools/extract_bilingual_scripts.py --gate 40      # 每部抽40对跑Sakura验钞（可重入断点）
  python tools/extract_bilingual_scripts.py --report       # 汇总台账

纪律：R18 内容只出聚合统计，报告/日志不含显性原句。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from opencc import OpenCC  # noqa: E402

_t2s = OpenCC("t2s").convert  # 繁→简：与简体 Sakura 输出逐字比对前归一

WORKDIR = Path(r"D:/Downloads/asmr-script-align")
LEDGER = WORKDIR / "pair_bilingual.jsonl"
GATE_CACHE = WORKDIR / "gate_cache_bilingual_v2.json"  # v2=bge-m3对齐（v1字符DP的旧缓存已废弃）

KANA_RE = re.compile(r"[ぁ-んァ-ヶ]")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")
PUNCT = re.compile(r"[\s、。！？：；「」『』（）(),.!?;:\"'…―ー～·，♪♡×○\-\[\]]+")
SENT_SPLIT = re.compile(r"(?<=[。！？…」』])")
ANCHOR_RE = re.compile(r"[0-9]+|[a-zA-Z]+")
README_RE = re.compile(r"read.?me|読んで|説明|使い方", re.I)
USER_SUB_RE = re.compile(r"用户投稿")
SE_MARK_RE = re.compile(r"無音效|无效果音|無效果|SEoff|SE\s*off|音声なし", re.I)


def read_txt_any(p: Path) -> tuple[str, str]:
    """编码回退读取：utf-8-sig → utf-16(RJ419781实锤, 记事本'Unicode'格式) → cp932(Shift-JIS) → gb18030 → 替换。
    返回 (文本, 实际编码)。"""
    raw = p.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16"), "utf-16"
    for enc in ("utf-8-sig", "cp932", "gb18030"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace"), "replace"


def norm_zh(t: str) -> str:
    return PUNCT.sub("", _t2s((t or "").strip()))


def norm_ja(t: str) -> str:
    """片假名→平假名 + 去标点（与对齐管线同口径）。"""
    out = []
    for c in (t or "").strip():
        code = ord(c)
        if 0x30A1 <= code <= 0x30F6:
            c = chr(code - 0x60)
        out.append(c)
    return PUNCT.sub("", "".join(out))


def fingerprint(text: str) -> str:
    """语言指纹：假名占比>5% → 日文；汉字占比>10% 且几乎无假名 → 中文。"""
    n = max(len(text), 1)
    kana = len(KANA_RE.findall(text)) / n
    if kana > 0.05:
        return "ja"
    han = len(HAN_RE.findall(text)) / n
    return "zh" if han > 0.10 and kana < 0.02 else "unk"


def split_sentences(text: str) -> list[str]:
    """按行拆句（整行括号舞台指示剔出——中日语境下都无翻译价值，且两侧行文不对称）。"""
    sents = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        stripped = re.sub(r"[（(【\[].*?[）)】\]]", "", ln).strip()
        if not stripped or not (KANA_RE.search(stripped) or HAN_RE.search(stripped)):
            continue
        for piece in SENT_SPLIT.split(ln):
            piece = piece.strip()
            if len(piece) >= 2:
                sents.append(piece)
    return sents


def track_key(stem: str) -> str:
    """配对键：轨号/章节号 token（最多两位数字，防 '06 4回目'→'064' 碰撞，同金级管线教训）。"""
    s = unicodedata.normalize("NFKC", stem).strip().lower()
    s = PUNCT.sub("", s)
    m = re.search(r"(?:track|tr|#)?\d{1,2}", s)
    return m.group(0) if m else ""


def anchor_set(s: str) -> frozenset[str]:
    return frozenset(ANCHOR_RE.findall(s))


def pair_score(ja: str, zh: str) -> float:
    """句对匹配分：锚点重合（数字/拉丁串）为主 + 日中长度比先验（日文≈中文1.2~1.8倍）为辅。"""
    nj, nz = norm_ja(ja), norm_zh(zh)
    if not nj or not nz:
        return 0.0
    aj, az = anchor_set(nj), anchor_set(nz)
    if aj or az:
        inter = len(aj & az)
        s_anchor = 2 * inter / (len(aj) + len(az))
    else:
        s_anchor = None  # 无锚点=中性，不奖不罚
    r = len(nj) / max(len(nz), 1)
    s_len = pow(2.718281828, -((r - 1.4) / 0.6) ** 2)
    return round(0.6 * s_anchor + 0.4 * s_len, 3) if s_anchor is not None else round(0.5 * s_len, 3)


def dp_align(ja_sents: list[str], zh_sents: list[str],
             gap: float = -0.05, accept: float = 0.30) -> list[tuple[int, int, float]]:
    """单调 DP 对齐（Ndroplett-Wunsch 式）：允许跳句（两侧舞台指示删减不对称），
    只回收对角且得分≥accept 的句对。复杂度 O(n*m)。
    v1 字符启发式（锚点+长度比）——仅作 bge-m3 不可用时的兜底；
    实证：行级平行作品(句数比≈1)可用，结构不对称作品(比>2)直通率≈0（2026-09-29 验钞）。"""
    n, m = len(ja_sents), len(zh_sents)
    # 滚动数组求值 + 回溯矩阵
    SCORE = [[0.0] * (m + 1) for _ in range(n + 1)]
    FROM = [[0] * (m + 1) for _ in range(n + 1)]  # 1=对角 2=上 3=左
    for i in range(1, n + 1):
        si = ja_sents[i - 1]
        row, prev = SCORE[i], SCORE[i - 1]
        fro_i, fro_p = FROM[i], FROM[i - 1]
        for j in range(1, m + 1):
            s = pair_score(si, zh_sents[j - 1])
            best, frm = prev[j - 1] + (s if s >= accept else gap - 0.35), 1
            if prev[j] + gap > best:
                best, frm = prev[j] + gap, 2
            if row[j - 1] + gap > best:
                best, frm = row[j - 1] + gap, 3
            row[j], fro_i[j] = best, frm
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        f = FROM[i][j]
        if f == 1:
            s = pair_score(ja_sents[i - 1], zh_sents[j - 1])
            if s >= accept:
                pairs.append((i - 1, j - 1, s))
            i, j = i - 1, j - 1
        elif f == 2:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


EMB_MODEL = "bge-m3"
OLLAMA = "http://localhost:11434"
EMB_CACHE = WORKDIR / "emb_cache"


def embed_texts(texts: list[str], tag: str) -> "list[list[float]]":
    """ollama /api/embed 批量嵌入（bge-m3 多语言向量），按文件缓存到 .npy。"""
    import hashlib
    import urllib.request

    import numpy as np

    EMB_CACHE.mkdir(exist_ok=True)
    h = hashlib.sha1(("|".join(texts)).encode("utf-8")).hexdigest()[:16]
    cache_f = EMB_CACHE / f"{tag}_{h}.npy"
    if cache_f.exists():
        return np.load(cache_f)
    vecs = []
    for k in range(0, len(texts), 32):
        batch = texts[k:k + 32]
        req = urllib.request.Request(
            f"{OLLAMA}/api/embed",
            data=json.dumps({"model": EMB_MODEL, "input": batch}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        vecs.extend(data["embeddings"])
    arr = np.asarray(vecs, dtype=np.float32)
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
    np.save(cache_f, arr)
    return arr


def match_embed(ja_sents: list[str], zh_sents: list[str], tag: str,
                accept: float = 0.55) -> list[tuple[int, int, float]]:
    """语义向量对齐：bge-m3 余弦矩阵 → 分数降序贪心 + 单调一致性约束（1:1 配对）。
    替代 v1 字符启发式 DP 的主路径。"""
    import numpy as np

    J = np.asarray(embed_texts(ja_sents, tag + "_ja"))
    Z = np.asarray(embed_texts(zh_sents, tag + "_zh"))
    S = J @ Z.T
    cands = []
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            s = float(S[i, j])
            if s >= accept:
                cands.append((s, i, j))
    cands.sort(reverse=True)
    used_i, used_j, acc = set(), set(), []
    for s, i, j in cands:
        if i in used_i or j in used_j:
            continue
        if any((a < i and b > j) or (a > i and b < j) for a, b in acc):
            continue  # 单调违反
        used_i.add(i)
        used_j.add(j)
        acc.append((i, j))
    acc.sort()
    return [(i, j, round(s, 3)) for (s, i, j) in sorted(cands, key=lambda c: (c[1], c[2]))
            if (i, j) in set(acc)]


def detect_pairs(work: Path) -> list[dict]:
    """一部作品的 台本目录 → 日文×中文文件对（含来源/去重/降级标记）。"""
    sdir = work / "script"
    if not sdir.is_dir():
        return []
    entries = []
    for f in sorted(sdir.glob("*.txt")):
        text, enc = read_txt_any(f)
        lang = fingerprint(text)
        if lang == "unk":
            continue
        entries.append({"file": f, "stem": f.stem, "lang": lang, "enc": enc,
                        "n": len(split_sentences(text)),
                        "readme": bool(README_RE.search(f.stem)),
                        "user": bool(USER_SUB_RE.search(f.stem)),
                        "se": bool(SE_MARK_RE.search(f.stem))})
    ja = [e for e in entries if e["lang"] == "ja"]
    zh = [e for e in entries if e["lang"] == "zh"]
    if not ja or not zh:
        return []
    # SE 有无双版本：同语言内按配对键去重（保留无 SE 标记、文件名短的一个）
    def dedup(files):
        best = {}
        for e in files:
            k = track_key(e["stem"])
            cur = best.get(k)
            if cur is None or (e["se"], len(e["stem"])) < (cur["se"], len(cur["stem"])):
                best[k] = e
        return best
    jk, zk = dedup(ja), dedup(zh)
    pairs = []
    lone = len(jk) == 1 and len(zk) == 1  # 全目录仅一对文件时放宽 readme 守卫（RJ1571147 实锤）
    for k in sorted(set(jk) & set(zk)):
        a, b = jk[k], zk[k]
        if not lone and a["readme"] != b["readme"] and (a["readme"] or b["readme"]):
            continue  # 台本×说明书 不配
        pairs.append({"ja": a, "zh": b, "key": k,
                      "source": "user_sub" if (a["user"] or b["user"]) else "official"})
    return pairs


def extract_work(work: Path) -> list[dict]:
    rj = work.name
    out = []
    for pr in detect_pairs(work):
        ja_s, zh_s = split_sentences(read_txt_any(pr["ja"]["file"])[0]), split_sentences(read_txt_any(pr["zh"]["file"])[0])
        if not ja_s or not zh_s:
            continue
        tag = f"{rj}__{pr['ja']['stem'][:40]}"
        try:
            aligned = match_embed(ja_s, zh_s, tag)
            aligner = "bge-m3"
        except Exception as e:  # noqa: BLE001 — ollama 不可用时兜底字符启发式
            print(f"[warn] {rj} 嵌入失败({str(e)[:40]})，回退 DP", flush=True)
            aligned = dp_align(ja_s, zh_s)
            aligner = "dp"
        for i, j, s in aligned:
            out.append({
                "key": f"{rj}:{pr['ja']['stem']}:{i}",
                "rj": rj, "track": pr["ja"]["stem"], "seg_id": i,
                "ja": ja_s[i], "human": zh_s[j],
                "align_score": s, "aligner": aligner,
                "file_pair": [pr["ja"]["stem"], pr["zh"]["stem"]],
                "source": pr["source"],
                "enc": pr["ja"]["enc"] + "/" + pr["zh"]["enc"],
            })
    return out


def run_gate(pairs: list[dict], sample_n: int, gate_min: float = 0.0) -> None:
    """Sakura×human中文 验钞门。gate_min>0 = 全量验钞对齐分≥gate_min 的桶；
    否则按部分层抽样 sample_n 对。断点续跑（缓存按 pair key）。"""
    cache = json.loads(GATE_CACHE.read_text(encoding="utf-8")) if GATE_CACHE.exists() else {}
    if gate_min > 0:
        todo = [p for p in pairs if p["align_score"] >= gate_min and p["key"] not in cache]
    else:
        todo = [p for p in pairs if p["key"] not in cache]
        if sample_n and sample_n > 0:
            by_work = defaultdict(list)
            for p in todo:
                by_work[p["rj"]].append(p)
            todo = []
            for rj, ps in sorted(by_work.items()):
                ps.sort(key=lambda p: p["align_score"])
                step = max(len(ps) // sample_n, 1)
                todo.extend(ps[::step][:sample_n])
    if not todo:
        print(f"[gate] 无新增待验（缓存 {len(cache)} 条）", flush=True)
        return
    from livesub.models import SakuraMT
    mt = SakuraMT(n_gpu_layers=int(__import__("os").environ.get("SAKURA_NGL", "99")))
    done = 0
    try:
        for p in todo:
            sakura_zh = mt.translate(p["ja"]).strip()
            sim = SequenceMatcher(None, norm_zh(sakura_zh), norm_zh(p["human"])).ratio()
            verdict = "gold" if sim >= 0.62 else ("review" if sim >= 0.42 else "isolated")
            cache[p["key"]] = {"sim": round(sim, 3), "verdict": verdict}
            p.update({"sim": round(sim, 3), "verdict": verdict})
            done += 1
            if done % 20 == 0:
                GATE_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                print(f"[gate] {done}/{len(todo)}", flush=True)
    finally:
        GATE_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    for p in pairs:
        if p["key"] in cache:
            c = cache[p["key"]]
            p.update({"sim": c["sim"], "verdict": c["verdict"]})


def export_mt(pairs: list[dict]) -> int:
    """直通对 → RJ*/mt/*.json（dataset_finetune 兼容：无时间轴，mode=script_text）。"""
    by_work = defaultdict(lambda: defaultdict(list))
    n = 0
    for p in pairs:
        if p.get("verdict") == "gold":
            by_work[p["rj"]][p["track"]].append(p)
            n += 1
    for rj, tracks in by_work.items():
        mtdir = WORKDIR / rj / "mt"
        mtdir.mkdir(parents=True, exist_ok=True)
        for track, ps in tracks.items():
            slug = re.sub(r'[\\/:*?"<>|]', "_", track)[:80]
            rows = [{"ja": p["ja"], "human": p["human"], "seg_id": f"s{p['seg_id']:04d}",
                     "track": p["track"], "rj": p["rj"], "mode": "script_text",
                     "sim": p.get("sim"), "align_score": p.get("align_score"),
                     "source": p.get("source")} for p in ps]
            (mtdir / f"{slug}.json").write_text(
                json.dumps({"pairs": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=int, default=0, help="每部抽样 N 对跑 Sakura 验钞（0=跳过）")
    ap.add_argument("--gate-min", type=float, default=0.0, help="全量验钞对齐分≥此值的桶（优先于抽样）")
    ap.add_argument("--export", action="store_true", help="直通对导出 RJ*/mt/*.json")
    ap.add_argument("--report", action="store_true", help="只汇总已有台账")
    args = ap.parse_args()

    if args.report:
        rows = [json.loads(l) for l in LEDGER.read_text(encoding="utf-8")]
        report(rows)
        return

    works = sorted(WORKDIR.glob("RJ*"))
    all_pairs = []
    for w in works:
        ps = extract_work(w)
        all_pairs.extend(ps)
        ja_f = {p["file_pair"][0] for p in ps}
        print(f"[extract] {w.name}: 文件对 {len({tuple(p['file_pair']) for p in ps})} → 句对 {len(ps)}"
              if ps else f"[extract] {w.name}: 无双语台本对", flush=True)
    tmp = WORKDIR / "pair_bilingual.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    tmp.replace(LEDGER)
    print(f"[total] {len(all_pairs)} 句对 → {LEDGER.name}", flush=True)

    if args.gate or args.gate_min > 0:
        run_gate(all_pairs, args.gate, args.gate_min)
        with LEDGER.open("w", encoding="utf-8") as f:
            for p in all_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if args.export:
        cache = json.loads(GATE_CACHE.read_text(encoding="utf-8")) if GATE_CACHE.exists() else {}
        for p in all_pairs:
            if p["key"] in cache:
                c = cache[p["key"]]
                p.update({"sim": c["sim"], "verdict": c["verdict"]})
        n = export_mt(all_pairs)
        print(f"[export] 直通对 {n} → RJ*/mt/", flush=True)
    report(all_pairs)


def report(rows: list[dict]) -> None:
    if not rows:
        print("[report] 无台账")
        return
    by_work = defaultdict(list)
    for r in rows:
        by_work[r["rj"]].append(r)
    print("=== 双语台本提取报告（聚合统计）===")
    print(f"作品 {len(by_work)} 部 | 句对 {len(rows)}")
    gate = [r for r in rows if r.get("verdict")]
    if gate:
        g = defaultdict(int)
        for r in gate:
            g[r["verdict"]] += 1
        sims = sorted(r["sim"] for r in gate)
        n = len(sims)
        print(f"验钞抽样 {n} 对 | 相似度中位 {sims[n//2]:.2f} | 直通(≥0.62) {g['gold']} ({g['gold']*100//n}%)"
              f" | 复核 {g['review']} | 隔离 {g['isolated']}")
        hi = [r["sim"] for r in gate if r["align_score"] >= 0.5]
        lo = [r["sim"] for r in gate if r["align_score"] < 0.5]
        if hi and lo:
            print(f"对齐分相关性：高分对(≥0.5,n={len(hi)}) sim均值 {sum(hi)/len(hi):.2f}"
                  f" vs 低分对(n={len(lo)}) {sum(lo)/len(lo):.2f}")
    src = defaultdict(int)
    for r in rows:
        src[r["source"]] += 1
    print("来源构成:", dict(src))
    print("\n每部句对数:")
    for rj, ps in sorted(by_work.items(), key=lambda kv: -len(kv[1])):
        gv = [p for p in ps if p.get("verdict")]
        gd = sum(1 for p in gv if p["verdict"] == "gold")
        print(f"  {rj}: {len(ps)} 句对" + (f"（验金 {gd}/{len(gv)}）" if gv else ""))


if __name__ == "__main__":
    main()
