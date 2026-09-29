#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ingest_gold_pairs.py — 金级双语对入库管线（猎金产物 → mt/*.json → Sakura×human 验钞门）。

以后猎到新金级作品自动走这条线：
  1. 扫 RJ*/subtitles/{zh,ja}/，同轨配对（stem 全等 → 归一化 → 嵌套 startswith）
  2. 两种形态自动识别：
     - 双语同轴：一个 cue 内日文行+中文行（官方中文版惯例）→ 直接拆行配对，offset=0
     - 双语分轴：ja/zh 各自字幕文件 → 偏移自校正（网格搜 offset 极大化重叠对）后
       1:1 时间重叠配对（重叠 >= min_overlap×zh cue 时长，口径同 collect_finetune.pair_gold）
  3. 产出 <hunt>/pair_gold/RJxxx/mt/<track>.json（mt 布局，兼容 clean_pairs.py --verdicts 二票）
  4. 验钞门：Sakura 本地翻译 ja，与 human 中文相似度分桶
     valid(>=0.62) / partial(0.42~0.62) / invalid(<0.42)；边界样本经人工抽检校准。
     产物 pair_gold.jsonl 全量留痕，不删任何对——入库分层而非丢弃。

用法：
  python tools/ingest_gold_pairs.py --hunt D:/Downloads/asmr-gold-hunt --report   # 只统计不落盘
  python tools/ingest_gold_pairs.py --hunt D:/Downloads/asmr-gold-hunt --rj 1308361 --rj 1187282
  python tools/ingest_gold_pairs.py --hunt D:/Downloads/asmr-gold-hunt --rj 1308361 --no-gate
"""
import argparse
import difflib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from opencc import OpenCC
_t2s = OpenCC("t2s").convert  # 繁→简：繁体人工字幕与简体 Sakura 输出逐字比对会被系统性压分

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from subtitles_to_gt import parse_lrc, parse_srt_vtt  # noqa: E402  复用同口径字幕解析

MIN_OVERLAP = 0.3
VALID_SIM, PARTIAL_SIM = 0.62, 0.42
KANA = re.compile(r"[ぁ-んァ-ヶ]")
PUNCT = re.compile(r"[\s、。！？：；「」『』（）(),.!?;:\"'…―ー～·，]")
SECUT = re.compile(r"[（(【\[]?se\s*cut[）)】\]]?$")
MEDIA_EXT = re.compile(r"\.(mp3|wav|flac|m4a)$", re.I)


def norm_key(stem: str) -> str:
    s = MEDIA_EXT.sub("", SECUT.sub("", stem.strip().lower()))
    return PUNCT.sub("", s)


def norm_zh(t: str) -> str:
    return PUNCT.sub("", _t2s((t or "").strip()))


def detect_lang(text: str) -> str:
    """含假名→ja；纯汉字/拉丁→zh。"""
    return "ja" if KANA.search(text) else "zh"


def track_token(stem: str):
    """轨号 token：tr01 / #7 / 06 等前导编号（最多两位数字，防 '06 4回目'→'064' 碰撞），
    供轨名被翻译后仍能配轨。"""
    m = re.search(r"(?:tr|#)?\d{1,2}", norm_key(stem), re.I)
    return m.group(0) if m else None


def pick_lang_files(d: Path):
    """语言目录内每个轨选一个字幕文件：lrc > vtt > srt > txt(内容是 WEBVTT)。"""
    prio = {".lrc": 0, ".vtt": 1, ".srt": 2, ".txt": 3}
    best = {}
    for f in sorted(d.iterdir()):
        if f.suffix.lower() not in prio or not f.is_file():
            continue
        k = f.stem
        if k not in best or prio[f.suffix.lower()] < prio[best[k].suffix.lower()]:
            best[k] = f
    return list(best.values())


def load_cues(p: Path):
    """字幕 -> [(t0, t1, text)]。.txt 实为 WEBVTT 的按 vtt 解析；纯文本无时间轴返回 []。
    注意 parse_srt_vtt 对多行 cue 逐行产出（同行双语=同时间戳两条），语言归类见 split_bilingual。"""
    try:
        raw = p.read_text(encoding="utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        try:  # 部分汉化组字幕是 GBK 系编码（RJ1079813 实锤），errors=replace 会静默产乱码
            raw = p.read_text(encoding="gb18030", errors="strict")
        except Exception:  # noqa: BLE001
            return []
    except Exception:  # noqa: BLE001
        return []
    ext = p.suffix.lower()
    if ext == ".lrc":
        pts = parse_lrc(raw)
        return [(float(t0), float(pts[i + 1][0] if i + 1 < len(pts) else t0 + 10.0), str(t))
                for i, (t0, t) in enumerate(pts)]
    if ext in (".vtt", ".srt") or raw.lstrip().upper().startswith("WEBVTT"):
        return sorted([(float(a), float(b), str(t)) for a, b, t in parse_srt_vtt(raw)])
    return []


def split_bilingual(cues):
    """双语同轴：同时间戳一组里含假名行(ja)与非假名行(zh)，或单 cue 内多行混排。
    parse_srt_vtt 把同行双语拆成同时间戳两条（官方中文版惯例）；组内配对，offset=0。"""
    groups = defaultdict(list)
    for t0, t1, text in cues:
        for ln in str(text).splitlines():
            if ln.strip():
                groups[(round(t0, 2), round(t1, 2))].append(ln.strip())
    out = []
    for (t0, t1), lines in sorted(groups.items()):
        ja_lines = [ln for ln in lines if KANA.search(ln)]
        zh_lines = [ln for ln in lines if not KANA.search(ln)]
        if ja_lines and zh_lines:
            out.append((t0, t1, " ".join(ja_lines), " ".join(zh_lines)))
    return out


def mono_ratio(cues):
    """按假名行占比定文件语言。"""
    n_ja = sum(1 for _, _, t in cues if KANA.search(t))
    return "ja" if n_ja >= max(1, len(cues) * 0.3) else "zh"


def pair_1to1(zh_cues, ja_cues, offset, min_overlap=MIN_OVERLAP):
    """zh 轴平移 offset 后与 ja 轴按重叠贪心 1:1 配对（重对只配一次，防重复 ja 目标）。"""
    cands = []
    for zi, (z0, z1, zt) in enumerate(zh_cues):
        zz0, zz1 = z0 + offset, z1 + offset
        for ji, (j0, j1, jt) in enumerate(ja_cues):
            ov = min(zz1, j1) - max(zz0, j0)
            if ov >= min_overlap * max(z1 - z0, 1e-6):
                cands.append((ov, zi, ji))
    cands.sort(reverse=True)
    used_z, used_j, pairs = set(), set(), []
    for ov, zi, ji in cands:
        if zi in used_z or ji in used_j:
            continue
        used_z.add(zi)
        used_j.add(ji)
        z0, z1, zt = zh_cues[zi]
        pairs.append({"t0": round(z0, 2), "t1": round(z1, 2),
                      "ja": ja_cues[ji][2], "human": zt})
    pairs.sort(key=lambda p: p["t0"])
    return pairs


def offset_search(zh_cues, ja_cues, max_off=20.0):
    """偏移自校正：ja_start-zh_start 差值 0.25s 桶聚类取众数候选，配对数多者胜；
    与 offset=0 平手时取 0（不无故偏移）。返回 (offset, n_pairs)。"""
    if not zh_cues or not ja_cues:
        return 0.0, 0
    buckets = defaultdict(int)
    for z0, z1, _ in zh_cues:
        for j0, j1, _ in ja_cues:
            d = j0 - z0
            if abs(d) <= max_off:
                buckets[round(d / 0.25) * 0.25] += 1
    best = max(buckets.items(), key=lambda kv: (kv[1], -abs(kv[0]))) if buckets else (0.0, 0)
    n_best = len(pair_1to1(zh_cues, ja_cues, best[0]))
    n_zero = len(pair_1to1(zh_cues, ja_cues, 0.0))
    if n_zero >= n_best:
        return 0.0, n_zero
    return best[0], n_best


def extract_work(rj_dir: Path):
    """单作品提取：返回 [(track, mode, offset, pairs)]。"""
    zh_dir, ja_dir = rj_dir / "subtitles" / "zh", rj_dir / "subtitles" / "ja"
    zh_files = pick_lang_files(zh_dir) if zh_dir.exists() else []
    ja_files = pick_lang_files(ja_dir) if ja_dir.exists() else []

    results, used_ja = [], set()

    # 形态A：双语同轴文件（任一语言目录里都可能出现）
    for f in zh_files + ja_files:
        cues = load_cues(f)
        bi = split_bilingual(cues)
        n_groups = len({(round(a, 2), round(b, 2)) for a, b, _ in cues})
        # 双语组占优才算同轴轨，防个别双语注释误判
        if len(bi) >= 10 and len(bi) >= 0.5 * max(n_groups, 1):
            results.append((f.stem, "bilingual", 0.0,
                            [{"t0": round(a, 2), "t1": round(b, 2), "ja": j, "human": z}
                             for a, b, j, z in bi]))
            used_ja.add(f)

    # 形态B：分轴文件，按语言各自归类后配轨
    zh_mono, ja_mono = [], []
    for f in zh_files + ja_files:
        if f in used_ja:
            continue
        cues = load_cues(f)
        if len(cues) < 5:
            continue
        (ja_mono if mono_ratio(cues) == "ja" else zh_mono).append((f, cues))

    ja_by_key = {}
    for f, cues in ja_mono:
        ja_by_key.setdefault(norm_key(f.stem), (f, cues))
    ja_by_token = defaultdict(list)
    for f, cues in ja_mono:
        tk = track_token(f.stem)
        if tk:
            ja_by_token[tk].append((f, cues))
    for f, cues in zh_mono:
        k = norm_key(f.stem)
        hit = ja_by_key.get(k)
        if hit is None:  # 嵌套名/后缀差异：前后缀互为兜底（防短误配，要求 >=6 字符）
            for jk, (jf, jc) in ja_by_key.items():
                if len(jk) >= 6 and len(k) >= 6 and \
                        (jk.startswith(k) or k.startswith(jk) or jk.endswith(k) or k.endswith(jk)):
                    hit = (jf, jc)
                    break
        if hit is None:  # 轨名两边各有翻译（tr01_見習い… vs tr01_见习女仆…）：轨号 token 唯一匹配
            tk = track_token(f.stem)
            cands = ja_by_token.get(tk, []) if tk else []
            if len(cands) == 1:
                hit = cands[0]
        if hit is None:
            continue
        jf, ja_cues = hit
        off, _ = offset_search(cues, ja_cues)
        pairs = pair_1to1(cues, ja_cues, off)
        if pairs:
            results.append((f.stem, "cross", off, pairs))

    # 跨文件去重：主轨+特典重发同轴字幕很常见，同 (t0,ja,zh) 全同者只留一次
    seen, dedup = set(), []
    for track, mode, off, pairs in results:
        uniq = []
        for p in pairs:
            k = (round(p["t0"], 1), p["ja"], p["human"])
            if k in seen:
                continue
            seen.add(k)
            uniq.append(p)
        if uniq:
            dedup.append((track, mode, off, uniq))
    return dedup


def run_gate(pairs, cache, mt):
    """Sakura×human 验钞门（模型只加载一次，由调用方传入）。
    cache 键=rj/track:seg_id，重跑跳过已判。"""
    from livesub.models import SakuraMT  # noqa: F401  仅保留接口提示
    for p in pairs:
        key = f"{p['rj']}/{p['track']}:{p['seg_id']}"
        if key in cache:
            p.update(cache[key])
            continue
        sakura_zh = mt.translate(p["ja"]).strip()
        sim = difflib.SequenceMatcher(None, norm_zh(sakura_zh), norm_zh(p["human"])).ratio()
        verdict = "valid" if sim >= VALID_SIM else "partial" if sim >= PARTIAL_SIM else "invalid"
        p.update({"sakura_zh": sakura_zh, "sim": round(sim, 3), "verdict": verdict})
        cache[key] = {"sakura_zh": sakura_zh, "sim": round(sim, 3), "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hunt", default=r"D:/Downloads/asmr-gold-hunt")
    ap.add_argument("--rj", action="append", type=int, help="指定 RJ 号（默认全部）")
    ap.add_argument("--report", action="store_true", help="只统计配对，不落盘不进门")
    ap.add_argument("--no-gate", action="store_true", help="只产出 mt/*.json，不跑 Sakura 验钞")
    ap.add_argument("--limit", type=int, default=None, help="每部最多对数（冒烟用）")
    args = ap.parse_args()

    hunt = Path(args.hunt)
    rj_dirs = ([hunt / f"RJ{r}" for r in args.rj] if args.rj
               else sorted(hunt.glob("RJ*")))
    rj_dirs = [d for d in rj_dirs if d.is_dir()]
    if not rj_dirs:
        raise SystemExit(f"[NG] {hunt} 下无 RJ* 目录")

    cache_path = hunt / "pair_gold" / "gate_cache.json"
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    out_root = hunt / "pair_gold"
    grand = defaultdict(lambda: defaultdict(int))
    mt = None
    if not args.report and not args.no_gate:
        from livesub.models import SakuraMT
        mt = SakuraMT(n_gpu_layers=int(os.environ.get("SAKURA_NGL", "99")))
    for d in rj_dirs:
        rj = d.name
        tracks = extract_work(d)
        all_pairs = []
        for track, mode, off, pairs in tracks:
            if args.limit:
                pairs = pairs[:args.limit]
            for i, p in enumerate(pairs):
                p.update({"seg_id": f"s{i:04d}", "track": track, "mode": mode,
                          "offset": round(off, 2), "rj": rj})
            all_pairs.extend(pairs)
            grand[rj]["pairs"] += len(pairs)
            grand[rj]["tracks"] += 1
        if args.report:
            print(f"{rj}: 轨 {grand[rj]['tracks']} 对 {grand[rj]['pairs']}")
            continue
        if not all_pairs:
            print(f"[skip] {rj}: 零配对")
            continue

        wdir = out_root / rj
        (wdir / "mt").mkdir(parents=True, exist_ok=True)
        by_track = defaultdict(list)
        for p in all_pairs:
            by_track[p["track"]].append(p)
        for track, ps in by_track.items():
            slug = re.sub(r'[\\/:*?"<>|]', "_", track)[:80]
            (wdir / "mt" / f"{slug}.json").write_text(
                json.dumps({"pairs": ps}, ensure_ascii=False, indent=1), encoding="utf-8")

        if args.no_gate:
            print(f"[ok] {rj}: mt 对 {len(all_pairs)}（未验钞）")
            continue
        stats = defaultdict(int)
        with open(out_root / "pair_gold.jsonl", "a", encoding="utf-8") as fj:
            for p in all_pairs:
                run_gate([p], cache, mt)
                stats[p["verdict"]] += 1
                fj.write(json.dumps(p, ensure_ascii=False) + "\n")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        n = len(all_pairs)
        print(f"[done] {rj}: {n} 对 valid={stats['valid']} "
              f"partial={stats['partial']} invalid={stats['invalid']}", flush=True)

    if not args.report:
        return
    print("\n=== 配对总表 ===")
    for rj, g in sorted(grand.items(), key=lambda kv: -kv[1]["pairs"]):
        print(f"{rj}: 轨 {g['tracks']:2d} 对 {g['pairs']:4d}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[elapsed] {time.time()-t0:.1f}s", file=sys.stderr)
