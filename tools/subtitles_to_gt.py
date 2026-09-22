"""字幕/台本文件 → 数据集 gt_ja.json（金级，时间戳权威）。

为什么需要：gold 级真值此前只有人工听写的台本 txt（覆盖不全、对齐靠模糊匹配，
26 段里 5 段匹配不上）。带时间戳的字幕（srt/ass/vtt）是更好的金级来源——
时间戳就是权威对齐，不猜。双语字幕（JA+ZH）还能顺带提供**人工译文基准**，
补上"绝对翻译质量不可测"的缺口。

输入格式与策略：
  srt/vtt  顺序 cue，按时间戳切段；双语常表现为同时段两条 cue 或一条两行
  ass/ssa  Dialogue 行，Style 常区分 JA/ZH（如 Default vs 中文）；无法区分时
           按行内字符集判定（含汉字为主=ZH，含假名=JA）
  txt      无时间戳，只能走 align_official_script.py 模糊对齐（退回旧路，
           本脚本不支持，会明确报错）

清洗规则（与 align_official_script.clean_script_line 同宗）：
  去角色名标记 【未夜】/[Mika]/(耳かき)、去舞台指示 （…）;
  丢弃纯非言语 cue（[音乐]/[音效]/♪  only）——真值里不留无法评的内容;
  空行 cue 丢弃。

输出：dataset/<name>.gt_ja.json，schema 与 align 产物一致
（seg_id/t0/t1/official_ja/excluded/exclude_reason），tier=gold，
gt_source=subtitles_timestamped。双语时另写 <name>.human_zh.json
（seg_id/t0/t1/zh），provenance 记录来源文件。
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _has_kana(s):
    return bool(re.search(r"[぀-ヿ]", s))


def _has_han(s):
    return bool(re.search(r"[一-鿿]", s))


def clean_line(line: str) -> str:
    line = re.sub(r"^[【\[\(（].*?[】\]\)）]\s*", "", line)
    line = re.sub(r"（.*?）|\(.*?\)", "", line)
    return line.strip()


def is_nondialogue(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    # 纯音乐/音效标记或纯符号
    if re.fullmatch(r"[\[\(【（].*?[\]\)】）]|[♪♫～~…\s]+", t):
        return True
    return False


def parse_ts(s: str):
    """12:34:56.789 / 01:02:03,456 / 00:00:01.000 -> 秒"""
    s = s.strip().replace(",", ".")
    parts = s.split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] if parts else None


def parse_srt_vtt(text: str):
    """-> [(t0, t1, line), ...]

    cue 体内多行逐行产出（同行双语 srt：JA 行与 ZH 行同时间戳，交给下游
    按语言配对），不拼接成一条。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text)
    out = []
    for b in blocks:
        lines = [l.strip() for l in b.strip().splitlines() if l.strip()]
        if not lines:
            continue
        m = None
        for i, l in enumerate(lines[:3]):
            if "-->" in l:
                m = (i, l)
                break
        if not m:
            continue
        idx, tl = m
        seg = tl.split("-->")
        t0, t1 = parse_ts(seg[0]), parse_ts(seg[1].split()[0] if seg[1].split() else "0")
        if t0 is None or t1 is None:
            continue
        for body_line in lines[idx + 1:]:
            for sub in body_line.split("\\N"):
                sub = sub.strip()
                if sub:
                    out.append((t0, t1, sub))
    return out


def parse_ass(text: str):
    out = []
    for line in text.replace("\r\n", "\n").splitlines():
        line = line.strip()
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        t0, t1 = parse_ts(parts[1]), parse_ts(parts[2])
        body = parts[9].replace("\\N", "\n").replace("\\n", "\n")
        for sub in body.splitlines():
            sub = re.sub(r"\{[^}]*\}", "", sub).strip()
            if t0 is not None and t1 is not None and sub:
                out.append((t0, t1, sub))
    return out


# 粉丝字幕噪声行（汉化组署名/招群/转载声明），不进真值也不进人工译文
CREDIT_RE = re.compile(r"汉化组|字幕组|翻译组|转载请注明|仅供学习|交流群|群\s*\d{4,}|"
                       r"Arctime|字幕软件|仅学习交流")


def parse_lrc(text: str):
    """-> [(t0, line)]；LRC 无结束时间，t1 由调用方顺延。多时间戳行逐戳产出。"""
    out = []
    for ln in text.replace(chr(13) + chr(10), chr(10)).splitlines():
        ln = ln.strip().lstrip("﻿")
        if not ln or ln.startswith("[ti:") or ln.startswith("[ar:")                 or ln.startswith("[al:") or ln.startswith("[by:")                 or ln.startswith("[ve:") or ln.startswith("[re:")                 or ln.startswith("[offset:"):
            continue
        m = re.match(r"((?:\[\d{1,2}:\d{1,2}(?:[.,]\d{1,3})?\])+)(.*)", ln)
        if not m:
            continue
        body = m.group(2).strip()
        if not body or CREDIT_RE.search(body):
            continue
        for ts in re.findall(r"\[(\d{1,2}:\d{1,2}(?:[.,]\d{1,3})?)\]", m.group(1)):
            t0 = parse_ts(ts)
            if t0 is not None:
                out.append((t0, body))
    out.sort(key=lambda x: x[0])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sub", required=True, help="字幕文件 (srt/ass/ssa/vtt)")
    ap.add_argument("--name", required=True, help="场景名，如 scene_new_work")
    ap.add_argument("--out-dir", default="dataset", help="输出目录")
    ap.add_argument("--max-gap", type=float, default=10.0,
                    help="相邻 cue 间隔超过此秒数则中间视为无语音（仅记录，不删段）")
    ap.add_argument("--human-zh", action="store_true",
                    help="字幕是人工中文译文：产出 <name>.human_zh.json（绝对锚点）而非 JA 真值")
    args = ap.parse_args()

    sub_p = Path(args.sub)
    raw = sub_p.read_text(encoding="utf-8", errors="replace")
    ext = sub_p.suffix.lower()
    if ext in (".srt", ".vtt"):
        cues = parse_srt_vtt(raw)
    elif ext in (".ass", ".ssa"):
        cues = parse_ass(raw)
    elif ext == ".lrc":
        # LRC 无结束时间：t1 顺延到下一条时间戳（最后一条 +3s 估值）
        lrc = parse_lrc(raw)
        cues = [(t0, lrc[i + 1][0] if i + 1 < len(lrc) else t0 + 3.0, body)
                for i, (t0, body) in enumerate(lrc)]
    else:
        sys.exit(f"不支持的格式 {ext}：无时间戳的 txt 请走 align_official_script.py")

    if args.human_zh:
        # 人工译文模式：字幕即人工中文译文，产出 human_zh.json（绝对质量锚点）
        segs = [{"idx": i, "seg_id": i, "t0": c[0], "t1": c[1],
                 "zh": clean_line(c[2])}
                for i, c in enumerate(cues, 1)
                if not is_nondialogue(clean_line(c[2]))]
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        zh_p = out_dir / f"{args.name}.human_zh.json"
        zh_p.write_text(json.dumps({
            "script_file": sub_p.name,
            "source": "asmr_one_human_subtitle",
            "total_segments": len(segs),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "segments": segs,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {zh_p.name}: {len(segs)} 段人工译文（asmr.one 字幕，"
              f"绝对质量锚点）")
        return

    groups, order = {}, []
    for t0, t1, t in cues:
        key = (round(t0, 2), round(t1, 2))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(t)
    segs, human = [], []
    n_zh_only = 0
    for i, key in enumerate(order, 1):
        t0, t1 = key
        texts = groups[key]
        ja = [t for t in texts if _has_kana(t)]
        zh = [t for t in texts if _has_han(t) and not _has_kana(t)]
        if ja and zh:
            ja_t, zh_t = " ".join(ja), " ".join(zh)
        elif ja:
            ja_t, zh_t = " ".join(ja), None
        else:
            # 纯中文 cue：交替式双语里 ZH 译文行的时间戳常与 JA 行错开，
            # 无法可靠配对——跳过并计数，绝不收进 JA 真值
            n_zh_only += 1
            continue
        ja_clean = clean_line(ja_t)
        if is_nondialogue(ja_clean):
            continue
        segs.append({
            "idx": i, "seg_id": i, "t0": t0, "t1": t1,
            "official_ja": ja_clean, "live_ja": "",
            "match_score": None, "excluded": False, "exclude_reason": "",
        })
        if zh_t:
            human.append({"idx": i, "seg_id": i, "t0": t0, "t1": t1,
                          "zh": clean_line(zh_t)})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = {
        "script_file": sub_p.name,
        "audio_file": None,
        "total_segments": len(segs),
        "full_official_ja": "".join(s["official_ja"] for s in segs),
        "gt_source": "subtitles_timestamped",
        "tier": "gold",
        "segments": segs,
    }
    gt_p = out_dir / f"{args.name}.gt_ja.json"
    gt_p.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {gt_p.name}: {len(segs)} 段金级真值（时间戳权威，无模糊匹配）"
          f"；跳过纯中文 cue {n_zh_only} 条"
          + ("（双语文件，建议确认配对是否符合预期）" if n_zh_only else ""))

    if human:
        zh = {
            "script_file": sub_p.name,
            "audio_file": None,
            "total_segments": len(human),
            "source": "human_subtitles_bilingual",
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "segments": human,
        }
        zh_p = out_dir / f"{args.name}.human_zh.json"
        zh_p.write_text(json.dumps(zh, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {zh_p.name}: {len(human)} 段人工译文基准"
              f"（绝对翻译质量第一次可测——evaluate 的 --mt-ref 可指向它）")


if __name__ == "__main__":
    main()
