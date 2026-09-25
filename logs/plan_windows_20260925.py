# -*- coding: utf-8 -*-
"""扩建锚点场景 2026-09-25：窗口规划（只读分析，不写文件）。
复用 tools/subtitles_to_gt.py 的解析与清洗函数。
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(r"D:/ADLINK/asmr-live-sub")
sys.path.insert(0, str(ROOT))
from tools.subtitles_to_gt import (  # noqa: E402
    parse_lrc, parse_srt_vtt, clean_line, is_nondialogue,
)

import av  # noqa: E402

ZH_CORPUS = Path(r"D:/Downloads/asmr-zh-corpus")
SUB_CORPUS = Path(r"D:/Downloads/asmr-sub-corpus")
WIN = 180.0


def audio_duration(p: Path) -> float:
    c = av.open(str(p))
    try:
        d = c.duration
        if d and d > 0:
            return d / 1e6
        # fallback: count
        s = c.streams.audio[0]
        return float(s.duration or 0) / s.time_base / 1e6 if s.duration else 0.0
    finally:
        c.close()


def lrc_cues(path: Path):
    """-> [(t0, t1, body)] 人工 cue（LRC: t1 顺延到下一条）"""
    raw = path.read_text(encoding="utf-8", errors="replace")
    lrc = parse_lrc(raw)
    return [(t0, (lrc[i + 1][0] if i + 1 < len(lrc) else t0 + 3.0), body)
            for i, (t0, body) in enumerate(lrc)]


def timed_cues(path: Path):
    """srt/vtt -> [(t0, t1, body)]"""
    raw = path.read_text(encoding="utf-8", errors="replace")
    return parse_srt_vtt(raw)


def human_cues(path: Path):
    """按现有 human_zh 口径过滤后的 cue（clean_line + is_nondialogue）"""
    if path.suffix.lower() == ".lrc":
        cues = lrc_cues(path)
    else:
        cues = timed_cues(path)
    out = []
    for t0, t1, body in cues:
        zh = clean_line(body)
        if is_nondialogue(zh) or not zh:
            continue
        out.append((t0, t1, zh))
    return out


def densest_window(cues, lo, hi, dur, avoid, win=WIN):
    """在 [lo, hi] 区间内找 cue 最密的 win 秒窗（窗口须完整落在 [lo,hi]∩[0,dur]），
    与 avoid 中已有窗口不重叠；平手取最早。返回 (start, count) 或 None。"""
    lo = max(lo, 0.0)
    hi = min(hi, dur)
    if hi - lo < win:
        return None
    best = None
    starts = [x * 0.5 for x in range(int(lo * 2), int((hi - win) * 2) + 1)]
    if not starts:
        starts = [lo]
    for a in starts:
        b = a + win
        if any(a < pb and b > pa for pa, pb in avoid):
            continue
        n = sum(1 for (t0, _t1, _z) in cues if a <= t0 < b)
        if best is None or n > best[1]:
            best = (a, n)
    return best


def scan_tracks():
    """每部作品枚举 (audio, subtitle) 候选。"""
    works = {}

    def add(work, audio, sub):
        works.setdefault(work, []).append((audio, sub))

    # 现有 3 部：zh-corpus lrc
    for rid in ("RJ299717", "RJ416809", "RJ416816"):
        for lrc in (ZH_CORPUS / rid / "subtitles" / "zh").glob("*.lrc"):
            base = lrc.stem
            for aud in (ZH_CORPUS / rid / "audio").glob("*"):
                if aud.stem == base:
                    add(rid, aud, lrc)

    # 新 6 部
    # RJ01449384
    for aud in (SUB_CORPUS / "RJ01449384").rglob("*.wav"):
        v = aud.with_name(aud.name + ".vtt")
        if v.exists():
            add("RJ01449384", aud, v)
    # RJ01463510
    for aud in (SUB_CORPUS / "RJ01463510").glob("*.wav"):
        v = aud.with_name(aud.name + ".vtt")
        if v.exists():
            add("RJ01463510", aud, v)
    # RJ01467825 -> RJ324799
    for aud in (SUB_CORPUS / "RJ01467825" / "RJ324799").rglob("*.wav"):
        v = aud.with_name(aud.name + ".vtt")
        if v.exists():
            add("RJ324799", aud, v)
    # RJ01497366: WAV + srt
    srt_dir = SUB_CORPUS / "RJ01497366" / "（中国語台本・字幕）メイドのサラさんに癒してもらおう" / "中国語字幕"
    for aud in (SUB_CORPUS / "RJ01497366" / "WAV").glob("*.wav"):
        s = srt_dir / (aud.stem + ".srt")
        if s.exists():
            add("RJ01497366", aud, s)
    # RJ01527130
    for aud in (SUB_CORPUS / "RJ01527130" / "wav").glob("*.wav"):
        v = aud.with_name(aud.name + ".vtt")
        if v.exists():
            add("RJ01527130", aud, v)
    # RJ01528043 -> RJ01521586
    for aud in (SUB_CORPUS / "RJ01528043" / "RJ01521586").rglob("*.wav"):
        v = aud.with_name(aud.name + ".vtt")
        if v.exists():
            add("RJ01521586", aud, v)
    return works


KNOWN_W1 = {  # 现有 3 部的窗 1（human_zh note 记录）
    "RJ299717": ("05_お茶の時間", 412.0),
    "RJ416809": ("トレント&トリエステの夏の水着応援", 1093.0),
    "RJ416816": ("二人のエムデンの両耳安眠誘導", 109.0),
}


def main():
    works = scan_tracks()
    report = []
    for work, pairs in sorted(works.items()):
        tracks = []
        for aud, sub in pairs:
            try:
                dur = audio_duration(aud)
            except Exception as e:
                tracks.append({"work": work, "audio": aud.name, "error": str(e)})
                continue
            cues = human_cues(sub)
            tracks.append({"work": work, "audio": aud, "sub": sub, "dur": dur,
                           "n_cues": len(cues), "cues": cues})
        ok = [t for t in tracks if "dur" in t]
        ok.sort(key=lambda t: (-t["n_cues"], t["audio"].name))
        if not ok:
            report.append({"work": work, "error": "no usable track"})
            continue
        best = ok[0]
        report.append({"work": work, "selected": best["audio"].name,
                       "dur": round(best["dur"], 1), "n_cues": best["n_cues"],
                       "cues": best["cues"], "sub": best["sub"].name,
                       "alternatives": [(t["audio"].name, t["n_cues"]) for t in ok[1:5]]})

    # 打印选择结果 + 验证 w1 复现
    for r in report:
        if "error" in r:
            print(f"{r['work']}: ERROR {r['error']}")
            continue
        print(f"{r['work']}: selected {r['selected']} dur={r['dur']}s cues={r['n_cues']} sub={r['sub']}")
        if r.get("alternatives"):
            print(f"    others: {r['alternatives']}")
        cues = r["cues"]
        dur = r["dur"]
        if r["work"] in KNOWN_W1:
            name, w1 = KNOWN_W1[r["work"]]
            n = sum(1 for (t0, _t1, _z) in cues if w1 <= t0 < w1 + WIN)
            chk = densest_window(cues, 0, dur, dur, [])
            print(f"    w1 known {w1}-{w1+WIN} count={n}; recompute best={chk}")

    # 窗口规划
    print("\n=== windows ===")
    plan = {}
    for r in report:
        if "error" in r:
            continue
        work, cues, dur = r["work"], r["cues"], r["dur"]
        if work in KNOWN_W1:
            w1 = KNOWN_W1[work][1]
            avoid = [(w1, w1 + WIN)]
            thirds = [(dur / 3, 2 * dur / 3), (2 * dur / 3, dur)]
            labels = ["w2", "w3"]
            wins = {"w1": w1}
        else:
            chk = densest_window(cues, 0, dur, dur, [])
            w1 = chk[0]
            avoid = [(w1, w1 + WIN)]
            thirds = [(dur / 3, 2 * dur / 3), (2 * dur / 3, dur)]
            labels = ["w2", "w3"]
            wins = {"w1": w1}
        for (lo, hi), lab in zip(thirds, labels):
            got = densest_window(cues, lo, hi, dur, avoid)
            if got is None:
                # 回退：允许窗从 third_start 起延伸（不越过 [0,dur]），不与前窗重叠
                got = densest_window(cues, lo, min(hi, dur), dur, avoid) if hi - lo >= WIN else None
                if got is None:
                    # 最后回退：在 third 起点固定尝试（可越界到下一段，但不与前窗重叠）
                    for a in [lo, lo - 1, lo - 2]:
                        b = a + WIN
                        if a >= 0 and b <= dur and not any(a < pb and b > pa for pa, pb in avoid):
                            n = sum(1 for (t0, _t1, _z) in cues if a <= t0 < b)
                            got = (a, n)
                            break
                if got is None:
                    print(f"  {work} {lab}: 无法安排（记录少切一窗）")
                    continue
            wins[lab] = got[0]
            avoid.append((got[0], got[0] + WIN))
        plan[work] = {"audio": r["selected"], "dur": dur, "wins": wins,
                      "counts": {k: sum(1 for (t0, _t1, _z) in cues if v <= t0 < v + WIN)
                                 for k, v in wins.items()},
                      "n_cues_total": r["n_cues"]}
        print(f"  {work}: " + "  ".join(
            f"{k}={v:.1f}s(+{plan[work]['counts'][k]})" for k, v in wins.items()))

    out = {w: {k: (v if k != "cues" else None) for k, v in r.items()}
           for w, r in zip([r["work"] for r in report], report)}
    (ROOT / "logs" / "window_plan_20260925.json").write_text(
        json.dumps({w: {"audio": str(p["audio"]) if hasattr(p.get("audio"), "__str__") else None,
                        "sub": str(p["sub"]) if p.get("sub") else None,
                        "dur": p["dur"], "wins": p["wins"], "counts": p["counts"],
                        "n_cues_total": p["n_cues_total"]}
                    for w, p in plan.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("\nsaved logs/window_plan_20260925.json")


if __name__ == "__main__":
    main()
