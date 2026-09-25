# -*- coding: utf-8 -*-
"""锚点场景扩建 2026-09-25：选轨 → 切窗（48kHz mono s16le 180s）→ human_zh.json。
切窗规则见 docs/test_system_design.md §2；解析/清洗复用 tools/subtitles_to_gt.py。
"""
import json
import re
import sys
import time
import wave
from pathlib import Path

ROOT = Path(r"D:/ADLINK/asmr-live-sub")
sys.path.insert(0, str(ROOT))
from tools.subtitles_to_gt import (  # noqa: E402
    parse_lrc, parse_srt_vtt, clean_line, is_nondialogue,
)

import av  # noqa: E402
import numpy as np  # noqa: E402

ZH_CORPUS = Path(r"D:/Downloads/asmr-zh-corpus")
SUB_CORPUS = Path(r"D:/Downloads/asmr-sub-corpus")
STD180 = ZH_CORPUS / "std180s"
WIN = 180.0
RATE = 48000

# workid(场景用) -> (曲库目录, 轨道枚举器, 轨道号提取)
def tracks_zh(rid):
    out = []
    for lrc in (ZH_CORPUS / rid / "subtitles" / "zh").glob("*.lrc"):
        for aud in (ZH_CORPUS / rid / "audio").glob("*"):
            if aud.stem == lrc.stem:
                out.append((aud, lrc))
    return out


def vtt_next_to(pattern):
    out = []
    for aud in pattern:
        v = aud.with_name(aud.name + ".vtt")
        if v.exists():
            out.append((aud, v))
    return out


def track_no(work, stem):
    if work == "RJ01449384":
        return f"t{int(stem[:2]):02d}"
    if work == "RJ01463510":
        return f"t{int(re.search(r'Tr(\d+)', stem).group(1)):02d}"
    if work == "RJ324799":
        return f"t{int(re.search(r'rn02_(\d+)', stem).group(1)):02d}"
    if work == "RJ01497366":
        return f"t{int(stem[:2]):02d}"
    if work == "RJ01527130":
        return f"t{int(re.search(r'n18_(\d+)', stem).group(1)):02d}"
    if work == "RJ01521586":
        m = re.match(r"(ex)?0*(\d+)", stem)
        return (f"tex{int(m.group(2)):02d}" if m.group(1)
                else f"t{int(m.group(2)):02d}")
    raise ValueError(work)


NEW_WORKS = {
    "RJ01449384": lambda: vtt_next_to((SUB_CORPUS / "RJ01449384").rglob("*.wav")),
    "RJ01463510": lambda: vtt_next_to((SUB_CORPUS / "RJ01463510").glob("*.wav")),
    "RJ324799": lambda: vtt_next_to((SUB_CORPUS / "RJ01467825" / "RJ324799").rglob("*.wav")),
    "RJ01497366": lambda: [
        (a, Path(r"D:/Downloads/asmr-sub-corpus/RJ01497366/"
                 "（中国語台本・字幕）メイドのサラさんに癒してもらおう/中国語字幕") / (a.stem + ".srt"))
        for a in sorted((SUB_CORPUS / "RJ01497366" / "WAV").glob("*.wav"))
        if (Path(r"D:/Downloads/asmr-sub-corpus/RJ01497366/"
                 "（中国語台本・字幕）メイドのサラさんに癒してもらおう/中国語字幕") / (a.stem + ".srt")).exists()
    ],
    "RJ01527130": lambda: vtt_next_to((SUB_CORPUS / "RJ01527130" / "wav").glob("*.wav")),
    "RJ01521586": lambda: vtt_next_to((SUB_CORPUS / "RJ01528043" / "RJ01521586").rglob("*.wav")),
}

EXISTING = {  # workid -> (轨道字幕名, w1 起点)  现有窗1不动
    "RJ299717": ("05_お茶の時間", 412.0),
    "RJ416809": ("トレント&トリエステの夏の水着応援", 1093.0),
    "RJ416816": ("二人のエムデンの両耳安眠誘導", 109.0),
}


def human_cues(sub: Path):
    if sub.suffix.lower() == ".lrc":
        lrc = parse_lrc(sub.read_text(encoding="utf-8", errors="replace"))
        raw = [(t0, (lrc[i + 1][0] if i + 1 < len(lrc) else t0 + 3.0), b)
               for i, (t0, b) in enumerate(lrc)]
    else:
        raw = parse_srt_vtt(sub.read_text(encoding="utf-8", errors="replace"))
    out = []
    for t0, t1, body in raw:
        zh = clean_line(body)
        if not zh or is_nondialogue(zh):
            continue
        out.append((t0, t1, zh))
    return out


def dur_of(p: Path) -> float:
    c = av.open(str(p))
    try:
        return (c.duration or 0) / 1e6
    finally:
        c.close()


def densest(cues, lo, hi, dur, avoid):
    lo, hi = max(lo, 0.0), min(hi, dur)
    if hi - lo < WIN:
        return None
    best = None
    n_start = int(np.ceil(lo * 2) / 2)
    for a in [n_start + i * 0.5 for i in range(int((hi - WIN - n_start) * 2) + 1)]:
        b = a + WIN
        if any(a < pb and b > pa for pa, pb in avoid):
            continue
        n = sum(1 for (t0, _t1, _z) in cues if a <= t0 < b)
        if best is None or n > best[1]:
            best = (a, n)
    return best


def cut_audio(src: Path, start: float, out: Path):
    c = av.open(str(src))
    stream = c.streams.audio[0]
    rs = av.AudioResampler(format="s16", layout="mono", rate=RATE)
    chunks = []
    for frame in c.decode(stream):
        for rf in rs.resample(frame):
            chunks.append(np.frombuffer(bytes(rf.planes[0]), dtype=np.int16))
    c.close()
    pcm = np.concatenate(chunks)
    n0, n1 = int(round(start * RATE)), int(round((start + WIN) * RATE))
    seg = pcm[n0:n1]
    if len(seg) < int(WIN * RATE):
        seg = np.pad(seg, (0, int(WIN * RATE) - len(seg)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(seg.tobytes())
    return float(len(seg)) / RATE


def main():
    plan = {}
    # --- 现有 3 部：w1 固定，补 w2/w3 ---
    for work, (subname, w1) in EXISTING.items():
        pairs = tracks_zh(work)
        aud, sub = next((a, s) for a, s in pairs if s.stem == subname)
        tname = "t02" if work == "RJ299717" else "t01"
        dur = dur_of(aud)
        cues = human_cues(sub)
        wins = {"w1": w1}
        avoid = [(w1, w1 + WIN)]
        thirds = [("w2", dur / 3, 2 * dur / 3), ("w3", 2 * dur / 3, dur)]
        for lab, lo, hi in thirds:
            got = densest(cues, lo, hi, dur, avoid)
            if got:
                wins[lab] = got[0]
                avoid.append((got[0], got[0] + WIN))
            else:
                wins[lab] = None
        plan[work] = {"t": tname, "audio": aud, "sub": sub, "dur": dur,
                      "cues": cues, "wins": wins, "new_only": ["w2", "w3"]}

    # --- 新 6 部：选轨（cue 最多；同数优先非无音效/靠前）→ w1/w2/w3 ---
    for work, enum in NEW_WORKS.items():
        cands = []
        for aud, sub in enum():
            try:
                d = dur_of(aud)
            except Exception as e:
                print(f"[skip] {aud.name}: {e}")
                continue
            cues = human_cues(sub)
            cands.append({"audio": aud, "sub": sub, "dur": d, "cues": cues,
                          "n": len(cues)})
        if not cands:
            plan[work] = None
            continue
        cands.sort(key=lambda c: (-c["n"], "无音效" in c["audio"].name, c["audio"].name))
        best = cands[0]
        print(f"{work}: 选轨 {best['audio'].name} cues={best['n']} dur={best['dur']:.1f}s")
        chk = densest(best["cues"], 0, best["dur"], best["dur"], [])
        w1 = chk[0]
        wins = {"w1": w1}
        avoid = [(w1, w1 + WIN)]
        for lab, lo, hi in [("w2", best["dur"] / 3, 2 * best["dur"] / 3),
                            ("w3", 2 * best["dur"] / 3, best["dur"])]:
            got = densest(best["cues"], lo, hi, best["dur"], avoid)
            if got:
                wins[lab] = got[0]
                avoid.append((got[0], got[0] + WIN))
            else:
                wins[lab] = None
        plan[work] = {"t": track_no(work, best["audio"].stem), "audio": best["audio"],
                      "sub": best["sub"], "dur": best["dur"], "cues": best["cues"],
                      "wins": wins,
                      "new_only": ["w1", "w2", "w3"]}

    # --- 切音频 + human_zh ---
    summary = []
    for work, p in plan.items():
        if p is None:
            summary.append((work, "NO TRACK"))
            continue
        for lab in p["new_only"]:
            start = p["wins"].get(lab)
            if start is None:
                summary.append((f"{work} {lab}", "SKIPPED(无法安排不重叠窗)"))
                continue
            if lab == "w1" and work in EXISTING:
                continue  # 现有 3 部的 w1 已存在，不动
            workid = work[2:].lstrip("0") or work[2:]
            scene = f"scene_asmr{workid}_{p['t']}"
            if lab != "w1":
                scene += f"_{lab}"
            wav = STD180 / f"{scene}.wav"
            got_dur = cut_audio(p["audio"], start, wav)
            # human_zh：cue 过滤到窗内，时间戳归零，重编号
            segs = []
            for t0, t1, zh in p["cues"]:
                if start <= t0 < start + WIN:
                    segs.append({"idx": len(segs) + 1, "seg_id": len(segs) + 1,
                                 "t0": round(t0 - start, 2), "t1": round(t1 - start, 2),
                                 "zh": zh})
            if p["sub"].suffix.lower() == ".lrc":
                source = "asmr_one_human_subtitle"
            elif p["sub"].suffix.lower() == ".srt":
                source = "dlsite_builtin_srt"
            else:
                source = "dlsite_builtin_vtt"
            third = ("音轨全轨最密窗" if lab == "w1" and work not in EXISTING else
                     ("音轨中段三分之一最密窗" if lab == "w2" else "音轨尾段三分之一最密窗"))
            doc = {
                "script_file": p["sub"].name,
                "source": source,
                "total_segments": len(segs),
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "segments": segs,
                "note": (f"标准化 180s 锚点场景扩建（2026-09-25）：{lab} = 音轨 "
                         f"{start:.1f}~{start + WIN:.1f}s 窗口（{third}，与前窗不重叠），"
                         f"时间戳已归零；音轨 {p['audio']}；字幕 {p['sub']}"),
            }
            out = ROOT / "dataset" / f"{scene}.human_zh.json"
            out.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            summary.append((scene, f"win={start:.1f}s cues={len(segs)} wav={got_dur:.1f}s"))
    for s in summary:
        print(s)
    # 落盘 plan 供复核
    dump = {w: (None if p is None else {
        "t": p["t"], "dur": round(p["dur"], 2),
        "audio": str(p["audio"]), "sub": str(p["sub"]),
        "wins": {k: (v if v is None else round(v, 2)) for k, v in p["wins"].items()},
        "n_cues": len(p["cues"])}) for w, p in plan.items()}
    (ROOT / "logs" / "scene_build_20260925.json").write_text(
        json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
