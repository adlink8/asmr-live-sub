#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""asr_checkup.py — 识别模型弱点体检（三套测量，产出"最烂维度"排行）。

A. CER：双语作品 ja cue 切片，模型转写 vs 人工转写（量吞字）
B. 无参照：响度五档 × 字幕cue有无 分桶，幻觉率/重复率/空输出率（量水声幻觉）
C. SE A/B：双版本作品同段各转写一遍，两版一致率 = 音效干扰净贡献

用法：python tools/asr_checkup.py --out logs/asr_checkup.md
只做诊断不进回归门；切片不含锚点作品也无妨（A 用的 RJ299717 是诊断专用）。
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from collect_finetune import decode_audio, parse_cues  # noqa: E402
from asmrone_collect import detect_lang  # noqa: E402
from livesub.audio import TARGET_SR  # noqa: E402

SR = TARGET_SR
PUNCT = set("、。！？!?,.…・「」『』（）()【】~〜ー─\n\r\t 　")


def norm_text(s):
    return "".join(c for c in s if c not in PUNCT).strip()


def cer(hyp, ref):
    """字符编辑距离/参考长度；任一为空返回 None。"""
    h, r = norm_text(hyp), norm_text(ref)
    if not r:
        return None
    if not h:
        return 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def repeat_ratio(text):
    """输出自重复度：字符 4-gram 中重复所占比例 0~1。"""
    t = norm_text(text)
    if len(t) < 8:
        return 0.0
    grams = [t[i:i + 4] for i in range(len(t) - 3)]
    return round(1 - len(set(grams)) / len(grams), 3)


class ASR:
    def __init__(self):
        from faster_whisper import WhisperModel
        self.m = WhisperModel(str(ROOT / "models" / "anime-whisper-ct2"),
                              device="cuda", compute_type="float16")

    def __call__(self, x):
        kw = dict(language="ja", task="transcribe", beam_size=1,
                  condition_on_previous_text=False, no_speech_threshold=0.6,
                  patience=2)
        segs, _ = self.m.transcribe(x, **kw)
        return "".join(s.text for s in segs).strip()


def rms_of(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def cue_covered(cues, t0, t1):
    """该时间段是否被人工字幕 cue 覆盖（≥30% 重叠）。"""
    for a, b, _ in cues:
        if min(t1, b) - max(t0, a) >= 0.3 * (t1 - t0):
            return True
    return False


def rms_bucket(r):
    if r < 0.004: return "0_极静音"
    if r < 0.010: return "1_气声"
    if r < 0.020: return "2_轻声"
    if r < 0.040: return "3_正常"
    return "4_响"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "logs" / "asr_checkup.md"))
    ap.add_argument("--max-per-bucket", type=int, default=12)
    args = ap.parse_args()

    asr = ASR()
    lines = [f"# ASR 弱点体检 {datetime.now():%Y-%m-%d %H:%M}\n"]

    # ---------- A. CER（真日语转写切片；RJ299717 的"ja"目录实测是中文，必须校验语言） ----------
    print("== A: CER ==", flush=True)
    a_rows = defaultdict(list)
    w299 = Path("D:/Downloads/asmr-zh-corpus/RJ299717")
    for ja_sub in sorted((w299 / "subtitles" / "ja").glob("*.lrc")):
        aud = w299 / "audio" / (ja_sub.stem + ".mp3")
        if not aud.exists():
            continue
        cues = [c for c in parse_cues(ja_sub)
                if 1.5 <= c[1] - c[0] <= 15.0 and detect_lang(c[2]) == "ja"]
        if not cues:
            print(f"  A {ja_sub.stem[:14]} 无有效 ja 参考（字幕实为中文），跳过", flush=True)
            continue
        cues = cues[:: max(1, len(cues) // args.max_per_bucket)][: args.max_per_bucket]
        x = decode_audio(aud)
        for t0, t1, ref in cues:
            hyp = asr(x[int(t0 * SR):int(t1 * SR) + SR // 2])
            c = cer(hyp, ref)
            if c is not None:
                a_rows["全量"].append(c)
            print(f"  A {ja_sub.stem[:14]} cer={c if c is None else round(c, 3)}", flush=True)
    lines.append("## A. CER（vs 人工 ja 转写）\n")
    if not a_rows:
        lines.append("- 本地素材无真日语转写参照，A 测量空缺（需 gold_ready 作品音频）\n")
    for k, v in a_rows.items():
        lines.append(f"- {k}: n={len(v)} 中位CER={np.median(v):.3f} "
                     f"p90={np.percentile(v, 90):.3f}\n")

    # ---------- B. 幻觉/重复/空输出（响度五档 × 字幕覆盖；覆盖桶=有cue窗，空隙桶=cue间隙≥12s） ----------
    print("== B: 无参照 ==", flush=True)
    b_rows = defaultdict(lambda: {"n": 0, "hallu": 0, "empty": 0, "rep": []})
    for wdir in (Path("D:/tmp/gold-test/RJ304908"),
                 Path("D:/Downloads/asmr-drill-staging/RJ358382")):
        auds = sorted((wdir / "audio").glob("*.mp3"))
        for aud in auds[:: max(1, len(auds) // 4)][:3]:
            zh_dir = wdir / "subtitles" / "zh"
            sub = next((zh_dir / f"{aud.stem}{e}" for e in (".lrc", ".vtt")
                        if (zh_dir / f"{aud.stem}{e}").exists()), None)
            cues = parse_cues(sub) if sub else []
            x = decode_audio(aud)
            dur = len(x) / SR
            # 字幕覆盖桶：cue 密集区网格采样
            cov_windows = [t for t in np.arange(0, dur - 10, 10)
                           if cue_covered(cues, t, t + 10)]
            cov_windows = cov_windows[:: max(1, len(cov_windows) // 8)][:8]
            # 无字幕桶：cue 间隙 ≥12s 的区域（间奏/喘息长段=该输出空的地方）
            gap_windows = []
            cs = sorted(cues)
            for (a1, b1, _), (a2, _, _) in zip(cs, cs[1:]):
                if a2 - b1 >= 12:
                    gap_windows.append((b1 + 1.0, b1 + 1.0 + 10))
            gap_windows = gap_windows[:8]
            for t0, tag in [(t, "有字幕") for t in cov_windows] + \
                           [(t, "cue间隙") for t in gap_windows]:
                t1 = t0 + 10
                seg = x[int(t0 * SR):int(t1 * SR)]
                r = rms_bucket(rms_of(seg))
                hyp = asr(seg)
                row = b_rows[(r, tag)]
                row["n"] += 1
                row["rep"].append(repeat_ratio(hyp))
                if not hyp:
                    row["empty"] += 1
                elif tag == "cue间隙":
                    row["hallu"] += 1  # 字幕空隙输出非空=疑似幻觉
    lines.append("\n## B. 幻觉/重复/空输出（响度×字幕覆盖）\n")
    lines.append("| 响度 | 区域 | n | 幻觉率 | 空输出率 | 重复率中位 |\n"
                 "|---|---|---|---|---|---|\n")
    for (r, cov), v in sorted(b_rows.items()):
        lines.append(f"| {r} | {cov} | {v['n']} | "
                     f"{v['hallu'] / v['n']:.0%} | {v['empty'] / v['n']:.0%} | "
                     f"{np.median(v['rep']):.2f} |\n")

    # ---------- C. SE A/B（RJ358382 7 对双版本） ----------
    print("== C: SE A/B ==", flush=True)
    c_rows = defaultdict(lambda: {"n": 0, "agree": 0, "rep": [], "empty": 0})
    w358 = Path("D:/Downloads/asmr-drill-staging/RJ358382/audio")
    pairs = defaultdict(dict)
    for f in sorted(w358.glob("*.mp3")):
        base = f.stem.replace("（効果音無し）", "")
        pairs[base][("なし" if "効果音無し" in f.stem else "あり")] = f
    for base, v in pairs.items():
        if len(v) < 2:
            continue
        xs = {k: decode_audio(f) for k, f in v.items()}
        dur = min(len(x) for x in xs.values()) / SR
        for t0 in (30.0, min(120.0, dur / 2)):
            hyps = {}
            for k, x in xs.items():
                hyps[k] = asr(x[int(t0 * SR):int((t0 + 30) * SR)])
                row = c_rows[k]
                row["n"] += 1
                row["rep"].append(repeat_ratio(hyps[k]))
                if not hyps[k]:
                    row["empty"] += 1
            same = norm_text(hyps["あり"]) == norm_text(hyps["なし"])
            c_rows["_对比"]["n"] += 1
            c_rows["_对比"]["agree"] += same
        print(f"  C {base[:20]} あり≠なし: {not same}", flush=True)
    lines.append("\n## C. SE 音效干扰 A/B\n")
    for k, v in c_rows.items():
        if k == "_对比":
            lines.append(f"- 有SE vs 无SE 转写一致率: {v['agree'] / v['n']:.0%} "
                         f"(n={v['n']} 段)\n")
        else:
            lines.append(f"- {k}: n={v['n']} 空输出率={v['empty'] / v['n']:.0%} "
                         f"重复率中位={np.median(v['rep']):.2f}\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
