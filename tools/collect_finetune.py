#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collect_finetune.py — 微调数据采集编排（下载用 tools/asmrone_collect.py，本工具只做采集段）。

流程：作品目录(audio/ + subtitles/zh/) → 全轨连续切 180s 窗 → live_sub 快速回放(内容确定性，
<100 段规避 seg_q 驱逐) → realtime_compare 自动对齐(字幕漂移检测) → 产出训练对 →
原件保留（mp3 即压缩音频，~1MB/min = ASR 微调弹药，训练时按 (路径,t0,t1) 直接读取）→
窗 wav 为临时文件用完即删；--delete-source 才真正删原件（确认不再要 ASR 弹药时用）。

产物 dataset_finetune/<RJ>/：
  tmp/                               窗 wav 临时文件（回放完即删，不占长期空间）
  asr/<track>__wNNN.json             ASR 伪标签段 [{src,t0,t1,ja}]（t 为作品全轨绝对秒）
  mt/<track>__wNNN.json              (asr_ja, human_zh) 配对 + offset 元数据
  cmp/<track>__wNNN/                 realtime_compare 原始报告（对齐质量审计用）
  provenance.json                    作品级来源记录

硬门：锚点作品 RJ 全窗排除（防泄漏，26 窗清单见 ANCHOR_RJS）。
对齐失败（realtime_compare 退出码 2/3）的窗只产 asr/，不产 mt/。

用法：
  python tools/collect_finetune.py --work D:/Downloads/asmr-zh-corpus/RJ300204 \
      --limit-windows 2 --limit-tracks 1          # 试跑
"""
import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ANCHOR_RJS = {1449384, 1463510, 1497366, 1521586, 1527130, 299717, 324799,
              401391, 416809, 416816}
WIN_S = 180.0
SR = 16000

sys.path.insert(0, str(ROOT / "tools"))


def run(cmd, timeout=900):
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def decode_audio(path):
    import av
    chunks = []
    src_rate = 16000  # 实际值取自音频流；44.1k/48k 音源若误传 16k 会产出"松鼠音"（时长虚高 2.76 倍+音调上飘）
    with av.open(str(path)) as c:
        for frame in c.decode(audio=0):
            src_rate = frame.sample_rate
            x = frame.to_ndarray().astype(np.float32)
            scale = 32768.0 if any(s in frame.format.name for s in ("s16", "s32")) else 1.0
            if x.ndim > 1:
                x = x.mean(axis=0)
            chunks.append(x / scale)
    from livesub.audio import resample_to_16k
    return resample_to_16k(np.concatenate(chunks), src_rate)


def write_wav(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)


def _list_audio(d):
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a")) if d.exists() else []


def _list_audio_count(work):
    """全部音频数（含 sidecar 布局），用于零配对报错信息。"""
    n = len(_list_audio(work / "audio"))
    if n:
        return n
    return sum(1 for p in sorted(work.rglob("*"))
               if p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a"))


def _find_sub(audio_path, zh_dir):
    for ext in (".lrc", ".vtt", ".srt"):
        c = zh_dir / f"{audio_path.stem}{ext}"
        if c.exists():
            return c
    # asmr.one 嵌套命名：字幕文件名含音频全名（如 "Track01：xxx.mp3.vtt"）
    for f in zh_dir.iterdir():
        if f.name.startswith(audio_path.name) and \
                f.suffix.lower() in (".lrc", ".vtt", ".srt"):
            return f
    return None


def parse_cues(sub_path):
    """字幕文件 -> [(t0, t1, text)]（轨内绝对秒）。LRC 无结束时间由下一 cue 顺延；
    无时间戳的纯文本返回 []。供金级层（双语字幕直接配对）使用。"""
    from subtitles_to_gt import parse_lrc, parse_srt_vtt
    p = Path(sub_path)
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return []
    ext = p.suffix.lower()
    if ext == ".lrc":
        pts = parse_lrc(raw)
        cues = []
        for i, (t0, txt) in enumerate(pts):
            t1 = pts[i + 1][0] if i + 1 < len(pts) else t0 + 10.0
            cues.append((float(t0), float(t1), txt))
        return cues
    if ext in (".vtt", ".srt"):
        return [(float(a), float(b), t) for a, b, t in parse_srt_vtt(raw)]
    return []


def pair_gold(zh_cues, ja_cues, win_t0, win_t1, min_overlap=0.3):
    """金级配对：zh cue 与 ja cue 同一人工时间轴，按时间重叠配对——
    零 ASR、零偏移校正、零对齐误差。要求重叠 >= min_overlap×zh cue 时长。"""
    pairs = []
    for z0, z1, zt in zh_cues:
        if z1 <= win_t0 or z0 >= win_t1:
            continue
        best_ja, best_ov = None, 0.0
        for j0, j1, jt in ja_cues:
            ov = min(z1, j1) - max(z0, j0)
            if ov > best_ov:
                best_ov, best_ja = ov, jt
        if best_ja and best_ov >= min_overlap * max(z1 - z0, 1e-6):
            pairs.append({"t0": round(z0, 2), "t1": round(z1, 2),
                          "ja": best_ja, "human": zt})
    return pairs


def _find_ja_sub(audio_path, work):
    """金级层素材：同作品 subtitles/ja/ 下同 stem 的带时间戳日文字幕。"""
    jd = Path(work) / "subtitles" / "ja"
    for ext in (".lrc", ".vtt", ".srt"):
        c = jd / f"{audio_path.stem}{ext}"
        if c.exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="作品目录（含 audio/ 与 subtitles/zh/）")
    ap.add_argument("--out", default=str(ROOT / "dataset_finetune"))
    ap.add_argument("--limit-tracks", type=int, default=None)
    ap.add_argument("--limit-windows", type=int, default=None)
    ap.add_argument("--delete-source", action="store_true",
                    help="采集成功后删除原始音频（默认永不删）")
    ap.add_argument("--allow-anchor", action="store_true",
                    help="验证模式：允许锚点作品，产物标 validation_only 不进训练集")
    args = ap.parse_args()

    work = Path(args.work)
    import re as _re
    m = _re.search(r"RJ(\d+)", str(work))
    rj = int(m.group(1)) if m else 0
    out = Path(args.out) / work.name
    allow_anchor = args.allow_anchor
    if rj in ANCHOR_RJS and not allow_anchor:
        sys.exit(f"[SKIP] RJ{rj} 是锚点作品（防泄漏硬门）。验证用 --allow-anchor，产物标 validation_only。")
    for d in ("asr", "mt", "cmp", "tmp"):
        (out / d).mkdir(parents=True, exist_ok=True)

    # 布局自适应：asmr-zh 风格(audio/ + subtitles/zh/) 或 sidecar(音视频同目录同名 .vtt/.srt)
    if (work / "subtitles" / "zh").exists():
        tracks = [(p, _find_sub(p, work / "subtitles" / "zh"))
                  for p in _list_audio(work / "audio")]
        tracks = [(a, s) for a, s in tracks if s]
    else:  # sidecar：递归找音频，字幕=同 stem 的 .vtt/.srt
        tracks = []
        for p in sorted(work.rglob("*")):
            if p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a"):
                s = next((Path(str(p) + e) for e in (".vtt", ".srt", ".lrc")
                          if Path(str(p) + e).exists()), None)
                if s:
                    tracks.append((p, s))
    if args.limit_tracks:
        tracks = tracks[:args.limit_tracks]
    if not tracks:
        # 零配对=布局不识别或素材残缺：快速失败（非零退出），nightly 据此不标记已采
        sys.exit(f"[FAIL] {work.name} 音轨-字幕零配对（共 {_list_audio_count(work)} 条音频无字幕），"
                 f"拒绝产出空数据")
    prov = {"rj": rj, "work": str(work),
            "validation_only": bool(allow_anchor and rj in ANCHOR_RJS),
            "tracks": [], "config":
            {"win_s": WIN_S, "strategy": "adaptive", "max_s": 5.0, "hang_s": 2.0}}

    for aud, sub in tracks:
        stem = aud.stem
        ja_sub = _find_ja_sub(aud, work)
        zh_cues = parse_cues(sub)
        ja_cues = parse_cues(ja_sub) if ja_sub else []

        x = decode_audio(aud)
        n_win = int(len(x) / SR // WIN_S)
        if args.limit_windows:
            n_win = min(n_win, args.limit_windows)
        tr = {"track": stem, "subtitle": sub.name,
              "ja_subtitle": ja_sub.name if ja_sub else None,
              "windows": [], "duration_s": round(len(x) / SR, 1)}
        print(f"[track] {stem} {tr['duration_s']}s -> {n_win} 窗"
              + (f" [金级：双语字幕 {ja_sub.name}]" if ja_cues else ""), flush=True)

        for wi in range(n_win):
            t0, t1 = wi * WIN_S, (wi + 1) * WIN_S
            name = f"{stem}__w{wi:03d}"
            wav_p = out / "tmp" / f"{name}.wav"
            log_p = out / f"{name}.txt"
            seg = x[int(t0 * SR):int(t1 * SR)]
            # 静音/环境音窗跳过：低于 ASMR 门(0.008)的窗回放必然零产出，纯浪费 GPU
            rms = float(np.sqrt(np.mean(np.square(seg)))) if len(seg) else 0.0
            win = {"window": wi, "t0": t0, "t1": t1, "rms": round(rms, 5)}
            if rms < 0.004:
                win["status"] = "silent_skip"
                tr["windows"].append(win)
                continue
            write_wav(wav_p, seg)
            log_p.with_suffix(".jsonl").unlink(missing_ok=True)  # WorkLog 追加模式，防串台
            log_p.unlink(missing_ok=True)
            code, log = run([sys.executable, str(ROOT / "live_sub.py"),
                             "--source-audio", str(wav_p), "--replay", "--fast",
                             "--strategy", "adaptive", "--max-s", "5.0", "--hang-s", "2.0",
                             "--model", "anime", "--mt", "sakura", "--log", str(log_p)])
            wav_p.unlink(missing_ok=True)  # 临时窗用完即删
            win["replay_exit"] = code
            jsonl = log_p.with_suffix(".jsonl")
            if code != 0 or not jsonl.exists():
                win["status"] = "replay_failed"
                tr["windows"].append(win)
                continue
            # ASR 伪标签
            segs = []
            for line in jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("kind") == "asr" and (e.get("ja") or "").strip():
                    segs.append({"src": str(aud), "t0": round(float(e["t_start"]) + t0, 2),
                                 "t1": round(float(e["t_end"]) + t0, 2),
                                 "ja": e["ja"].strip()})
            (out / "asr" / f"{name}.json").write_text(
                json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
            win["asr_segs"] = len(segs)

            # 金级层：zh/ja 双人工字幕同时间轴，cue 直接按重叠配对——
            # 零 ASR 依赖、零偏移校正。ASR 伪标签照收（弹药），但 MT 对用金级。
            if ja_cues and zh_cues:
                gp = pair_gold(zh_cues, ja_cues, t0, t1)
                if gp:
                    (out / "mt" / f"{name}.json").write_text(json.dumps(
                        {"track": stem, "window": wi, "t0": t0, "t1": t1,
                         "tier": "gold", "pairs": gp},
                        ensure_ascii=False, indent=1), encoding="utf-8")
                    win["tier"] = "gold"
                    win["mt_pairs"] = len(gp)
                    tr["windows"].append(win)
                    print(f"  w{wi:03d}: asr={len(segs)} gold_pairs={len(gp)}", flush=True)
                    continue

            win["tier"] = "teacher"

            # 自动对齐 + MT 对
            cmp_out = out / "cmp" / name
            # 两遍自校正：pass1 offset=t0 → 用配对 cue 的 latency 中位估计字幕超前量 → pass2 修正
            def _compare(off):
                return run([sys.executable, str(ROOT / "tools" / "realtime_compare.py"),
                            str(jsonl), "--human", str(sub), "--out", str(cmp_out),
                            "--offset", str(off)])
            ccode, _ = _compare(t0)
            rep = Path(str(cmp_out) + ".json")
            try:
                cues = json.loads(rep.read_text(encoding="utf-8"))["cues"]
                lats = sorted(c["latency_s"] for c in cues
                              if c.get("status") == "paired" and c.get("latency_s") is not None)
            except Exception:
                lats = []
            off = t0
            if lats and lats[len(lats) // 2] > 4.0:  # 中位时差≫流水线固有小延迟 → 字幕超前，修正重配
                off = t0 + (lats[len(lats) // 2] - 1.0)
                ccode, _ = _compare(off)
                win["offset_corrected"] = round(off - t0, 2)
            win["align_exit"] = ccode
            # 统计门：对齐质量不达标的窗降级——只出 ASR 伪标签，不出 MT 对（宁缺毋脏）
            gate = {"status": "ok"}
            if win.get("offset_corrected", 0) and abs(win["offset_corrected"]) > 30:
                gate = {"status": "downgraded", "reason": "offset_jump"}
            elif len(lats) < 3:
                gate = {"status": "low_confidence", "reason": "few_paired_cues"}
            elif lats:
                p10, p90 = lats[max(0, int(len(lats) * 0.1))], lats[min(len(lats) - 1, int(len(lats) * 0.9))]
                if p90 - p10 > 15.0:
                    gate = {"status": "downgraded", "reason": "latency_spread",
                            "p10": round(p10, 1), "p90": round(p90, 1)}
            win["gate"] = gate
            ji = Path(str(cmp_out) + ".judge_input.json")
            if gate["status"] == "ok" and ccode == 0 and ji.exists():
                pairs = json.loads(ji.read_text(encoding="utf-8"))["pairs"]
                (out / "mt" / f"{name}.json").write_text(json.dumps(
                    {"track": stem, "window": wi, "t0": t0, "t1": t1,
                     "pairs": pairs}, ensure_ascii=False, indent=1), encoding="utf-8")
                win["mt_pairs"] = len(pairs)
            else:
                win["mt_pairs"] = 0
            tr["windows"].append(win)
            print(f"  w{wi:03d}: asr={win.get('asr_segs')} align_exit={ccode} mt_pairs={win.get('mt_pairs')}", flush=True)

        if args.delete_source:
            aud.unlink()
            tr["source_deleted"] = True
        prov["tracks"].append(tr)

    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
    n_mt = sum(w.get("mt_pairs") or 0 for t in prov["tracks"] for w in t["windows"])
    n_asr = sum(w.get("asr_segs") or 0 for t in prov["tracks"] for w in t["windows"])
    print(f"[done] RJ{rj}: asr_segs={n_asr} mt_pairs={n_mt} -> {out}")


if __name__ == "__main__":
    main()
