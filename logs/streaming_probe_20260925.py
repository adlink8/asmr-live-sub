# -*- coding: utf-8 -*-
"""流式起翻可行性探针（只读实验，不改 livesub/tools/dataset）。

问题：不等段收齐就把半截音频喂给 anime-whisper，识别文本还能用吗？
方法：取生产基线（logs/benchmark_runs，adaptive/2.0/5.0）5 窗的全部 asr 段，
对每段分别喂 50%/70%/100% 截断音频，同参数解码，量化截断结果相对全段结果的：
  - sim：difflib 字符相似度（0~1）
  - prefix：截断结果是否近似全段结果的前缀（流式场景截断结果可作"先行字幕"）
  - 空识别率
判读：50%/70% 的 sim 高且 prefix 率高 → 流式起翻可行；sim 崩 → 只能微调路线。
"""
import difflib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/ADLINK/asmr-live-sub")
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "logs" / "benchmark_runs"
OUT = ROOT / "logs" / "streaming_probe_20260925"
OUT.mkdir(exist_ok=True)

SUBSET = ["scene_asmr299717_t02", "scene_asmr416809_t01", "scene_asmr416816_t01",
          "scene_asmr1521586_t01", "scene_asmr1463510_t05"]
FRACS = [0.5, 0.7]
SR = 16000

import argparse
import live_sub  # CUDA DLL 注册副作用
from livesub.audio import resample_to_16k

def load_wav(path):
    import av
    chunks = []
    with av.open(str(path)) as c:
        for frame in c.decode(audio=0):
            x = frame.to_ndarray().astype(np.float32)
            scale = 32768.0 if any(s in frame.format.name for s in ("s16", "s32")) else 1.0
            if x.ndim > 1:
                x = x.mean(axis=0)
            chunks.append(x / scale)
    rate0 = 16000
    return resample_to_16k(np.concatenate(chunks), rate0)

def decode(model, audio):
    segs, _ = model.transcribe(audio, language="ja", task="transcribe", beam_size=1,
                               condition_on_previous_text=False,
                               no_speech_threshold=0.6, patience=2)
    return "".join(s.text for s in segs).strip()

def main():
    args = argparse.Namespace(model="anime", device="cuda")
    t0 = time.time()
    model = live_sub.load_model(args)
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    results = []
    for scene in SUBSET:
        wav = load_wav(Path(r"D:/Downloads/asmr-zh-corpus/std180s") / f"{scene}.wav")
        segs = []
        for line in (RUNS / f"{scene}.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("kind") == "asr" and (e.get("ja") or "").strip():
                segs.append((float(e["t_start"]), float(e["t_end"])))
        for i, (a, b) in enumerate(segs):
            ia, ib = int(a * SR), int(b * SR)
            full_audio = wav[ia:ib]
            ref = decode(model, full_audio)
            row = {"scene": scene, "seg": i, "audio_s": round((ib - ia) / SR, 2), "full": ref}
            for f in FRACS:
                cut = ia + int((ib - ia) * f)
                if cut - ia < int(0.8 * SR):  # <0.8s 不测
                    row[f"t{int(f*100)}"] = None
                    continue
                txt = decode(model, wav[ia:cut])
                row[f"t{int(f*100)}"] = txt
                row[f"sim{int(f*100)}"] = round(difflib.SequenceMatcher(None, txt, ref).ratio(), 3) if ref or txt else None
            results.append(row)
            print(f"{scene}#{i} {row['audio_s']}s full={ref[:24]!r} t70={row.get('t70','')[:20] if row.get('t70') else None!r} sim70={row.get('sim70')}", flush=True)
        (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    # 汇总
    summary = {}
    for f in FRACS:
        k = f"t{int(f*100)}"
        sims = [r[f"sim{int(f*100)}"] for r in results if r.get(f"sim{int(f*100)}") is not None]
        pref = sum(1 for r in results if r.get(k) and r["full"].startswith(r[k][:max(1, len(r[k]) - 2)]))
        empty = sum(1 for r in results if r.get(k) == "")
        summary[k] = {"n": len(sims), "sim_mean": round(sum(sims) / len(sims), 3),
                      "sim_ge08": round(sum(1 for s in sims if s >= 0.8) / len(sims), 3),
                      "prefix_like": pref, "empty": empty}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)

if __name__ == "__main__":
    main()
