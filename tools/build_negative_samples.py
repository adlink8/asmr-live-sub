"""构建负样本对照段（no-speech controls）。

目的：防幻觉过滤器（纯吟唱丢弃、no-speech 门槛、压缩率上限）在全人声的
benchmarks 上不可观测——一个永远不显示的废过滤器能拿满分。负样本段补上
这个对照组：段内无人声，管线的合格表现是**零字幕**。

两类来源：
  1. 真实段：用 livesub 自己的 Segmenter 扫现有 wav，取"整段无字幕产出"
     的 ≥35s 空隙居中切 30s。分三级：数字静音 / 门限下弱音 / 有能量但不成句。
  2. 合成分（确定性种子）：粉噪（门限之上，考 ASR 不乱识）、类吟唱调幅
     220Hz（考 sound-only 过滤器）。manifest 里标 source=synthetic。

输出：benchmarks/neg_*.wav + benchmarks/negative_samples.json
（每项：name/source/category/duration_s/expect_subtitles=0）
"""
import json
import queue as q
import wave
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))  # 脚本在 tools/ 下，livesub 包在仓库根

from livesub.audio import Segmenter

BENCH = ROOT / "benchmarks"
SR = 16000
WINDOW_S = 30.0
MIN_GAP_S = 35.0
MARGIN_S = 2.5

SOURCES = [
    BENCH / "soak_15min.wav",
    BENCH / "soak_30min.wav",
    BENCH / "soak_60min.wav",
    BENCH / "scene_a_dialogue.wav",
    BENCH / "scene_b_interactive.wav",
    BENCH / "scene_c_whisper_soft.wav",
    BENCH / "scene_dense_talk.wav",
    BENCH / "scene_random_talk.wav",
]


def decode_wav(path: Path):
    with av.open(str(path)) as c:
        chunks = []
        for frame in c.decode(audio=0):
            x = frame.to_ndarray().astype(np.float32)
            scale = 32768.0 if any(s in frame.format.name for s in ("s16", "s32")) else 1.0
            if x.ndim > 1:
                x = x.mean(axis=0)
            chunks.append(x / scale)
    return np.concatenate(chunks)


def emitted_ranges(audio: np.ndarray):
    """用真 Segmenter 跑一遍，返回产出字幕的 [t0, t1] 列表。"""
    out = q.Queue()
    seg = Segmenter(out, SR)
    chunk_n = int(SR * 0.1)
    for i in range(0, len(audio), chunk_n):
        seg.feed(audio[i:i + chunk_n])
    seg.flush()
    ranges = []
    while not out.empty():
        _, t0, t1 = out.get()
        ranges.append((float(t0), float(t1)))
    return sorted(ranges)


def gaps_of(ranges, duration_s):
    gaps, cur = [], 0.0
    for t0, t1 in ranges:
        if t0 - cur >= MIN_GAP_S:
            gaps.append((cur, t0))
        cur = max(cur, t1)
    if duration_s - cur >= MIN_GAP_S:
        gaps.append((cur, duration_s))
    return gaps


def classify(audio: np.ndarray):
    frame = int(SR * 0.51)
    n = len(audio) // frame
    if n == 0:
        return "digital_silence"
    rms = [float(np.sqrt(np.mean(audio[i * frame:(i + 1) * frame] ** 2))) for i in range(n)]
    peak = max(rms)
    if peak < 1e-4:
        return "digital_silence"
    if peak < 0.008:
        return "below_gate_weak"
    return "energy_but_no_speech"


def save_wav(path: Path, audio: np.ndarray):
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def synth_pink(rng, n):
    """粉噪：1/f 近似，RMS 归一到 0.02（asmr 门限 0.008 之上）。"""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    freqs[0] = freqs[1]
    spec /= np.sqrt(freqs)
    pink = np.fft.irfft(spec, n)
    return pink / np.sqrt(np.mean(pink ** 2)) * 0.02


def synth_am_moan(rng, n):
    """类吟唱：220Hz 载波 + 4Hz 幅度调制，RMS≈0.05（足够翻进 loud 模式）。"""
    t = np.arange(n) / SR
    am = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t)
    tone = np.sin(2 * np.pi * 220.0 * t) * am
    return tone / np.sqrt(np.mean(tone ** 2)) * 0.05


def main():
    rng = np.random.default_rng(20260922)
    manifest = []

    real = []
    for src in SOURCES:
        if not src.exists():
            continue
        audio = decode_wav(src)
        dur = len(audio) / SR
        for g0, g1 in gaps_of(emitted_ranges(audio), dur):
            mid = (g0 + g1) / 2
            t0 = mid - WINDOW_S / 2
            if t0 < g0 + 0.1 or t0 + WINDOW_S > g1 - 0.1:
                continue
            seg = audio[int(t0 * SR):int((t0 + WINDOW_S) * SR)]
            if len(seg) < int(WINDOW_S * SR) - SR // 10:
                continue
            cat = classify(seg)
            real.append((src.name, t0, cat, seg))
            break  # 每个源取最靠前的一段就够
        if len(real) >= 3:
            break

    for i, (src_name, t0, cat, seg) in enumerate(real[:3], 1):
        name = f"neg_{i}_real_{cat}.wav"
        save_wav(BENCH / name, seg)
        manifest.append({"name": name, "source": f"real:{src_name}@{t0:.1f}s",
                         "category": cat, "duration_s": WINDOW_S,
                         "expect_subtitles": 0})
        print(f"[real] {name}  <- {src_name} @ {t0:.1f}s  ({cat})")

    # 合成分量不受"真实空隙是否存在"影响，始终构建：
    # 粉噪与类吟唱调幅考的是真实空隙隔离不出来的两条路径
    # （ASR 对非语音能量的幻觉、sound-only 过滤器），必须常备。
    n = int(WINDOW_S * SR)
    synths = [("neg_synth_pink_noise.wav", "energy_no_speech_synth", synth_pink(rng, n)),
              ("neg_synth_am_moan.wav", "moan_like_tone_synth", synth_am_moan(rng, n))]
    for name, cat, seg in synths:
        save_wav(BENCH / name, seg)
        manifest.append({"name": name, "source": "synthetic",
                         "category": cat, "duration_s": WINDOW_S,
                         "expect_subtitles": 0})
        print(f"[synth] {name}  ({cat})")

    # 数字静音对照无论如何都留一个（最基础的"无输入无输出"）
    name = "neg_synth_digital_silence.wav"
    save_wav(BENCH / name, np.zeros(int(WINDOW_S * SR), dtype=np.float32))
    manifest.append({"name": name, "source": "synthetic",
                     "category": "digital_silence", "duration_s": WINDOW_S,
                     "expect_subtitles": 0})
    print(f"[synth] {name}  (digital_silence)")

    with open(BENCH / "negative_samples.json", "w", encoding="utf-8") as f:
        json.dump({"window_s": WINDOW_S, "samples": manifest}, f, ensure_ascii=False, indent=2)
    print(f"[✓] {len(manifest)} 个负样本段已写入 {BENCH / 'negative_samples.json'}")


if __name__ == "__main__":
    main()
