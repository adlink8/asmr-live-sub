"""decode_audio 采样率回归：44.1kHz 音源若被误当 16k 传给重采样，
产出时长虚高 2.76 倍的"松鼠音"（2026-09-25 演练事故，全量采集数据作废的根因）。"""
import wave

import numpy as np

from tools.collect_finetune import decode_audio, write_wav


def test_decode_audio_resamples_44k1_to_16k(tmp_path):
    sr = 44100
    t = np.arange(sr, dtype=np.float32) / sr  # 1.0 秒
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    p = tmp_path / "sine44k.wav"
    write_wav_at(p, x, sr)

    y = decode_audio(p)
    dur = len(y) / 16000
    assert abs(dur - 1.0) < 0.05, f"解码时长 {dur:.3f}s，应为 ~1.0s"


def test_decode_audio_passes_through_16k(tmp_path):
    t = np.arange(16000, dtype=np.float32) / 16000
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    p = tmp_path / "sine16k.wav"
    write_wav_at(p, x, 16000)

    y = decode_audio(p)
    dur = len(y) / 16000
    assert abs(dur - 1.0) < 0.05, f"解码时长 {dur:.3f}s，应为 ~1.0s"


def write_wav_at(path, x, sr):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
