"""Shared fixtures: duck-typed stubs and synthetic audio builders.

The pipeline takes every dependency as a parameter (models, translator,
queues, loggers), so stubs stand in for the real thing without any mocking
framework. Each stub implements exactly the attributes the production code
touches — that surface is itself pinned by tests/test_contracts.py.
"""
import json
import queue as q
from pathlib import Path

import numpy as np
import pytest

from livesub.audio import Segmenter
from livesub.logging import LineLog, WorkLog


class StubSegment:
    """Mimics a faster-whisper segment object (only what asr_decode reads)."""

    def __init__(self, text, avg_logprob=-0.5, no_speech_prob=0.1,
                 compression_ratio=1.0, start=0.0, end=1.0):
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.compression_ratio = compression_ratio
        self.start = start
        self.end = end


class StubWhisperInfo:
    language = "ja"
    language_probability = 0.99


class StubWhisperModel:
    """Records every transcribe() call. segments may be a list or a
    callable(audio) -> list, for per-segment scripting."""

    def __init__(self, segments=None):
        self._segments = segments if segments is not None else [StubSegment("こんにちは")]
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append({"audio": audio, "kw": kw})
        segs = self._segments(audio) if callable(self._segments) else self._segments
        return list(segs), StubWhisperInfo()


class StubTranslator:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.last_stats = {}

    def translate(self, text):
        return self.mapping.get(text, f"译:{text}")


class RaisingTranslator:
    last_stats = {}

    def translate(self, text):
        raise RuntimeError("boom")


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def rate():
    return 16000


@pytest.fixture
def logs(tmp_path):
    """(LineLog, WorkLog) writing into a tmp dir; auto-closed."""
    line = LineLog(tmp_path / "test.txt")
    work = WorkLog(tmp_path / "work.jsonl")
    yield line, work
    line.close()
    work.close()


@pytest.fixture
def work_events(tmp_path):
    """WorkLog plus a parsed-events accessor."""
    work = WorkLog(tmp_path / "work.jsonl")
    yield work
    work.close()


def events_of(work: WorkLog, kind: str) -> list[dict]:
    """Parse the JSONL trace and return all events of one kind."""
    if not work.path.exists():
        return []
    out = []
    for line in work.path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") == kind:
            out.append(rec)
    return out


def make_frames_audio(pattern, rate=16000, frame_s=0.51, amp=0.5):
    """Build a float32 array from a per-frame pattern string.

    'v' = voiced (440 Hz sine, RMS ~0.35, far above the 0.008 gate),
    's' = silence (zeros). amp is overridable for gate-threshold tests.
    Plain function (not a fixture) so non-test code like drive_replay can
    call it directly — pytest forbids calling fixtures directly.
    """
    n = int(rate * frame_s)
    parts = []
    for ch in pattern:
        if ch == "v":
            t = np.arange(n) / rate
            parts.append((amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
        else:
            parts.append(np.zeros(n, dtype=np.float32))
    return np.concatenate(parts)


@pytest.fixture
def make_frames():
    return make_frames_audio


@pytest.fixture
def run_segmenter(make_frames, tmp_path):
    """Feed a frame pattern through a Segmenter.

    Returns (emitted, events): emitted is the list of raw
    (audio16, t_start, t_end) tuples; events is every JSONL trace record.
    Feeding uses realistic 100 ms chunks like the live feeder, unless
    chunk_s=None feeds the whole array in one call.
    """

    def _run(pattern, rate=16000, frame_s=0.51, amp=0.5, chunk_s=0.1, **seg_kw):
        out_q: q.Queue = q.Queue(maxsize=seg_kw.pop("out_q_size", 0) or 1000)
        work = WorkLog(tmp_path / "w.jsonl")
        seg = Segmenter(out_q, rate, frame_s=frame_s, work=work, **seg_kw)
        audio = make_frames(pattern, rate=rate, frame_s=frame_s, amp=amp)
        if chunk_s is None:
            seg.feed(audio)
        else:
            n = int(rate * chunk_s)
            for i in range(0, len(audio), n):
                seg.feed(audio[i:i + n])
        seg.flush()
        items = []
        while True:
            try:
                items.append(out_q.get_nowait())
            except q.Empty:
                break
        work.close()
        events = [json.loads(line) for line in
                  work.path.read_text(encoding="utf-8").splitlines()]
        return items, events

    return _run
