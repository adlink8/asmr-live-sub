"""INTEGRATION — real modules wired together, stubs only at the model boundary.

Everything here runs the production code paths (Segmenter.feed, asr_loop,
mt_loop, FrameBuffer) with duck-typed models. Queue sizes mirror production
(seg_q=100, mt_q=2, late_q=200, result_q=50).
"""
import json
import queue as q
import subprocess
import sys
import threading

import numpy as np
import pytest

from conftest import StubSegment, StubTranslator, StubWhisperModel, make_frames_audio
from livesub.audio import FrameBuffer, Segmenter
from livesub.config import ROOT
from livesub.logging import LineLog, WorkLog
from livesub.pipeline import asr_loop, mt_loop


def production_queues():
    return (q.Queue(maxsize=100), q.Queue(maxsize=2),
            q.Queue(maxsize=200), q.Queue(maxsize=50))


def counting_model(prefix="段"):
    state = {"n": 0}

    def segs(audio):
        state["n"] += 1
        return [StubSegment(f"{prefix}{state['n']}のセリフ", avg_logprob=-0.5)]

    return StubWhisperModel(segs)


def drive_replay(pattern, rate=16000, **seg_kw):
    """Whole pipeline in replay mode: Segmenter -> asr_loop -> mt_loop.

    Returns (subtitles, events). subtitles are (zh, ja, dt) in order.
    """
    from conftest import make_frames_audio
    seg_q, mt_q, late_q, result_q = production_queues()
    done = threading.Event()
    logs_path = seg_kw.pop("logs", None) or ROOT / "logs" / "itest.jsonl"
    work = WorkLog(logs_path)
    log = LineLog(logs_path.with_suffix(".txt"))
    asr_t = threading.Thread(target=asr_loop,
                             args=(seg_q, mt_q, late_q, counting_model(), log,
                                   work, 1, True), daemon=True)
    mt_t = threading.Thread(target=mt_loop,
                            args=(mt_q, late_q, result_q, StubTranslator(),
                                  log, work, True, done), daemon=True)
    asr_t.start()
    mt_t.start()

    seg = Segmenter(seg_q, rate, work=work, **seg_kw)
    audio = make_frames_audio(pattern, rate=rate)
    n = int(rate * 0.1)
    for i in range(0, len(audio), n):
        seg.feed(audio[i:i + n])
    seg.flush()
    seg_q.put(None)
    assert done.wait(timeout=10), "pipeline did not drain"
    asr_t.join(timeout=5)
    mt_t.join(timeout=5)
    subs = []
    while not result_q.empty():
        subs.append(result_q.get_nowait())
    log.close()
    work.close()
    events = [json.loads(line) for line in
              work.path.read_text(encoding="utf-8").splitlines()]
    return subs, events


class TestFullPipeline:
    def test_two_sentences_two_subtitles(self, tmp_path):
        # 两个 3 帧句子 + 3 帧静音间隔：两段都过 MIN_SEG_S
        subs, events = drive_replay("vvvsssvvvsss",
                                    logs=tmp_path / "w.jsonl")
        assert len(subs) == 2
        assert [s[0] for s in subs] == ["译:段1のセリフ", "译:段2のセリフ"]

    def test_short_breath_dropped_end_to_end(self, tmp_path):
        # 0.51s 喘息在段层就被丢，根本到不了 asr/mt
        subs, events = drive_replay("vsssvvvsss", logs=tmp_path / "w.jsonl")
        assert len(subs) == 1
        drops = [e for e in events if e["kind"] == "seg-drop"]
        assert len(drops) == 1
        assert drops[0]["reason"] == "too-short"

    def test_event_chain_joins_by_seg_id(self, tmp_path):
        subs, events = drive_replay("vvvsssvvvsss", logs=tmp_path / "w.jsonl")
        asr_by_id = {e["seg_id"]: e for e in events if e["kind"] == "asr"}
        mt_evs = [e for e in events if e["kind"] == "mt"]
        assert len(mt_evs) == len(subs) == 2
        for ev in mt_evs:
            assert ev["seg_id"] in asr_by_id          # 每条 mt 都能 join 回 asr
            assert asr_by_id[ev["seg_id"]]["ja"] == ev["ja"]

    def test_subtitle_ja_matches_asr_ja(self, tmp_path):
        subs, events = drive_replay("vvvsssvvvsss", logs=tmp_path / "w.jsonl")
        asr_jas = [e["ja"] for e in events
                   if e["kind"] == "asr" and e["ja"]]
        assert [s[1] for s in subs] == asr_jas

    def test_segment_boundaries_flow_into_events(self, tmp_path):
        # asr 事件的 t_start/t_end 来自 Segmenter 的真实段边界
        subs, events = drive_replay("vvvsssvvvsss", logs=tmp_path / "w.jsonl")
        asr_evs = [e for e in events if e["kind"] == "asr"]
        assert [e["t_start"] for e in asr_evs] == [0.0, 3.06]
        assert [e["t_end"] for e in asr_evs] == [3.06, 6.12]
        assert [e["audio_s"] for e in asr_evs] == [3.06, 3.06]


class TestCapturePlumbing:
    def test_buffer_plus_feeder_equals_direct_feed(self, tmp_path, make_frames):
        # live 路径：PortAudio callback 写 FrameBuffer，feeder 按 100ms 读出
        # 转 float 喂 Segmenter。与直接整段喂入必须产出完全相同的段。
        # 注意：feeder 读不满一个 chunk 的尾部会留在 buffer 里永远喂不进去
        # （固有特性），所以把音频补齐到 chunk 整数倍，保证两条路采样流一致。
        rate = 48000  # 同时 exercised 重采样路径
        chunk_bytes = int(rate * 0.1) * 2  # mono int16, 100ms
        audio = make_frames("vvvsssvvvsss", rate=rate)
        pad = (-len(audio)) % (chunk_bytes // 2)
        audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])

        def run(feeder_style):
            seg_q: q.Queue = q.Queue()
            seg = Segmenter(seg_q, rate, work=None)
            if feeder_style:
                buf = FrameBuffer()
                raw = (audio * 32767).astype(np.int16).tobytes()
                pos = 0
                while pos < len(raw):
                    buf.write(raw[pos:pos + 4096])
                    pos += 4096
                    data = buf.read(chunk_bytes)
                    if len(data) < chunk_bytes:
                        continue
                    x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    seg.feed(x)
            else:
                seg.feed(audio)
            seg.flush()
            out = []
            while True:
                try:
                    out.append(seg_q.get_nowait())
                except q.Empty:
                    return out

        direct = run(False)
        via_buffer = run(True)
        assert len(direct) == len(via_buffer) == 2
        for (a1, s1, e1), (a2, s2, e2) in zip(direct, via_buffer):
            assert s1 == pytest.approx(s2, abs=1e-9)
            assert e1 == pytest.approx(e2, abs=1e-9)
            assert len(a1) == len(a2)
            assert np.allclose(a1, a2, atol=1e-3)  # int16 量化误差

    def test_emitted_audio_is_16k_float32_mono(self, make_frames):
        seg_q: q.Queue = q.Queue()
        seg = Segmenter(seg_q, 48000)
        seg.feed(make_frames("vvvsss", rate=48000))
        seg.flush()
        audio, t0, t1 = seg_q.get_nowait()
        assert audio.dtype == np.float32
        # 3v+3s 不剪静音（trail=0）：6 帧 @48k 重采样后 3.06s @16k
        assert abs(len(audio) / 16000 - 3.06) < 0.05


class TestReplayFidelity:
    def test_long_sequence_loses_nothing(self, tmp_path):
        # 10 个句子，回放模式必须一条不丢且保序
        pattern = "vvvsss" * 10
        subs, events = drive_replay(pattern, logs=tmp_path / "w.jsonl")
        assert len(subs) == 10
        assert [s[0] for s in subs] == [f"译:段{i}のセリフ" for i in range(1, 11)]
        asr_ids = [e["seg_id"] for e in events if e["kind"] == "asr"]
        assert asr_ids == list(range(10))  # seg_id 连续无洞

    def test_adaptive_strategy_runs_end_to_end(self, tmp_path):
        subs, _ = drive_replay("vvvvvv" + "s" + "vvvvvv" + "s",
                               strategy="adaptive", logs=tmp_path / "w.jsonl")
        assert len(subs) == 2


class TestCliSubprocess:
    def test_help_exits_zero_with_key_flags(self):
        r = subprocess.run(
            [sys.executable, "live_sub.py", "--help"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=120)
        assert r.returncode == 0
        for flag in ("--strategy", "--max-s", "--hang-s", "--layer",
                     "--source-audio", "--replay", "--mt"):
            assert flag in r.stdout, flag

    def test_compat_import_registers_cuda_dlls(self):
        # 新解释器里 import live_sub：shim -> livesub -> config 注册 DLL 路径
        r = subprocess.run(
            [sys.executable, "-c",
             "import live_sub; print(live_sub.HAVE_CUDA_DLLS, "
             "callable(live_sub.load_model), callable(live_sub.SakuraMT))"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=120)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip().endswith("True True")
