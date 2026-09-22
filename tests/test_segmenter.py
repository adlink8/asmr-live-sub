"""UNIT + REGRESSION — livesub.audio.Segmenter.

Frame constants: frame_s=0.51 -> 8160 samples @16k; one frame = 0.51s.
These tests pin the segmentation semantics that were tuned against real
session logs — including two quirks that are intentionally frozen:

  * hang_s=2.0 only yields hang_frames=int(2.0/0.51)=3 (1.53s, not 2.0s)
  * baseline max_s counts VOICED frames, so audio_s can exceed max_s
    (60.3% of real segments were force-cut this way; latency is 79% of it)
"""
import queue as q

import numpy as np
import pytest

from livesub.audio import Segmenter

FRAME_S = 0.51
FRAME_N = 8160  # int(16000 * 0.51)


def make_seg(out_q, rate=16000, **kw):
    kw.setdefault("frame_s", FRAME_S)
    return Segmenter(out_q, rate, **kw)


def drain(out_q):
    items = []
    while True:
        try:
            items.append(out_q.get_nowait())
        except q.Empty:
            return items


class TestSilence:
    def test_pure_silence_emits_nothing(self, run_segmenter):
        items, events = run_segmenter("ssssss")
        assert items == []
        assert [e for e in events if e["kind"] == "seg-drop"] == []


class TestMinSegGate:
    def test_one_voiced_frame_dropped(self, run_segmenter):
        # 0.51s < MIN_SEG_S=1.5 —— 这就是 09-22 日志里 331 次 seg-drop 的形态
        items, events = run_segmenter("vsss")
        assert items == []
        drops = [e for e in events if e["kind"] == "seg-drop"]
        assert len(drops) == 1
        assert drops[0]["reason"] == "too-short"
        assert drops[0]["voiced_s"] == pytest.approx(0.51, abs=0.01)

    def test_two_voiced_frames_dropped(self, run_segmenter):
        # 1.02s 仍低于门槛：相槌（はい/うん）全灭的根因
        items, events = run_segmenter("vvsss")
        assert items == []
        assert events[-1]["voiced_s"] == pytest.approx(1.02, abs=0.01)

    def test_three_voiced_frames_kept(self, run_segmenter):
        # 1.53s >= 1.5 过线
        items, _ = run_segmenter("vvvsss")
        assert len(items) == 1

    def test_drop_event_has_full_boundary_fields(self, run_segmenter):
        _, events = run_segmenter("vsss")
        drop = [e for e in events if e["kind"] == "seg-drop"][0]
        for key in ("reason", "voiced_s", "audio_s", "t_start", "t_end", "rms"):
            assert key in drop


class TestTrailingSilenceTrim:
    def test_hang_silence_trimmed_when_speech_dominates(self, run_segmenter):
        # 6 voiced + 3 hang silence: trail=3 < speech=6 -> 剪掉尾部静音
        items, _ = run_segmenter("vvvvvvsss")
        audio, t0, t1 = items[0]
        assert len(audio) / 16000 == pytest.approx(3.06, abs=0.02)  # 6 frames
        # 尾部确实是语音而不是静音
        assert float(np.max(np.abs(audio[-FRAME_N:]))) > 0.1

    def test_no_trim_when_silence_ge_speech(self, run_segmenter):
        # 3 voiced + 3 silence: trail(3) < speech(3) 为假 -> 不剪（现状如此）
        items, _ = run_segmenter("vvvsss")
        audio, t0, t1 = items[0]
        assert len(audio) / 16000 == pytest.approx(3.06, abs=0.02)


class TestHangFrames:
    def test_two_silence_frames_not_enough(self, make_frames):
        # hang_s=2.0 -> hang_frames=3：两帧静音不切
        out_q: q.Queue = q.Queue()
        seg = make_seg(out_q)
        seg.feed(make_frames("vvvss"))
        assert drain(out_q) == []

    def test_third_silence_frame_cuts(self, make_frames):
        out_q: q.Queue = q.Queue()
        seg = make_seg(out_q)
        seg.feed(make_frames("vvvss"))
        seg.feed(make_frames("s"))
        assert len(drain(out_q)) == 1

    def test_hang_s_floor_is_int_truncated(self, make_frames):
        # 锁定 int(2.0/0.51)=3（1.53s）而非 2.0s 的取整现状
        out_q: q.Queue = q.Queue()
        seg = make_seg(out_q)
        assert seg.hang_frames == 3


class TestMaxFramesForceCut:
    def test_fifteen_voiced_frames_force_emit(self, run_segmenter):
        # baseline: max_s=8.0 -> max_frames=15，只数有声帧
        items, _ = run_segmenter("v" * 15, chunk_s=None)
        assert len(items) == 1

    def test_interleaved_silence_lets_audio_exceed_max_s(self, run_segmenter):
        # 名不副实锁定：静音帧不进 _speech_frames，audio_s 可远超 max_s=8.0
        items, _ = run_segmenter("vs" * 15, chunk_s=None)
        assert len(items) == 1
        audio, t0, t1 = items[0]
        audio_s = len(audio) / 16000
        assert audio_s == pytest.approx(14.79, abs=0.05)  # 29 frames
        assert audio_s > 8.0  # max_s=8.0 根本没兜住

    def test_voiced_s_is_frame_count_not_audio_length(self, run_segmenter):
        items, _ = run_segmenter("vs" * 15, chunk_s=None)
        audio, t0, t1 = items[0]
        assert len(audio) / 16000 == pytest.approx(14.79, abs=0.05)


class TestAdaptiveLadder:
    def test_before_3s_needs_full_hang(self, make_frames):
        # adaptive: <3.0s 时 hang_limit 仍是 3 帧。
        # 不能走 run_segmenter——它会 flush()，会把没等到挂起的段吐出来。
        out_q: q.Queue = q.Queue()
        seg = make_seg(out_q, strategy="adaptive")
        seg.feed(make_frames("vvvvvs"))
        assert drain(out_q) == []
        seg.feed(make_frames("ss"))  # 静音累计到 3 帧才切
        assert len(drain(out_q)) == 1

    def test_after_3s_single_silence_frame_cuts(self, run_segmenter):
        # >=3.0s 后 1 帧静音(~0.5s 换气)即切断
        items, _ = run_segmenter("vvvvvvvs", strategy="adaptive", chunk_s=None)
        assert len(items) == 1

    def test_full_ladder_sequence(self, run_segmenter):
        items, _ = run_segmenter("vvvvvv" + "s" + "vvvvvv" + "s",
                                 strategy="adaptive", chunk_s=None)
        assert len(items) == 2


class TestHardcap:
    def test_wall_clock_cap_fires(self, run_segmenter):
        # hardcap: time_up 按墙钟。注意纯 voiced 流下帧配额
        # (floor(max_s/frame_s)) 总是先触发；墙钟上限的价值在静音交错时——
        # 有声帧被静音稀释、攒不满配额，段会被一直拖着，time_up 兜底。
        items, _ = run_segmenter("vs" * 9, strategy="hardcap", max_s=4.0,
                                 chunk_s=None)
        # 注意：夹具结尾会 flush()，墙钟切断后的残余帧会被合法吐出，
        # 所以只断言第一段（被 time_up 切断的那段）。
        first = items[0]
        _, t0, t1 = first
        # 第 9 帧(0-indexed 8)处 curr=4.08s >= 4.0 触发，段含 0..8 共 9 帧
        assert t1 - t0 == pytest.approx(4.59, abs=0.02)
        assert len(first[0]) / 16000 == pytest.approx(4.59, abs=0.02)

    def test_baseline_lets_same_audio_run_longer(self, run_segmenter):
        # 对照：baseline 没有 time_up，同一 pattern 靠 7 帧有声配额拖到更晚才切
        items, _ = run_segmenter("vs" * 9, strategy="baseline", max_s=4.0,
                                 chunk_s=None)
        _, t0, t1 = items[0]
        assert t1 - t0 == pytest.approx(6.63, abs=0.02)  # 13 帧

    def test_wall_clock_cap_beats_voiced_frame_quota(self, run_segmenter):
        # max_s=2.0 -> max_frames=3：小上限时配额本身就更早，切出更多段
        items, _ = run_segmenter("v" * 15, strategy="hardcap", max_s=2.0,
                                 chunk_s=None)
        assert len(items) >= 2


class TestFlush:
    def test_flush_emits_pending_speech(self, make_frames):
        out_q: q.Queue = q.Queue()
        seg = make_seg(out_q)
        seg.feed(make_frames("vvv"))  # 没等到挂起静音
        assert drain(out_q) == []
        seg.flush()
        assert len(drain(out_q)) == 1

    def test_flush_empty_is_noop(self, make_frames):
        out_q: q.Queue = q.Queue()
        seg = make_seg(out_q)
        seg.feed(make_frames("sss"))
        seg.flush()
        assert drain(out_q) == []


class TestBoundaries:
    def test_first_segment_starts_at_zero(self, run_segmenter):
        items, _ = run_segmenter("vvvsss")
        assert items[0][1] == pytest.approx(0.0)

    def test_second_segment_starts_where_first_ends(self, run_segmenter):
        items, _ = run_segmenter("vvvsssvvvsss")
        assert len(items) == 2
        _, t0a, t1a = items[0]
        _, t0b, t1b = items[1]
        assert t0b == pytest.approx(t1a, abs=0.11)  # 100ms 喂饭粒度内的连续

    def test_t_end_matches_cut_frame_not_feed_entry(self, run_segmenter):
        # 回归：t_end 必须是切段那一帧的结束位置。
        # 旧实现用 feed() 入口时的 _total_samples，100ms 喂饭下系统性偏旧，
        # 单次大数组喂入时甚至恒为 0。
        items, _ = run_segmenter("vvvsss", chunk_s=None)
        _, t0, t1 = items[0]
        assert t1 == pytest.approx(3.06, abs=0.02)

    def test_t_end_matches_cut_frame_chunked_feed(self, run_segmenter):
        # 同上，但走 100ms 分块喂入（live 路径的真实形态）
        items, _ = run_segmenter("vvvsss", chunk_s=0.1)
        _, t0, t1 = items[0]
        assert t1 == pytest.approx(3.06, abs=0.02)


class TestGate:
    def test_loud_audio_switches_mode_and_threshold(self, run_segmenter):
        # amp=0.5 -> RMS≈0.354，p50>=0.03 进 loud，thr=max(0.008, min(0.06, .33*p90))
        _, events = run_segmenter("vvvvvv", amp=0.5)
        gates = [e for e in events if e["kind"] == "gate"]
        assert gates and gates[-1]["mode"] == "loud"
        assert gates[-1]["thr"] == pytest.approx(0.06, abs=1e-6)

    def test_quiet_audio_stays_asmr_baseline(self, run_segmenter):
        _, events = run_segmenter("vvvvvv", amp=0.02)
        gates = [e for e in events if e["kind"] == "gate"]
        assert gates == []  # 从未离开 asmr 基线，无模式切换事件
        # 阈值保持 base 0.008：amp=0.02 -> RMS 0.014 > 0.008 仍是 voiced

    def test_hysteresis_band_does_not_flip(self, run_segmenter):
        # 先进 loud，再喂 p50≈0.02（滞回区 0.015~0.03 内）不得切回 asmr
        _, events = run_segmenter("vvvvv" + "v" * 5, amp=0.5)  # 全 loud
        assert [e for e in events if e["kind"] == "gate"][-1]["mode"] == "loud"
        out_q: q.Queue = q.Queue()
        seg = Segmenter(out_q, 16000, frame_s=FRAME_S)
        mid = (0.028 * np.sin(2 * np.pi * 440 *
                              np.arange(FRAME_N) / 16000)).astype(np.float32)
        loud = (0.5 * np.sin(2 * np.pi * 440 *
                             np.arange(FRAME_N) / 16000)).astype(np.float32)
        for _ in range(5):
            seg.feed(loud)
        assert seg.mode == "loud"
        for _ in range(5):
            seg.feed(mid)
        assert seg.mode == "loud"  # 0.02 落在滞回区，不切

    def test_gate_resets_to_asmr_when_truly_quiet(self, run_segmenter):
        out_q: q.Queue = q.Queue()
        seg = Segmenter(out_q, 16000, frame_s=FRAME_S)
        loud = (0.5 * np.sin(2 * np.pi * 440 *
                             np.arange(FRAME_N) / 16000)).astype(np.float32)
        quiet = (0.002 * np.sin(2 * np.pi * 440 *
                                np.arange(FRAME_N) / 16000)).astype(np.float32)
        for _ in range(5):
            seg.feed(loud)
        for _ in range(6):
            seg.feed(quiet)
        assert seg.mode == "asmr"


class TestQueueOverflow:
    def test_out_q_newest_wins(self, run_segmenter):
        items, _ = run_segmenter("vvvsssvvvsss", out_q_size=1)
        # maxsize=1：第二段把第一段顶掉（newest-wins，设计如此）
        assert len(items) == 1
        assert items[0][1] == pytest.approx(3.06, abs=0.11)
