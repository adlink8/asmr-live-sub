"""UNIT — livesub.audio primitives: FrameBuffer, resampler, newest-wins queue."""
import queue as q

import numpy as np
import pytest

from livesub.audio import FrameBuffer, _put_or_bump, enumerate_loopbacks, resample_to_16k


class TestFrameBuffer:
    def test_write_then_read_exact(self):
        buf = FrameBuffer()
        buf.write(b"0123456789")
        assert buf.read(10) == b"0123456789"

    def test_read_short_returns_empty(self):
        buf = FrameBuffer()
        buf.write(b"abc")
        assert buf.read(10) == b""

    def test_partial_read_keeps_rest(self):
        buf = FrameBuffer()
        buf.write(b"0123456789")
        assert buf.read(4) == b"0123"
        assert buf.read(6) == b"456789"

    def test_overflow_drops_oldest(self):
        buf = FrameBuffer(max_bytes=10)
        buf.write(b"0123456789ABCDE")  # 15 bytes -> keep last 10
        assert buf.read(10) == b"56789ABCDE"

    def test_overflow_is_silent(self):
        # 设计行为：溢出静默从头删（会撕裂帧且无日志——已知限制，勿"修"成抛错）
        buf = FrameBuffer(max_bytes=4)
        buf.write(b"ABCDEFGH")
        assert len(buf.read(4)) == 4


class TestResample:
    def test_identity_same_object(self):
        x = np.zeros(100, dtype=np.float32)
        assert resample_to_16k(x, 16000) is x

    def test_48k_to_16k_halves_length(self):
        x = np.zeros(48000, dtype=np.float32)
        out = resample_to_16k(x, 48000)
        assert len(out) == 16000
        assert out.dtype == np.float32

    def test_preserves_signal_roughly(self):
        # 线性插值在慢变信号上应保持幅度量级
        t = np.arange(48000) / 48000
        x = (0.5 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
        out = resample_to_16k(x, 48000)
        assert float(np.max(out)) == pytest.approx(0.5, abs=0.05)


class TestPutOrBump:
    def test_put_when_space(self):
        q_: q.Queue = q.Queue(maxsize=2)
        _put_or_bump(q_, "a")
        assert q_.get_nowait() == "a"

    def test_full_queue_drops_oldest_keeps_newest(self):
        q_: q.Queue = q.Queue(maxsize=2)
        q_.put("old1")
        q_.put("old2")
        _put_or_bump(q_, "new")
        assert q_.qsize() == 2
        assert list(q_.queue) == ["old2", "new"]

    def test_never_blocks(self):
        q_: q.Queue = q.Queue(maxsize=1)
        for i in range(100):
            _put_or_bump(q_, i)  # must not raise Full
        assert list(q_.queue) == [99]


class TestEnumerateLoopbacks:
    def test_returns_list_from_generator(self):
        class FakePa:
            def get_loopback_device_info_generator(self):
                yield {"index": 3, "name": "fake"}

        assert enumerate_loopbacks(FakePa()) == [{"index": 3, "name": "fake"}]
