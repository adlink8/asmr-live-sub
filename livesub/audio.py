"""Audio plumbing: newest-wins queue helper, decoupling frame buffer,
resampling to 16 kHz, the RMS energy-gate Segmenter, and WASAPI loopback
device enumeration.
"""
import queue
import threading
import time
from collections import deque

import numpy as np

from livesub.config import MIN_SEG_S, TARGET_SR


def _put_or_bump(q, item):
    """Newest-wins: never block the producer if the consumer is behind."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


class FrameBuffer:
    def __init__(self, max_bytes=2_000_000):
        self._buf = bytearray()
        self._lock = threading.Lock()
        self.max_bytes = max_bytes

    def write(self, data: bytes):
        with self._lock:
            self._buf.extend(data)
            overflow = len(self._buf) - self.max_bytes
            if overflow > 0:
                del self._buf[:overflow]

    def read(self, n_bytes: int) -> bytes:
        with self._lock:
            if len(self._buf) < n_bytes:
                return b""
            out = bytes(self._buf[:n_bytes])
            del self._buf[:n_bytes]
            return out


def resample_to_16k(x: np.ndarray, rate: int) -> np.ndarray:
    # 线性插值而非 scipy.signal.resample_poly：48k->16k 的混叠失真 Whisper
    # （log-Mel 谱前端）可以容忍，不值得为此引入 scipy。换 ASR 模型或更高
    # 采样率音源时，优先改回 resample_poly。
    if rate == TARGET_SR:
        return x
    n_out = int(len(x) * TARGET_SR / rate)
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def enumerate_loopbacks(pya):
    return list(pya.get_loopback_device_info_generator())


class Segmenter:
    """RMS energy gate. ASMR keeps a low floor; loud audio raises the cut
    so a dip relative to recent peak ends a sentence instead of waiting for max_s."""

    # hang_s=2.0 对齐 faster-whisper Silero VAD 的 min_silence_duration_ms 默认值。
    # 原值 0.9s 会把换气/喘息当成句尾，把一句话切成两半：实测段长中位数 1.02s，
    # 而 0.8~1.5s 的碎片有 73% 是空识别（纯浪费算力）。
    def __init__(self, out_q, rate, frame_s=0.51, threshold=0.008,
                 hang_s=2.0, max_s=8.0, strategy="baseline", log=None, work=None):
        self.out_q = out_q
        self.rate = rate
        self.frame_s = frame_s
        self.frame = int(rate * frame_s)
        self.base_threshold = threshold
        self.threshold = threshold
        self.hang_s = hang_s
        self.max_s = max_s
        self.strategy = strategy
        self.hang_frames = max(1, int(hang_s / frame_s))
        self.max_frames = int(max_s / frame_s)
        self.log = log
        self.work = work
        self.mode = "asmr"
        self._rms_hist = deque(maxlen=max(8, int(6.0 / frame_s)))
        self._last_gate_log = 0.0
        self._pending = np.zeros(0, dtype=np.float32)
        self._speech: list[np.ndarray] = []
        self._speech_frames = 0
        self._silence_run = 0
        self._voiced_frames = 0
        self._total_samples = 0          # 已喂入的总采样数（绝对位置）
        self._seg_start_sample = None    # 当前积累中的段起点（绝对）
        self._session_t0 = time.time()   # 流起点墙钟，用于换算真实时刻

    def _update_gate(self, rms: float):
        self._rms_hist.append(rms)
        if len(self._rms_hist) < 4:
            return
        xs = sorted(self._rms_hist)
        p50 = xs[len(xs) // 2]
        peak = xs[int((len(xs) - 1) * 0.9)]
        prev = self.mode
        if self.mode == "asmr":
            if p50 >= 0.03:
                self.mode = "loud"
        elif p50 < 0.015:
            self.mode = "asmr"
        if self.mode == "asmr":
            self.threshold = self.base_threshold
        else:
            self.threshold = max(self.base_threshold, min(0.06, 0.33 * peak))
        if self.mode == prev:
            return
        if self.log:
            self.log.write(
                f"[gate] {self.mode} thr={self.threshold:.4f} p50={p50:.4f} peak={peak:.4f}")
        if self.work:
            self.work.event(
                "gate", mode=self.mode, thr=round(self.threshold, 5),
                p50=round(p50, 5), peak=round(peak, 5), rms=round(rms, 5))

    def feed(self, chunk: np.ndarray):
        # carry partial frames across calls: chunks arrive ~100ms, frames are ~510ms
        # carry = 上一轮遗留、尚未消费的采样数。绝对位置必须把它减掉：
        # _total_samples 到 feed() 末尾才 +=，_pending 也是末尾才裁剪，
        # 循环中直接用 _total_samples + pos 会让段边界最多虚高一个帧长（0.51s）。
        carry = len(self._pending)
        self._pending = np.concatenate((self._pending, chunk))
        pos = 0
        while len(self._pending) - pos >= self.frame:
            frame = self._pending[pos:pos + self.frame]
            frame_abs = self._total_samples - carry + pos
            pos += self.frame
            rms = float(np.sqrt(np.mean(frame * frame)))
            self._update_gate(rms)
            voiced = rms > self.threshold

            # 计算当前积攒段的物理总时长（秒）
            curr_seg_s = ((frame_abs - self._seg_start_sample) / self.rate) if self._seg_start_sample is not None else 0.0

            if self.strategy == "hardcap":
                time_up = curr_seg_s >= self.max_s
                hang_limit = self.hang_frames
            elif self.strategy == "adaptive":
                time_up = curr_seg_s >= self.max_s
                # 前 3 秒严格静音保护完整句；3 秒以上只要 1 帧静音(~0.5s换气)即切断
                hang_limit = self.hang_frames if curr_seg_s < 3.0 else 1
            else:
                time_up = False
                hang_limit = self.hang_frames

            if voiced:
                if self._seg_start_sample is None:
                    self._seg_start_sample = frame_abs
                self._silence_run = 0
                self._speech.append(frame)
                self._speech_frames += 1
                self._voiced_frames += 1
                if self._speech_frames >= self.max_frames or time_up:
                    self._emit(frame_abs + self.frame)
            elif self._speech:
                self._speech.append(frame)
                self._silence_run += 1
                if self._silence_run >= hang_limit or self._speech_frames >= self.max_frames or time_up:
                    self._emit(frame_abs + self.frame)
        self._pending = self._pending[pos:]
        self._total_samples += len(chunk)

    def _emit(self, end_abs=None):
        # 在拼音频之前先把"有效人声时长"算出来。_speech 里最后几帧是挂起静音，
        # 判门槛用的是 self._voiced_frames（只有真正 voiced 的帧），
        # 而不是 len(_speech)。
        voiced_s = self._voiced_frames * (self.frame / self.rate)
        # 尾部静音要剪掉再送 ASR：留着纯属浪费算力，还会把 avg_logprob 拉低，
        # 让正常短句更容易撞上 low-conf。挂起静音只负责"判定句子结束"，
        # 没有理由进入识别。
        trail = self._silence_run if self._silence_run < self._speech_frames else 0
        keep = len(self._speech) - trail
        seg = np.concatenate(self._speech[:keep]) if keep > 0 else np.zeros(0, np.float32)
        self._speech = []
        self._speech_frames = 0
        self._silence_run = 0
        self._voiced_frames = 0
        # 段边界（相对流起点的秒）：起点是记录下的采样位置，终点是当前帧的结束位置
        # （feed() 调用方传入）。flush() 无当前帧，用全流长度。
        t_start = (self._seg_start_sample / self.rate
                   if self._seg_start_sample is not None else None)
        t_end = (end_abs if end_abs is not None else self._total_samples) / self.rate
        self._seg_start_sample = None
        audio16 = resample_to_16k(seg, self.rate)
        dur = len(audio16) / TARGET_SR
        # ASMR 是 ~2/3 的语气词（ん/はぁ）：太短的段几乎不可能成为一行字幕，
        # 还会把真正的句子埋掉。下限按"有效人声时长"判，不按打包后的总长——
        # 后者含静音垫，会让门槛永久失效。配合 hang_s=2.0 一起抬到 1.5s。
        if voiced_s < MIN_SEG_S:
            # 被丢弃的段以前在日志里完全不可见：看不到它，就永远不知道
            # 1.5s 这个下限砍掉了多少、砍的什么。这是调参的前提数据。
            if self.work:
                self.work.event("seg-drop", reason="too-short",
                                voiced_s=round(voiced_s, 3),
                                audio_s=round(dur, 3),
                                t_start=round(t_start, 3) if t_start is not None else None,
                                t_end=round(t_end, 3),
                                rms=round(float(np.sqrt(np.mean(seg * seg))) if len(seg) else 0.0, 5))
            return
        _put_or_bump(self.out_q, (audio16, t_start, t_end))

    def flush(self):
        """流结束时把积压中的段吐出来。否则最后一句会因为没等到挂起静音而丢掉。"""
        if self._speech:
            self._emit()
