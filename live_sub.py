"""WASAPI loopback -> faster-whisper (ja listen, zh out) -> always-on-top subtitle window.

Usage:
  python live_sub.py                     # live capture, small model
  python live_sub.py --model medium
  python live_sub.py --list-devices
  python live_sub.py --source-audio x.mp3   # offline smoke test, no GUI
"""
import argparse
import json
import os
import queue
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("XDG_DATA_HOME", str(ROOT / "argosdata"))
os.environ.setdefault("XDG_CONFIG_HOME", str(ROOT / "argosconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "argoscache"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TARGET_SR = 16000
# 段长下限：低于此长度的音频不送 ASR。见 Segmenter._emit 注释。
MIN_SEG_S = 1.5
# low-conf 的段长门槛：短句的 avg_logprob 天然偏低，只有长段才据此判幻觉。
LOWCONF_MIN_S = 4.0
# low-conf 的置信度阈值。实测正常段均值 -0.54~-0.61，正常常用语会擦到 -1.0，
# 故收紧到 -1.15（与长段幻觉的实际水平对齐）。
LOWCONF_LOGPROB = -1.15


def _add_cuda_dlls() -> bool:
    site = ROOT / ".venv" / "Lib" / "site-packages" / "nvidia"
    dirs = [site / lib / "bin" for lib in ("cublas", "cudnn")]
    found = all(d.is_dir() for d in dirs)
    handles = []
    if found:
        import ctypes
        # ctranslate2 loads cublas by name via WinAPI; without this the
        # AddDllDirectory dirs are not consulted (0x1000 = SEARCH_DEFAULT_DIRS)
        ctypes.windll.kernel32.SetDefaultDllDirectories(0x1000)
        for d in dirs:
            handles.append(os.add_dll_directory(str(d)))
    _add_cuda_dlls.handles = handles  # keep alive: closing removes dir from search
    return found


HAVE_CUDA_DLLS = _add_cuda_dlls()

import numpy as np  # noqa: E402

# layer 1: community post-filters (WhisperJAV / LocalLLaMA / zenn.jp)
# い excluded from moan set so はい survives; kanji always keeps the line
HALLU_RE = re.compile(
    r"ご(視聴|覧).{0,16}(ありがとう|感謝)"
    r"|チャンネル登録"
    r"|(最後まで|どうも).{0,16}(ありがとう|お(聞き|読み))"
    r"|(お疲れ様|おつかれさま)でし(た|て)"
    r"|thank you for watching"
    r"|thanks for watching",
    re.I,
)
# い excluded so はい survives; か/さ/た/な/ま/や/ら/わ 行常用词不进表
_MOAN = set(
    "あぁぅうぇえぉおんっはぁふひへほ"
    "ぐグぎギぷプぶブぺペぽポゅュょョヴ"
    "きキァゥェォンッハヒフヘホー～〜"
)
_PUNCT = re.compile(r"[\s\-ー～〜…・。、，,！!？?~゛゜「」『』（）()【】\[\]]+")
_LOOP_UNIT = re.compile(r"(.{1,4})\1{5,}")

# 翻译侧拒答检测。Sakura 偶尔不翻译而是回一段"我是AI我无法…"，
# 这类输出 zh_len 正常、zh_empty=False，光看长度完全正常，
# 但会被原样塞进字幕显示给用户——属于用户可见的故障。
# 只抓高置信度特征串，宁可漏也不误杀：正常翻译里不会出现这些整句。
# 注意「我无法/我不能」这类**不能**裸匹配：「我无法用语言形容」是正常译文。
# 必须要求后面跟"拒绝类动词"，且句中不能是普通的日常表达。
REFUSAL_RE = re.compile(
    r"作为一个.{0,6}(AI|人工智能|语言模型)"
    r"|我是.{0,6}(AI|人工智能|语言模型|助手)"
    r"|(无法|不能)(翻译|协助|回答|提供|完成|满足|处理)(这|该|您|你的|此|本)?(个|段|项|要求|请求|内容|文本|问题)?"
    r"|抱歉[，,].{0,12}(无法|不能|帮不上)"
    r"|对不起[，,].{0,12}(无法|不能|帮不上)"
    r"|我(无法|不能|没有办法)理解.{0,6}(这|该|这段|该段)"
    r"|请(提供|给出).{0,8}(文本|日文|原文|内容)"
    r"|(翻译|内容).{0,4}(超出|违反).{0,6}(范围|规定|政策)",
    re.I,
)


def is_refusal(zh: str) -> bool:
    """译文中是否出现"模型拒答"特征。用于丢弃而不是当字幕显示。"""
    return bool(zh) and bool(REFUSAL_RE.search(zh))


def _ts():
    t = time.time()
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"


class LineLog:
    def __init__(self, path: Path | None):
        self.path = path
        self._f = path.open("a", encoding="utf-8") if path else None

    def write(self, msg: str):
        line = f"{_ts()} {msg}"
        print(line, flush=True)
        if self._f:
            self._f.write(line + "\n")
            self._f.flush()

    def close(self):
        if self._f:
            self._f.close()
            self._f = None


class WorkLog:
    """JSONL model-trace log. File only — not printed to the overlay console."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def event(self, kind: str, **kv):
        rec = {"ts": _ts(), "kind": kind}
        rec.update(kv)
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()

    def close(self):
        with self._lock:
            self._f.close()


def is_kana_loop(text: str) -> bool:
    compact = _PUNCT.sub("", text)
    if len(compact) < 12:
        return False
    freq: dict[str, int] = {}
    for c in compact:
        freq[c] = freq.get(c, 0) + 1
    ch, n = max(freq.items(), key=lambda kv: kv[1])
    if n / len(compact) >= 0.55 and ch in _MOAN:
        return True
    return _LOOP_UNIT.search(compact) is not None


def drop_reason(text: str, no_speech_prob=None, avg_logprob=None,
                audio_s=None, compression_ratio=None) -> str | None:
    t = text.strip()
    if not t:
        return "empty"
    if HALLU_RE.search(t):
        return "hallucination"
    if is_kana_loop(t):
        return "kana-loop"
    if no_speech_prob is not None and no_speech_prob > 0.6:
        if audio_s is None or audio_s <= 2.5:
            return "no-speech"
    # 原实现要求 avg_logprob<-1.0 与 no_speech_prob>0.6 同时成立才判 low-conf，
    # 实测 0 次触发；而 >6s 的长段平均 logprob 已是 -1.15，明显低质却全放行。
    # 置信度与"是否语音"是两个独立信号，分开判。
    # 阈值必须带段长门槛：短句的 avg_logprob 天然偏低，独立触发会误杀
    # "すごい""危ないよ"这类正常短句。
    # 阈值本身也从 -1.0 收紧到 -1.15：实测正常段的平均 logprob 是 -0.54~-0.61，
    # 而"お疲れ様です"(-1.03)、"今のがいいね"(-1.00) 这类正常常用语会擦线误杀，
    # -1.0 卡在真实分布边缘太危险。-1.15 与长段幻觉的实际水平对齐。
    if avg_logprob is not None and avg_logprob < LOWCONF_LOGPROB:
        if audio_s is None or audio_s > LOWCONF_MIN_S:
            return "low-conf"
    # compression_ratio 一直在采集却从未参与判断。Whisper 的经典重复幻觉信号
    # 就是该值偏高（>2.4），白捡的过滤条件。
    if compression_ratio is not None and compression_ratio > 2.4:
        return "repetition"
    core = _PUNCT.sub("", t)
    if not core:
        return "sound-only"
    if all(c in _MOAN for c in core) and len(core) <= 24:
        return "sound-only"
    return None


# ---------- audio plumbing ----------

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
    if rate == TARGET_SR:
        return x
    n_out = int(len(x) * TARGET_SR / rate)
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


class Segmenter:
    """RMS energy gate. ASMR keeps a low floor; loud audio raises the cut
    so a dip relative to recent peak ends a sentence instead of waiting for max_s."""

    # hang_s=2.0 对齐 faster-whisper Silero VAD 的 min_silence_duration_ms 默认值。
    # 原值 0.9s 会把换气/喘息当成句尾，把一句话切成两半：实测段长中位数 1.02s，
    # 而 0.8~1.5s 的碎片有 73% 是空识别（纯浪费算力）。
    def __init__(self, out_q, rate, frame_s=0.51, threshold=0.008,
                 hang_s=2.0, max_s=8.0, log=None, work=None):
        self.out_q = out_q
        self.rate = rate
        self.frame = int(rate * frame_s)
        self.base_threshold = threshold
        self.threshold = threshold
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
        # 真正有声的帧数（不含尾部挂起静音）。判"这段是否值得送 ASR"必须用它，
        # 不能用 len(audio16)：_emit 会把 hang_frames 帧静音一起打包，
        # 于是每个段都自带 ~1.53s 静音垫。用总长衡量的话 MIN_SEG_S 等于失效——
        # 哪怕只有 1 帧人声（0.51s），加上垫底静音也有 2.04s，必然越过 1.5s 门槛。
        # 实测：0.4s 爆音 → 入队段长 2.55s，MIN_SEG_S 完全没拦住。
        self._voiced_frames = 0
        # 调优用的时序追踪：feed() 收到的总采样数即流的绝对位置。
        # 没有它就无法算段间间隔，"两段是否真的紧邻"这类判断只能靠猜。
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
        self._pending = np.concatenate((self._pending, chunk))
        pos = 0
        while len(self._pending) - pos >= self.frame:
            frame = self._pending[pos:pos + self.frame]
            # 段起点的绝对采样位置：每次取帧前先记一份，被判定为 voiced 的
            # 那一帧即当前段的起点。用 _total_samples + pos 而非 _total_samples，
            # 因为一帧跨多次 feed 时，起点在本次 chunk 内部。
            frame_abs = self._total_samples + pos
            pos += self.frame
            rms = float(np.sqrt(np.mean(frame * frame)))
            self._update_gate(rms)
            voiced = rms > self.threshold
            if voiced:
                if self._seg_start_sample is None:
                    self._seg_start_sample = frame_abs
                self._silence_run = 0
                self._speech.append(frame)
                self._speech_frames += 1
                self._voiced_frames += 1
                # must cut while still loud: max_s used to be checked only on
                # silence, so a loud track grew one clip forever and froze ASR
                if self._speech_frames >= self.max_frames:
                    self._emit()
            elif self._speech:
                self._speech.append(frame)
                self._silence_run += 1
                if self._silence_run >= self.hang_frames or self._speech_frames >= self.max_frames:
                    self._emit()
        self._pending = self._pending[pos:]
        self._total_samples += len(chunk)

    def _emit(self):
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
        # 段边界（相对流起点的秒）：起点是记录下的采样位置，终点是当前绝对位置。
        t_start = (self._seg_start_sample / self.rate
                   if self._seg_start_sample is not None else None)
        t_end = self._total_samples / self.rate
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


# ---------- transcription ----------

# 本地 CT2 模型的别名表：--model anime 即加载这个目录。
# quantumcookie/anime-whisper-ct2 是 litagin/anime-whisper（756M）的 CTranslate2 版，
# 底座 kotoba-whisper-v2.0，5300h 动漫语音微调；官方特性：抗幻觉、识别非言语声
# （笑/叹气/呼吸）、支持 NSFW 音频。日语 ASMR 场景比 Whisper small 对口得多。
# 注意：官方要求不要加 initial_prompt，会掉点。
LOCAL_ASR_MODELS = {
    "anime": ROOT / "models" / "anime-whisper-ct2",
}


def load_model(args):
    from faster_whisper import WhisperModel
    device = args.device
    if device == "cuda" and not HAVE_CUDA_DLLS:
        print("[warn] cuBLAS/cuDNN runtime not found, using CPU", file=sys.stderr)
        device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    # 先查别名表：命中的话直接走本地目录，不联网
    src = LOCAL_ASR_MODELS.get(args.model)
    if src is not None:
        if not (src / "model.bin").exists():
            raise FileNotFoundError(
                f"本地模型 {args.model} 不完整：缺少 {src / 'model.bin'}")
        print(f"[asr] 本地 CT2 模型 {args.model} -> {src}")
        return WhisperModel(str(src), device=device, compute_type=compute,
                            local_files_only=True)
    return WhisperModel(args.model, device=device, compute_type=compute,
                        download_root=str(ROOT / "models"))


class SakuraMT:
    """Sakura-7B Q4 on CPU/RAM. GPU stays with Whisper."""
    SYS = ("你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，"
           "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。")
    GGUF = ROOT / "models" / "sakura" / "sakura-7b-qwen2.5-v1.0-iq4xs.gguf"

    def __init__(self, n_gpu_layers=0):
        from llama_cpp import Llama
        if not self.GGUF.exists():
            raise FileNotFoundError(f"missing {self.GGUF}")
        n_threads = max(4, min(12, (os.cpu_count() or 8) - 2))
        t0 = time.time()
        self.llm = Llama(
            model_path=str(self.GGUF),
            n_ctx=2048,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        print(f"[mt] Sakura-7B IQ4XS 已加载 (cpu threads={n_threads} ngl={n_gpu_layers} {time.time()-t0:.1f}s)")
        self.last_stats = {"load_s": round(time.time() - t0, 3), "threads": n_threads, "ngl": n_gpu_layers}

    def translate(self, text):
        text = (text or "").strip()
        if not text:
            self.last_stats = {}
            return ""
        prompt = (
            f"<|im_start|>system\n{self.SYS}<|im_end|>\n"
            f"<|im_start|>user\n将下面的日文文本翻译成中文：{text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        t0 = time.time()
        max_tokens = min(80, max(12, len(text) * 3 + 8))
        out = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.3,
            repeat_penalty=1.05,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        zh = (out["choices"][0]["text"] or "").strip()
        usage = out.get("usage") or {}
        tim = out.get("timings") or {}
        self.last_stats = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "pred_ms": tim.get("predicted_ms"),
            "prompt_ms": tim.get("prompt_ms"),
            "pred_per_s": tim.get("predicted_per_second"),
            "gen_s": round(time.time() - t0, 3),
            "in_chars": len(text),
            "out_chars": len(zh),
            "max_tokens": max_tokens,
        }
        return zh


class NllbMT:
    """Dedicated ja→zh MT (NLLB-200 distilled 600M). No chat prompt, no safety filter."""
    MODEL_ID = "facebook/nllb-200-distilled-600M"
    SRC = "jpn_Jpan"
    TGT = "zho_Hans"

    def __init__(self):
        import ctranslate2
        from transformers import AutoTokenizer
        ct2_dir = ROOT / "models" / "nllb"
        if not (ct2_dir / "model.bin").exists():
            raise FileNotFoundError(f"missing {ct2_dir / 'model.bin'}; convert first")
        cache = str(ROOT / "models")
        self.tok = AutoTokenizer.from_pretrained(
            self.MODEL_ID, src_lang=self.SRC, cache_dir=cache)
        self.tr = ctranslate2.Translator(
            str(ct2_dir), device="cuda", compute_type="float16")
        print("[mt] NLLB-200-600M 已加载 (cuda fp16, ja→zh)")
        self.last_stats = {}

    def translate(self, text):
        text = (text or "").strip()
        if not text:
            return ""
        source = self.tok.convert_ids_to_tokens(self.tok.encode(text))
        max_len = min(160, max(12, len(source) * 3))
        results = self.tr.translate_batch(
            [source], target_prefix=[[self.TGT]],
            beam_size=1, repetition_penalty=1.3,
            no_repeat_ngram_size=3, disable_unk=True,
            max_decoding_length=max_len)
        hyp = results[0].hypotheses[0]
        if hyp and hyp[0] == self.TGT:
            hyp = hyp[1:]
        ids = self.tok.convert_tokens_to_ids(hyp)
        out = self.tok.decode(ids, skip_special_tokens=True).strip()
        core = re.sub(r"[\s,，.。、!！?？~]+", "", out)
        if len(core) >= 8 and len(set(core)) <= 2:
            return ""
        return out


class QwenMT:
    SYS = ("你是日中字幕翻译。把用户输入的日语文本逐句直译成简体中文："
           "不得增删原文没有的内容；拟声词、喘息、语气词按原样用中文对应字保留；"
           "粗俗或暧昧用语直译，不要美化或改写；无法确定含义的词按字面音译。"
           "只输出译文。")

    def __init__(self, path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float16, device_map="cuda")
        self.model.eval()
        print("[mt] Qwen2.5-1.5B 已加载 (cuda fp16)")
        self.last_stats = {}

    def translate(self, text):
        msgs = [{"role": "system", "content": self.SYS},
                {"role": "user", "content": text}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=160, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        gen = out[0][enc["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


def load_translator(args):
    if args.mt == "none":
        return None
    if args.mt == "sakura":
        try:
            return SakuraMT(n_gpu_layers=getattr(args, "mt_ngl", 0))
        except Exception as e:
            print(f"[mt-warn] Sakura 加载失败({e})，只显示日语原文", file=sys.stderr)
            return None
    if args.mt == "qwen":
        path = ROOT / "models" / "qwen"
        if not (path / "config.json").exists():
            print("[mt] 未找到 models/qwen，只显示日语原文", file=sys.stderr)
            return None
        try:
            return QwenMT(str(path))
        except Exception as e:
            print(f"[mt-warn] Qwen 加载失败({e})，只显示日语原文", file=sys.stderr)
            return None
    try:
        return NllbMT()
    except Exception as e:
        print(f"[mt-warn] NLLB 加载失败({e})，只显示日语原文", file=sys.stderr)
        return None


def drain_all(q):
    """Pop the blocking head plus everything already queued. None still means shutdown."""
    first = q.get()
    items = [first]
    if first is None:
        return items
    while True:
        try:
            nxt = q.get_nowait()
        except queue.Empty:
            break
        items.append(nxt)
        if nxt is None:
            break
    return items


def asr_decode(audio, model, log: LineLog, work: WorkLog, layer: int, last_ja: str,
               role: str, qinfo: dict, t_start=None, t_end=None, seg_id=None):
    """Return (ja, asr_s, last_ja). ja is None if filtered/empty (already logged)."""
    t0 = time.time()
    audio_s = round(len(audio) / TARGET_SR, 3) if audio is not None else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio is not None and len(audio) else 0.0
    kw = dict(language="ja", task="transcribe", beam_size=1,
              condition_on_previous_text=False, no_speech_threshold=0.6,
              # patience 放宽 beam 搜索的收敛条件，降低乱猜概率。
              # 注意：suppress_tokens="" 只适用于 OpenAI 原版 CLI；faster-whisper
              # 走 ctranslate2，该参数必须是 List[int]（默认 [-1]），传空串会直接抛
              # incompatible function arguments。想完全关闭用 []，不要用 ""。
              patience=2)
    if layer >= 2:
        # 显式给出 VAD 参数。min_silence_duration_ms 与 Segmenter.hang_s 保持一致，
        # 避免两处分段标准打架。
        kw["vad_filter"] = True
        kw["vad_parameters"] = dict(
            threshold=0.5,
            min_speech_duration_ms=250,
            max_speech_duration_s=30,
            min_silence_duration_ms=2000,
            speech_pad_ms=400,
        )
    segments, info = model.transcribe(audio, **kw)
    segs = list(segments)
    asr_s = round(time.time() - t0, 3)
    raw = []
    for s in segs:
        raw.append({
            "text": s.text.strip(),
            "start": round(getattr(s, "start", 0) or 0, 3),
            "end": round(getattr(s, "end", 0) or 0, 3),
            "avg_logprob": round(getattr(s, "avg_logprob", 0) or 0, 4),
            "no_speech_prob": round(getattr(s, "no_speech_prob", 0) or 0, 4),
            "compression_ratio": round(getattr(s, "compression_ratio", 0) or 0, 3),
        })
    drop_why = None
    seg_drops: list[str] = []  # 逐 segment 的丢弃原因，之前只写进 rec 里没汇总
    ja = ""
    if layer >= 1:
        kept = []
        for rec, s in zip(raw, segs):
            why = drop_reason(rec["text"], rec["no_speech_prob"], rec["avg_logprob"],
                              audio_s, rec["compression_ratio"])
            rec["drop"] = why
            if why:
                seg_drops.append(why)
                log.write(f"[drop] {why} | {rec['text']}")
                continue
            kept.append(rec["text"])
        ja = " ".join(kept).strip()
        max_nsp = max((r["no_speech_prob"] for r in raw), default=None) if raw else None
        max_cr = max((r["compression_ratio"] for r in raw), default=None) if raw else None
        drop_why = drop_reason(ja, max_nsp, None, audio_s, max_cr) if ja else "empty"
        if drop_why:
            if ja:
                log.write(f"[drop] {drop_why} | {ja}")
            ja = None
        elif ja in last_ja:
            # 原实现只跟"上一条"比，导致 ありがとう 这类真实短句连续出现会被全部放行。
            # last_ja 现在是最近 N 条的历史窗口。
            drop_why = "repeat"
            log.write(f"[drop] repeat | {ja}")
            ja = None
    else:
        ja = " ".join(r["text"] for r in raw).strip() or None
    work.event(
        "asr",
        role=role,
        # 段的绝对边界（相对流起点的秒）与段长。以前日志里只有 audio_s（段长），
        # 没有边界，导致两条字幕"时间上是否相邻"只能靠列表位置猜——句重组那版
        # 测试因此得出"顺序错乱"的错误结论。段间间隔必须用 t_start 差值算。
        seg_id=seg_id,
        t_start=round(t_start, 3) if t_start is not None else None,
        t_end=round(t_end, 3) if t_end is not None else None,
        audio_s=audio_s,
        rms=round(rms, 5),
        asr_s=asr_s,
        lang=getattr(info, "language", None),
        lang_p=round(getattr(info, "language_probability", 0) or 0, 4),
        nseg=len(raw),
        segments=raw,
        ja=ja,
        # drop 是"整段最终结论"；seg_drops 汇总逐 segment 原因。
        # 旧版只写 drop，段内被丢的 segment 原因在 ja 非空时全部丢失，
        # 导致按 JSONL 统计丢弃分布会严重失真（实测只剩 empty 一类）。
        drop=drop_why,
        seg_drops=seg_drops or None,
        **qinfo,
    )
    if ja is None:
        return None, asr_s, last_ja
    # 历史窗口：最近 5 条，用于去重（只存不含标点的核心文本）
    hist = list(last_ja)
    core_new = _PUNCT.sub("", ja)
    if core_new and core_new not in hist:
        hist = (hist + [core_new])[-5:]
    return ja, asr_s, hist


def asr_loop(seg_q, mt_q, late_q, model, log: LineLog, work: WorkLog, layer: int):
    last_ja: list[str] = []  # 最近 5 条去重历史窗口
    seg_seq = 0              # 段序号：把 asr 与后续 mt 事件串起来的相关 ID
    while True:
        items = drain_all(seg_q)
        halt = items[-1] is None
        # 队列里现在是 (audio16, t_start, t_end) 三元组。为了兼容离线灌流
        # （-source-audio 直接 put 裸数组），这里统一包一层解包。
        clips = [it for it in items if it is not None]
        clips = [it if isinstance(it, tuple) else (it, None, None) for it in clips]
        if not clips:
            mt_q.put(None)
            late_q.put(None)
            break
        qinfo = {"q_seg": seg_q.qsize(), "q_mt": mt_q.qsize(), "q_late": late_q.qsize(),
                 "batch": len(clips)}
        work.event("asr-batch", olds=max(0, len(clips) - 1), **qinfo)
        *olds, live = clips
        try:
            audio, t0_, t1_ = live
            ja, asr_s, last_ja = asr_decode(
                audio, model, log, work, layer, last_ja, "live", qinfo,
                t_start=t0_, t_end=t1_, seg_id=seg_seq)
            seg_seq += 1
            if ja:
                _put_or_bump(mt_q, (ja, asr_s, seg_seq - 1))
            for audio, t0_, t1_ in olds:
                ja, asr_s, last_ja = asr_decode(
                    audio, model, log, work, layer, last_ja, "late", qinfo,
                    t_start=t0_, t_end=t1_, seg_id=seg_seq)
                seg_seq += 1
                if ja:
                    log.write(f"[late-ja] {ja}   (asr {asr_s:.1f}s)")
                    _put_or_bump(late_q, (ja, asr_s, seg_seq - 1))
        except Exception as e:
            log.write(f"[asr-error] {e}")
            work.event("asr-error", err=str(e))
        if halt:
            mt_q.put(None)
            late_q.put(None)
            break


def mt_loop(mt_q, late_q, result_q, translator, log: LineLog, work: WorkLog):
    def translate_one(item, live: bool):
        # item 是 (ja, asr_s[, seg_id])。seg_id 兼容旧的两元组，缺省为 None。
        if len(item) >= 3:
            ja, asr_s, seg_id = item[0], item[1], item[2]
        else:
            ja, asr_s = item
            seg_id = None
        t0 = time.time()
        zh = translator.translate(ja) if translator else ""
        mt_s = round(time.time() - t0, 3)
        stats = dict(getattr(translator, "last_stats", None) or {})
        # 拒答检测：命中就把这条丢掉，宁可少一行字幕也不给用户看"我是AI我无法…"。
        # 保留中文原文在日志里，方便回看是哪些日文触发了拒答（可能是调 prompt 的线索）。
        refused = is_refusal(zh)
        if refused:
            tag = "[mt-refusal]" if live else "[mt-refusal-late]"
            log.write(f"{tag} {zh}\n      └ {ja}   (asr {asr_s:.1f}s mt {mt_s:.1f}s)")
            zh_shown = ""
        else:
            tag = "[sub]" if live else "[late]"
            log.write(f"{tag} {zh or ja}\n      └ {ja}   (asr {asr_s:.1f}s mt {mt_s:.1f}s)")
            zh_shown = zh
        work.event(
            "mt",
            live=live,
            # seg_id 把同一条内容在 asr 与 mt 两个事件里串起来：
            # 没有它就只能靠 ja 文本做模糊匹配，重复句子会串行。
            seg_id=seg_id,
            ja=ja,
            zh=zh,
            # 翻译质量的可见标记：空译文说明 translator 返回了空串
            # （超时/上下文溢出/被截断），以前只有空字符串躺在日志里没人看。
            zh_empty=(not zh),
            zh_len=len(zh),
            # 拒答单独一个布尔位，不和"正常但长"混在一起：
            # 否则事后统计只能靠翻 zh 文本肉眼认。
            zh_refusal=refused or None,
            asr_s=round(asr_s, 3),
            mt_s=mt_s,
            q_mt=mt_q.qsize(),
            q_late=late_q.qsize(),
            **stats,
        )
        if live and zh_shown:
            _put_or_bump(result_q, (zh_shown, ja, asr_s + mt_s))

    while True:
        got_live = False
        live_item = None
        try:
            live_item = mt_q.get(timeout=0.15)
            got_live = True
        except queue.Empty:
            pass
        if got_live:
            if live_item is None:
                break
            dumped = []
            while True:
                try:
                    nxt = mt_q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    dumped.append(live_item)
                    live_item = None
                    break
                dumped.append(live_item)
                live_item = nxt
            for d in dumped:
                if d is not None:
                    _put_or_bump(late_q, d)
            if live_item is not None:
                try:
                    translate_one(live_item, True)
                except Exception as e:
                    log.write(f"[mt-error] {e}")
                    work.event("mt-error", err=str(e), live=True)
            if live_item is None and not dumped:
                break
            if live_item is None:
                break
            continue
        try:
            late = late_q.get_nowait()
        except queue.Empty:
            continue
        if late is None:
            break
        try:
            translate_one(late, False)
        except Exception as e:
            log.write(f"[mt-error] {e}")
            work.event("mt-error", err=str(e))


# ---------- capture ----------

def enumerate_loopbacks(pya):
    return list(pya.get_loopback_device_info_generator())


# ---------- GUI ----------

class SubtitleWindow:
    def __init__(self, result_q, on_close):
        import tkinter as tk
        self.q = result_q
        self.on_close = on_close
        self.root = tk.Tk()
        self.root.title("asmr-live-sub")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.85)
        w, h = 1120, 116
        self.w = w
        x = self.root.winfo_screenwidth() - w - 50
        y = self.root.winfo_screenheight() - h - 170
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.canvas = tk.Canvas(self.root, width=w, height=h, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.t_now = self.canvas.create_text(w // 2, 42, fill="white", font=("Microsoft YaHei", 21, "bold"), width=w - 40)
        self.t_prev = self.canvas.create_text(w // 2, 88, fill="#9a9a9a", font=("Microsoft YaHei", 12), width=w - 40)
        self.t_status = self.canvas.create_text(10, 6, anchor="w", fill="#4ec94e", font=("Consolas", 9))
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.root.bind("<Escape>", lambda e: self.close())
        self._drag_off = None

    def _press(self, ev):
        self._drag_off = (ev.x_root - self.root.winfo_x(), ev.y_root - self.root.winfo_y())
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.canvas.unbind("<B1-Motion>"))

    def _drag(self, ev):
        if self._drag_off:
            self.root.geometry(f"+{ev.x_root - self._drag_off[0]}+{ev.y_root - self._drag_off[1]}")

    def poll(self):
        status = "listening..."
        try:
            zh, ja, dt = self.q.get_nowait()
            self.canvas.itemconfig(self.t_prev, text=ja)
            self.canvas.itemconfig(self.t_now, text=zh or ja)
            status = f"ok {dt:.1f}s"
        except queue.Empty:
            pass
        self.canvas.itemconfig(self.t_status, text=status)
        self.root.after(120, self.poll)

    def run(self):
        self.root.after(120, self.poll)
        self.root.mainloop()
        self.on_close.set()

    def close(self):
        self.root.destroy()


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small",
                    help="small/medium/large-v3/... 或本地别名 anime(=anime-whisper-ct2)")
    ap.add_argument("--mt", default="sakura", choices=["sakura", "nllb", "qwen", "none"],
                    help="zh translation: sakura=7B CPU, nllb=600M GPU, qwen=1.5B GPU, none=ja only")
    ap.add_argument("--mt-ngl", type=int, default=0, help="llama.cpp n_gpu_layers (0=CPU/RAM only)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--source-audio", help="offline: decode file, transcribe, print zh text")
    ap.add_argument("--replay", action="store_true",
                    help="配合 --source-audio：原速经 Segmenter 回放，产出真实段边界与 seg-drop")
    ap.add_argument("--device-index", type=int, default=None, help="loopback device index (default: system default output)")
    ap.add_argument("--layer", type=int, default=1,
                    help="0=raw, 1=text filters(推荐), 2=+silero VAD(ASMR 上不要用! 见下)")
    ap.add_argument("--duration", type=float, default=None, help="seconds of capture then exit (after models load)")
    ap.add_argument("--log", default=None, help="append [sub]/[drop] lines to this file")
    args = ap.parse_args()
    if args.device == "auto":
        args.device = "cuda" if HAVE_CUDA_DLLS else "cpu"

    # layer>=2 会在每段音频上加 Silero VAD。实测（2026-09-20 真实 A/B，同一场直播）：
    #   layer 2 -> 空识别 82%，字幕 0.97 条/分钟
    #   layer 1 -> 空识别  0%，字幕 4.50 条/分钟
    # 原因是 ASMR 大量为气声/耳语/呼吸，能量集中在 300Hz 以下，
    # Silero 按正常语音训练，把这整类内容判为"没人说话"直接丢弃。
    if args.layer >= 2:
        print("[warn] layer>=2 的 Silero VAD 在 ASMR 上是负收益 "
              "(实测空识别 82% vs layer1 的 0%)，强烈建议改用 --layer 1",
              file=sys.stderr)

    if args.list_devices:
        import pyaudiowpatch as pyaudio
        pya = pyaudio.PyAudio()
        for d in enumerate_loopbacks(pya):
            print(f"  [{d['index']}] {d['name']}  {int(d['defaultSampleRate'])}Hz  ch={d['maxInputChannels']}")
        pya.terminate()
        return

    if args.log:
        log_path = Path(args.log)
    elif args.duration:
        log_path = ROOT / "logs" / f"layer{args.layer}.txt"
    else:
        log_path = ROOT / "logs" / f"live-{time.strftime('%Y%m%d')}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = LineLog(log_path)
    work_path = log_path.parent / f"work-{time.strftime('%Y%m%d')}.jsonl"
    work = WorkLog(work_path)
    log.write(f"[meta] layer={args.layer} model={args.model} mt={args.mt} duration={args.duration} work={work_path.name}")
    work.event("start", layer=args.layer, model=args.model, mt=args.mt,
               ngl=getattr(args, "mt_ngl", 0), duration=args.duration)

    seg_q: queue.Queue = queue.Queue(maxsize=100)
    result_q: queue.Queue = queue.Queue(maxsize=50)

    print(f"[load] model={args.model} device={args.device} layer={args.layer} (模型缓存在 models/，首次自动下载)")
    model = load_model(args)
    translator = load_translator(args)
    mt_q: queue.Queue = queue.Queue(maxsize=2)
    late_q: queue.Queue = queue.Queue(maxsize=200)
    work.event("loaded", asr=args.model, mt=args.mt, translator=type(translator).__name__ if translator else None)
    threading.Thread(target=asr_loop, args=(seg_q, mt_q, late_q, model, log, work, args.layer), daemon=True).start()
    threading.Thread(target=mt_loop, args=(mt_q, late_q, result_q, translator, log, work), daemon=True).start()

    if args.source_audio:
        rate0, chunks = None, []
        import av
        with av.open(args.source_audio) as c:
            stream = c.streams.audio[0]
            rate0 = int(stream.rate) if hasattr(stream, "rate") else None
            for frame in c.decode(audio=0):
                if rate0 is None:
                    rate0 = frame.sample_rate
                x = frame.to_ndarray().astype(np.float32)
                scale = 32768.0 if any(s in frame.format.name for s in ("s16", "s32")) else 1.0
                if x.ndim > 1:
                    x = x.mean(axis=0)
                chunks.append(x / scale)
        full = np.concatenate(chunks)
        if args.replay:
            # 走真正的 Segmenter，模拟实时流：按 100ms 一块喂，原速推进。
            # 与直接 put 一整段不同，这条路径会产生真实的段边界（t_start/t_end）
            # 和 seg-drop 事件，是验证分段/重组参数唯一可信的离线手段。
            print(f"[replay] {args.source_audio} 经 Segmenter 原速回放 ({len(full)/rate0:.1f}s)")
            segr = Segmenter(seg_q, rate0, log=log, work=work)
            chunk_n = int(rate0 * 0.1)
            t_next = time.time()
            for i in range(0, len(full), chunk_n):
                segr.feed(full[i:i + chunk_n])
                t_next += 0.1
                dt = t_next - time.time()
                if dt > 0:
                    time.sleep(dt)
            # 冲掉最后一段
            segr.flush()
            seg_q.put(None)
            # 等管线排空
            deadline = time.time() + 240
            got = 0
            while time.time() < deadline:
                try:
                    zh, en, dt = result_q.get(timeout=2)
                    got += 1
                    print(f"[zh] {zh}\n[ja] {en}   ({dt:.1f}s)")
                except queue.Empty:
                    if got and mt_q.empty() and result_q.empty():
                        break
                    if time.time() > deadline - 230 and not got:
                        break
            print(f"[done] 共 {got} 条字幕")
            log.close()
            work.close()
            return
        # 离线灌流也走三元组：seg_id 之后由 asr_loop 统一分配。
        seg_q.put((resample_to_16k(full, rate0), 0.0, len(full) / rate0))
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                zh, en, dt = result_q.get(timeout=3)
                print(f"[zh] {zh}\n[ja] {en}   ({dt:.1f}s)")
                break
            except queue.Empty:
                continue
        print("[done] 无结果=识别失败或该段无语音")
        log.close()
        work.close()
        return

    import pyaudiowpatch as pyaudio
    pya = pyaudio.PyAudio()
    loops = enumerate_loopbacks(pya)
    if not loops:
        print("[error] 没有可用的 WASAPI 环回设备", file=sys.stderr)
        sys.exit(1)
    if args.device_index is None:
        dev = pya.get_default_wasapi_loopback()
    else:
        dev = next(d for d in loops if d["index"] == args.device_index)
    rate, ch = int(dev["defaultSampleRate"]), max(1, int(dev["maxInputChannels"]))
    print(f"[capture] {dev['name']} @ {rate}Hz ch={ch}")
    buffer = FrameBuffer()
    segmenter = Segmenter(seg_q, rate, log=log, work=work)

    def callback(in_data, frame_count, time_info, status):
        buffer.write(in_data)
        return (None, pyaudio.paContinue)

    stream = pya.open(format=pyaudio.paInt16, channels=ch, rate=rate, input=True,
                      input_device_index=dev["index"], frames_per_buffer=1024,
                      stream_callback=callback)
    stream.start_stream()

    stop = threading.Event()

    def feeder():
        chunk_n = int(rate * 0.1)  # 100ms
        bpf = chunk_n * 2 * ch
        while not stop.is_set():
            data = buffer.read(bpf)
            if len(data) < bpf:
                time.sleep(0.02)
                continue
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                x = x.reshape(-1, ch).mean(axis=1)
            segmenter.feed(x)
        seg_q.put(None)

    threading.Thread(target=feeder, daemon=True).start()
    win = SubtitleWindow(result_q, stop)
    log.write("[run] 正在监听系统音频（播放 asmr.one 即可），Esc 关闭"
              + (f"，{args.duration:.0f}s 后自动停" if args.duration else ""))
    work.event("run", capture="loopback")

    if args.duration:
        def _timeout():
            if stop.wait(args.duration):
                return
            log.write(f"[done] duration {args.duration:.0f}s reached")
            try:
                win.root.after(0, win.close)
            except Exception:
                stop.set()
        threading.Thread(target=_timeout, daemon=True).start()

    win.run()
    stop.set()
    time.sleep(0.3)
    stream.stop_stream()
    stream.close()
    pya.terminate()
    work.event("stop")
    log.close()
    work.close()


if __name__ == "__main__":
    main()
