"""CLI and process composition root (main).

Three modes:
  --list-devices            print WASAPI loopback devices and exit
  --source-audio FILE       offline: decode, transcribe, translate, print
                            (add --replay to push audio through the real
                            Segmenter at original speed instead of one blob)
  (default)                 live WASAPI loopback capture + overlay window

This module owns argument parsing, logging paths, queue wiring and thread
startup — the composition of the other modules. Everything heavy is
imported lazily (av, pyaudiowpatch, tkinter) so offline/eval paths stay cheap.
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from livesub.audio import FrameBuffer, Segmenter, enumerate_loopbacks, resample_to_16k
from livesub.config import HAVE_CUDA_DLLS, ROOT
from livesub.gui import SubtitleWindow
from livesub.logging import LineLog, WorkLog
from livesub.models import load_model, load_translator
from livesub.pipeline import asr_loop, mt_loop


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
    ap.add_argument("--strategy", default="baseline", choices=["baseline", "hardcap", "adaptive"],
                    help="分段策略: baseline=原有逻辑, hardcap=严格物理墙钟上限, adaptive=阶梯衰减切块")
    ap.add_argument("--max-s", type=float, default=8.0, help="单段最大时长上限(秒)")
    ap.add_argument("--hang-s", type=float, default=2.0, help="静音挂起断句时长(秒)")
    ap.add_argument("--fast", action="store_true", help="配合 --replay：不 sleep 原速等待，全速推流加速测评")
    ap.add_argument("--duration", type=float, default=None, help="seconds of capture then exit (after models load)")
    ap.add_argument("--screen", type=int, default=None, help="显示器索引 (0=主屏, 1=外接屏, 默认自动选外接屏若存在)")
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
        work_path = log_path.with_suffix(".jsonl")
    elif args.duration:
        log_path = ROOT / "logs" / f"layer{args.layer}.txt"
        work_path = log_path.parent / f"work-{time.strftime('%Y%m%d')}.jsonl"
    else:
        log_path = ROOT / "logs" / f"live-{time.strftime('%Y%m%d')}.txt"
        work_path = log_path.parent / f"work-{time.strftime('%Y%m%d')}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = LineLog(log_path)
    work = WorkLog(work_path)
    log.write(f"[meta] layer={args.layer} model={args.model} mt={args.mt} strategy={args.strategy} max_s={args.max_s} hang_s={args.hang_s} duration={args.duration} work={work_path.name}")
    work.event("start", layer=args.layer, model=args.model, mt=args.mt,
               strategy=args.strategy, max_s=args.max_s, hang_s=args.hang_s,
               ngl=getattr(args, "mt_ngl", 0), duration=args.duration)

    seg_q: queue.Queue = queue.Queue(maxsize=100)
    result_q: queue.Queue = queue.Queue(maxsize=50)

    print(f"[load] model={args.model} device={args.device} layer={args.layer} strategy={args.strategy} (模型缓存在 models/，首次自动下载)")
    model = load_model(args)
    translator = load_translator(args)
    mt_q: queue.Queue = queue.Queue(maxsize=2)
    late_q: queue.Queue = queue.Queue(maxsize=200)
    mt_done = threading.Event()
    threading.Thread(target=asr_loop, args=(seg_q, mt_q, late_q, model, log, work, args.layer, bool(args.replay)), daemon=True).start()
    threading.Thread(target=mt_loop, args=(mt_q, late_q, result_q, translator, log, work, bool(args.replay), mt_done), daemon=True).start()

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
            print(f"[replay] {args.source_audio} 经 Segmenter 回放 ({len(full)/rate0:.1f}s, fast={args.fast} strategy={args.strategy})")
            segr = Segmenter(seg_q, rate0, log=log, work=work,
                             strategy=args.strategy, max_s=args.max_s, hang_s=args.hang_s)
            chunk_n = int(rate0 * 0.1)
            t_next = time.time()
            for i in range(0, len(full), chunk_n):
                segr.feed(full[i:i + chunk_n])
                if not args.fast:
                    t_next += 0.1
                    dt = t_next - time.time()
                    if dt > 0:
                        time.sleep(dt)
            # 冲掉最后一段
            segr.flush()
            seg_q.put(None)
            # 等管线排空
            deadline = time.time() + 600
            got = 0
            while time.time() < deadline:
                try:
                    zh, en, dt = result_q.get(timeout=0.5)
                    got += 1
                    print(f"[zh] {zh}\n[ja] {en}   ({dt:.1f}s)")
                except queue.Empty:
                    if mt_done.is_set() and result_q.empty():
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
    segmenter = Segmenter(seg_q, rate, log=log, work=work,
                          strategy=args.strategy, max_s=args.max_s, hang_s=args.hang_s)

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
    win = SubtitleWindow(result_q, stop, screen_idx=getattr(args, "screen", None))
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
