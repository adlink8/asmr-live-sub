"""Pipeline threads: ASR decode + filtering (asr_loop) and translation (mt_loop).

Both loops are plain functions taking their queues, models and loggers as
parameters — no globals, no hidden state — so the offline replay path
(--source-audio --replay) runs the exact same code as live capture.

Degradation policy (by design, see logs for trigger counts):
  asr_loop  drains the whole seg_q and transcribes only the newest clip
            "live"; older clips go to late_q and are never shown.
  mt_loop   keeps only the newest mt_q item; backlog is bumped to late_q.
late_q content is logged but never displayed — the overlay always shows the
freshest line rather than a stale queue.
"""
import queue
import threading
import time

import numpy as np

from livesub.audio import _put_or_bump
from livesub.config import TARGET_SR
from livesub.filters import drop_reason, is_refusal, strip_punct
from livesub.logging import LineLog, WorkLog


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
        elif strip_punct(ja) in last_ja:
            # 原实现只跟"上一条"比，导致 ありがとう 这类真实短句连续出现会被全部放行。
            # last_ja 现在是最近 N 条的历史窗口。
            # 比较必须同口径：窗口里存的是去标点的核心文本，所以 ja 也要去了标点再比
            # （原实现拿带标点的 ja 比核心窗口，"ありがとう" vs "ありがとう。" 漏判）。
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
    core_new = strip_punct(ja)
    if core_new and core_new not in hist:
        hist = (hist + [core_new])[-5:]
    return ja, asr_s, hist


def asr_loop(seg_q, mt_q, late_q, model, log: LineLog, work: WorkLog, layer: int, replay: bool = False):
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
            if halt:
                mt_q.put(None)
                late_q.put(None)
                break
            continue
        qinfo = {"q_seg": seg_q.qsize(), "q_mt": mt_q.qsize(), "q_late": late_q.qsize(),
                 "batch": len(clips)}
        work.event("asr-batch", olds=0 if replay else max(0, len(clips) - 1), **qinfo)

        if replay:
            # 测评/回放模式：严格保序 FIFO，逐段送入 MT，绝不把早前段当做 olds 丢弃或颠倒顺序
            for audio, t0_, t1_ in clips:
                try:
                    ja, asr_s, last_ja = asr_decode(
                        audio, model, log, work, layer, last_ja, "live", qinfo,
                        t_start=t0_, t_end=t1_, seg_id=seg_seq)
                    seg_seq += 1
                    if ja:
                        mt_q.put((ja, asr_s, seg_seq - 1))
                except Exception as e:
                    log.write(f"[asr-error] {e}")
                    work.event("asr-error", err=str(e))
        else:
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


def mt_loop(mt_q, late_q, result_q, translator, log: LineLog, work: WorkLog, replay: bool = False, done_event: threading.Event = None, stream_q: queue.Queue = None):
    try:
        def translate_one(item, live: bool):
            # item 是 (ja, asr_s[, seg_id])。seg_id 兼容旧的两元组，缺省为 None。
            if len(item) >= 3:
                ja, asr_s, seg_id = item[0], item[1], item[2]
            else:
                ja, asr_s = item
                seg_id = None
            t0 = time.time()

            def on_token(piece):
                if stream_q is not None and live:
                    _put_or_bump(stream_q, piece)

            if translator:
                try:
                    zh = translator.translate(ja, on_token=on_token)
                except TypeError:
                    zh = translator.translate(ja)
            else:
                zh = ""

            if stream_q is not None and live:
                _put_or_bump(stream_q, None)

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
                seg_id=seg_id,
                ja=ja,
                zh=zh,
                zh_empty=(not zh),
                zh_len=len(zh),
                zh_refusal=refused or None,
                asr_s=round(asr_s, 3),
                mt_s=mt_s,
                q_mt=mt_q.qsize(),
                q_late=late_q.qsize(),
                **stats,
            )
            if (live or replay) and zh_shown:
                _put_or_bump(result_q, (zh_shown, ja, asr_s + mt_s))

        if replay:
            # 测评/回放模式：顺序消费 mt_q，不主动丢弃或下沉
            while True:
                try:
                    item = mt_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    break
                try:
                    translate_one(item, True)
                except Exception as e:
                    log.write(f"[mt-error] {e}")
                    work.event("mt-error", err=str(e), live=True)
            return

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
                halt = False
                while True:
                    try:
                        nxt = mt_q.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        # 关机哨兵紧随后台项到达：当前 live 项仍是最新的，
                        # 必须翻译完再退。旧实现把它丢进 late_q 后直接 break，
                        # 最后一条 live 字幕会既不翻译也不上屏（静默丢失）。
                        halt = True
                        break
                    dumped.append(live_item)
                    live_item = nxt
                for d in dumped:
                    if d is not None:
                        _put_or_bump(late_q, d)
                try:
                    translate_one(live_item, True)
                except Exception as e:
                    log.write(f"[mt-error] {e}")
                    work.event("mt-error", err=str(e), live=True)
                if halt:
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
    finally:
        if done_event is not None:
            done_event.set()
