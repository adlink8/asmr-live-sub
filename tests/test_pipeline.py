"""UNIT + REGRESSION — livesub.pipeline: asr_decode rules, loop degradation.

Threads are exercised for real (daemon threads + queues) because the loops
take everything as parameters; stubs stand in for whisper/MT.
"""
import queue as q
import threading
import time

import numpy as np
import pytest

from conftest import StubSegment, StubTranslator, StubWhisperModel, events_of
from livesub.pipeline import asr_decode, asr_loop, mt_loop


def decode(model, layer=1, last_ja=None, log=None, work=None, **kw):
    return asr_decode(
        audio=np.zeros(16000, dtype=np.float32),
        model=model, log=log, work=work, layer=layer,
        last_ja=last_ja if last_ja is not None else [],
        role="live", qinfo=kw.pop("qinfo", {}), **kw)


def counting_model(prefix="文"):
    """每次 transcribe 返回不同文本的 stub——避开去重窗口的干扰。"""
    state = {"n": 0}

    def segs(audio):
        state["n"] += 1
        return [StubSegment(f"{prefix}{state['n']}")]

    return StubWhisperModel(segs)


class TestAsrDecodeLayers:
    def test_layer0_keeps_everything(self, logs):
        log, work = logs
        model = StubWhisperModel([StubSegment("はぁ…はぁ…")])
        ja, _, _ = decode(model, layer=0, log=log, work=work)
        assert ja == "はぁ…はぁ…"

    def test_layer1_filters_sound_only(self, logs):
        log, work = logs
        model = StubWhisperModel([StubSegment("はぁ…はぁ…")])
        ja, _, _ = decode(model, layer=1, log=log, work=work)
        assert ja is None
        ev = [e for e in work_events(work, "asr")][0]
        assert ev["drop"] == "empty"
        assert ev["seg_drops"] == ["sound-only"]

    def test_layer1_keeps_speech(self, logs):
        log, work = logs
        model = StubWhisperModel([StubSegment("こんにちは")])
        ja, _, _ = decode(model, layer=1, log=log, work=work)
        assert ja == "こんにちは"

    def test_layer2_adds_vad_parameters(self, logs):
        log, work = logs
        model = StubWhisperModel([StubSegment("こんにちは")])
        decode(model, layer=2, log=log, work=work)
        kw = model.calls[0]["kw"]
        assert kw["vad_filter"] is True
        assert kw["vad_parameters"]["min_silence_duration_ms"] == 2000

    def test_no_suppress_tokens_empty_string(self, logs):
        # 回归：suppress_tokens="" 在 ctranslate2 下会崩；当前实现根本不传该键
        log, work = logs
        model = StubWhisperModel([StubSegment("こんにちは")])
        decode(model, layer=1, log=log, work=work)
        assert "suppress_tokens" not in model.calls[0]["kw"]


class TestAsrDecodeDedup:
    def test_identical_repeat_dropped(self, logs):
        log, work = logs
        model = StubWhisperModel([StubSegment("ありがとう")])
        ja1, _, hist = decode(model, log=log, work=work)
        assert ja1 == "ありがとう"
        ja2, _, hist = decode(model, last_ja=hist, log=log, work=work)
        assert ja2 is None
        ev = [e for e in work_events(work, "asr")][1]
        assert ev["drop"] == "repeat"

    def test_punctuation_variant_also_dropped(self, logs):
        # 回归：窗口存的是去标点核心文本，比较也必须同口径。
        # 旧实现拿带标点的 ja 比核心窗口，"ありがとう" vs "ありがとう。" 漏判。
        log, work = logs
        m1 = StubWhisperModel([StubSegment("ありがとう")])
        _, _, hist = decode(m1, log=log, work=work)
        m2 = StubWhisperModel([StubSegment("ありがとう。")])
        ja, _, _ = decode(m2, last_ja=hist, log=log, work=work)
        assert ja is None

    def test_window_evicts_after_five(self, logs):
        log, work = logs
        texts = ["いち", "に", "さん", "よん", "ご", "ろく"]
        hist = []
        for t in texts:
            ja, _, hist = decode(StubWhisperModel([StubSegment(t)]),
                                 last_ja=hist, log=log, work=work)
            assert ja == t
        # 窗口只留最近 5 条：いち 已掉出窗口，重复应放行
        ja, _, _ = decode(StubWhisperModel([StubSegment("いち")]),
                          last_ja=hist, log=log, work=work)
        assert ja == "いち"


class TestAsrDecodeEvent:
    def test_asr_event_fields(self, logs):
        log, work = logs
        model = StubWhisperModel([StubSegment("こんにちは", avg_logprob=-0.6,
                                              no_speech_prob=0.05)])
        decode(model, seg_id=7, t_start=1.5, t_end=3.0, qinfo={"q_seg": 2},
               log=log, work=work)
        ev = [e for e in work_events(work, "asr")][0]
        for key in ("seg_id", "t_start", "t_end", "audio_s", "rms", "asr_s",
                    "lang", "lang_p", "nseg", "segments", "ja", "drop",
                    "seg_drops", "q_seg"):
            assert key in ev, key
        assert ev["seg_id"] == 7
        assert ev["t_start"] == 1.5
        assert ev["q_seg"] == 2
        assert ev["segments"][0]["avg_logprob"] == -0.6


def work_events(work, kind):
    return events_of(work, kind)


def start_loop(target, args):
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t


def drain(q_, timeout=2.0):
    items = []
    end = time.time() + timeout
    while time.time() < end:
        try:
            items.append(q_.get(timeout=0.1))
        except q.Empty:
            if items:
                break
    return items


class TestAsrLoop:
    def test_live_mode_keeps_only_newest(self, logs):
        log, work = logs
        seg_q, mt_q, late_q = q.Queue(), q.Queue(), q.Queue()
        model = counting_model()
        for i in range(3):
            seg_q.put((np.zeros(16000, dtype=np.float32), float(i), float(i) + 1))
        seg_q.put(None)
        t = start_loop(asr_loop, (seg_q, mt_q, late_q, model, log, work, 1, False))
        t.join(timeout=5)
        assert not t.is_alive()
        live = [x for x in drain(mt_q) if x is not None]
        late = [x for x in drain(late_q) if x is not None]
        assert len(live) == 1          # 只有最新段上屏通路
        assert len(late) == 2          # 旧段下沉 late
        # 口径提醒：live 项是最先被解码的（stub 计数器因此给它文1），
        # 判断"谁是新段"要看 asr 事件的 role 与 t_start。
        asr_evs = work_events(work, "asr")
        live_evs = [e for e in asr_evs if e["role"] == "live"]
        late_evs = [e for e in asr_evs if e["role"] == "late"]
        assert len(live_evs) == 1 and len(late_evs) == 2
        assert live_evs[0]["t_start"] == 2.0  # 最后喂入的段走 live
        assert sorted(e["t_start"] for e in late_evs) == [0.0, 1.0]
        batch = work_events(work, "asr-batch")[0]
        assert batch["olds"] == 2

    def test_replay_mode_is_fifo(self, logs):
        log, work = logs
        seg_q, mt_q, late_q = q.Queue(), q.Queue(), q.Queue()
        model = counting_model()
        for i in range(3):
            seg_q.put((np.zeros(16000, dtype=np.float32), float(i), float(i) + 1))
        seg_q.put(None)
        t = start_loop(asr_loop, (seg_q, mt_q, late_q, model, log, work, 1, True))
        t.join(timeout=5)
        assert not t.is_alive()
        live = [x for x in drain(mt_q) if x is not None]
        assert len(live) == 3
        assert [x[2] for x in live] == [0, 1, 2]  # seg_id 保序
        assert [x[0] for x in live] == ["文1", "文2", "文3"]
        assert work_events(work, "asr-batch")[0]["olds"] == 0

    def test_legacy_bare_array_accepted(self, logs):
        # 兼容：离线灌流直接 put 裸数组（旧两/单元组格式）
        log, work = logs
        seg_q, mt_q, late_q = q.Queue(), q.Queue(), q.Queue()
        model = StubWhisperModel([StubSegment("生声")])
        seg_q.put(np.zeros(16000, dtype=np.float32))
        seg_q.put(None)
        t = start_loop(asr_loop, (seg_q, mt_q, late_q, model, log, work, 1, False))
        t.join(timeout=5)
        live = [x for x in drain(mt_q) if x is not None]
        assert len(live) == 1

    def test_halt_propagates_none_to_both_queues(self, logs):
        log, work = logs
        seg_q, mt_q, late_q = q.Queue(), q.Queue(), q.Queue()
        seg_q.put(None)
        t = start_loop(asr_loop, (seg_q, mt_q, late_q, StubWhisperModel(),
                                  log, work, 1, False))
        t.join(timeout=5)
        assert not t.is_alive()
        assert mt_q.get_nowait() is None
        assert late_q.get_nowait() is None

    def test_model_exception_logged_loop_survives(self, logs):
        log, work = logs

        class Boom:
            def transcribe(self, audio, **kw):
                raise RuntimeError("asr boom")

        seg_q, mt_q, late_q = q.Queue(), q.Queue(), q.Queue()
        seg_q.put((np.zeros(16000, dtype=np.float32), 0.0, 1.0))
        seg_q.put(None)
        t = start_loop(asr_loop, (seg_q, mt_q, late_q, Boom(), log, work, 1, False))
        t.join(timeout=5)
        assert not t.is_alive()
        errs = work_events(work, "asr-error")
        assert errs and "asr boom" in errs[0]["err"]


class TestMtLoop:
    def test_normal_translation_reaches_result_q(self, logs):
        log, work = logs
        mt_q, late_q, result_q, done = q.Queue(), q.Queue(), q.Queue(), threading.Event()
        mt_q.put(("こんにちは", 0.4, 3))
        mt_q.put(None)
        t = start_loop(mt_loop, (mt_q, late_q, result_q, StubTranslator(),
                                 log, work, False, done))
        t.join(timeout=5)
        assert done.is_set()
        item = result_q.get_nowait()
        zh, ja, dt = item
        assert zh == "译:こんにちは"
        assert ja == "こんにちは"
        ev = work_events(work, "mt")[0]
        assert ev["seg_id"] == 3
        assert ev["zh_empty"] is False
        assert ev["zh_refusal"] is None
        assert ev["asr_s"] == 0.4

    def test_result_dt_is_asr_plus_mt_only(self, logs):
        # 契约：dt = asr_s + mt_s，不含段长（端到端延迟要另外加段长）
        log, work = logs
        mt_q, late_q, result_q, done = q.Queue(), q.Queue(), q.Queue(), threading.Event()
        mt_q.put(("文", 1.0, 0))
        mt_q.put(None)
        t = start_loop(mt_loop, (mt_q, late_q, result_q, StubTranslator(),
                                 log, work, False, done))
        t.join(timeout=5)
        zh, ja, dt = result_q.get_nowait()
        ev = work_events(work, "mt")[0]
        assert dt == pytest.approx(1.0 + ev["mt_s"], abs=0.05)

    def test_refusal_dropped_not_shown(self, logs):
        log, work = logs
        mt_q, late_q, result_q, done = q.Queue(), q.Queue(), q.Queue(), threading.Event()
        tr = StubTranslator({"x": "作为一个AI语言模型，我无法翻译这段内容"})
        mt_q.put(("x", 0.1, 0))
        mt_q.put(None)
        t = start_loop(mt_loop, (mt_q, late_q, result_q, tr, log, work, False, done))
        t.join(timeout=5)
        assert result_q.empty()
        ev = work_events(work, "mt")[0]
        assert ev["zh_refusal"] is True

    def test_two_tuple_item_compat(self, logs):
        # 兼容旧的两元组 (ja, asr_s)，seg_id 缺省 None
        log, work = logs
        mt_q, late_q, result_q, done = q.Queue(), q.Queue(), q.Queue(), threading.Event()
        mt_q.put(("文", 0.2))
        mt_q.put(None)
        t = start_loop(mt_loop, (mt_q, late_q, result_q, StubTranslator(),
                                 log, work, False, done))
        t.join(timeout=5)
        zh, ja, dt = result_q.get_nowait()
        assert zh == "译:文"
        assert work_events(work, "mt")[0]["seg_id"] is None

    def test_backlog_keeps_newest_bumps_rest_to_late(self, logs):
        log, work = logs
        mt_q, late_q, result_q, done = q.Queue(), q.Queue(), q.Queue(), threading.Event()
        tr = StubTranslator()
        for i in range(3):
            mt_q.put((f"文{i}", 0.1, i))
        t = start_loop(mt_loop, (mt_q, late_q, result_q, tr, log, work, False, done))
        time.sleep(1.0)
        mt_q.put(None)
        t.join(timeout=5)
        shown = []
        while not result_q.empty():
            shown.append(result_q.get_nowait())
        assert len(shown) == 1
        assert shown[0][1] == "文2"  # 只有最新段上屏
        # 旧两段被下沉 late 并在后台翻译（live=False），只进日志不上屏
        mts = work_events(work, "mt")
        live_mts = [e for e in mts if e["live"]]
        late_mts = [e for e in mts if not e["live"]]
        assert [e["ja"] for e in live_mts] == ["文2"]
        assert sorted(e["ja"] for e in late_mts) == ["文0", "文1"]

    def test_sentinel_right_after_item_still_shows_it(self, logs):
        # 回归：关机哨兵 None 紧跟在 live 项后面入队时，该项必须照常翻译上屏。
        # 旧实现在此把 live 项丢进 late_q 后直接 break——最后一条字幕静默丢失。
        log, work = logs
        mt_q, late_q, result_q, done = q.Queue(), q.Queue(), q.Queue(), threading.Event()
        mt_q.put(("最後の文", 0.3, 9))
        mt_q.put(None)
        t = start_loop(mt_loop, (mt_q, late_q, result_q, StubTranslator(),
                                 log, work, False, done))
        t.join(timeout=5)
        assert not result_q.empty()
        assert result_q.get_nowait()[0] == "译:最後の文"

    def test_late_items_never_shown(self, logs):
        log, work = logs
        mt_q, late_q, result_q, done = q.Queue(), q.Queue(), q.Queue(), threading.Event()
        late_q.put(("旧文", 0.1, 0))
        late_q.put(None)
        t = start_loop(mt_loop, (mt_q, late_q, result_q, StubTranslator(),
                                 log, work, False, done))
        t.join(timeout=5)
        assert result_q.empty()
        ev = work_events(work, "mt")[0]
        assert ev["live"] is False

    def test_translate_exception_live_logged(self, logs):
        log, work = logs

        class Boom:
            last_stats = {}

            def translate(self, text):
                raise RuntimeError("mt boom")

        mt_q, late_q, result_q, done = q.Queue(), q.Queue(), q.Queue(), threading.Event()
        mt_q.put(("一", 0.1, 0))
        mt_q.put(None)
        t = start_loop(mt_loop, (mt_q, late_q, result_q, Boom(),
                                 log, work, False, done))
        t.join(timeout=5)
        errs = work_events(work, "mt-error")
        assert errs and "mt boom" in errs[0]["err"]
        assert errs[0]["live"] is True
        assert done.is_set()  # 异常后循环仍能干净退出

    def test_translate_exception_late_path_continues(self, logs):
        # late 通路抛异常不该杀死循环：坏项记日志，后续项继续处理
        log, work = logs

        class Flaky:
            last_stats = {}
            calls = 0

            def translate(self, text):
                Flaky.calls += 1
                if Flaky.calls == 1:
                    raise RuntimeError("late boom")
                return "好"

        mt_q, late_q, result_q, done = q.Queue(), q.Queue(), q.Queue(), threading.Event()
        late_q.put(("坏", 0.1, 0))
        late_q.put(("好", 0.1, 1))
        late_q.put(None)
        t = start_loop(mt_loop, (mt_q, late_q, result_q, Flaky(),
                                 log, work, False, done))
        t.join(timeout=5)
        errs = work_events(work, "mt-error")
        assert errs and "late boom" in errs[0]["err"]
        assert errs[0].get("live") in (None, False)
        # 抛异常的项在 translate 阶段就炸了，不产生 mt 事件，只留 mt-error；
        # 后续好项照常产生 mt 事件——循环没有被杀死
        mts = work_events(work, "mt")
        assert len(mts) == 1
        assert mts[0]["zh"] == "好"

    def test_none_translator_shows_nothing(self, logs):
        # 已知歧义（勿改）：--mt none 帮助文本说 "ja only"，但当前实现
        # zh_shown 为空就不上屏，字幕窗反而什么都不显示。
        # GUI 里有 zh or ja 的回落却永远等不到空 zh 的 item。
        # 此处锁定现状；是否改成显示日文待用户决策。
        log, work = logs
        mt_q, late_q, result_q, done = q.Queue(), q.Queue(), q.Queue(), threading.Event()
        mt_q.put(("こんにちは", 0.1, 0))
        mt_q.put(None)
        t = start_loop(mt_loop, (mt_q, late_q, result_q, None,
                                 log, work, False, done))
        t.join(timeout=5)
        assert result_q.empty()
