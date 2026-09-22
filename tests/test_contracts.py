"""CONTRACT — cross-module interface shapes.

These tests don't check behavior; they check that the shape of every
exchange stays stable, because downstream consumers depend on it:

  * JSONL event fields  -> tools/evaluate_accuracy.py / tools/run_*_eval.py /
    all the post-hoc analysis in .workbuddy/memory reads these keys by name
  * queue item shapes   -> Segmenter -> asr_loop -> mt_loop -> GUI
  * MT backend protocol -> any translator with translate()+last_stats works
  * live_sub compat surface -> tools/build_teacher_groundtruth.py and friends
"""
import json
import queue as q
import subprocess
import sys
import threading

import numpy as np
import pytest

from conftest import (StubSegment, StubTranslator, StubWhisperModel,
                      events_of, make_frames_audio)
from livesub.audio import Segmenter
from livesub.config import ROOT
from livesub.models import (LOCAL_ASR_MODELS, NllbMT, QwenMT, SakuraMT,
                            load_translator)
from livesub.pipeline import asr_decode, asr_loop, mt_loop


# ------------------------------------------------------------- queue shapes

class TestSegQueueContract:
    def test_item_is_audio_boundaries_triple(self, tmp_path):
        seg_q: q.Queue = q.Queue()
        from livesub.logging import WorkLog
        work = WorkLog(tmp_path / "w.jsonl")
        seg = Segmenter(seg_q, 16000, work=work)
        seg.feed(make_frames_audio("vvvsss"))
        seg.flush()
        work.close()
        item = seg_q.get_nowait()
        assert isinstance(item, tuple) and len(item) == 3
        audio, t_start, t_end = item
        assert audio.dtype == np.float32
        assert isinstance(t_start, float) and isinstance(t_end, float)
        assert 0.0 <= t_start <= t_end
        # audio_s 与段边界自洽：剪掉尾部静音后 audio_s <= t_end - t_start
        audio_s = len(audio) / 16000
        assert audio_s <= (t_end - t_start) + 1e-6

    def test_legacy_bare_array_still_accepted(self, tmp_path):
        # 契约：asr_loop 兼容裸数组入队（旧离线灌流格式）
        from livesub.logging import LineLog, WorkLog
        log, work = LineLog(None), WorkLog(tmp_path / "w.jsonl")
        seg_q, mt_q, late_q = q.Queue(), q.Queue(), q.Queue()
        model = StubWhisperModel([StubSegment("生声")])
        seg_q.put(np.zeros(16000, dtype=np.float32))
        seg_q.put(None)
        t = threading.Thread(target=asr_loop,
                             args=(seg_q, mt_q, late_q, model, log, work, 1, False),
                             daemon=True)
        t.start()
        t.join(timeout=5)
        work.close()
        item = mt_q.get_nowait()
        assert item is not None and len(item) == 3  # (ja, asr_s, seg_id)


class TestMtQueueContract:
    def test_item_is_ja_asr_segid_triple(self, tmp_path):
        from livesub.logging import LineLog, WorkLog
        log, work = LineLog(None), WorkLog(tmp_path / "w.jsonl")
        seg_q, mt_q, late_q = q.Queue(), q.Queue(), q.Queue()
        seg_q.put((np.zeros(16000, dtype=np.float32), 0.0, 1.0))
        seg_q.put(None)
        t = threading.Thread(target=asr_loop,
                             args=(seg_q, mt_q, late_q,
                                   StubWhisperModel([StubSegment("文")]),
                                   log, work, 1, False), daemon=True)
        t.start()
        t.join(timeout=5)
        work.close()
        ja, asr_s, seg_id = mt_q.get_nowait()
        assert ja == "文"
        assert isinstance(asr_s, float)
        assert isinstance(seg_id, int)

    def test_result_item_is_zh_ja_dt_triple(self, tmp_path):
        from livesub.logging import LineLog, WorkLog
        log, work = LineLog(None), WorkLog(tmp_path / "w.jsonl")
        mt_q, late_q, result_q = q.Queue(), q.Queue(), q.Queue()
        mt_q.put(("文", 0.5, 1))
        mt_q.put(None)
        t = threading.Thread(target=mt_loop,
                             args=(mt_q, late_q, result_q, StubTranslator(),
                                   log, work, False, threading.Event()),
                             daemon=True)
        t.start()
        t.join(timeout=5)
        work.close()
        item = result_q.get_nowait()
        assert len(item) == 3
        zh, ja, dt = item
        assert zh == "译:文" and ja == "文"
        mt_ev = events_of(work, "mt")[0]
        assert dt == pytest.approx(0.5 + mt_ev["mt_s"], abs=0.05)


# ------------------------------------------------------------- event schema

REQUIRED_FIELDS = {
    "seg-drop": {"reason", "voiced_s", "audio_s", "t_start", "t_end", "rms"},
    "gate": {"mode", "thr", "p50", "peak", "rms"},
    "asr-batch": {"olds", "q_seg", "q_mt", "q_late", "batch"},
    "asr": {"role", "seg_id", "t_start", "t_end", "audio_s", "rms", "asr_s",
            "lang", "lang_p", "nseg", "segments", "ja", "drop", "seg_drops",
            "q_seg", "q_mt", "q_late", "batch"},
    "mt": {"live", "seg_id", "ja", "zh", "zh_empty", "zh_len", "zh_refusal",
           "asr_s", "mt_s", "q_mt", "q_late"},
}


def collect_all_kinds(tmp_path):
    """Drive each producer minimally and return {kind: [events]}."""
    from livesub.logging import LineLog, WorkLog
    log, work = LineLog(None), WorkLog(tmp_path / "w.jsonl")
    out = {}

    # seg-drop + gate: Segmenter with a loud prefix, silence to close that
    # segment, then a short blip that must be dropped on its own
    seg_q: q.Queue = q.Queue()
    seg = Segmenter(seg_q, 16000, log=log, work=work)
    seg.feed(make_frames_audio("vvvv", amp=0.5))   # -> gate loud
    seg.feed(make_frames_audio("sss"))             # 收掉第一段（2.04s, 保留）
    seg.feed(make_frames_audio("vsss"))            # -> seg-drop too-short
    seg.flush()

    # asr + asr-batch: one live clip through asr_loop
    seg_q2, mt_q, late_q = q.Queue(), q.Queue(), q.Queue()
    seg_q2.put((np.zeros(16000, dtype=np.float32), 0.0, 1.0))
    seg_q2.put(None)
    t = threading.Thread(target=asr_loop,
                         args=(seg_q2, mt_q, late_q,
                               StubWhisperModel([StubSegment("契約テスト")]),
                               log, work, 1, False), daemon=True)
    t.start()
    t.join(timeout=5)

    # mt: one translation through mt_loop
    mt_q2, late_q2, result_q = q.Queue(), q.Queue(), q.Queue()
    mt_q2.put(("契約テスト", 0.3, 0))
    mt_q2.put(None)
    t2 = threading.Thread(target=mt_loop,
                          args=(mt_q2, late_q2, result_q, StubTranslator(),
                                log, work, False, threading.Event()),
                          daemon=True)
    t2.start()
    t2.join(timeout=5)
    log.close()
    work.close()

    out = {}
    for line in work.path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        out.setdefault(rec["kind"], []).append(rec)
    return out


class TestEventSchema:
    def test_every_event_has_ts_and_kind(self, tmp_path):
        kinds = collect_all_kinds(tmp_path)
        for kind, evs in kinds.items():
            for ev in evs:
                assert "ts" in ev and ev["kind"] == kind

    @pytest.mark.parametrize("kind", sorted(REQUIRED_FIELDS))
    def test_required_fields_present(self, tmp_path, kind):
        kinds = collect_all_kinds(tmp_path)
        assert kind in kinds, f"no {kind} event produced"
        ev = kinds[kind][0]
        missing = REQUIRED_FIELDS[kind] - set(ev)
        assert not missing, f"{kind} missing fields: {missing}"

    def test_every_line_is_valid_json(self, tmp_path):
        collect_all_kinds(tmp_path)
        for line in (tmp_path / "w.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)  # raises if corrupted/interleaved
            assert isinstance(rec, dict)


# ------------------------------------------------------------- MT protocol

class TestMtBackendProtocol:
    @pytest.mark.parametrize("cls", [SakuraMT, NllbMT, QwenMT])
    def test_backend_exposes_translate(self, cls):
        # 类级检查：不加载权重，只验证协议成员存在
        assert callable(getattr(cls, "translate", None))

    def test_minimal_duck_typed_translator_suffices(self, tmp_path):
        # 管线只碰 translate() 和 last_stats 两个成员——任何实现这两个的
        # 对象都能当 MT 后端（这就是 models.py 里三个后端能互换的原因）
        from livesub.logging import LineLog, WorkLog

        class Minimal:
            def __init__(self):
                self.last_stats = {"custom": 1}

            def translate(self, text):
                return f"[min]{text}"

        log, work = LineLog(None), WorkLog(tmp_path / "w.jsonl")
        mt_q, late_q, result_q = q.Queue(), q.Queue(), q.Queue()
        mt_q.put(("文", 0.1, 0))
        mt_q.put(None)
        t = threading.Thread(target=mt_loop,
                             args=(mt_q, late_q, result_q, Minimal(),
                                   log, work, False, threading.Event()),
                             daemon=True)
        t.start()
        t.join(timeout=5)
        work.close()
        assert result_q.get_nowait()[0] == "[min]文"
        ev = events_of(work, "mt")[0]
        assert ev["custom"] == 1  # last_stats 被合并进事件

    def test_load_translator_none_is_none(self):
        # --mt none 不加载任何模型
        class Args:
            mt = "none"
        assert load_translator(Args()) is None

    def test_asr_alias_table_shape(self):
        # --model anime 别名表：指向 ROOT/models 下的本地 CT2 目录
        assert "anime" in LOCAL_ASR_MODELS
        assert str(LOCAL_ASR_MODELS["anime"]).startswith(str(ROOT))
        assert LOCAL_ASR_MODELS["anime"].name == "anime-whisper-ct2"


# ------------------------------------------------------------- CLI surface

COMPAT_NAMES = [
    "main", "load_model", "load_translator", "SakuraMT", "NllbMT", "QwenMT",
    "Segmenter", "FrameBuffer", "resample_to_16k", "enumerate_loopbacks",
    "asr_loop", "mt_loop", "asr_decode", "drain_all",
    "drop_reason", "is_refusal", "is_kana_loop",
    "LineLog", "WorkLog", "SubtitleWindow",
    "ROOT", "TARGET_SR", "MIN_SEG_S", "LOWCONF_MIN_S", "LOWCONF_LOGPROB",
    "HAVE_CUDA_DLLS",
]


class TestCompatSurface:
    def test_live_sub_shim_exposes_legacy_names(self):
        import live_sub
        missing = [n for n in COMPAT_NAMES if not hasattr(live_sub, n)]
        assert not missing, f"live_sub missing: {missing}"

    def test_package_all_matches_exports(self):
        import livesub
        for name in livesub.__all__:
            assert hasattr(livesub, name), name

    def test_layer2_warning_printed(self):
        # layer>=2 必须在启动时大声警告（ASMR 上是负收益）
        r = subprocess.run(
            [sys.executable, "live_sub.py", "--layer", "2", "--list-devices"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=120)
        assert "layer>=2" in r.stderr
