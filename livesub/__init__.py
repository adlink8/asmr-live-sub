"""asmr-live-sub: WASAPI loopback -> faster-whisper (ja) -> MT (zh) -> overlay window.

Importing this package registers the CUDA DLL search paths (livesub.config),
which must happen before torch / faster_whisper / llama_cpp / ctranslate2
are imported — those are all imported lazily inside functions, so importing
this package never loads a model and never touches the GPU.

Module map (see docs/module_map.md):
  config    paths, env vars, tunables, CUDA DLL registration
  logging   LineLog (console/file) + WorkLog (JSONL trace)
  filters   text-layer drop rules (hallucination / kana-loop / low-conf / ...)
  audio     frame buffer, resampler, energy-gate Segmenter, device enum
  models    ASR loader + MT backends (Sakura / NLLB / Qwen)
  pipeline  asr_loop / mt_loop threads and per-segment decode
  gui       tkinter overlay window
  cli       argparse + composition root (main)
"""
from livesub import config  # noqa: F401  — import order matters: registers CUDA DLL paths first

from livesub.audio import (  # noqa: F401
    FrameBuffer,
    Segmenter,
    enumerate_loopbacks,
    resample_to_16k,
)
from livesub.config import (  # noqa: F401
    HAVE_CUDA_DLLS,
    LOWCONF_LOGPROB,
    LOWCONF_MIN_S,
    MIN_SEG_S,
    ROOT,
    TARGET_SR,
)
from livesub.filters import drop_reason, is_refusal, is_kana_loop  # noqa: F401
from livesub.gui import SubtitleWindow  # noqa: F401
from livesub.logging import LineLog, WorkLog  # noqa: F401
from livesub.models import (  # noqa: F401
    NllbMT,
    QwenMT,
    SakuraMT,
    load_model,
    load_translator,
)
from livesub.pipeline import asr_decode, asr_loop, drain_all, mt_loop  # noqa: F401
from livesub.cli import main  # noqa: F401

__all__ = [
    # entry point
    "main",
    # models
    "load_model", "load_translator", "SakuraMT", "NllbMT", "QwenMT",
    # pipeline core
    "Segmenter", "FrameBuffer", "resample_to_16k", "enumerate_loopbacks",
    "asr_loop", "mt_loop", "asr_decode",
    # filters & logging
    "drop_reason", "is_refusal", "is_kana_loop",
    "LineLog", "WorkLog",
    # GUI
    "SubtitleWindow",
    # constants
    "ROOT", "TARGET_SR", "MIN_SEG_S", "LOWCONF_MIN_S", "LOWCONF_LOGPROB",
    "HAVE_CUDA_DLLS",
]
