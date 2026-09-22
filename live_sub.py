"""Entry point and backward-compatibility shim.

The implementation lives in the livesub package (see docs/module_map.md);
this file stays because scripts such as build_teacher_groundtruth.py do
`import live_sub` and reach for load_model / SakuraMT directly, and because
start.bat invokes `python live_sub.py`.

Importing this module also performs the CUDA DLL registration (via
livesub.config) — required before importing faster_whisper/llama_cpp in
ad-hoc experiment scripts:
    import live_sub
    model = live_sub.load_model(args)
"""
from livesub import (  # noqa: F401  — backward-compatible re-exports
    HAVE_CUDA_DLLS,
    LOWCONF_LOGPROB,
    LOWCONF_MIN_S,
    MIN_SEG_S,
    ROOT,
    TARGET_SR,
    FrameBuffer,
    LineLog,
    NllbMT,
    QwenMT,
    SakuraMT,
    Segmenter,
    SubtitleWindow,
    WorkLog,
    asr_decode,
    asr_loop,
    drain_all,
    drop_reason,
    enumerate_loopbacks,
    is_kana_loop,
    is_refusal,
    load_model,
    load_translator,
    main,
    mt_loop,
    resample_to_16k,
)

if __name__ == "__main__":
    main()
