"""Process-wide setup: paths, environment variables, tunables, CUDA DLL registration.

Every other module in the package imports this one (directly or transitively),
and livesub/__init__ imports it first. That ordering is load-bearing: the
environment variables and the CUDA DLL search paths must be in place before
any model library (torch / faster_whisper / llama_cpp / ctranslate2) is
imported. All model libraries are imported lazily inside functions/methods,
so importing this package never loads a model.
"""
import ctypes
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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
    site = ROOT / ".venv" / "Lib" / "site-packages"
    dirs = [
        site / "nvidia" / "cublas" / "bin",
        site / "nvidia" / "cudnn" / "bin",
        site / "torch" / "lib",
        site / "llama_cpp" / "lib",
    ]
    handles = []
    ctypes.windll.kernel32.SetDefaultDllDirectories(0x1000)
    for d in dirs:
        if d.is_dir():
            handles.append(os.add_dll_directory(str(d)))
    _add_cuda_dlls.handles = handles  # keep alive: closing removes dir from search
    return len(handles) > 0


HAVE_CUDA_DLLS = _add_cuda_dlls()
