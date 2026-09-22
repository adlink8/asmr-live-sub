"""Logging: human-readable console/file lines (LineLog) and the JSONL model
trace (WorkLog). WorkLog records every pipeline decision so tuning and
post-hoc analysis never have to guess.
"""
import json
import threading
import time
from pathlib import Path


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
