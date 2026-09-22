"""Poll a live_sub log. stdout is only DONE / FAILED / ACTION_REQUIRED."""
import argparse
import os
import time
from pathlib import Path


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("pid", type=int)
    ap.add_argument("--load-s", type=int, default=180)
    ap.add_argument("--audio-s", type=int, default=180)
    args = ap.parse_args()
    log = Path(args.log)
    t0 = time.time()
    run_t = None
    warned_audio = False
    diag = Path(args.log + ".watch")

    def dump(tag):
        txt = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        diag.write_text(txt[-4000:], encoding="utf-8")
        return txt

    while True:
        txt = dump("poll")
        dead = not alive(args.pid)
        if "CUDA out of memory" in txt or "[asr-error]" in txt and "CUDA" in txt:
            print("FAILED: CUDA error")
            return 1
        if "[done] duration" in txt:
            print("DONE: duration reached")
            return 0
        if "[run]" in txt and run_t is None:
            run_t = time.time()
        if run_t is None and time.time() - t0 > args.load_s:
            if dead:
                print("FAILED: process died before [run]")
                return 1
            print("FAILED: models did not reach [run] in time")
            return 1
        if run_t is not None and not warned_audio:
            after = txt[txt.find("[run]"):]
            if "[sub]" in after or "[drop]" in after:
                warned_audio = True
            elif time.time() - run_t > args.audio_s:
                print("ACTION_REQUIRED: [run] but no [sub]/[drop] — play ASMR into the loopback device")
                warned_audio = True
        if dead and "[done] duration" not in txt:
            print("FAILED: process exited early")
            return 1
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
