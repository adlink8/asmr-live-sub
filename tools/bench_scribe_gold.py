"""ElevenLabs Scribe API 转写实测（金标窗）。key 从 ~/.zcode/elevenlabs.key 读，
严禁写进仓库。2026-09-26 首测：v1=v2=25.8%（anime-whisper 16.3%，large-v3 32.8%）。
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from bench_large_v3_local import cer  # noqa: E402

KEY_PATH = Path.home() / ".zcode" / "elevenlabs.key"
WAV = ROOT / "benchmarks" / "scene_random_talk.wav"
GT = ROOT / "dataset" / "scene_random_talk.gt_ja.json"


def scribe(model_id, wav):
    key = KEY_PATH.read_text().strip()
    boundary = "----zbench"
    data = wav.read_bytes()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="file"; filename="{wav.name}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n").encode() + data + \
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model_id\"\r\n\r\n{model_id}\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/speech-to-text", data=body, method="POST",
        headers={"xi-api-key": key,
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main():
    gt = json.loads(GT.read_text(encoding="utf-8"))
    gt = gt if isinstance(gt, list) else gt.get("segments", [])
    ref_all = "".join(s["official_ja"] for s in gt)
    for mid in ("scribe_v1", "scribe_v2"):
        try:
            d = scribe(mid, WAV)
            hyp = d.get("text", "")
            print(f"{mid}: CER {cer(ref_all, hyp)*100:.1f}% "
                  f"| 字数 {len(hyp)} | 计费 {d.get('audio_duration_secs')}s")
        except Exception as e:  # noqa: BLE001
            print(f"{mid}: FAIL {e}")


if __name__ == "__main__":
    main()
