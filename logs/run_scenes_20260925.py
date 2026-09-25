# -*- coding: utf-8 -*-
"""2026-09-25 锚点场景扩建：串行 GPU 实跑 22 个新场景。
纪律：绝不并行、不加 --fast；每次跑前删同名 .txt/.jsonl（WorkLog 追加模式）；
跑前检查显存（空闲 <5000MiB 视为被占用 → 等 2 分钟重试，最多 3 次后跳过）。
注：设计文档写 ≥6GB，但本机桌面常态占用 ~2.1GB（总 8GB → 空闲 ~5.8GB），
历史成功跑也是在 ~5.1GB 空闲下完成的，故以 5000MiB 为"被占用"判定线，
偏差已记录在最终报告。
"""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(r"D:/ADLINK/asmr-live-sub")
PY = ROOT / ".venv" / "Scripts" / "python.exe"
RUNS = ROOT / "logs" / "benchmark_runs"
STD180 = Path(r"D:/Downloads/asmr-zh-corpus/std180s")

SCENES = [
    "scene_asmr1521586_t01",
    "scene_asmr1521586_t01_w2",
    "scene_asmr1521586_t01_w3",
]
RESUME_ONLY = True
_SCENES_FULL = [
    "scene_asmr299717_t02_w2",
    "scene_asmr416809_t01_w2",
    "scene_asmr416809_t01_w3",
    "scene_asmr416816_t01_w3",
    "scene_asmr1449384_t03",
    "scene_asmr1449384_t03_w2",
    "scene_asmr1449384_t03_w3",
    "scene_asmr1463510_t05",
    "scene_asmr1463510_t05_w2",
    "scene_asmr1463510_t05_w3",
    "scene_asmr324799_t03",
    "scene_asmr324799_t03_w2",
    "scene_asmr324799_t03_w3",
    "scene_asmr1497366_t07",
    "scene_asmr1497366_t07_w2",
    "scene_asmr1497366_t07_w3",
    "scene_asmr1527130_t04",
    "scene_asmr1527130_t04_w2",
    "scene_asmr1527130_t04_w3",
    "scene_asmr1521586_t01",
    "scene_asmr1521586_t01_w2",
    "scene_asmr1521586_t01_w3",
]


def gpu_free_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().splitlines()
    try:
        return int(out[0])
    except Exception:
        return -1


def wait_gpu():
    for attempt in range(3):
        free = gpu_free_mib()
        if free >= 5000:
            return True, free
        print(f"[gpu] free={free}MiB <5000，等 2 分钟重试（{attempt + 1}/3）", flush=True)
        time.sleep(120)
    return False, gpu_free_mib()


def run_one(scene):
    txt, jsonl = RUNS / f"{scene}.txt", RUNS / f"{scene}.jsonl"
    for p in (txt, jsonl):
        if p.exists():
            p.unlink()
    ok, free = wait_gpu()
    if not ok:
        return {"scene": scene, "status": "skipped_gpu_busy", "free_mib": free}
    cmd = [str(PY), "live_sub.py", "--model", "anime", "--mt", "sakura",
           "--mt-ngl", "99", "--strategy", "adaptive", "--max-s", "5.0",
           "--hang-s", "2.0", "--layer", "1",
           "--source-audio", str(STD180 / f"{scene}.wav"),
           "--replay", "--log", str(txt)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=900)
        rc, tail = r.returncode, (r.stdout or "")[-400:] + (r.stderr or "")[-400:]
    except subprocess.TimeoutExpired:
        rc, tail = "timeout", ""
    elapsed = round(time.time() - t0, 1)
    n_sub = n_mt = 0
    if txt.exists():
        n_sub = sum(1 for ln in txt.read_text(encoding="utf-8", errors="ignore").splitlines()
                    if "[sub]" in ln)
    if jsonl.exists():
        for ln in jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                if json.loads(ln).get("kind") == "mt":
                    n_mt += 1
            except Exception:
                pass
    status = "ok" if (rc == 0 and n_mt > 0) else f"check_rc={rc}"
    return {"scene": scene, "status": status, "elapsed_s": elapsed,
            "sub_lines": n_sub, "mt_events": n_mt, "tail": tail[-200:]}


def main():
    results = []
    scenes = SCENES if RESUME_ONLY else _SCENES_FULL
    for i, s in enumerate(scenes, 1):
        print(f"=== [{i}/{len(SCENES)}] {s} start {time.strftime('%H:%M:%S')} ===", flush=True)
        r = run_one(s)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)
    (ROOT / "logs" / "batch_20260925_resume_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
