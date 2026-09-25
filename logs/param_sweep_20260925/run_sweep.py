# -*- coding: utf-8 -*-
"""分段参数 A/B 扫描驱动（2026-09-25）。

跑法与既有 22 窗批量（logs/run_scenes_20260925.py）完全一致：
  live_sub.py --model anime --mt sakura --mt-ngl 99 --layer 1  + 原速回放（不加 --fast），
  仅变 --strategy / --max-s / --hang-s。保证与既有 26 窗 adaptive/max5 产物同口径可比。
每窗独立输出文件名（<cfg>_<scene>.txt/.jsonl），跑前先删（WorkLog 追加模式）。
跑完立即调 tools/realtime_compare.py（既有 missing/coverage 口径）出 cmp 产物。
只读仓库代码与数据，不修改 livesub/ tools/ dataset/。
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:/ADLINK/asmr-live-sub")
PY = ROOT / ".venv" / "Scripts" / "python.exe"
STD180 = Path(r"D:/Downloads/asmr-zh-corpus/std180s")
SWEEP = ROOT / "logs" / "param_sweep_20260925"

# cfg 名 -> (strategy, max_s, hang_s)
CONFIGS = {
    "cfg_base":   ("baseline", 8.0, 2.0),
    "cfg_h15":    ("baseline", 8.0, 1.5),
    "cfg_h10":    ("baseline", 8.0, 1.0),
    "cfg_h07":    ("baseline", 8.0, 0.7),
    "cfg_h10_m6": ("baseline", 6.0, 1.0),
    "cfg_adapt":  ("adaptive", 8.0, 1.0),
}

SCENES5 = [
    "scene_asmr299717_t02",
    "scene_asmr416809_t01",
    "scene_asmr416816_t01",
    "scene_asmr1521586_t01",
    "scene_asmr1463510_t05",
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


def run_one(cfg, scene, timeout_s=420):
    strategy, max_s, hang_s = CONFIGS[cfg]
    txt = SWEEP / f"{cfg}_{scene}.txt"
    jsonl = SWEEP / f"{cfg}_{scene}.jsonl"
    for p in (txt, jsonl):
        if p.exists():
            p.unlink()
    ok, free = wait_gpu()
    if not ok:
        return {"cfg": cfg, "scene": scene, "status": "skipped_gpu_busy", "free_mib": free}
    cmd = [str(PY), "live_sub.py", "--model", "anime", "--mt", "sakura",
           "--mt-ngl", "99", "--layer", "1",
           "--strategy", strategy, "--max-s", str(max_s), "--hang-s", str(hang_s),
           "--source-audio", str(STD180 / f"{scene}.wav"),
           "--replay", "--log", str(txt)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout_s)
        rc = r.returncode
        (SWEEP / f"{cfg}_{scene}.stdout.txt").write_text(
            (r.stdout or "") + "\n===STDERR===\n" + (r.stderr or ""), encoding="utf-8")
    except subprocess.TimeoutExpired:
        rc, tail = "timeout", ""
        (SWEEP / f"{cfg}_{scene}.stdout.txt").write_text("TIMEOUT", encoding="utf-8")
    elapsed = round(time.time() - t0, 1)

    # 跑完立即用既有口径出 coverage/missing 产物
    cmp_rc = None
    if jsonl.exists():
        cmp_r = subprocess.run(
            [str(PY), "tools/realtime_compare.py", str(jsonl),
             "--human", str(ROOT / "dataset" / f"{scene}.human_zh.json"),
             "--out", str(SWEEP / "cmp" / f"{cfg}_{scene}")],
            cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120)
        cmp_rc = cmp_r.returncode
        if cmp_rc != 0:
            print(f"[cmp-err] {cfg}/{scene} rc={cmp_rc}: {(cmp_r.stdout or '')[-300]}"
                  f"{(cmp_r.stderr or '')[-300:]}", flush=True)

    n_mt = 0
    if jsonl.exists():
        for ln in jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                if json.loads(ln).get("kind") == "mt":
                    n_mt += 1
            except Exception:
                pass
    status = "ok" if rc == 0 else f"check_rc={rc}"
    return {"cfg": cfg, "scene": scene, "status": status, "elapsed_s": elapsed,
            "mt_events": n_mt, "cmp_rc": cmp_rc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="逗号分隔 cfg 名")
    ap.add_argument("--scenes", required=True, help="逗号分隔 scene 名")
    ap.add_argument("--out", required=True, help="逐窗结果 jsonl 追加文件")
    args = ap.parse_args()

    out_path = Path(args.out)
    jobs = [(c, s) for c in args.configs.split(",") for s in args.scenes.split(",")]
    print(f"[sweep] jobs={len(jobs)} configs={args.configs} scenes={args.scenes}", flush=True)
    for i, (c, s) in enumerate(jobs, 1):
        print(f"=== [{i}/{len(jobs)}] {c}/{s} start {time.strftime('%H:%M:%S')} ===", flush=True)
        r = run_one(c, s)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
