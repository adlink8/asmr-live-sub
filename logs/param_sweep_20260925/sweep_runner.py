# -*- coding: utf-8 -*-
"""分段参数 A/B 扫描批量跑（只读仓库代码，产物全部写本目录）。

跑法对齐 logs/run_scenes_20260925.py（22 窗批量）：
  live_sub.py --model anime --mt sakura --mt-ngl 99 --layer 1
              --source-audio <wav> --replay --log <txt>
差异（显式记录）：
  - 本次加 --fast（任务要求；Segmenter 逐帧确定性与墙钟无关，段数 <100 时
    seg_q(100) 无驱逐，段边界与实时回放等价）；
  - 分段参数（--hang-s/--max-s/--strategy）按配置组变化；
  - 跑前删同名 .txt/.jsonl（WorkLog 追加模式，必须清旧）；
  - 跑前检查显存（空闲 <5000MiB → 等 2 分钟重试，最多 3 次后跳过该任务）。

用法：
  python sweep_runner.py phase1            # 5 窗 x 6 配置
  python sweep_runner.py phase2 <cfg>      # 指定配置跑其余 21 窗
进度逐任务追加写到 sweep_progress.jsonl。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:/ADLINK/asmr-live-sub")
PY = ROOT / ".venv" / "Scripts" / "python.exe"
OUT = ROOT / "logs" / "param_sweep_20260925"
STD180 = Path(r"D:/Downloads/asmr-zh-corpus/std180s")

SUBSET5 = [
    "scene_asmr299717_t02",
    "scene_asmr416809_t01",
    "scene_asmr416816_t01",
    "scene_asmr1521586_t01",
    "scene_asmr1463510_t05",
]
WINDOWS26 = [
    "scene_asmr1449384_t03", "scene_asmr1449384_t03_w2", "scene_asmr1449384_t03_w3",
    "scene_asmr1463510_t05", "scene_asmr1463510_t05_w2", "scene_asmr1463510_t05_w3",
    "scene_asmr1497366_t07", "scene_asmr1497366_t07_w2", "scene_asmr1497366_t07_w3",
    "scene_asmr1521586_t01", "scene_asmr1521586_t01_w2", "scene_asmr1521586_t01_w3",
    "scene_asmr1527130_t04", "scene_asmr1527130_t04_w2", "scene_asmr1527130_t04_w3",
    "scene_asmr299717_t02", "scene_asmr299717_t02_w2",
    "scene_asmr324799_t03", "scene_asmr324799_t03_w2", "scene_asmr324799_t03_w3",
    "scene_asmr401391_t01",
    "scene_asmr416809_t01", "scene_asmr416809_t01_w2", "scene_asmr416809_t01_w3",
    "scene_asmr416816_t01", "scene_asmr416816_t01_w3",
]

# 配置组：name -> (hang_s, max_s, strategy)
CFGS = {
    "cfg_base":   (2.0, 8.0, "baseline"),
    "cfg_h15":    (1.5, 8.0, "baseline"),
    "cfg_h10":    (1.0, 8.0, "baseline"),
    "cfg_h07":    (0.7, 8.0, "baseline"),
    "cfg_h10_m6": (1.0, 6.0, "baseline"),
    "cfg_adapt":  (1.0, 8.0, "adaptive"),
}


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


def run_one(cfg, scene):
    hang, mx, strat = CFGS[cfg]
    txt = OUT / f"{cfg}_{scene}.txt"
    jsonl = OUT / f"{cfg}_{scene}.jsonl"
    for p in (txt, jsonl):
        if p.exists():
            p.unlink()
    ok, free = wait_gpu()
    if not ok:
        return {"cfg": cfg, "scene": scene, "status": "skipped_gpu_busy", "free_mib": free}
    cmd = [str(PY), "live_sub.py", "--model", "anime", "--mt", "sakura",
           "--mt-ngl", "99", "--layer", "1",
           "--strategy", strat, "--max-s", str(mx), "--hang-s", str(hang),
           "--source-audio", str(STD180 / f"{scene}.wav"),
           "--replay", "--fast", "--log", str(txt)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=1200)
        rc = r.returncode
        err_tail = (r.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        rc, err_tail = "timeout", ""
    elapsed = round(time.time() - t0, 1)
    n_mt = 0
    if jsonl.exists():
        for ln in jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if e.get("kind") == "mt" and (e.get("zh") or "").strip() and not e.get("zh_refusal"):
                n_mt += 1
    status = "ok" if rc == 0 else f"rc={rc}"
    return {"cfg": cfg, "scene": scene, "status": status, "elapsed_s": elapsed,
            "n_subs": n_mt, "err": err_tail[-150:]}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "phase1"
    jobs = []
    if mode == "phase1":
        for cfg in CFGS:
            for s in SUBSET5:
                jobs.append((cfg, s))
    elif mode == "phase2":
        cfg = sys.argv[2]
        if cfg not in CFGS:
            raise SystemExit(f"unknown cfg: {cfg}")
        rest = [s for s in WINDOWS26 if s not in SUBSET5]
        for s in rest:
            jobs.append((cfg, s))
    else:
        raise SystemExit(f"unknown mode: {mode}")

    prog = OUT / "sweep_progress.jsonl"
    for i, (cfg, s) in enumerate(jobs, 1):
        print(f"=== [{i}/{len(jobs)}] {cfg}_{s} start {time.strftime('%H:%M:%S')} ===",
              flush=True)
        r = run_one(cfg, s)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        with prog.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
