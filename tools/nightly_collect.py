#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nightly_collect.py — 夜间微调数据采集循环（00:00-07:00，一次一部，GPU 串行）。

循环：扫描 asmr.one 带字幕新作品(成人标题过滤,排除锚点与已采) → fetch 一部 →
collect_finetune 全轨采集(自校正+统计门) → 早报追加 → 下一部，到 07:00 收工。
原件 mp3 保留=ASR 训练弹药(~1MB/min)；窗 wav 临时即删；磁盘预算见 REPORT。

用法：python tools/nightly_collect.py --staging D:/Downloads/asmr-collect-staging
      --deadline 07:00 --pages 5 --max-audio-mb 800
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 锚点 RJ 防泄漏硬门（与 tools/collect_finetune.py 同一清单）
ANCHOR = {1449384, 1463510, 1497366, 1521586, 1527130, 299717, 324799, 401391, 416809, 416816}


def sh(cmd, timeout=7200):
    return subprocess.run([sys.executable] + cmd, cwd=str(ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)


def collected_rjs(out_root: Path):
    done = set()
    if (out_root / "collected_rjs.txt").exists():
        done = {int(x) for x in out_root.joinpath("collected_rjs.txt").read_text().split()}
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default="D:/Downloads/asmr-collect-staging")
    ap.add_argument("--out", default=str(ROOT / "dataset_finetune"))
    ap.add_argument("--deadline", default="07:00")
    ap.add_argument("--pages", type=int, default=5)
    ap.add_argument("--max-audio-mb", type=int, default=800)
    args = ap.parse_args()

    staging = Path(args.staging)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    done = collected_rjs(out)
    h, m = map(int, args.deadline.split(":"))
    report = out / f"nightly_{datetime.now():%Y%m%d}.md"
    log_lines = [f"# 夜间采集早报 {datetime.now():%Y-%m-%d %H:%M}\n"]

    inv = staging / "inventory.json"
    if not inv.exists():
        r = sh(["tools/asmrone_collect.py", "scan", "--pages", str(args.pages),
                "--out", str(staging)], timeout=3600)
        log_lines.append(f"scan exit={r.returncode}\n")

    while datetime.now() < datetime.now().replace(hour=h, minute=m, second=0):
        works = json.loads(inv.read_text(encoding="utf-8"))["works"] if inv.exists() else []
        cand = [w for w in works
                if int(w["id"]) not in ANCHOR and int(w["id"]) not in done
                and w.get("audio_bytes", 0) <= args.max_audio_mb * 1e6]
        if not cand:
            log_lines.append("候选耗尽，重扫描\n")
            sh(["tools/asmrone_collect.py", "scan", "--pages", str(args.pages + 5),
                "--out", str(staging)], timeout=3600)
            continue
        w = cand[0]
        rid = int(w["id"])
        # 单作品临时清单：保证 fetch 拉的正是我们选中的这部
        one = staging / "_one.json"
        one.write_text(json.dumps({"works": [w]}, ensure_ascii=False), encoding="utf-8")
        t0 = time.time()
        log_lines.append(f"\n## RJ{rid} {w['title'][:40]}\n")
        r = sh(["tools/asmrone_collect.py", "fetch", "--from-inv", str(one),
                "--limit", "1", "--audio", "--max-audio-mb", str(args.max_audio_mb),
                "--out", str(staging)], timeout=3600)
        log_lines.append(f"fetch exit={r.returncode} {time.time()-t0:.0f}s\n")
        wdir = next((p for p in staging.glob(f"RJ{rid}") if p.is_dir()), None)
        if wdir is None:
            done.add(rid)
            log_lines.append("fetch 无产物，跳过\n")
            continue
        r = sh(["tools/collect_finetune.py", "--work", str(wdir), "--out", str(out)],
               timeout=4 * 3600)
        log_lines.append(f"collect exit={r.returncode}\n```\n{r.stdout[-1500:]}\n```\n")
        done.add(rid)
        (out / "collected_rjs.txt").write_text(
            "\n".join(str(x) for x in sorted(done)), encoding="utf-8")

    log_lines.append(f"\n收工 {datetime.now():%H:%M}")
    report.write_text("\n".join(log_lines), encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
