#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nightly_collect.py — 夜间微调数据采集循环（00:00-07:00，一次一部，GPU 串行）。

循环：扫描 asmr.one 带字幕新作品(成人标题过滤,排除锚点与已采) → 按标签缺口打分选部 →
fetch 全音轨(mp3/wav 去重,mp3优先) → collect_finetune 全轨采集(自校正+统计门) →
早报逐部落盘 → 下一部，到 07:00 收工。

标签补全：目标 tag 集 = 扫描全集的 tags 并集；已覆盖 = 收藏种子(favorites.json,
若存在) + 已采作品 meta.json 的 tags 并集；候选按"未覆盖 tag 数"降序、下载量次序。

磁盘水位：staging 所在盘剩余 < --min-free-gb 时停止接新作品（防全音轨吃满盘）。

用法：python tools/nightly_collect.py --staging D:/Downloads/asmr-collect-staging
      --deadline 07:00 --pages 5 --max-audio-mb 800 --min-free-gb 50
"""
import argparse
import json
import shutil
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


def covered_tags(out_root: Path, seed_path: Path | None = None) -> set:
    """已覆盖标签 = 收藏种子(favorites.json) + 已采作品 meta.json 的 tags 并集。"""
    tags = set()
    for m in out_root.glob("RJ*/meta.json"):
        try:
            tags |= set(json.loads(m.read_text(encoding="utf-8")).get("tags") or [])
        except Exception:  # noqa: BLE001
            continue
    if seed_path and seed_path.exists():
        try:
            for w in json.loads(seed_path.read_text(encoding="utf-8"))["works"]:
                tags |= set(w.get("tags") or [])
        except Exception:  # noqa: BLE001
            pass
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default="D:/Downloads/asmr-collect-staging")
    ap.add_argument("--out", default=str(ROOT / "dataset_finetune"))
    ap.add_argument("--deadline", default="07:00")
    ap.add_argument("--pages", type=int, default=5)
    ap.add_argument("--max-audio-mb", type=int, default=800, help="单条音轨体积上限")
    ap.add_argument("--min-free-gb", type=int, default=50, help="磁盘剩余低于此值收工")
    args = ap.parse_args()

    # 解释器自检：子进程全用 sys.executable，系统 Python 缺 av 会让采集全崩
    # 还照标记已采（2026-09-25 演练实证），所以启动时就地拦下。
    try:
        import av  # noqa: F401
    except ImportError:
        print("[FATAL] 当前解释器缺 av 模块——请用 .venv/Scripts/python.exe 启动", file=sys.stderr)
        sys.exit(2)

    staging = Path(args.staging)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    done = collected_rjs(out)
    failed = set()  # 本次运行内采失败的部不重试（下晚重来）
    h, m = map(int, args.deadline.split(":"))
    deadline = datetime.now().replace(hour=h, minute=m, second=0)
    if deadline <= datetime.now():
        deadline += timedelta(days=1)  # 23:xx 手动启动 → 明早 07:00 收工
    report = out / f"nightly_{datetime.now():%Y%m%d}.md"
    log_lines = [f"# 夜间采集早报 {datetime.now():%Y-%m-%d %H:%M}\n"]

    def flush_report():
        report.write_text("\n".join(log_lines), encoding="utf-8")

    inv = staging / "inventory.json"
    if not inv.exists():
        r = sh(["tools/asmrone_collect.py", "scan", "--pages", str(args.pages),
                "--out", str(staging)], timeout=3600)
        log_lines.append(f"scan exit={r.returncode}\n")

    # 收藏种子：拉账号收藏清单（tags 基线 + 采集优先队列）。失败不阻塞，退化为纯扫描打分。
    fav_path = staging / "favorites.json"
    fav_ids = set()
    if not fav_path.exists():
        r = sh(["tools/asmrone_collect.py", "favorites", "--out", str(staging)],
               timeout=3600)
        log_lines.append(f"favorites exit={r.returncode}\n")
    if fav_path.exists():
        try:
            fav_ids = {int(w["id"])
                       for w in json.loads(fav_path.read_text(encoding="utf-8"))["works"]}
        except Exception:  # noqa: BLE001
            pass
        log_lines.append(f"收藏种子 {len(fav_ids)} 部（其中出现在扫描清单的才会被采："
                         f"扫描已保证带中文字幕）\n")

    while datetime.now() < deadline:
        free_gb = shutil.disk_usage(staging).free / 1e9
        if free_gb < args.min_free_gb:
            log_lines.append(f"\n磁盘剩余 {free_gb:.0f}GB < {args.min_free_gb}GB，水位门收工")
            break
        works = json.loads(inv.read_text(encoding="utf-8"))["works"] if inv.exists() else []
        cap_bytes = args.max_audio_mb * 1024 * 1024
        cand = [w for w in works
                if int(w["id"]) not in ANCHOR and int(w["id"]) not in done
                and int(w["id"]) not in failed
                and w.get("audio_bytes_unique", w.get("audio_bytes", 0)) <= cap_bytes]
        if not cand:
            log_lines.append("候选耗尽，重扫描\n")
            flush_report()
            r = sh(["tools/asmrone_collect.py", "scan", "--pages", str(args.pages + 5),
                    "--out", str(staging)], timeout=3600)
            if r.returncode != 0 or not inv.exists():
                log_lines.append("重扫描失败，10 分钟后重试\n")
                flush_report()
                time.sleep(600)
            continue
        # 标签补全打分：收藏优先，未覆盖 tag 数降序，下载量次之
        cov = covered_tags(out, fav_path)
        cand.sort(key=lambda w: (0 if int(w["id"]) in fav_ids else 1,
                                 -len(set(w.get("tags") or []) - cov),
                                 -w.get("dl_count", 0)))
        w = cand[0]
        rid = int(w["id"])
        # 单作品临时清单：保证 fetch 拉的正是我们选中的这部
        one = staging / "_one.json"
        one.write_text(json.dumps({"works": [w]}, ensure_ascii=False), encoding="utf-8")
        t0 = time.time()
        log_lines.append(f"\n## RJ{rid} {w['title'][:40]}\n")
        log_lines.append(f"fav={'是' if rid in fav_ids else '否'} "
                         f"tags={','.join((w.get('tags') or [])[:8])}\n")
        r = sh(["tools/asmrone_collect.py", "fetch", "--from-inv", str(one),
                "--limit", "1", "--audio-full", "--max-audio-mb", str(args.max_audio_mb),
                "--out", str(staging)], timeout=3600)
        log_lines.append(f"fetch exit={r.returncode} {time.time()-t0:.0f}s\n")
        flush_report()
        wdir = next((p for p in staging.glob(f"RJ{rid}") if p.is_dir()), None)
        if wdir is None:
            failed.add(rid)
            log_lines.append("fetch 无产物，跳过\n")
            flush_report()
            continue
        r = sh(["tools/collect_finetune.py", "--work", str(wdir), "--out", str(out)],
               timeout=4 * 3600)
        log_lines.append(f"collect exit={r.returncode}\n```\n{r.stdout[-1500:]}\n```\n")
        if r.returncode == 0:
            done.add(rid)
            (out / "collected_rjs.txt").write_text(
                "\n".join(str(x) for x in sorted(done)), encoding="utf-8")
        else:
            failed.add(rid)
            log_lines.append("collect 失败，不标记已采（下晚重试）\n")
        flush_report()

    log_lines.append(f"\n收工 {datetime.now():%H:%M}")
    flush_report()
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
