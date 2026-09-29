#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fetch_script_txts.py — 双语台本深扫的 txt 拉取器（纯文本 KB 级，不碰音频）。

数据源：D:/Downloads/asmr-collect-staging/text_inventory.jsonl 的 texts[] 节点（title+url）。
范围：1627 部 txt 质量扫描池 ∪ 43 部候选中 script 目录缺失/不全的（②补残）。
落盘：D:/Downloads/asmr-script-align/RJ{id}/script/{safe_name}.txt
纪律：R18 内容不落日志；限速冷却按 handover 教训（429→冷却15分钟+降线程）。
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from asmrone_collect import api_get  # noqa: E402

INV = Path(r"D:/Downloads/asmr-collect-staging/text_inventory.jsonl")
SCAN = Path(r"D:/Downloads/asmr-script-align/txt_quality_scan.jsonl")
CAND = Path(r"D:/Downloads/asmr-gold-hunt/script_full_sample.json")
WORKDIR = Path(r"D:/Downloads/asmr-script-align")
DONE_LOG = WORKDIR / "fetch_txts_done.jsonl"

MAX_FILES_PER_WORK = 40
WORKERS = 6
GAP = 0.3
lock = threading.Lock()


def safe_name(name: str) -> str:
    for a, b in (("<", "＜"), (">", "＞"), (":", "："), ('"', "”"),
                 ("/", "／"), ("\\", "＼"), ("|", "｜"), ("?", "？"), ("*", "＊")):
        name = name.replace(a, b)
    return name[:100]


def load_targets() -> dict[int, list[dict]]:
    """work_id → texts[]（.txt 节点，截断到上限）。"""
    want: set[int] = set()
    for ln in SCAN.read_text(encoding="utf-8").splitlines():
        try:
            want.add(json.loads(ln)["id"])
        except Exception:  # noqa: BLE001
            continue
    for ln in CAND.read_text(encoding="utf-8").splitlines() if CAND.exists() else []:
        try:
            want.add(json.loads(ln)["id"])
        except Exception:  # noqa: BLE001
            continue
    inv = {}
    for ln in INV.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        if "error" in r or r["id"] not in want:
            continue
        txts = [t for t in r.get("texts", []) if t["title"].lower().endswith(".txt") and t.get("url")]
        if txts:
            inv[r["id"]] = txts[:MAX_FILES_PER_WORK]
    return inv


def already_done() -> set[int]:
    if not DONE_LOG.exists():
        return set()
    ids = set()
    for ln in DONE_LOG.read_text(encoding="utf-8").splitlines():
        try:
            ids.add(json.loads(ln)["id"])
        except Exception:  # noqa: BLE001
            continue
    return ids


def fetch_work(wid: int, txts: list[dict]) -> tuple[int, int, str]:
    """下载一部作品的全部 txt → script/。返回 (新增文件数, 跳过数, 状态)。"""
    sdir = WORKDIR / f"RJ{wid}" / "script"
    added = skipped = 0
    for t in txts:
        dest = sdir / safe_name(t["title"])
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        sdir.mkdir(parents=True, exist_ok=True)
        try:
            raw = api_get(t["url"], binary=True)
            dest.write_bytes(raw)
            added += 1
        except Exception as e:  # noqa: BLE001
            return added, skipped, f"err:{str(e)[:50]}"
        time.sleep(GAP)
    return added, skipped, "ok"


def main():
    inv = load_targets()
    done = already_done()
    todo = {k: v for k, v in inv.items() if k not in done}
    total_files = sum(len(v) for v in todo.values())
    print(f"[list] 待扫 {len(todo)} 部 / {total_files} 个txt（已完成跳过 {len(done)} 部）", flush=True)
    if not todo:
        return
    t0 = time.time()
    n_ok = n_err = n_files = 0
    cooldown_until = 0.0
    with DONE_LOG.open("a", encoding="utf-8") as flog:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(fetch_work, wid, txts): wid for wid, txts in todo.items()}
            for fu in as_completed(futs):
                wid = futs[fu]
                while time.time() < cooldown_until:
                    time.sleep(5)
                try:
                    added, skipped, status = fu.result()
                except Exception as e:  # noqa: BLE001
                    added, skipped, status = 0, 0, f"exc:{str(e)[:50]}"
                with lock:
                    if status == "ok":
                        n_ok += 1
                        n_files += added
                        flog.write(json.dumps({"id": wid}) + "\n")
                        flog.flush()
                    else:
                        n_err += 1
                        print(f"[fail] RJ{wid}: {status}", flush=True)
                    if (n_ok + n_err) % 50 == 0:
                        rate = (n_ok + n_err) / max(time.time() - t0, 1)
                        print(f"[prog] {n_ok+n_err}/{len(todo)} 部 新增txt {n_files} err {n_err} "
                              f"({rate:.1f}部/s)", flush=True)
                    if status.startswith(("err:429", "exc:HTTP Error 429")):
                        cooldown_until = time.time() + 900
                        print("[cool] 429 限速 → 冷却15分钟", flush=True)
    print(f"[done] ok {n_ok} err {n_err} 新增txt {n_files} 用时 {(time.time()-t0)/60:.0f}分钟", flush=True)


if __name__ == "__main__":
    main()
