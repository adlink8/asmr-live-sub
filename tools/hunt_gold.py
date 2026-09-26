"""全池金级猎手：从 registry 全池登记里按信号排序，逐部现场检测找
"ja 字幕带时间戳 + zh 字幕"的双语作品（gold），命中即下载全部字幕文件。

候选优先级（探矿信号）：
  1. circle="大家一起来翻译"（RJ1018238 同类官方多语言版，已实证带 ja VTT）
  2. translation.is_volunteer=True（志愿者翻译组织，双语可能都放）
  3. 官方多语言版（langs 含 JPN+CHI）按 dl_count 降序
  4. 翻译社团 10 家排除（探矿实证 ja 字幕 0/10）
  5. 其余官方中文版按 dl_count 降序

断点续扫：progress.txt 记已扫 id；命中追加 gold_found.jsonl + 字幕落盘。

    .venv\\Scripts\\python.exe tools/hunt_gold.py \\
        --registry D:/Downloads/asmr-collect-staging/registry_full.json \\
        --top 300 --out dataset_gold
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from asmrone_collect import (THROTTLE, TRANSLATION_CIRCLES, api_get,
                             classify_work)

_write_lock = threading.Lock()
_print_lock = threading.Lock()


def rank_candidates(reg_works):
    tier1, tier2, tier3, tier5 = [], [], [], []
    for w in reg_works:
        circle = (w.get("circle") or {}).get("name") or ""
        if circle in TRANSLATION_CIRCLES:
            continue  # 探矿实证 ja=0，排除
        tr = w.get("translation") or {}
        if circle == "大家一起来翻译":
            tier1.append(w)
        elif tr.get("is_volunteer"):
            tier2.append(w)
        langs = [str(x) for x in (w.get("langs") or [])]
        if "JPN" in langs and any(x.startswith("CHI") for x in langs):
            tier3.append(w)
        else:
            tier5.append(w)
    for lst in (tier1, tier2, tier3, tier5):
        lst.sort(key=lambda w: -w.get("dl_count", 0))
    return tier1 + tier2 + tier3 + tier5


def download_subs(info, out_dir):
    wdir = out_dir / f"RJ{info['id']}" / "subtitles"
    n = 0
    for lang, urls in (("zh", info.get("zh_urls") or []),
                       ("ja", info.get("ja_urls") or [])):
        d = wdir / lang
        d.mkdir(parents=True, exist_ok=True)
        seen = set()
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            import urllib.parse
            name = urllib.parse.unquote(url.rsplit("/", 1)[-1]) or "sub.txt"
            try:
                (d / name).write_bytes(api_get(url, binary=True))
                n += 1
                time.sleep(THROTTLE)
            except Exception as e:  # noqa: BLE001
                print(f"    [warn] {name} 下载失败: {e}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry", required=True)
    ap.add_argument("--top", type=int, default=300, help="最多现场检测的部数")
    ap.add_argument("--workers", type=int, default=4,
                    help="并发线程数（网络 IO 密集，4~6 合理）")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--shard", type=int, default=0,
                    help="分片编号（多进程并行扫同一池，各领一片互不重复）")
    ap.add_argument("--shards", type=int, default=1,
                    help="总分片数；>1 时断点记 progress_shard{N}.txt，"
                         "并合并读取所有 progress* 文件去重")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    for pf in out.glob("progress*.txt"):
        done_ids |= {int(x) for x in pf.read_text().split()}
    found_path = out / "gold_found.jsonl"

    reg = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    ranked = rank_candidates(reg.get("works") or [])
    queue = [w for w in ranked if int(w["id"]) not in done_ids]
    if args.shards > 1:
        queue = queue[args.shard::args.shards]
    queue = queue[:args.top]
    progress = out / ("progress.txt" if args.shards <= 1
                      else f"progress_shard{args.shard}.txt")
    print(f"分片 {args.shard}/{args.shards}: 队列 {len(queue)} 部 × "
          f"{args.workers} 并发，全池历史已扫 {len(done_ids)}", flush=True)

    n_gold = 0
    n_done = 0

    def record(wid, line=None, gold_meta=None):
        nonlocal n_gold
        with _write_lock:
            with progress.open("a", encoding="utf-8") as f:
                f.write(f"{wid}\n")
            if gold_meta is not None:
                with found_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(gold_meta, ensure_ascii=False) + "\n")
                n_gold += 1

    def probe(i, w):
        nonlocal n_done
        wid = int(w["id"])
        title = w.get("title", "")
        try:
            info = classify_work(wid, title)
        except Exception as e:  # noqa: BLE001
            with _print_lock:
                print(f"[{i+1}/{len(queue)}] RJ{wid} 检测失败: {e}", flush=True)
            record(wid)
            n_done += 1
            return
        if info and info.get("gold_ready") \
                and (info.get("sub_coverage") or 0) >= args.min_coverage:
            n_dl = download_subs(info, out)
            meta = {"id": wid, "title": title,
                    "circle": (w.get("circle") or {}).get("name"),
                    "tags": w.get("tags") or [],
                    "sub_coverage": info.get("sub_coverage"),
                    "zh_count": len(info.get("zh_urls") or []),
                    "ja_count": len(info.get("ja_urls") or []),
                    "subs_downloaded": n_dl}
            record(wid, gold_meta=meta)
            with _print_lock:
                print(f"[{i+1}/{len(queue)}] [GOLD] RJ{wid} "
                      f"zh×{meta['zh_count']} ja×{meta['ja_count']} "
                      f"覆盖率={meta['sub_coverage']:.0%} 已下 {n_dl} 字幕 "
                      f"{title[:30]}", flush=True)
        else:
            why = "无zh" if info is None else (
                "无ja轴" if not info.get("gold_ready")
                else f"覆盖率{info.get('sub_coverage'):.0%}")
            record(wid)
            with _print_lock:
                print(f"[{i+1}/{len(queue)}] RJ{wid} 未中（{why}）{title[:28]}",
                      flush=True)
        n_done += 1
        time.sleep(THROTTLE)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(probe, i, w) for i, w in enumerate(queue)]
        for f in as_completed(futures):
            f.result()  # 异常已在 probe 内捕获；此处兜底传播

    print(f"\n[done] 本轮检测 {n_done} 部（{args.workers} 并发），"
          f"新命中金级 {n_gold} 部 -> {found_path}", flush=True)


if __name__ == "__main__":
    main()
