"""长时评测（30m/60m）：实时回放 + 完整条件/结果记录。

与 run_soak_evaluation.py / run_marathon_eval.py 的区别（那两份历史产物已作废，
原因见 docs/issues_and_solutions.md 问题八与 dataset/README.md）：
  - 默认**实时回放**（不加 --fast）：推流速度=音频速度，队列不积压，
    seg_q 驱逐与 result_q 显示丢失两个回放缺陷均不触发——长时测量因此可信；
  - 生产配置（与 start.bat 一致）：anime + sakura ngl=99 + layer1 + adaptive；
  - 每个测试产出条件快照（音频哈希/时长/显存/完整命令）+ 结果统计 + md 报告，
    文件名带日期，不覆盖历史。

用法：
  python tools/run_long_eval.py --name 30m_realtime_20260923 --audio benchmarks/soak_30min.wav
  python tools/run_long_eval.py --name 60m_fast_20260923 --audio benchmarks/soak_60min.wav --fast
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def audio_info(p: Path):
    import av
    with av.open(str(p)) as c:
        st = c.streams.audio[0]
        return {"rate": int(st.rate), "channels": int(st.channels),
                "codec": st.codec_context.name}


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=15).stdout.strip()
        return out
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi 不可用: {e}"


def pct(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return round(s[f] + (k - f) * (s[c] - f), 2) if False else round(s[f] + (k - f) * (s[c] - s[f]), 2)


def analyze(jsonl: Path):
    evs = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evs.append(json.loads(line))
        except Exception:
            pass
    asr = [e for e in evs if e.get("kind") == "asr"]
    mt = [e for e in evs if e.get("kind") == "mt"]
    drops = [e for e in evs if e.get("kind") == "seg-drop"]
    batch = [e for e in evs if e.get("kind") == "asr-batch"]
    mt_ok = [m for m in mt if m.get("zh") and not m.get("zh_refusal")]
    roles = {}
    for a in asr:
        roles[a.get("role", "?")] = roles.get(a.get("role", "?"), 0) + 1
    amap = {a["seg_id"]: a for a in asr if a.get("seg_id") is not None}
    delays, durs, asr_ts = [], [], []
    for m in mt_ok:
        a = amap.get(m.get("seg_id"))
        if a:
            delays.append(round((a.get("audio_s") or 0) + (a.get("asr_s") or 0)
                                + (m.get("mt_s") or 0), 2))
    for a in asr:
        if a.get("audio_s") is not None:
            durs.append(a["audio_s"])
        if a.get("asr_s") is not None:
            asr_ts.append(a["asr_s"])
    ids = sorted(a["seg_id"] for a in asr if a.get("seg_id") is not None)
    gaps = [(x, y) for x, y in zip(ids, ids[1:]) if y - x > 1]
    reasons = {}
    for d in drops:
        reasons[d.get("reason", "?")] = reasons.get(d.get("reason", "?"), 0) + 1
    audio_covered_s = max((a.get("t_end") or 0) for a in asr) if asr else 0
    return {
        "asr_segments": len(asr),
        "asr_with_text": sum(1 for a in asr if (a.get("ja") or "").strip()),
        "asr_roles": roles,
        "mt_total": len(mt),
        "mt_valid": len(mt_ok),
        "mt_empty": sum(1 for m in mt if not (m.get("zh") or "").strip()),
        "mt_refusal": sum(1 for m in mt if m.get("zh_refusal")),
        "seg_drops": len(drops), "seg_drop_reasons": reasons,
        "asr_batch_events": len(batch),
        "asr_batch_max": max((b.get("batch") or 0) for b in batch) if batch else 0,
        "seg_id_range": [ids[0], ids[-1]] if ids else None,
        "seg_id_gaps": gaps[:5],
        "audio_covered_s": round(audio_covered_s, 1),
        "subtitles_per_min": round(len(mt_ok) / (audio_covered_s / 60), 2) if audio_covered_s else 0,
        "delay_p50_p90_max": [pct(delays, .5), pct(delays, .9), max(delays) if delays else 0],
        "seg_dur_p50_p90_max": [pct(durs, .5), pct(durs, .9), max(durs) if durs else 0],
        "asr_time_p50_p90_max": [pct(asr_ts, .5), pct(asr_ts, .9), max(asr_ts) if asr_ts else 0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "logs" / "soak_runs"))
    ap.add_argument("--fast", action="store_true", help="对照组：fast 推流（已知会触发回放缺陷）")
    ap.add_argument("--strategy", default="adaptive")
    ap.add_argument("--max-s", type=float, default=5.0)
    ap.add_argument("--hang-s", type=float, default=2.0)
    ap.add_argument("--analyze-only", action="store_true",
                    help="不重跑管线，只分析已存在的 jsonl（报告脚本崩溃后恢复用）")
    ap.add_argument("--displayed", type=int, default=None,
                    help="analyze-only 时人工提供的 [done] 显示数")
    ap.add_argument("--wall", type=float, default=None, help="analyze-only 时人工提供的墙钟秒数")
    args = ap.parse_args()

    audio = Path(args.audio)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"{args.name}.txt"
    work_file = out_dir / f"{args.name}.jsonl"

    info = audio_info(audio)
    cond = {
        "name": args.name, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audio": {"path": str(audio), "sha256_16": sha256(audio),
                  "size_mb": round(audio.stat().st_size / 1e6, 1), **info},
        "mode": "fast" if args.fast else "realtime",
        "config": {"model": "anime", "mt": "sakura", "mt_ngl": 99, "layer": 1,
                   "strategy": args.strategy, "max_s": args.max_s, "hang_s": args.hang_s,
                   "note": "生产配置（与 start.bat 一致）"},
        "gpu_at_start": gpu_info(),
        "python": sys.version.split()[0],
    }
    cmd = [str(PYTHON), "live_sub.py", "--model", "anime", "--mt", "sakura",
           "--mt-ngl", "99", "--layer", "1", "--strategy", args.strategy,
           "--max-s", str(args.max_s), "--hang-s", str(args.hang_s),
           "--source-audio", str(audio), "--replay"]
    if args.fast:
        cmd.append("--fast")
    cmd += ["--log", str(log_file)]
    cond["command"] = " ".join(cmd)

    if args.analyze_only:
        cond["reconstructed"] = ("报告脚本首次运行崩在 [done] 解析（管线本身正常跑完），"
                                 "本文件由 --analyze-only 从现存 jsonl 重建；"
                                 "displayed/wall 由运行现场恢复")
        cond["exit_code"] = 0
        cond["wall_seconds"] = args.wall
        p_stdout = ""
    else:
        for f in (log_file, work_file):
            if f.exists():
                f.unlink()  # WorkLog 是追加模式，必须先清（历史污染事故的教训）
        print(f"[*] {args.name} 启动（{'fast' if args.fast else 'realtime'}）"
              f" 音频 {cond['audio']['size_mb']}MB")
        t0 = time.time()
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        cond["wall_seconds"] = round(time.time() - t0, 1)
        cond["exit_code"] = p.returncode
        cond["gpu_at_end"] = gpu_info()
        p_stdout = p.stdout or ""
    done = ""
    for line in p_stdout.splitlines():
        if "条字幕" in line:
            done = line.strip()
    displayed = args.displayed
    if displayed is None and done:
        m = re.search(r"(\d+)", done)
        displayed = int(m.group(1)) if m else None

    stats = analyze(work_file)
    stats["done_line"] = done or f"[done] 共 {displayed} 条字幕（运行现场恢复）"
    stats["displayed_vs_translated"] = {"translated": stats["mt_valid"],
                                        "displayed": displayed}

    md = [f"# 长时评测报告 {args.name}", "",
          f"- 时间：{cond['date']}｜退出码 {cond['exit_code']}｜墙钟 {cond['wall_seconds']}s",
          f"- 音频：{audio.name} {cond['audio']['size_mb']}MB "
          f"sha256:{cond['audio']['sha256_16']} {info['rate']}Hz {info['codec']}",
          f"- 模式：**{cond['mode']}**（realtime=推流速度=音频速度；fast=全速推流，"
          f"已知触发 seg_q 驱逐/result_q 显示丢失）",
          f"- 配置：{json.dumps(cond['config'], ensure_ascii=False)}",
          f"- 命令：`{cond['command']}`",
          f"- GPU：start[{cond['gpu_at_start']}] end[{cond.get('gpu_at_end', '未记录(重建)')}]", "",
          "## 结果", ""]
    for k, v in stats.items():
        md.append(f"- {k}: {json.dumps(v, ensure_ascii=False)}")
    md += ["", "## 口径说明", "",
           "- **realtime**：推流速度=音频速度。seg_q(100) 不积压（asr_batch_max=1、"
           "无 late 角色、seg_id 连续即证据），段级数据完整可信；**但 result_q(50) "
           "显示丢失仍然发生**——主线程整个推流期间都在 feed 循环，排水循环推流结束才启动，"
           "期间 result_q 无消费者，只保住最后约 50 条。故 jsonl 统计（asr/mt/延迟/丢弃）"
           "可信，`[done] 显示数`被问题八缺陷污染（translated vs displayed 即缺陷代价）；",
           "- **fast**：全速推流，两个缺陷同时触发（seg_q 驱逐最旧段 + result_q 逐出），"
           "仅作缺陷量化对照，不作性能结论；",
           "- 语义正确度不在本轨道度量范围（soak 素材无人工字幕），"
           "准确度看 dataset/ 的 180s 标准化锚点场景。"]
    (out_dir / f"{args.name}.report.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / f"{args.name}.conditions.json").write_text(
        json.dumps({**cond, "stats": stats}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {args.name}: asr={stats['asr_segments']} mt_valid={stats['mt_valid']} "
          f"done=[{done}] wall={cond['wall_seconds']}s")
    print(f"     报告: {out_dir / (args.name + '.report.md')}")


if __name__ == "__main__":
    main()
