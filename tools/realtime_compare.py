#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""realtime_compare.py — 层④真实场景对照：会话 WorkLog × 人工字幕时间轴（事后分析）。

场景：用户正常播放一部带人工字幕的 ASMR 作品（live 环回采集，真实使用状态），
live_sub 实时采集翻译并记 WorkLog。本工具事后拿该会话的 WorkLog jsonl 与该
作品的人工字幕（原始 LRC/VTT/SRT，或已归零的 human_zh.json）对齐比对，回答：
  1. 本地字幕比人工字幕慢多少（逐条时差 + 中位数/p90/最大）
  2. 覆盖：会话范围内多少人工 cue 没被任何本地段覆盖（missing 方向）
  3. 多余：多少条本地字幕没配到任何人工 cue（多余/疑似幻觉方向）
  4. 语义比对输入：人工 zh vs 本地 zh 逐对打包成 judge_input（与
     tools/build_anchor_scene.py 产物同构），供现有裁判管线复用

时间轴与偏移（核心概念）：
  - 本地音频时间轴：WorkLog asr 事件的 t_start/t_end（回放/采集起点为 0）
  - 作品时间轴：人工字幕文件的时间戳
  - offset：作品时间 = 本地音频时间 + offset。
    自动检测：本地字幕（mt 中文）与人工 cue 逐对模糊匹配（difflib，无新依赖），
    高相似对给出偏移候选（cue.t0 - seg.t_start），聚类取主簇中心，再在
    1s 粒度上滑窗搜索使"时间接近且文本相似"配对数最大的偏移。
    用 asr 音频时间而非上屏墙钟做时间接近性判据：墙钟匹配会把流水线延迟
    吸收进偏移；音频时间轴匹配与回放速度无关（原速/快速回放均可对齐）。
  - --offset 可人工指定。自动检测置信度低（文本配对段数 < --min-pairs）时
    拒绝出报告并要求人工指定（会话中暂停/拖进度条会破坏对齐，只能人工兜底）。

上屏时刻度量（对回放速度与墙钟起点假设均稳健）：
  某段字幕的实际出现时刻（本地音频时间轴）
      ≈ t_end + asr_s + (mt.ts - asr.ts)
  即"段收口 → ASR 完成 → 翻译完成"全部计入，且只用同一会话内的墙钟差分，
  不假设回放起点墙钟、不假设原速。实际出现（作品时间轴）= 上式 + offset；
  时差 = 实际（作品轴）- 应在（cue 的 t0）。

用法：
  python tools/realtime_compare.py logs/benchmark_runs/scene_asmr299717_t02.jsonl ^
      --human dataset/scene_asmr299717_t02.human_zh.json ^
      --out logs/realtime_compare/scene_asmr299717_t02

  python tools/realtime_compare.py session.jsonl ^
      --human "D:/Downloads/asmr-zh-corpus/RJ299717/subtitles/zh/05_お茶の時間.lrc" ^
      --offset 412

退出码：0 成功；2 自动检测置信度低（需 --offset）；3 检测完全失败（无候选）。
"""
import argparse
import json
import re
import statistics
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from subtitles_to_gt import parse_lrc, parse_srt_vtt  # noqa: E402

# ---- 可调常数（默认值针对现有场景数据校准） ----
SIM_CAND = 0.50      # 生成偏移粗候选所需的最小相似度
SIM_MATCH = 0.45     # 计分时"文本相似配对"阈值
TOL_START = 8.0      # 段起点与 cue 起点的时间接近容忍（秒，吸收切块边界）
SCAN_RADIUS = 45.0   # 粗候选中心附近的滑窗半径（秒）
CLUSTER_GAP = 15.0   # 偏移候选聚类间隙（秒）
SPAN_TAIL_S = 8.0    # 会话尾部容差：最后一段结束后这么久内开始的 cue 仍算范围内
                     # （容纳录音末尾静音收尾；再往后的 cue 属"会话没听到"，不计 missing）
MIN_SPAN_OVERLAP = 1.0  # cue 窗与会话统计范围重叠低于此秒数视为范围外
                        # （gapless 字幕的 cue 窗含行间静音，切片级重叠不算覆盖）
MIN_LAT_OVERLAP = 0.5   # 时差选段要求段与 cue 窗重叠不低于此秒数（无则退化全取）
LRC_TAIL = 3.0       # 最后一条 LRC cue 的估值时长（LRC 无结束时间）

_PUNCT_RE = re.compile(r"[\s，。？！?!~～…、·「」『』（）()【】\[\]:：;；,.\"'“”‘’—\-ー]+")

DAY_S = 86400.0


def norm(text: str) -> str:
    return _PUNCT_RE.sub("", text or "").lower()


def sim_ratio(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def parse_wallclock(ts: str) -> float:
    """WorkLog 的 ts（"HH:MM:SS.mmm" 墙钟）→ 当日秒数。"""
    parts = ts.strip().split(":")
    sec = float(parts[-1])
    minute = int(parts[-2]) if len(parts) > 1 else 0
    hour = int(parts[-3]) if len(parts) > 2 else 0
    return hour * 3600 + minute * 60 + sec


def load_worklog(path: Path):
    """读 WorkLog jsonl → {seg_id: seg}；seg 含音频窗/ja/zh/上屏时刻口径所需字段。"""
    segs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        kind, sid = e.get("kind"), e.get("seg_id")
        if kind == "asr" and sid is not None:
            s = segs.setdefault(sid, {"seg_id": sid})
            s["t_start"] = float(e.get("t_start") or 0.0)
            s["t_end"] = float(e.get("t_end") or 0.0)
            s["ja"] = (e.get("ja") or "").strip()
            s["asr_ts"] = parse_wallclock(e["ts"])
            s["asr_s"] = float(e.get("asr_s") or 0.0)
        elif kind == "mt" and sid is not None:
            s = segs.setdefault(sid, {"seg_id": sid})
            s["mt_ts"] = parse_wallclock(e["ts"])
            s["mt_s"] = float(e.get("mt_s") or 0.0)
            s["zh"] = (e.get("zh") or "").strip()
            s["zh_refusal"] = e.get("zh_refusal")
    for s in segs.values():
        if "mt_ts" in s and "asr_ts" in s:
            d = s["mt_ts"] - s["asr_ts"]
            if d < 0:  # 跨午夜会话
                d += DAY_S
            s["mt_delay"] = d  # ASR 完成 → 翻译完成（墙钟差分）
            s["display_local"] = s["t_end"] + s.get("asr_s", 0.0) + d
    return segs


def displayed(segs):
    """已上屏的段：有 ja、有 zh 且非拒答、可算上屏时刻。"""
    return [s for s in segs.values()
            if s.get("ja") and s.get("zh") and not s.get("zh_refusal")
            and "display_local" in s]


def load_cues(path: Path):
    """人工字幕 → [{seg_id, t0, t1, zh}]（t0/t1 保持文件自身时间轴，不归零）。"""
    p = Path(path)
    ext = p.suffix.lower()
    raw = p.read_text(encoding="utf-8-sig", errors="replace")
    if ext == ".json":  # human_zh schema（subtitles_to_gt --human-zh 产物）
        data = json.loads(raw)
        cues = [{"seg_id": s.get("seg_id", i + 1), "t0": float(s["t0"]),
                 "t1": float(s["t1"]), "zh": (s.get("zh") or "").strip()}
                for i, s in enumerate(data["segments"])]
        fmt = "human_zh.json"
    elif ext == ".lrc":  # LRC 无结束时间：t1 顺延到下一条时间戳
        pts = parse_lrc(raw)
        cues = []
        for i, (t0, body) in enumerate(pts):
            t1 = pts[i + 1][0] if i + 1 < len(pts) else t0 + LRC_TAIL
            cues.append({"seg_id": i + 1, "t0": t0, "t1": max(t1, t0 + 0.5),
                         "zh": body})
        fmt = "lrc"
    elif ext in (".vtt", ".srt"):
        pts = parse_srt_vtt(raw)
        cues = [{"seg_id": i + 1, "t0": t0, "t1": t1, "zh": body}
                for i, (t0, t1, body) in enumerate(pts)]
        fmt = ext.lstrip(".")
    else:
        raise SystemExit(f"[ERR] 不支持的人工字幕格式: {ext}（支持 .json/.lrc/.srt/.vtt）")
    return cues, fmt


def detect_offset(disp_segs, cues):
    """滑窗搜索最优偏移。返回 (offset 或 None, 文本配对段数)。

    评分：offset 处，每条已上屏段找"时间接近（cue 窗平移后与段窗重叠，且
    段起点与 cue 起点差 <= TOL_START）且文本相似 >= SIM_MATCH"的最佳 cue，
    计配对段数与相似度和；先段数后相似和，取最大。"""
    seg_sims, cands = [], []
    for s in disp_segs:
        lst = []
        for c in cues:
            r = sim_ratio(s["zh"], c["zh"])
            if r >= SIM_MATCH:
                lst.append((c, r))
            if r >= SIM_CAND:
                cands.append(c["t0"] - s["t_start"])
        seg_sims.append((s, lst))
    if not cands:
        return None, 0, seg_sims
    cands.sort()
    clusters = [[cands[0]]]
    for v in cands[1:]:
        if v - clusters[-1][-1] <= CLUSTER_GAP:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    main = max(clusters, key=len)
    center = sorted(main)[len(main) // 2]

    def score(off):
        cnt, tot, dist = 0, 0.0, 0.0
        for s, lst in seg_sims:
            best, best_d = 0.0, 0.0
            for c, r in lst:
                c0, c1 = c["t0"] - off, c["t1"] - off
                if c0 < s["t_end"] and c1 > s["t_start"] \
                        and abs(s["t_start"] - c0) <= TOL_START and r > best:
                    best, best_d = r, abs(s["t_start"] - c0)
            if best >= SIM_MATCH:
                cnt += 1
                tot += best
                dist += best_d
        return cnt, tot, dist

    best_off, best_key = None, (-1, -1.0, float("inf"))
    steps = int(SCAN_RADIUS * 2) + 1
    for i in range(steps):
        off = round(center - SCAN_RADIUS + i)  # 1s 粒度
        key = score(off)
        # 先段数、后相似和、再对齐距离和（真偏移处配对段起点与 cue 起点最贴）
        if (key[0], key[1], -key[2]) > (best_key[0], best_key[1], -best_key[2]):
            best_off, best_key = off, key
    return best_off, best_key[0], seg_sims


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def pct(sorted_vals, q):
    """q 分位（0-1]，ceil 索引取值，n 小时不做插值。"""
    if not sorted_vals:
        return None
    import math
    return sorted_vals[max(0, math.ceil(q * len(sorted_vals)) - 1)]


def main():
    ap = argparse.ArgumentParser(
        prog="realtime_compare.py",
        description="层④真实场景对照：会话 WorkLog × 人工字幕时间轴（事后分析）。",
        epilog="示例：\n"
               "  python tools/realtime_compare.py logs/benchmark_runs/scene_asmr299717_t02.jsonl "
               "--human dataset/scene_asmr299717_t02.human_zh.json\n"
               "  python tools/realtime_compare.py session.jsonl "
               "--human 05_お茶の時間.lrc --offset 412\n"
               "offset 语义：作品时间 = 本地音频时间 + offset（秒）。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("worklog", help="live_sub 会话的 WorkLog jsonl（live 环回/回放产物）")
    ap.add_argument("--human", required=True,
                    help="人工字幕文件：human_zh.json / 原始 .lrc / .srt / .vtt（时间轴不归零）")
    ap.add_argument("--offset", type=float, default=None,
                    help="人工指定偏移（秒）：作品时间 = 本地音频时间 + offset；"
                         "缺省时自动检测（置信度低则拒绝并要求本参数）")
    ap.add_argument("--min-pairs", type=int, default=3,
                    help="自动检测的置信度阈值：文本配对段数低于该值视为低置信（默认 3）")
    ap.add_argument("--out", default=None,
                    help="输出前缀（生成 <prefix>.md 与 <prefix>.json）；"
                         "默认 <worklog 同目录>/<worklog名>.realtime_compare")
    ap.add_argument("--judge-out", default=None,
                    help="语义比对输入 json 路径（与 build_anchor_scene 的 judge_input 同构）；"
                         "默认 <prefix>.judge_input.json")
    args = ap.parse_args()

    wl_path = Path(args.worklog)
    if not wl_path.is_file():
        raise SystemExit(f"[ERR] WorkLog 不存在: {wl_path}")
    human_path = Path(args.human)
    if not human_path.is_file():
        raise SystemExit(f"[ERR] 人工字幕不存在: {human_path}")

    segs = load_worklog(wl_path)
    if not segs:
        raise SystemExit("[ERR] WorkLog 中没有 asr/mt 事件")
    disp = displayed(segs)
    cues, fmt = load_cues(human_path)
    if not cues:
        raise SystemExit("[ERR] 人工字幕解析出 0 条 cue")

    # ---- 偏移 ----
    n_match = None
    if args.offset is not None:
        offset, offset_source = args.offset, "manual"
    else:
        offset, n_match, _ = detect_offset(disp, cues)
        offset_source = "auto"
        if offset is None:
            print("[ERR] 自动检测失败：本地字幕与人工 cue 无任何文本相似候选，"
                  "请用 --offset 人工指定偏移（作品时间 = 本地音频时间 + offset）")
            sys.exit(3)
        if n_match < args.min_pairs:
            print(f"[WARN] 自动检测置信度低：仅 {n_match} 段文本配对（阈值 {args.min_pairs}）。"
                  f"疑似会话含暂停/拖进度条或字幕不匹配。请人工核对后用 --offset 指定；"
                  f"当前检测值 {offset:.1f}s 仅供参考，本次不出报告。")
            sys.exit(2)
    lo = min(s["t_start"] for s in segs.values() if "t_start" in s)
    hi = max(s["t_end"] for s in segs.values() if "t_end" in s)
    span_work = (lo + offset, hi + offset)  # 会话覆盖范围（作品时间轴）
    range_work = (span_work[0], span_work[1] + SPAN_TAIL_S)  # 统计范围（含尾部容差）

    # ---- 会话范围内的 cue / 范围外 cue ----
    in_range = [c for c in cues
                if overlap(c["t0"], c["t1"], range_work[0], range_work[1]) >= MIN_SPAN_OVERLAP]
    out_range = [c for c in cues if c not in in_range]

    # ---- 逐 cue 配对（口径同 build_anchor_scene：时间重叠聚合）+ 时差 ----
    pairs, missing, rows = [], [], []
    for cue in in_range:
        c0, c1 = cue["t0"] - offset, cue["t1"] - offset  # cue → 本地音频时间轴
        hits = [(overlap(s["t_start"], s["t_end"], c0, c1), sid)
                for sid, s in segs.items() if s.get("ja")]
        hits = [h for h in hits if h[0] > 0]
        if not hits:
            missing.append(cue)
            rows.append({"seg_id": cue["seg_id"], "t0": cue["t0"], "t1": cue["t1"],
                         "human": cue["zh"], "status": "missing",
                         "local_zh": "", "local_ja": "", "asr_seg_ids": [],
                         "display_work_s": None, "latency_s": None})
            continue
        sids = [sid for _, sid in sorted(hits, key=lambda h: segs[h[1]]["t_start"])]
        ja = " / ".join(segs[s]["ja"] for s in sids)
        zh = " / ".join(segs[s]["zh"] for s in sids
                        if segs[s].get("zh") and not segs[s].get("zh_refusal"))
        pairs.append({"seg_id": cue["seg_id"], "t0": cue["t0"], "t1": cue["t1"],
                      "ja": ja, "sakura_live": zh, "human": cue["zh"],
                      "asr_seg_ids": sids,
                      "overlap_s": round(sum(h[0] for h in hits), 2)})
        # 时差：取与本 cue 文本最相似且重叠足够的已上屏段（并列取重叠更大/更早段）
        disp_hits = [(overlap(segs[sid]["t_start"], segs[sid]["t_end"], c0, c1), sid)
                     for sid in sids
                     if segs[sid].get("zh") and not segs[sid].get("zh_refusal")
                     and "display_local" in segs[sid]]
        solid = [h for h in disp_hits if h[0] >= MIN_LAT_OVERLAP] or disp_hits
        if solid:
            best = max(solid, key=lambda h: (sim_ratio(segs[h[1]]["zh"], cue["zh"]), h[0],
                                             -segs[h[1]]["t_start"]))[1]
            s = segs[best]
            rows.append({"seg_id": cue["seg_id"], "t0": cue["t0"], "t1": cue["t1"],
                         "human": cue["zh"], "status": "paired",
                         "local_zh": s["zh"], "local_ja": s["ja"],
                         "asr_seg_ids": sids,
                         "display_work_s": round(s["display_local"] + offset, 2),
                         "latency_s": round(s["display_local"] - c0, 2)})
        else:
            rows.append({"seg_id": cue["seg_id"], "t0": cue["t0"], "t1": cue["t1"],
                         "human": cue["zh"], "status": "paired_no_display",
                         "local_zh": "", "local_ja": ja, "asr_seg_ids": sids,
                         "display_work_s": None, "latency_s": None})

    # ---- 多余：已上屏但与任何范围内 cue 零重叠 ----
    extra = []
    for s in disp:
        if all(overlap(s["t_start"], s["t_end"], c["t0"] - offset, c["t1"] - offset) == 0
               for c in in_range):
            extra.append({"seg_id": s["seg_id"], "local_s": [s["t_start"], s["t_end"]],
                          "work_s": [round(s["t_start"] + offset, 2),
                                     round(s["t_end"] + offset, 2)],
                          "ja": s["ja"], "zh": s["zh"]})
    no_zh = [s["seg_id"] for s in segs.values()
             if s.get("ja") and not (s.get("zh") and not s.get("zh_refusal"))]

    # ---- 时差统计 ----
    lats = sorted(r["latency_s"] for r in rows if r["latency_s"] is not None)
    lat_stats = {"n": len(lats),
                 "median_s": round(statistics.median(lats), 2) if lats else None,
                 "p90_s": round(pct(lats, 0.9), 2) if lats else None,
                 "max_s": round(max(lats), 2) if lats else None}
    n_disp = len(disp)
    coverage = round(len(pairs) / len(in_range), 3) if in_range else None

    # ---- 输出 ----
    prefix = Path(args.out) if args.out else wl_path.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    judge_path = Path(args.judge_out) if args.judge_out else \
        prefix.with_name(prefix.name + ".judge_input.json")
    meta = {"worklog": str(wl_path), "human": str(human_path), "human_format": fmt,
            "offset_s": round(offset, 2), "offset_source": offset_source,
            "text_matched_segs": n_match, "displayed_segs": n_disp,
            "segs_no_zh": len(no_zh),
            "session_local_s": [round(lo, 2), round(hi, 2)],
            "session_work_s": [round(span_work[0], 2), round(span_work[1], 2)],
            "cues_total": len(cues), "cues_in_range": len(in_range),
            "cues_out_of_range": len(out_range),
            "pairs": len(pairs), "missing": len(missing), "extra_displayed": len(extra),
            "coverage": coverage, "latency": lat_stats}

    (prefix.with_name(prefix.name + ".json")).write_text(
        json.dumps({"meta": meta, "cues": rows, "missing": missing, "extra": extra},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    judge_path.write_text(
        json.dumps({"pairs": pairs}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # ---- md 报告 ----
    md = []
    md.append("# realtime_compare 真实场景对照报告\n")
    md.append(f"- 会话 WorkLog: `{wl_path}`")
    md.append(f"- 人工字幕: `{human_path}`（{fmt}，共 {len(cues)} 条 cue，"
              f"会话范围内 {len(in_range)} 条）")
    src = f"自动检测（文本配对 {n_match} 段）" if offset_source == "auto" else "人工指定"
    md.append(f"- 偏移 offset: **{offset:.1f}s**（{src}；作品时间 = 本地音频时间 + offset）")
    md.append(f"- 会话覆盖（作品时间轴）: {span_work[0]:.1f} ~ {span_work[1]:.1f}s"
              f"（本地 {lo:.1f} ~ {hi:.1f}s）\n")
    md.append("## 时差汇总（实际 - 应在）\n")
    if lats:
        md.append(f"- 配对且已上屏 {lat_stats['n']} 条：中位数 **{lat_stats['median_s']}s**，"
                  f"p90 **{lat_stats['p90_s']}s**，最大 **{lat_stats['max_s']}s**")
    else:
        md.append("- 无可计算时差的配对")
    md.append(f"- 覆盖率: {len(pairs)}/{len(in_range)} = "
              f"{coverage * 100:.1f}%" if coverage is not None else "- 覆盖率: n/a")
    md.append(f"- missing（范围内无任何本地段覆盖）: **{len(missing)}**")
    md.append(f"- 多余/未配对字幕（已上屏但没配到任何 cue）: **{len(extra)}**"
              f"（另有 {len(no_zh)} 段有 ja 无 zh 未上屏）")
    md.append(f"- 会话范围外 cue（不计 missing）: {len(out_range)}\n")
    md.append("## 逐条明细\n")
    md.append("| cue | 应在(作品s) | 实际(作品s) | 时差(s) | 人工字幕 | 本地字幕(zh) | 本地ja |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["status"] == "missing":
            md.append(f"| {r['seg_id']} | {r['t0']:.2f} | — | — | {r['human']} | "
                      f"**missing** | — |")
        elif r["latency_s"] is None:
            md.append(f"| {r['seg_id']} | {r['t0']:.2f} | — | — | {r['human']} | "
                      f"(未上屏) | {r['local_ja'][:24]} |")
        else:
            md.append(f"| {r['seg_id']} | {r['t0']:.2f} | {r['display_work_s']:.2f} | "
                      f"{r['latency_s']:+.2f} | {r['human']} | {r['local_zh']} | "
                      f"{r['local_ja'][:24]} |")
    if missing:
        md.append("\n### missing 明细\n")
        for c in missing:
            md.append(f"- cue {c['seg_id']} @ {c['t0']:.2f}~{c['t1']:.2f}s: {c['zh']}")
    if extra:
        md.append("\n### 多余/未配对字幕明细\n")
        for e in extra:
            md.append(f"- seg {e['seg_id']} @ 作品 {e['work_s'][0]}~{e['work_s'][1]}s: "
                      f"{e['zh']}（{e['ja']}）")
    if out_range:
        md.append("\n### 会话范围外 cue（不计入统计）\n")
        for c in out_range:
            md.append(f"- cue {c['seg_id']} @ {c['t0']:.2f}s: {c['zh']}")
    md.append("\n---\n"
              "口径说明：实际出现时刻 = 段收口(t_end) + ASR 耗时(asr_s) + "
              "ASR完成→翻译完成墙钟差(mt.ts - asr.ts)，映射到作品时间轴后与 cue t0 相减；"
              "gapless 字幕的 cue t0 是上一条的结束边界，时差天然含正偏置。\n")
    (prefix.with_name(prefix.name + ".md")).write_text("\n".join(md), encoding="utf-8")

    # ---- stdout 摘要 ----
    print(f"[OK] offset={offset:.1f}s（{src}）  会话(作品轴) {span_work[0]:.1f}~{span_work[1]:.1f}s")
    print(f"[OK] cue 范围内 {len(in_range)}（范围外 {len(out_range)}）："
          f"配对 {len(pairs)}、missing {len(missing)}、多余 {len(extra)}，"
          f"覆盖率 {coverage * 100:.1f}%" if coverage is not None else "[OK] 无范围内 cue")
    if lats:
        print(f"[OK] 时差 n={lat_stats['n']}  中位 {lat_stats['median_s']}s  "
              f"p90 {lat_stats['p90_s']}s  最大 {lat_stats['max_s']}s")
    print(f"[OK] 报告: {prefix.with_name(prefix.name + '.md')}")
    print(f"[OK] 数据: {prefix.with_name(prefix.name + '.json')}")
    print(f"[OK] judge_input: {judge_path}（{len(pairs)} 对）")


if __name__ == "__main__":
    main()
