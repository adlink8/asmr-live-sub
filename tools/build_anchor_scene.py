"""asmr.one 人工字幕锚点场景构建：live jsonl + human_zh.json → live_ja.json + judge_input.json。

口径（与 scene_asmr299717_t01 同轨，差异点显式记录）：
- JA 输入 = 实时 ASR 输出（**非真值**，本类作品无日文字幕）；参考 = 人工中文字幕（绝对锚点）；
- 以人工 cue 为分母逐个配对：取与其时间窗重叠的 ASR 段，ja / 译文按段序拼接；
- 无任何有效 ASR 段覆盖的 cue 记入 missing（该处端到端零产出，判不可验证）；
- 不出 stepflash 云端参考：无 API key 时不造假，judge_input 只含 sakura_live 与 human。

用法：
  python tools/build_anchor_scene.py scene_asmr401391_t01 ^
    --live logs/benchmark_runs/scene_asmr401391_t01.jsonl ^
    --human dataset/scene_asmr401391_t01.human_zh.json ^
    --audio D:/Downloads/asmr-zh-corpus/RJ401391/audio/Track01_夢の中で出会い.wav
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_events(jsonl_path: Path):
    asr, mt = {}, {}
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        kind = e.get("kind")
        sid = e.get("seg_id")
        if kind == "asr" and sid is not None:
            asr[sid] = {"seg_id": sid, "t0": e.get("t_start") or 0.0,
                        "t1": e.get("t_end") or 0.0, "ja": (e.get("ja") or "").strip()}
        elif kind == "mt" and sid is not None:
            if e.get("zh") and not e.get("zh_refusal"):
                mt[sid] = e["zh"].strip()
    return asr, mt


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--live", required=True, help="live_sub 回放产出的 work jsonl")
    ap.add_argument("--human", required=True, help="subtitles_to_gt --human-zh 产出")
    ap.add_argument("--audio", default="", help="音轨路径（仅记录 provenance）")
    ap.add_argument("--out-dir", default=str(ROOT / "dataset"))
    args = ap.parse_args()

    asr, mt = load_events(Path(args.live))
    human = json.loads(Path(args.human).read_text(encoding="utf-8"))

    # live_ja.json：ASR 段形状（与 scene_asmr299717_t01 同 schema；本类作品无 JA 字幕，
    # official_ja 恒为空——JA 真值不存在，端到端裁判只回答翻译意思对不对）
    live_segs = []
    for sid in sorted(asr):
        s = asr[sid]
        live_segs.append({
            "seg_id": sid, "t0": round(s["t0"], 2), "t1": round(s["t1"], 2),
            "official_ja": "", "live_ja": s["ja"],
            "excluded": not s["ja"], "exclude_reason": "" if s["ja"] else "asr_empty",
        })
    live_ja = {
        "script_file": Path(args.human).name,
        "audio_file": args.audio,
        "total_segments": len(live_segs),
        "gt_source": "none (asmr.one work has no JA subtitle; JA input is live ASR output)",
        "note": "翻译实验输入：实时 ASR 日文段，非真值",
        "segments": live_segs,
    }

    # judge_input：以人工 cue 为分母，按时间重叠聚合 ASR 段
    pairs, missing = [], []
    for cue in human["segments"]:
        c0, c1 = float(cue["t0"]), float(cue["t1"])
        hits = [(overlap(s["t0"], s["t1"], c0, c1), sid)
                for sid, s in asr.items() if s["ja"]]
        hits = sorted((h for h in hits if h[0] > 0), key=lambda h: -h[0])
        if not hits:
            missing.append({"seg_id": cue["seg_id"], "t0": c0, "t1": c1,
                            "human": cue["zh"]})
            continue
        sids = [sid for _, sid in sorted(hits, key=lambda h: asr[h[1]]["t0"])]
        ja = " / ".join(asr[s]["ja"] for s in sids)
        zh = " / ".join(mt[s] for s in sids if s in mt)
        pairs.append({
            "seg_id": cue["seg_id"], "t0": c0, "t1": c1,
            "ja": ja, "sakura_live": zh, "human": cue["zh"],
            "asr_seg_ids": sids,
            "overlap_s": round(sum(h[0] for h in hits), 2),
        })

    judge_input = {"pairs": pairs}
    out_dir = Path(args.out_dir)
    (out_dir / f"{args.name}.live_ja.json").write_text(
        json.dumps(live_ja, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{args.name}.judge_input.json").write_text(
        json.dumps(judge_input, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {args.name}.live_ja.json: {len(live_segs)} ASR 段")
    print(f"[OK] {args.name}.judge_input.json: {len(pairs)} 对（missing {len(missing)}）")
    if missing:
        print("     missing cue:", [(m['seg_id'], m['human'][:20]) for m in missing[:8]])


if __name__ == "__main__":
    main()
