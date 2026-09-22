"""regroup 规则离线验证：用带时间边界的真实日志跑，看合并是否合理。

用法：
    python validate_regroup.py logs/work-YYYYMMDD.jsonl

只读日志，不改 live_sub.py，不联网。
关键：只吃**带 t_start/t_end** 的 asr 事件（新格式）。旧格式没有时间信息，
按规则会被 gap_s=None 全部拒绝合并——这正是我们要的保守行为。
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from regroup import regroup, should_merge, MAX_GAP_S


def load_sessions(path: str) -> list[list[dict]]:
    """按 start 事件切会话，收集带时间边界的 asr 段。"""
    sessions, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("kind") == "start":
            if cur:
                sessions.append(cur)
            cur = []
            continue
        if (r.get("kind") == "asr" and not r.get("drop") and r.get("ja")
                and r.get("t_start") is not None):
            cur.append({"ja": r["ja"], "t_start": r["t_start"], "t_end": r["t_end"]})
    if cur:
        sessions.append(cur)
    return [s for s in sessions if len(s) >= 2]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/work-20260920.jsonl"
    sessions = load_sessions(path)
    if not sessions:
        print(f"[skip] {path} 里没有带时间边界的 asr 段，无法验证。")
        print("       需要先用当前版本跑一场真实直播（会产生新格式日志）。")
        return

    total_in = total_out = 0
    reasons = Counter()
    gaps_merged, gaps_rejected = [], []

    for seq in sessions:
        merged = regroup(seq)
        total_in += len(seq)
        total_out += len(merged)
        for m in merged:
            if len(m["src"]) > 1:
                reasons[m["why"]] += 1
                gaps_merged.append(round(m["t_end"] - m["t_start"], 2))
            else:
                nf = m.get("_no_merge", "")
                if nf in ("gap-too-big", "prev-terminal"):
                    gaps_rejected.append(round(m["t_end"] - m["t_start"], 2))

    if not total_in:
        print("无数据")
        return

    print(f"输入段 {total_in} -> 合并后 {total_out}  (减少 {total_in-total_out}, "
          f"压缩 {100*(total_in-total_out)/total_in:.1f}%)")
    print(f"MAX_GAP_S = {MAX_GAP_S}s")
    print()
    print("合并触发理由分布:")
    for k, v in reasons.most_common():
        print(f"  {k}: {v}")
    print()
    print("=" * 72)
    print("抽样：实际合并的样子（检查是否合理）")
    print("=" * 72)
    shown = 0
    for seq in sessions:
        for m in regroup(seq):
            if len(m["src"]) > 1 and shown < 25:
                print(f"  gap 内合并 [{m['why']}]  t=[{m['t_start']},{m['t_end']}]")
                for s in m["src"]:
                    print(f"      - {s}")
                print(f"    => {m['text']}")
                print()
                shown += 1
        if shown >= 25:
            break


if __name__ == "__main__":
    main()
