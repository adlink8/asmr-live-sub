"""【已废弃 2026-09-20】regroup 规则离线验证（第一版，结论错误）。

!!! 不要再用这个文件的结果做决策 !!!

它失败的根本原因：**只看文本、不看时间**。日志里当时没有段边界，
它按"数组相邻 = 时间相邻"假设合并，结果把间隔 5~10 秒的两段也并了。
当时得出的"压缩 14.9%"是假象。

已被取代：
  - regroup.py              —— 带时间间隔约束的规则（纯函数模块）
  - logs/validate_regroup.py —— 用带 t_start/t_end 的日志验证，无时间戳则拒绝运行

**最终结论：sentence regrouping 本身就不该做。**
Segmenter 在音频层用 hang_s=2.0 已经完成了"按停顿合并"，
送进 ASR 的段间隔恒 >= 2.0s，文本层再做重组只会二选一：
阈值 <2.0 永不触发，>2.0 就并错。详见 .workbuddy/memory/MEMORY.md。

保留此文件仅作历史记录。
"""
import json
import re
import sys
from collections import Counter

# ---------- 规则定义（候选，待验证） ----------

# 终助词/语气词：出现在句尾说明话未说完，倾向与下一段合并
CONT_PARTICLES = ("ね", "よ", "わ", "の", "な", "か", "さ", "ぞ", "ぜ", "っ")
# 明确终结的标点：出现即认为句子完整，不合并
TERMINAL = ("。", "！", "!", "？", "?", "…", "ー")
# 相槌：短的纯应答，倾向并入上一段
AIZUCHI = {"うん", "はい", "ええ", "ううん", "そう", "そうですね", "なるほど",
           "へえ", "ほんと", "マジ", "うんうん", "はいはい"}
# 合并后的长度上限（字符）
MAX_MERGED = 42
# 相槌段的最长长度（超过就不算相槌）
AIZUCHI_MAX = 8


def tail_last(ja: str) -> str:
    """取末尾第一个非空白字符。"""
    s = ja.strip()
    return s[-1] if s else ""


def is_aizuchi(ja: str) -> bool:
    core = re.sub(r"[\s。、，,！!？?…ー~〜]+", "", ja)
    return len(core) <= AIZUCHI_MAX and (core in AIZUCHI or core in CONT_PARTICLES)


def should_merge(prev: str, cur: str) -> tuple[bool, str]:
    """判断 cur 是否应与 prev 合并。返回 (是否合并, 理由)。"""
    if not prev:
        return False, "no-prev"
    merged_len = len(prev) + len(cur)
    if merged_len > MAX_MERGED:
        return False, "too-long"
    t = tail_last(prev)
    # 规则 1：当前段是相槌/短语气词 → 并入上一段
    if is_aizuchi(cur):
        return True, "cur-aizuchi"
    # 规则 2：上一段以终助词结尾 → 话没说完
    if t in CONT_PARTICLES:
        return True, "prev-particle"
    # 规则 3：上一段以「、」结尾
    if t == "、" or t == ",":
        return True, "prev-comma"
    return False, "no-rule"


def regroup(seq: list[str]) -> list[tuple[str, list[str], str]]:
    """把段序列合并。返回 [(合并后文本, 来源段列表, 触发理由), ...]"""
    out = []
    buf = ""
    src: list[str] = []
    why = ""
    for ja in seq:
        if not buf:
            buf, src, why = ja, [ja], ""
            continue
        ok, reason = should_merge(buf, ja)
        if ok:
            buf = buf + ja
            src.append(ja)
            why = why or reason
        else:
            out.append((buf, src, why))
            buf, src, why = ja, [ja], ""
    if buf:
        out.append((buf, src, why))
    return out


def load_sessions(path: str) -> list[list[str]]:
    sessions, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") == "start":
                if cur:
                    sessions.append(cur)
                cur = []
                continue
            if r.get("kind") == "asr" and not r.get("drop") and r.get("ja"):
                cur.append(r["ja"])
    if cur:
        sessions.append(cur)
    return [s for s in sessions if len(s) >= 2]


def main():
    sessions = load_sessions("logs/work-20260920.jsonl")
    total_in = sum(len(s) for s in sessions)
    total_out = 0
    reasons = Counter()
    samples = []

    for si, seq in enumerate(sessions):
        merged = regroup(seq)
        total_out += len(merged)
        for text, src, why in merged:
            if len(src) > 1:
                reasons[why] += 1
                samples.append((si, src, text))

    print(f"输入段 {total_in} → 合并后 {total_out}  （减少 {total_in - total_out}，"
          f"压缩 {100*(total_in-total_out)/total_in:.1f}%）")
    print()
    print("合并触发理由分布:")
    for k, v in reasons.most_common():
        print(f"  {k}: {v}")
    print()
    print("=" * 70)
    print("抽样：合并样例（看是否合理）")
    print("=" * 70)
    for si, src, text in samples[:25]:
        print(f"[会话{si}] {' + '.join(repr(x) for x in src)}")
        print(f"    => {text}")
        print()


if __name__ == "__main__":
    main()
