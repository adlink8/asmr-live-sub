"""日语句子重组规则（离线验证版）。

背景：Segmenter 按能量切段，会把一句话切碎。典型是两种情况：
  1. 话说到一半换气 -> 终助词结尾（ね/よ/わ）却成了独立段
  2. 相槌（うん/はい）单独成段，跟上一句的语义其实连着

旧版规则（logs/test_regroup.py）失败的原因：**只看文本、不看时间**。
结果把间隔 5~10 秒的两段也合并了——那种间隔说明是两回事，不是一句话被切开。
现在 asr 事件带 t_start/t_end，可以算段间间隔了，用它做硬约束。

本模块只做纯函数，不依赖 live_sub，方便离线跑。
"""
import re

# 终助词：出现在句尾说明话未说完
CONT_PARTICLES = ("ね", "よ", "わ", "の", "な", "か", "さ", "ぞ", "ぜ", "っ")
# 明确终结的标点
TERMINAL = ("。", "！", "!", "？", "?", "…", "ー")
# 相槌
AIZUCHI = {"うん", "はい", "ええ", "ううん", "そう", "そうですね", "なるほど",
           "へえ", "ほんと", "マジ", "はいはい", "うんうん"}
# 合并后长度上限（字符）
MAX_MERGED = 42
# 相槌段最长长度
AIZUCHI_MAX = 8
# 【关键】段间静音间隔上限（秒）。超过这个值一定不合并。
# 依据：Segmenter 的 hang_s=2.0，也就是说一段结束到下一段起点之间，
# 若间隔接近或超过 2s，说明中间有真实停顿（话讲完了），不是被切开的半句。
MAX_GAP_S = 1.2

_CLEAN = re.compile(r"[\s。、，,！!？?…ー~〜]+")


def tail_char(ja: str) -> str:
    s = ja.strip()
    return s[-1] if s else ""


def core_of(ja: str) -> str:
    return _CLEAN.sub("", ja)


def is_aizuchi(ja: str) -> bool:
    core = core_of(ja)
    return len(core) <= AIZUCHI_MAX and (core in AIZUCHI or core in CONT_PARTICLES)


def should_merge(prev: str, cur: str, gap_s: float | None) -> tuple[bool, str]:
    """判断 cur 是否应与 prev 合并。返回 (是否合并, 理由)。

    gap_s: prev 结束到 cur 开始的静音间隔（秒）。None 表示不可知，此时最保守：不合并。
    """
    if not prev:
        return False, "no-prev"
    # 时间约束优先：间隔太大直接否掉，后面的文本规则不用看
    if gap_s is None:
        return False, "no-gap-info"
    if gap_s > MAX_GAP_S:
        return False, "gap-too-big"
    if len(prev) + len(cur) > MAX_MERGED:
        return False, "too-long"
    # prev 以终结标点结尾 -> 句子已完整，不合并（除非 cur 是相槌）
    t = tail_char(prev)
    if t in TERMINAL and not is_aizuchi(cur):
        return False, "prev-terminal"
    # 规则 1：cur 是相槌/短语气词
    if is_aizuchi(cur):
        return True, "cur-aizuchi"
    # 规则 2：prev 以终助词结尾 -> 话没说完
    if t in CONT_PARTICLES:
        return True, "prev-particle"
    # 规则 3：prev 以顿号结尾
    if t in ("、", ","):
        return True, "prev-comma"
    return False, "no-rule"


def regroup(items: list[dict]) -> list[dict]:
    """合并段序列。

    items: [{"ja": str, "t_start": float|None, "t_end": float|None}, ...]
    返回同结构，text 为合并后文本，src 为来源 ja 列表，why 为触发理由。
    """
    out: list[dict] = []
    buf = None
    for it in items:
        if buf is None:
            buf = {"text": it["ja"], "src": [it["ja"]],
                   "t_start": it.get("t_start"), "t_end": it.get("t_end"), "why": ""}
            continue
        gap = None
        if it.get("t_start") is not None and buf.get("t_end") is not None:
            gap = round(it["t_start"] - buf["t_end"], 3)
        ok, reason = should_merge(buf["text"], it["ja"], gap)
        if ok:
            buf["text"] = buf["text"] + it["ja"]
            buf["src"].append(it["ja"])
            buf["t_end"] = it.get("t_end")
            buf["why"] = buf["why"] or reason
        else:
            out.append(buf)
            buf = {"text": it["ja"], "src": [it["ja"]],
                   "t_start": it.get("t_start"), "t_end": it.get("t_end"),
                   "why": "", "_no_merge": reason}
    if buf is not None:
        out.append(buf)
    return out
