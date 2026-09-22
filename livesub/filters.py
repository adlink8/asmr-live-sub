"""Text-layer filters (layer >= 1): hallucination, kana-loop, sound-only,
low-confidence, repetition, and translation-refusal detection.

Thresholds were tuned against real session logs — see the comments on each
rule before changing anything; the defaults come from measured distributions,
not from paper values.
"""
import re

from livesub.config import LOWCONF_LOGPROB, LOWCONF_MIN_S

# layer 1: community post-filters (WhisperJAV / LocalLLaMA / zenn.jp)
# い excluded from moan set so はい survives; kanji always keeps the line
HALLU_RE = re.compile(
    r"ご(視聴|覧).{0,16}(ありがとう|感謝)"
    r"|チャンネル登録"
    r"|(最後まで|どうも).{0,16}(ありがとう|お(聞き|読み))"
    r"|(お疲れ様|おつかれさま)でし(た|て)"
    r"|thank you for watching"
    r"|thanks for watching",
    re.I,
)
# い excluded so はい survives; か/さ/た/な/ま/や/ら/わ 行常用词不进表
_MOAN = set(
    "あぁぅうぇえぉおんっはぁふひへほ"
    "ぐグぎギぷプぶブぺペぽポゅュょョヴ"
    "きキァゥェォンッハヒフヘホー～〜"
)
_PUNCT = re.compile(r"[\s\-ー～〜…・。、，,！!？?~゛゜「」『』（）()【】\[\]]+")
_LOOP_UNIT = re.compile(r"(.{1,4})\1{5,}")

# 翻译侧拒答检测。Sakura 偶尔不翻译而是回一段"我是AI我无法…"，
# 这类输出 zh_len 正常、zh_empty=False，光看长度完全正常，
# 但会被原样塞进字幕显示给用户——属于用户可见的故障。
# 只抓高置信度特征串，宁可漏也不误杀：正常翻译里不会出现这些整句。
# 注意「我无法/我不能」这类**不能**裸匹配：「我无法用语言形容」是正常译文。
# 必须要求后面跟"拒绝类动词"，且句中不能是普通的日常表达。
REFUSAL_RE = re.compile(
    r"作为一个.{0,6}(AI|人工智能|语言模型)"
    r"|我是.{0,6}(AI|人工智能|语言模型|助手)"
    r"|(无法|不能)(翻译|协助|回答|提供|完成|满足|处理)(这|该|您|你的|此|本)?(个|段|项|要求|请求|内容|文本|问题)?"
    r"|抱歉[，,].{0,12}(无法|不能|帮不上)"
    r"|对不起[，,].{0,12}(无法|不能|帮不上)"
    r"|我(无法|不能|没有办法)理解.{0,6}(这|该|这段|该段)"
    r"|请(提供|给出).{0,8}(文本|日文|原文|内容)"
    r"|(翻译|内容).{0,4}(超出|违反).{0,6}(范围|规定|政策)",
    re.I,
)


def is_refusal(zh: str) -> bool:
    """译文中是否出现"模型拒答"特征。用于丢弃而不是当字幕显示。"""
    return bool(zh) and bool(REFUSAL_RE.search(zh))


def strip_punct(text: str) -> str:
    """去掉标点与空白后的核心文本。sound-only 判定与去重窗口共用。"""
    return _PUNCT.sub("", text)


def is_kana_loop(text: str) -> bool:
    compact = strip_punct(text)
    if len(compact) < 12:
        return False
    freq: dict[str, int] = {}
    for c in compact:
        freq[c] = freq.get(c, 0) + 1
    ch, n = max(freq.items(), key=lambda kv: kv[1])
    if n / len(compact) >= 0.55 and ch in _MOAN:
        return True
    return _LOOP_UNIT.search(compact) is not None


def drop_reason(text: str, no_speech_prob=None, avg_logprob=None,
                audio_s=None, compression_ratio=None) -> str | None:
    t = text.strip()
    if not t:
        return "empty"
    if HALLU_RE.search(t):
        return "hallucination"
    if is_kana_loop(t):
        return "kana-loop"
    if no_speech_prob is not None and no_speech_prob > 0.6:
        if audio_s is None or audio_s <= 2.5:
            return "no-speech"
    # 原实现要求 avg_logprob<-1.0 与 no_speech_prob>0.6 同时成立才判 low-conf，
    # 实测 0 次触发；而 >6s 的长段平均 logprob 已是 -1.15，明显低质却全放行。
    # 置信度与"是否语音"是两个独立信号，分开判。
    # 阈值必须带段长门槛：短句的 avg_logprob 天然偏低，独立触发会误杀
    # "すごい""危ないよ"这类正常短句。
    # 阈值本身也从 -1.0 收紧到 -1.15：实测正常段的平均 logprob 是 -0.54~-0.61，
    # 而"お疲れ様です"(-1.03)、"今のがいいね"(-1.00) 这类正常常用语会擦线误杀，
    # -1.0 卡在真实分布边缘太危险。-1.15 与长段幻觉的实际水平对齐。
    if avg_logprob is not None and avg_logprob < LOWCONF_LOGPROB:
        if audio_s is None or audio_s > LOWCONF_MIN_S:
            return "low-conf"
    # compression_ratio 一直在采集却从未参与判断。Whisper 的经典重复幻觉信号
    # 就是该值偏高（>2.4），白捡的过滤条件。
    if compression_ratio is not None and compression_ratio > 2.4:
        return "repetition"
    core = strip_punct(t)
    if not core:
        return "sound-only"
    if all(c in _MOAN for c in core) and len(core) <= 24:
        return "sound-only"
    return None
