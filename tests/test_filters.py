"""UNIT — livesub.filters: every drop rule and its tuned thresholds.

Thresholds are pinned from measured distributions on real session logs
(see .workbuddy/memory). If a threshold must move, move it deliberately and
update the test — these numbers are load-bearing, not arbitrary.
"""
import pytest

from livesub.filters import (
    drop_reason,
    is_kana_loop,
    is_refusal,
    strip_punct,
)


class TestEmpty:
    def test_empty_string(self):
        assert drop_reason("") == "empty"

    def test_whitespace_only(self):
        assert drop_reason("   \n ") == "empty"


class TestHallucination:
    @pytest.mark.parametrize("text", [
        "ご視聴ありがとうございました",
        "チャンネル登録よろしくお願いします",
        "最後までご覧いただきありがとうございます",
        "お疲れ様でした",
        "Thank You For Watching",
    ])
    def test_known_hallucinations(self, text):
        assert drop_reason(text) == "hallucination"

    def test_normal_thanks_survives(self):
        # 日常台词里的"谢谢"不是幻觉特征串
        assert drop_reason("ありがとう") is None
        assert drop_reason("今日はありがとう") is None


class TestKanaLoop:
    def test_long_moan_repetition(self):
        assert is_kana_loop("じゅるるるるるるるじゅるじゅる") is True
        assert drop_reason("じゅるるるるるるるじゅるじゅる") == "kana-loop"

    def test_short_moan_not_loop(self):
        # <12 字符不判循环（短拟声是合法内容）
        assert is_kana_loop("じゅる") is False

    def test_single_char_dominance(self):
        # 单字符占比 >=55% 且该字符在拟声表内
        assert is_kana_loop("ん" * 20) is True

    def test_normal_text_not_loop(self):
        assert is_kana_loop("こんにちは今日はいい天気ですね") is False


class TestNoSpeech:
    def test_high_nsp_short_segment(self):
        assert drop_reason("なんか", no_speech_prob=0.8, audio_s=2.0) == "no-speech"

    def test_high_nsp_long_segment_passes(self):
        # 长段高 nsp 不判 no-speech（交给 low-conf 等规则）
        assert drop_reason("なんか", no_speech_prob=0.8, audio_s=3.0) is None

    def test_nsp_without_audio_s(self):
        assert drop_reason("なんか", no_speech_prob=0.8) == "no-speech"

    def test_low_nsp_passes(self):
        assert drop_reason("なんか", no_speech_prob=0.3, audio_s=2.0) is None


class TestLowConfidence:
    def test_long_low_logprob_dropped(self):
        assert drop_reason("そこまで焦って出てくることもないだろうに",
                           avg_logprob=-1.2, audio_s=5.0) == "low-conf"

    def test_short_segment_never_low_conf(self):
        # 短句 logprob 天然偏低，段长门槛保护正常短句（"すごい""危ないよ"）
        assert drop_reason("すごい", avg_logprob=-1.2, audio_s=2.0) is None

    def test_threshold_boundary(self):
        # -1.15 是严格小于：正好压在阈值上放行
        assert drop_reason("そこまで焦って出てくることもないだろうに",
                           avg_logprob=-1.15, audio_s=5.0) is None
        assert drop_reason("そこまで焦って出てくることもないだろうに",
                           avg_logprob=-1.1501, audio_s=5.0) == "low-conf"

    def test_normal_logprob_passes(self):
        # 实测正常段均值 -0.54~-0.61
        assert drop_reason("正常な文です", avg_logprob=-0.58, audio_s=5.0) is None


class TestRepetition:
    def test_high_compression_ratio(self):
        assert drop_reason("普通の文章", compression_ratio=2.5) == "repetition"

    def test_boundary(self):
        assert drop_reason("普通の文章", compression_ratio=2.4) is None


class TestSoundOnly:
    def test_pure_moan(self):
        assert drop_reason("はぁ…はぁ…") == "sound-only"

    def test_punctuation_only(self):
        assert drop_reason("……？！") == "sound-only"

    def test_kanji_always_kept(self):
        assert drop_reason("んあ……大豆") is None

    def test_hai_survives(self):
        # い 故意不在拟声表里：はい 必须活
        assert drop_reason("はい") is None

    def test_sound_only_24_char_boundary(self):
        # 全拟声且 <=24 字才判 sound-only；24 字（非重复串）命中
        assert drop_reason("はぁふひへほんっ" * 3) == "sound-only"

    def test_all_moan_over_24_chars_kept(self):
        # 25 字全拟声但非重复：超出 sound-only 长度窗，也不构成循环
        assert drop_reason("はぁふひへほんっ" * 3 + "あ") is None

    def test_long_moan_repetition_is_kana_loop_not_sound_only(self):
        # 判序：kana-loop 先于 sound-only。26 字纯 moan 重复串走循环规则
        assert drop_reason("はぁ" * 13) == "kana-loop"


class TestRefusal:
    def test_ai_refusal(self):
        assert is_refusal("作为一个AI语言模型，我无法翻译这段内容") is True

    def test_normal_unable_phrase_survives(self):
        # "我无法用语言形容" 是正常译文，不能误杀
        assert is_refusal("我无法用语言形容这种感觉") is False

    def test_empty(self):
        assert is_refusal("") is False

    def test_normal_translation(self):
        assert is_refusal("所谓的蜻蜓点水就是这么回事吧。") is False


class TestStripPunct:
    def test_basic(self):
        assert strip_punct("こんにちは。今日は！") == "こんにちは今日は"

    def test_fullwidth_and_ascii(self):
        assert strip_punct("a, b。c！d?") == "abcd"
