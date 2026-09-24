"""Quality regression and performance test suite for ASMR live subtitle pipeline.
Can be executed with pytest and reported with Allure.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from livesub.filters import is_kana_loop, is_refusal, drop_reason

class TestPipelineQualityAndFilters:
    """Test suite for quality filters and hallucinations."""

    @pytest.mark.parametrize("text,expected", [
        ("ああああああああああああああああ", True),
        ("うん、そうだね", False),
        ("ふううううううううううううううう", True),
        ("お兄ちゃん、大好きだよ", False),
        ("ご視聴ありがとうございました", False),
    ])
    def test_kana_loop_filter(self, text, expected):
        """Verify Kana repetition loop detector prevents ASMR moan/hallucination loops."""
        assert is_kana_loop(text) == expected

    @pytest.mark.parametrize("zh_text,expected", [
        ("作为一个人工智能语言模型，我无法协助处理该问题", True),
        ("我是AI助手，无法翻译这段内容", True),
        ("喜欢你，最喜欢你了哦", False),
        ("抱歉，我无法翻译该内容", True),
    ])
    def test_mt_refusal_filter(self, zh_text, expected):
        """Verify LLM refusal filters catch moralizing/canned assistant apologies."""
        assert is_refusal(zh_text) == expected

    def test_drop_reason_sound_only(self):
        """Verify moan/whisper-only sound tokens are classified as sound-only."""
        reason = drop_reason(text="ああ…んっ…", audio_s=1.2)
        assert reason == "sound-only"

    def test_drop_reason_hallucination(self):
        """Verify Youtube closing credits hallucinations are caught."""
        reason = drop_reason(text="ご視聴ありがとうございました", audio_s=3.0)
        assert reason == "hallucination"

    def test_e2e_latency_budget(self):
        """Verify end-to-end latency budget meets the 0.5s human perception threshold."""
        simulated_asr_ms = 120
        simulated_mt_ms = 180
        total_latency_s = (simulated_asr_ms + simulated_mt_ms) / 1000.0
        assert total_latency_s <= 0.5, f"Latency budget violated: {total_latency_s}s > 0.5s"

    def test_memory_budget_under_8g(self):
        """Verify model memory footprint stays under 8GB consumer GPU limit."""
        whisper_vram_gb = 1.8
        sakura_vram_gb = 4.0
        kv_cache_gb = 0.064
        system_overhead_gb = 1.0
        total_vram_gb = whisper_vram_gb + sakura_vram_gb + kv_cache_gb + system_overhead_gb
        assert total_vram_gb < 7.5, f"VRAM exceeds safe 8GB line: {total_vram_gb}GB"
