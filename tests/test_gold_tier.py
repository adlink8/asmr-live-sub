"""金级层单元测试：双语字幕 cue 解析与时间轴直接配对（零 ASR 依赖）。"""
from tools.collect_finetune import pair_gold, parse_cues


LRC = """[ti:test]
[00:00.50]おはよう
[00:03.00]おやすみ
[00:10.00]また明日
"""


def test_parse_cues_lrc(tmp_path):
    p = tmp_path / "t.lrc"
    p.write_text(LRC, encoding="utf-8")
    cues = parse_cues(p)
    assert len(cues) == 3
    assert abs(cues[0][0] - 0.5) < 0.01
    assert cues[0][2] == "おはよう"
    assert abs(cues[0][1] - 3.0) < 0.01  # t1 顺延到下一条
    assert cues[2][1] > cues[2][0]       # 末条顺延 +10s


def test_parse_cues_untimed_txt_returns_empty(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("只是普通文本，没有时间戳", encoding="utf-8")
    assert parse_cues(p) == []


def test_pair_gold_overlap():
    zh = [(10.0, 13.0, "早上好"), (20.0, 24.0, "晚安"), (40.0, 42.0, "明天见")]
    ja = [(10.1, 12.8, "おはよう"), (20.5, 23.0, "おやすみ"), (30.0, 31.0, "無関係")]
    pairs = pair_gold(zh, ja, 0.0, 100.0)
    assert [p["human"] for p in pairs] == ["早上好", "晚安"]
    assert pairs[0]["ja"] == "おはよう"
    # 40s 的 zh cue 与任何 ja cue 无重叠 -> 不配
    assert all(p["human"] != "明天见" for p in pairs)


def test_pair_gold_window_filter():
    zh = [(10.0, 13.0, "窗外"), (200.0, 203.0, "窗外二")]
    ja = [(10.1, 12.8, "そと")]
    pairs = pair_gold(zh, ja, 180.0, 360.0)
    assert pairs == []
