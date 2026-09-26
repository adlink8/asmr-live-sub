# -*- coding: utf-8 -*-
"""nightly 翻译社团队列：构建筛选与检测过门判定。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from nightly_collect import build_translated_queue, translated_passes

CIRC = "MYHONYAKU"  # TRANSLATION_CIRCLES 成员
OTHER = "普通社团"


def _w(rid, circle, dl=0):
    return {"id": rid, "circle": {"name": circle}, "dl_count": dl,
            "title": f"t{rid}"}


def test_queue_filters_circle_anchor_done_rejected():
    from nightly_collect import ANCHOR
    anchor_id = sorted(ANCHOR)[0]  # 真锚点号，锁死防泄漏
    works = [
        _w(anchor_id, CIRC, dl=999),          # 锚点必须排除（防泄漏硬门）
        _w(1001, CIRC, dl=50),
        _w(1002, CIRC, dl=80),
        _w(1003, OTHER, dl=70),               # 非翻译社团不进队
        _w(1004, CIRC, dl=60),                # 已采
        _w(1005, CIRC, dl=40),                # 已拒
    ]
    exclude = {1004, 1005}
    q = build_translated_queue(works, {anchor_id} | exclude)
    assert [w["id"] for w in q] == [1002, 1001]  # dl_count 降序


def test_queue_empty_inputs():
    assert build_translated_queue([], set()) == []
    assert build_translated_queue([_w(1, OTHER)], set()) == []


def test_translated_passes_gate():
    good = {"sub_coverage": 0.96, "audio_bytes_unique": 100 * 1024 * 1024}
    cap = 800 * 1024 * 1024
    assert translated_passes(good, 0.5, cap)
    assert not translated_passes(good, 0.99, cap)          # 覆盖率不达标
    assert not translated_passes(good, 0.5, 50 * 1024 * 1024)  # 体积超限
    assert not translated_passes(None, 0.5, cap)           # 检测失败
    assert not translated_passes({"sub_coverage": None}, 0.5, cap)  # 缺字段=拦
    # 纯音效作品 coverage=0.0 必须拦截（0.0 不是缺失）
    assert not translated_passes({"sub_coverage": 0.0}, 0.5, cap)
