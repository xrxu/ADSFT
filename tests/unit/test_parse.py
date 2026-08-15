"""US3 评分解析测试（FR-010, research R6）。"""

from __future__ import annotations

import pytest

from aphasia.infer.parse import parse_scores


@pytest.mark.parametrize(
    "text,info,flue",
    [
        ("信息量=7 流畅度=5", 7, 5),
        ("信息量：8，流畅度：6", 8, 6),
        ("信息量 9 分，流畅度 4 分", 9, 4),
        ("该病人信息量是10分，流畅度是0分", 10, 0),
        ("信息量为五，流畅度为三", 5, 3),
        # 反向顺序也能解析
        ("流畅度=5 信息量=7", 7, 5),
    ],
)
def test_parse_ok_cases(text, info, flue):
    r = parse_scores(text)
    assert r.ok is True, f"expected ok for: {text!r}"
    assert r.info == info
    assert r.flue == flue


def test_paired_match_in_think_section():
    """优先取 </think> 后的成对匹配。"""
    text = "<think>\n信息量=3 流畅度=3 (错误推理)\n</think>\n信息量=7 流畅度=5"
    r = parse_scores(text)
    assert r.ok is True
    assert r.info == 7
    assert r.flue == 5


def test_no_flue_in_think_section_rejected():
    """</think> 后只有信息量没有流畅度 → parse 失败。"""
    text = "<think>信息量=6</think>\n信息量是6分信息量是6分信息量是6分"
    r = parse_scores(text)
    assert r.ok is False, "should fail: no 流畅度 in final, repetitive output"


def test_scattered_matches_rejected():
    """信息量和流畅度相距太远（>100 字符）→ parse 失败。"""
    text = "信息量=5" + "x" * 200 + "流畅度=3"
    r = parse_scores(text)
    assert r.ok is False


@pytest.mark.parametrize(
    "text",
    [
        "",
        "我无法评分",
        "信息量=11 流畅度=5",
        "信息量=7",
        "流畅度=5",
        "评估结果：8分 和 6分",  # 无 "信息量"/"流畅度" 标签
    ],
)
def test_parse_fail_cases(text):
    r = parse_scores(text)
    assert r.ok is False


def test_individual_close_match_after_think():
    """</think> 后两分虽不成对但相距 ≤100 字符 → 仍可解析。"""
    text = "结论：信息量得分为7。根据以上分析，流畅度得分为5。"
    r = parse_scores(text)
    assert r.ok is True
    assert r.info == 7
    assert r.flue == 5
