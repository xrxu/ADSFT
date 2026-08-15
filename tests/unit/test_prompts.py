"""US1 prompt 保真测试（原则 I：文本逐字来自 xlsx）。"""

from __future__ import annotations

import pytest

from aphasia.data import prompts
from aphasia.data.gold_reader import read_gold


@pytest.fixture(scope="module")
def criterion():
    return read_gold().criterion


def test_system_content_contains_xlsx_fields_verbatim(criterion):
    content = prompts.build_system_content(criterion)
    assert criterion.system in content
    assert criterion.field in content
    assert criterion.dialogue_scope in content
    assert criterion.diag_info_criterion in content
    assert criterion.diag_flue_criterion in content


def test_user_content_contains_dialogue_and_surfix(criterion):
    dialogue = "医生：你今天好吗？病人：好。"
    content = prompts.build_user_content(criterion, dialogue, "score")
    assert dialogue in content
    assert criterion.question_surfix in content
    assert criterion.dialogue_content in content


def test_score_user_uses_format_instruction(criterion):
    content = prompts.build_user_content(criterion, "dummy", "score")
    assert prompts.SCORE_OUTPUT_INSTRUCTION in content


def test_reason_user_uses_format_instruction(criterion):
    content = prompts.build_user_content(criterion, "dummy", "reason")
    assert prompts.REASON_OUTPUT_INSTRUCTION in content


def test_format_score_output_roundtrips():
    out = prompts.format_score_output(7, 5)
    assert out == "信息量=7 流畅度=5"
