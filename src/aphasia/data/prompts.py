"""从 ScoringCriterion 字段拼装 ShareGPT messages（原则 I：文本逐字来自 xlsx，不硬编码摘要）。

每个对话产出两类 messages：
- score：assistant 为"信息量=X 流畅度=Y"
- reason：assistant 为评分理由
"""

from __future__ import annotations

from ..data.gold_reader import ScoringCriterion

# 要求模型把分数放进固定标记内，提升解析鲁棒性（与 infer/parse.py 对应）
SCORE_OUTPUT_INSTRUCTION = (
    "请只输出两个 0–10 的整数，格式严格为：信息量=X 流畅度=Y（X、Y 为整数），不要输出其他内容。"
)
REASON_OUTPUT_INSTRUCTION = "请给出该病人信息量与流畅度评分的详细理由。"


def build_system_content(criterion: ScoringCriterion) -> str:
    """system prompt：医生设定 + 图画场景 + 问题列表 + 两维评分标准（全部取自 xlsx）。"""
    return "\n".join(
        [
            criterion.system,
            criterion.field,
            criterion.dialogue_scope,
            criterion.diag_info_criterion,
            criterion.diag_flue_criterion,
        ]
    )


def build_user_content(criterion: ScoringCriterion, dialogue: str, qtype: str) -> str:
    """user content：对话内容 + question_surfix + 输出格式约束。"""
    instruction = (
        SCORE_OUTPUT_INSTRUCTION if qtype == "score" else REASON_OUTPUT_INSTRUCTION
    )
    return "\n".join(
        [
            criterion.dialogue_content,
            dialogue,
            criterion.question_surfix,
            instruction,
        ]
    )


def format_score_output(info_gold: int, flue_gold: int) -> str:
    """score 类样本的目标输出，必须可被 parse 还原。"""
    return f"信息量={info_gold} 流畅度={flue_gold}"
