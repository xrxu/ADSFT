"""评分输出解析（FR-010，research R6）。

将 score 问型的模型输出还原为 (info, flue)；两分均为 0–10 整数才 parse_ok=True。
针对带 <think> 推理过程的输出，取文本**末尾**的分数（模型最终答案），过滤中间推理噪音。

关键策略：
- 优先取 </think> 之后的匹配（模型最终输出）
- 要求信息量/流畅度两分**成对出现**（相距 ≤100 字符），避免从推理噪音中抓假值
- 不使用松散的 "X分" 兜底
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CN_NUM = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


@dataclass(frozen=True)
class ParseResult:
    info: int | None
    flue: int | None
    ok: bool


def _to_int(token: str) -> int | None:
    token = token.strip()
    if token.isdigit():
        v = int(token)
        return v if 0 <= v <= 10 else None
    if token in _CN_NUM:
        return _CN_NUM[token]
    return None


def _valid(v: int | None) -> bool:
    return v is not None and 0 <= v <= 10


_NUM = r"(\d{1,2}|[零一二两三四五六七八九十])"
_INFO_PAT = re.compile(r"信息量[^0-9零一二两三四五六七八九十]{0,6}" + _NUM)
_FLUE_PAT = re.compile(r"流畅度[^0-9零一二两三四五六七八九十]{0,6}" + _NUM)
# 成对模式：信息量=X ... 流畅度=Y（相距 ≤100 字符），两分都在 0–10 内
_PAIRED_PAT = re.compile(
    r"信息量[^0-9零一二两三四五六七八九十]{0,6}" + _NUM
    + r".{0,100}?"
    + r"流畅度[^0-9零一二两三四五六七八九十]{0,6}" + _NUM,
    re.DOTALL,
)


def _extract_paired(text: str) -> tuple[int | None, int | None]:
    """找成对的 '信息量=X ... 流畅度=Y'（相距 ≤100 字符），取最后一个匹配。"""
    matches = list(_PAIRED_PAT.finditer(text))
    if not matches:
        # 反向：流畅度在前
        rev_pat = re.compile(
            r"流畅度[^0-9零一二两三四五六七八九十]{0,6}" + _NUM
            + r".{0,100}?"
            + r"信息量[^0-9零一二两三四五六七八九十]{0,6}" + _NUM,
            re.DOTALL,
        )
        matches = list(rev_pat.finditer(text))
    if matches:
        m = matches[-1]  # 最后一个成对匹配
        groups = m.groups()
        # 判断顺序：第一个 group 的值
        info_val, flue_val = _to_int(groups[0]), _to_int(groups[1])
        # 如果反向匹配（流畅度在前），交换
        full_match = m.group()
        if full_match.startswith("流畅度"):
            info_val, flue_val = flue_val, info_val
        return info_val, flue_val
    return None, None


def _extract_last_individuals(text: str) -> tuple[int | None, int | None]:
    """找最后一个 info 和最后一个 flue（不成对要求），仅当它们相距 ≤100 字符时使用。"""
    info_matches = list(_INFO_PAT.finditer(text))
    flue_matches = list(_FLUE_PAT.finditer(text))
    if not info_matches or not flue_matches:
        return None, None
    i_last = info_matches[-1]
    f_last = flue_matches[-1]
    # 要求两分出现在相近位置（相距 ≤100 字符）
    if abs(i_last.start() - f_last.start()) > 100:
        return None, None
    return _to_int(i_last.group(1)), _to_int(f_last.group(1))


def parse_scores(text: str) -> ParseResult:
    """从模型输出解析信息量、流畅度分。"""
    if not text:
        return ParseResult(None, None, False)

    # 优先搜索 </think> 之后；无 </think> 则搜全文
    think_end = text.rfind("</think>")
    has_think = think_end >= 0
    after_think = text[think_end + len("</think>"):] if has_think else text

    # 1. 成对匹配（</think> 后 或 全文）
    info, flue = _extract_paired(after_think)
    if info is None and has_think:
        # </think> 后不成对 → 回退全文成对
        info, flue = _extract_paired(text)

    # 2. 成对失败 → 独立最后匹配（仅限 </think> 后，要求相距 ≤100）
    if info is None or flue is None:
        info, flue = _extract_last_individuals(after_think)
    if (info is None or flue is None) and has_think:
        info2, flue2 = _extract_last_individuals(text)
        if info is None:
            info = info2
        if flue is None:
            flue = flue2

    ok = _valid(info) and _valid(flue)
    return ParseResult(info if ok else info, flue if ok else flue, ok)
