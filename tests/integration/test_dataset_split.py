"""US1 数据集完整性测试（SC-001/002）。运行于真实 gold xlsx。"""

from __future__ import annotations

import pytest

from aphasia import config
from aphasia.data.build_dataset import build_records
from aphasia.data.gold_reader import read_gold


@pytest.fixture(scope="module")
def gold():
    return read_gold()


@pytest.fixture(scope="module")
def built(gold):
    records, log = build_records(gold, config.CONVERSATION_DIR)
    return records, log


def test_no_invalid_cases_in_any_split(gold, built):
    records, _ = built
    invalid_ids = {c.case_id for c in gold.cases if c.usage in config.USAGE_INVALID}
    used_ids = {r["case_id"] for r in records}
    assert invalid_ids.isdisjoint(used_ids), "无效病例混入了数据集"


def test_train_only_train_usage(gold, built):
    records, _ = built
    train_ids = {r["case_id"] for r in records if r["split"] == "train" and r["source"] == "real"}
    usage_by_id = {c.case_id: c.usage for c in gold.cases}
    for cid in train_ids:
        assert usage_by_id[cid] in config.USAGE_TRAIN, f"{cid} 不应在训练集"


def test_test_only_test_usage(gold, built):
    records, _ = built
    test_ids = {r["case_id"] for r in records if r["split"] == "test"}
    usage_by_id = {c.case_id: c.usage for c in gold.cases}
    for cid in test_ids:
        assert usage_by_id[cid] in config.USAGE_TEST, f"{cid} 不应在测试集"


def test_test_set_case_count(built):
    records, _ = built
    test_cases = {r["case_id"] for r in records if r["split"] == "test"}
    # 测试用途共 26 病例；缺金标准/缺转写者会被剔除，故 <= 26 且 > 0
    assert 0 < len(test_cases) <= 26


def test_gold_scores_match_summary(gold, built):
    records, _ = built
    gold_by_id = {c.case_id: (c.info_gold, c.flue_gold) for c in gold.cases}
    for r in records:
        if r["source"] != "real":
            continue
        info, flue = gold_by_id[r["case_id"]]
        assert r["info_gold"] == info
        assert r["flue_gold"] == flue


def test_scores_in_range(built):
    records, _ = built
    for r in records:
        assert 0 <= r["info_gold"] <= 10
        assert 0 <= r["flue_gold"] <= 10
