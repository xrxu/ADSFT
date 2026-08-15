"""US3 指标测试（FR-014, research R7）。"""

from __future__ import annotations

import math

from aphasia.eval.metrics import dim_metrics, route_metrics, success_verdict


def test_perfect_match():
    pred = [0, 2, 5, 7, 10]
    gold = [0, 2, 5, 7, 10]
    m = dim_metrics(pred, gold)
    assert m.exact_match == 1.0
    assert m.off_by_one == 1.0
    assert m.mae == 0.0
    assert m.rmse == 0.0
    assert m.qwk == 1.0
    assert abs(m.spearman - 1.0) < 1e-9
    assert abs(m.pearson - 1.0) < 1e-9


def test_mae_rmse_known():
    pred = [3, 5]
    gold = [5, 5]
    m = dim_metrics(pred, gold)
    assert m.mae == 1.0  # (|3-5|+|5-5|)/2 = 1
    assert abs(m.rmse - math.sqrt((4 + 0) / 2)) < 1e-9  # sqrt(2)


def test_off_by_one_vs_exact():
    pred = [4, 6, 8]
    gold = [5, 5, 8]
    m = dim_metrics(pred, gold)
    assert m.exact_match == 1 / 3  # only last matches
    assert m.off_by_one == 1.0  # all within ±1


def test_route_metrics_parse_rate():
    pairs = [
        {"info_pred": 5, "flue_pred": 5, "info_gold": 5, "flue_gold": 5, "parse_ok": True},
        {"info_pred": None, "flue_pred": None, "info_gold": 6, "flue_gold": 4, "parse_ok": False},
    ]
    rm = route_metrics(pairs)
    assert rm.parse_success_rate == 0.5
    assert rm.final_failed == 1
    assert rm.info.n == 1


def test_success_verdict_improvement():
    good_pairs = [
        {"info_pred": g, "flue_pred": g, "info_gold": g, "flue_gold": g, "parse_ok": True}
        for g in [0, 3, 5, 8, 10]
    ]
    bad_pairs = [
        {"info_pred": 5, "flue_pred": 5, "info_gold": g, "flue_gold": g, "parse_ok": True}
        for g in [0, 3, 5, 8, 10]
    ]
    ft = route_metrics(good_pairs)
    base = route_metrics(bad_pairs)
    v = success_verdict(ft, base)
    assert v["success"] is True
    assert v["primary_correlation_improved"] is True


def test_spearman_p_value():
    """Spearman 应返回有效 P 值。"""
    pred = [0, 3, 5, 8, 10]
    gold = [0, 3, 5, 7, 10]
    m = dim_metrics(pred, gold)
    assert m.spearman is not None
    assert m.spearman_p is not None
    assert 0.0 <= m.spearman_p <= 1.0


def test_icc_extended_stats():
    """ICC 应返回 CI95、F、df1、df2、P。"""
    pred = [0, 2, 5, 7, 10]
    gold = [0, 2, 5, 7, 10]
    m = dim_metrics(pred, gold)
    assert m.icc is not None
    assert m.icc_ci95 is not None
    assert len(m.icc_ci95) == 2
    assert m.icc_ci95[0] - 1e-9 <= m.icc <= m.icc_ci95[1] + 1e-9
    assert m.icc_f is not None
    assert m.icc_df1 is not None
    assert m.icc_df2 is not None
    assert m.icc_p is not None


def test_mae_bootstrap_ci():
    """MAE Bootstrap CI 应包围点估计值。"""
    pred = [0, 3, 5, 8, 10]
    gold = [0, 3, 5, 7, 10]
    m = dim_metrics(pred, gold)
    assert m.mae is not None
    assert m.mae_ci95 is not None
    assert len(m.mae_ci95) == 2
    assert m.mae_ci95[0] <= m.mae <= m.mae_ci95[1]
    assert m.mae_p is not None
    assert 0.0 <= m.mae_p <= 1.0


def test_new_fields_none_when_empty():
    """空输入时新字段应为 None。"""
    m = dim_metrics([], [])
    assert m.spearman_p is None
    assert m.icc_ci95 is None
    assert m.icc_f is None
    assert m.icc_p is None
    assert m.mae_ci95 is None
    assert m.mae_p is None
