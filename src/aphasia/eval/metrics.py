"""吻合度与运营指标（FR-014，research R7）。

吻合度（按信息量/流畅度两维 + 总体）：Spearman、Pearson、ICC、QWK、ExactMatch、±1、MAE、RMSE。
运营：解析成功率、重跑后最终失败数。
成功判据：相关性(首要)、QWK(次要) vs baseline（SC-009）。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import bootstrap, pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

from ..config import SEED


@dataclass
class DimMetrics:
    n: int
    spearman: float | None
    pearson: float | None
    icc: float | None
    qwk: float | None
    exact_match: float | None
    off_by_one: float | None
    mae: float | None
    rmse: float | None
    # 统计分析扩展字段（需求 13–15）
    spearman_p: float | None = None
    icc_ci95: list[float] | None = None
    icc_f: float | None = None
    icc_df1: int | None = None
    icc_df2: int | None = None
    icc_p: float | None = None
    mae_ci95: list[float] | None = None
    mae_p: float | None = None


def _corr(fn, pred, gold) -> tuple[float | None, float | None]:
    """计算相关系数，返回 (r, pvalue)。"""
    if len(pred) < 2:
        return None, None
    if len(set(pred)) < 2 or len(set(gold)) < 2:
        return None, None
    try:
        result = fn(pred, gold)
        r, p = result[0], result[1]
    except Exception:
        return None, None
    r = None if (r is None or math.isnan(r)) else float(r)
    p = None if (p is None or math.isnan(p)) else float(p)
    return r, p


def _compute_icc(pred: list[int], gold: list[int]) -> dict:
    """ICC(A,1) = ICC(2,1) two-way random model, single rater absolute agreement。

    返回 dict: icc, ci95([lower,upper]), f, df1, df2, pval。
    """
    if len(pred) < 2:
        return dict(icc=None, ci95=None, f=None, df1=None, df2=None, pval=None)
    if len(set(pred)) < 2 and len(set(gold)) < 2:
        return dict(icc=None, ci95=None, f=None, df1=None, df2=None, pval=None)
    try:
        df = pd.DataFrame({
            "target": gold + pred,
            "rater": ["gold"] * len(gold) + ["pred"] * len(pred),
            "subject": list(range(len(gold))) + list(range(len(pred))),
        })
        result = pg.intraclass_corr(
            data=df, targets="subject", raters="rater", ratings="target",
        ).set_index("Type")
        # ICC(A,1) = two-way random model, single rater, absolute agreement
        row = result.loc["ICC(A,1)"]
        icc_val = float(row["ICC"])
        if math.isnan(icc_val):
            return dict(icc=None, ci95=None, f=None, df1=None, df2=None, pval=None)
        ci95_raw = row["CI95"]
        pval = float(row["pval"])
        return dict(
            icc=icc_val,
            ci95=[float(ci95_raw[0]), float(ci95_raw[1])],
            f=float(row["F"]),
            df1=int(row["df1"]),
            df2=int(row["df2"]),
            pval=None if math.isnan(pval) else pval,
        )
    except Exception:
        return dict(icc=None, ci95=None, f=None, df1=None, df2=None, pval=None)


def _compute_mae_stats(pred: list[int], gold: list[int]) -> dict:
    """MAE Bootstrap 95% CI + Permutation P 值（需求 15）。

    CI: scipy.stats.bootstrap, 2000 resamples, BCa method。
    P: 双尾置换检验，H0 为随机配对不比模型预测差。
    """
    if len(pred) < 2:
        return dict(mae=None, ci95=None, pval=None)

    abs_errors = np.array([abs(p - g) for p, g in zip(pred, gold)])
    mae = float(np.mean(abs_errors))

    # Bootstrap 95% CI
    ci95: list[float] | None = None
    try:
        bs = bootstrap(
            (abs_errors,), np.mean,
            confidence_level=0.95,
            n_resamples=2000,
            method="BCa",
            random_state=np.random.RandomState(SEED),
        )
        ci95 = [float(bs.confidence_interval.low), float(bs.confidence_interval.high)]
    except Exception:
        pass

    # Permutation P 值（H0: 预测与金标准无关联）
    pval: float | None = None
    try:
        gold_arr = np.array(gold)
        n_perm = 2000
        rng = np.random.RandomState(SEED)
        perm_count = 0
        for _ in range(n_perm):
            shuffled = rng.permutation(pred)
            perm_mae = float(np.mean(np.abs(shuffled - gold_arr)))
            if perm_mae <= mae:
                perm_count += 1
        pval = perm_count / n_perm
    except Exception:
        pass

    return dict(mae=mae, ci95=ci95, pval=pval)


def dim_metrics(pred: list[int], gold: list[int]) -> DimMetrics:
    """单维度指标。pred/gold 为等长、已配对、均为有效整数的列表。"""
    n = len(pred)
    if n == 0:
        return DimMetrics(
            n=0,
            spearman=None, pearson=None, icc=None, qwk=None,
            exact_match=None, off_by_one=None, mae=None, rmse=None,
            spearman_p=None, icc_ci95=None, icc_f=None,
            icc_df1=None, icc_df2=None, icc_p=None,
            mae_ci95=None, mae_p=None,
        )
    exact = sum(p == g for p, g in zip(pred, gold)) / n
    off1 = sum(abs(p - g) <= 1 for p, g in zip(pred, gold)) / n
    mae = sum(abs(p - g) for p, g in zip(pred, gold)) / n
    rmse = math.sqrt(sum((p - g) ** 2 for p, g in zip(pred, gold)) / n)
    qwk: float | None
    try:
        qwk = float(
            cohen_kappa_score(gold, pred, weights="quadratic", labels=list(range(11)))
        )
        if math.isnan(qwk):
            qwk = None
    except Exception:
        qwk = None

    spearman_r, spearman_p = _corr(spearmanr, pred, gold)
    pearson_r, _ = _corr(pearsonr, pred, gold)

    icc_stats = _compute_icc(pred, gold)

    mae_stats = _compute_mae_stats(pred, gold)

    return DimMetrics(
        n=n,
        spearman=spearman_r,
        spearman_p=spearman_p,
        pearson=pearson_r,
        icc=icc_stats["icc"],
        icc_ci95=icc_stats["ci95"],
        icc_f=icc_stats["f"],
        icc_df1=icc_stats["df1"],
        icc_df2=icc_stats["df2"],
        icc_p=icc_stats["pval"],
        qwk=qwk,
        exact_match=exact,
        off_by_one=off1,
        mae=mae,
        mae_ci95=mae_stats["ci95"],
        mae_p=mae_stats["pval"],
        rmse=rmse,
    )


@dataclass
class RouteMetrics:
    info: DimMetrics
    flue: DimMetrics
    parse_success_rate: float | None
    final_failed: int

    def to_dict(self) -> dict:
        return {
            "info": asdict(self.info),
            "flue": asdict(self.flue),
            "parse_success_rate": self.parse_success_rate,
            "final_failed": self.final_failed,
        }


def route_metrics(pairs: list[dict]) -> RouteMetrics:
    """对单路推理结果（score 类记录列表）计算指标。

    pairs: 每项含 info_pred/flue_pred/info_gold/flue_gold/parse_ok。
    仅 parse_ok 的样本进入吻合度计算；parse_success_rate 与 final_failed 统计全部。
    """
    total = len(pairs)
    ok = [p for p in pairs if p.get("parse_ok")]
    info_pred = [p["info_pred"] for p in ok]
    info_gold = [p["info_gold"] for p in ok]
    flue_pred = [p["flue_pred"] for p in ok]
    flue_gold = [p["flue_gold"] for p in ok]
    psr = (len(ok) / total) if total else None
    return RouteMetrics(
        info=dim_metrics(info_pred, info_gold),
        flue=dim_metrics(flue_pred, flue_gold),
        parse_success_rate=psr,
        final_failed=total - len(ok),
    )


def success_verdict(finetuned: RouteMetrics, baseline: RouteMetrics) -> dict:
    """SC-009：相关性(首要)、QWK(次要) 微调后 vs baseline。"""

    def better(a, b) -> bool | None:
        # 双方都无值 → 无法判定
        if a is None and b is None:
            return None
        # finetuned 有值而 baseline 无值（如 baseline 预测无方差）→ 视为改善
        if b is None:
            return True
        if a is None:
            return False
        return a > b

    # 首要：相关性（两维 Spearman，任一改善即视为相关性改善方向）
    info_corr = better(finetuned.info.spearman, baseline.info.spearman)
    flue_corr = better(finetuned.flue.spearman, baseline.flue.spearman)
    info_qwk = better(finetuned.info.qwk, baseline.info.qwk)
    flue_qwk = better(finetuned.flue.qwk, baseline.flue.qwk)

    corr_improved = any(x for x in (info_corr, flue_corr) if x is not None)
    qwk_improved = any(x for x in (info_qwk, flue_qwk) if x is not None)
    return {
        "primary_correlation_improved": corr_improved,
        "secondary_qwk_improved": qwk_improved,
        "success": bool(corr_improved) or (corr_improved is None and bool(qwk_improved)),
        "detail": {
            "info_spearman": [baseline.info.spearman, finetuned.info.spearman],
            "flue_spearman": [baseline.flue.spearman, finetuned.flue.spearman],
            "info_qwk": [baseline.info.qwk, finetuned.info.qwk],
            "flue_qwk": [baseline.flue.qwk, finetuned.flue.qwk],
        },
    }
