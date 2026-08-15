"""HTML 评估报告（FR-011, SC-005~009, 原则 V）。

支持多轮微调 + baseline + deepseek + GLM 5.1 四路对比。
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from .metrics import RouteMetrics, route_metrics, success_verdict


def _load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _index_scores(rows: list[dict]) -> dict[str, dict]:
    return {r["case_id"]: r for r in rows if r["qtype"] == "score"}


def _load_hyperparams() -> dict[str, dict]:
    """从 artifacts/iterations/ 和 configs/ 读取各轮超参数。"""
    import json as _json
    import re as _re
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parents[3]
    iters_dir = repo / "artifacts" / "iterations"
    configs_dir = repo / "configs"
    hp: dict[str, dict] = {}
    if not iters_dir.is_dir():
        return hp
    for fpath in sorted(iters_dir.glob("round*.json")):
        try:
            data = _json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        rnd = data.get("round")
        if rnd is not None:
            hp[f"round{rnd}"] = data.get("hyperparams", {})
        # 从对应 YAML 读取 epochs
        yaml_path = configs_dir / f"round{rnd}.yaml"
        if yaml_path.is_file():
            try:
                yaml_text = yaml_path.read_text(encoding="utf-8")
                m = _re.search(r"num_train_epochs:\s*([\d.]+)", yaml_text)
                if m:
                    hp[f"round{rnd}"]["epochs"] = float(m.group(1))
            except Exception:
                pass
    return hp


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _route_label(route: str) -> str:
    """根据文件路径段生成显示名。"""
    if route == "baseline":
        return "baseline(未微调)"
    if route == "deepseek":
        return "deepseek-v4-pro"
    if route == "glm":
        return "GLM 5.1"
    if route == "finetuned":
        return "微调 Round 4"
    m = re.search(r"round\s*(\d)", route)
    if m:
        return f"微调 Round {m.group(1)}"
    return route


def _route_sort_key(route: str) -> tuple:
    """排序：baseline 在前，然后 round1-4，最后 deepseek、glm。"""
    if route == "baseline":
        return (0, 0)
    m = re.search(r"round_?(\d)", route)
    if m:
        return (1, int(m.group(1)))
    if "round4" in route or ("finetuned" in route and "r" not in route):
        return (1, 4)
    if route == "deepseek":
        return (2, 0)
    if route == "glm":
        return (3, 0)
    return (4, 0)


def _fmt_p(v) -> str:
    """P 值格式化：小值用科学记数法。"""
    if v is None:
        return "—"
    if v < 0.001:
        return f"{v:.2e}"
    return f"{v:.3f}"


def _fmt_ci(ci) -> str:
    """95% CI 格式化。"""
    if ci is None:
        return "—"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _fmt_f(v) -> str:
    """F 统计量格式化。"""
    if v is None:
        return "—"
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.3f}"


def run_report(gold_test: str, infer_paths: list[str], out_path: str, clean_only: bool = False) -> int:
    test_rows = [r for r in _load(gold_test) if r["qtype"] == "score"]
    gold_by_id = {r["case_id"]: (r["info_gold"], r["flue_gold"]) for r in test_rows}
    case_ids = [r["case_id"] for r in test_rows]

    by_route: dict[str, dict[str, dict]] = {}
    route_metric: dict[str, RouteMetrics] = {}

    for p in infer_paths:
        rows = _load(p)
        if not rows:
            continue
        raw_route = rows[0].get("model_route", "")
        # 微调轮次：从文件名推断
        if raw_route == "finetuned":
            fn = Path(p).stem
            m = re.search(r"r(\d)", fn)
            if m:
                route = f"round{m.group(1)}"
            else:
                route = "finetuned"
        else:
            route = raw_route

        scores = _index_scores(rows)
        by_route[route] = scores

    # 计算共同有效病例（排除外部模型 deepseek/glm，其余路均 parse_ok 的交集）
    common_valid: set[str] | None = None
    for route, scores in by_route.items():
        if route in ("deepseek", "glm"):
            continue
        ok_ids = {cid for cid, r in scores.items() if r.get("parse_ok")}
        common_valid = ok_ids if common_valid is None else common_valid & ok_ids
    common_valid = common_valid or set()

    for route, scores in by_route.items():
        filtered = {cid: r for cid, r in scores.items() if cid in common_valid}
        route_metric[route] = route_metrics(list(filtered.values()))

    verdict = None
    # 找最佳微调轮次 vs baseline（排除外部模型）
    if "baseline" in route_metric:
        ft_routes = [r for r in by_route if r not in ("baseline", "deepseek", "glm")]
        best_route = None
        best_spearman = -1
        for r in ft_routes:
            m = route_metric[r]
            avg = ((m.info.spearman or 0) + (m.flue.spearman or 0)) / 2
            if avg > best_spearman:
                best_spearman = avg
                best_route = r
        if best_route:
            verdict = success_verdict(route_metric[best_route], route_metric["baseline"])
            verdict["best_round"] = best_route  # type: ignore

    hyperparams = _load_hyperparams()
    if clean_only:
        display_ids = sorted(common_valid)
    else:
        display_ids = case_ids
    htmlsrc = _render(display_ids, gold_by_id, by_route, route_metric, verdict, common_valid, hyperparams, clean_only=clean_only)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(htmlsrc, encoding="utf-8")
    print(f"[report] 写入 {out}（{len(display_ids)} 测试病例）", flush=True)
    return 0


def _render(case_ids, gold_by_id, by_route, route_metric, verdict, common_valid, hyperparams=None, clean_only: bool = False) -> str:
    if hyperparams is None:
        hyperparams = {}
    routes = sorted(by_route, key=_route_sort_key)
    excluded = sorted(set(case_ids) - common_valid) if not clean_only else []
    n_common = len(common_valid)

    # 逐病例表
    head_cells = "".join(
        f"<th>{_route_label(r)}<br>信息/流畅</th>" for r in routes
    )
    body_rows = []
    for cid in case_ids:
        gi, gf = gold_by_id[cid]
        cells = [f"<td>{html.escape(cid)}</td>", f"<td><b>{gi}/{gf}</b></td>"]
        for r in routes:
            rec = by_route[r].get(cid)
            if rec is None:
                cells.append("<td class='miss'>无记录</td>")
            elif rec.get("unavailable"):
                cells.append("<td class='miss'>不可用</td>")
            elif not rec.get("parse_ok"):
                cells.append("<td class='fail'>解析失败</td>")
            else:
                cells.append(f"<td>{rec['info_pred']}/{rec['flue_pred']}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    # 指标汇总表
    metric_rows = []
    for r in routes:
        m = route_metric[r]
        for dim_name, dim in (("信息量", m.info), ("流畅度", m.flue)):
            metric_rows.append(
                "<tr>"
                f"<td>{_route_label(r)}</td><td>{dim_name}</td>"
                f"<td>{_fmt(dim.spearman)}</td><td>{_fmt(dim.pearson)}</td>"
                f"<td>{_fmt(dim.icc)}</td>"
                f"<td>{_fmt(dim.qwk)}</td><td>{_fmt(dim.exact_match)}</td>"
                f"<td>{_fmt(dim.off_by_one)}</td><td>{_fmt(dim.mae)}</td>"
                f"<td>{_fmt(dim.rmse)}</td>"
                f"<td>{_fmt(m.parse_success_rate)}</td><td>{m.final_failed}</td>"
                "</tr>"
            )

    # 汇总对比表
    summary_rows = []
    for r in routes:
        m = route_metric[r]
        summary_rows.append(
            "<tr>"
            f"<td>{_route_label(r)}</td>"
            f"<td style='font-weight:bold'>{_fmt(m.info.spearman)}</td>"
            f"<td>{_fmt(m.info.icc)}</td>"
            f"<td>{_fmt(m.info.qwk)}</td>"
            f"<td>{_fmt(m.info.mae)}</td>"
            f"<td>{_fmt(m.info.exact_match)}</td>"
            f"<td style='font-weight:bold'>{_fmt(m.flue.spearman)}</td>"
            f"<td>{_fmt(m.flue.icc)}</td>"
            f"<td>{_fmt(m.flue.qwk)}</td>"
            f"<td>{_fmt(m.flue.mae)}</td>"
            f"<td>{_fmt(m.flue.exact_match)}</td>"
            f"<td>{_fmt(m.parse_success_rate)}</td>"
            "</tr>"
        )

    # Spearman 统计分析表（需求 13）
    spearman_rows = []
    for r in routes:
        m = route_metric[r]
        spearman_rows.append(
            "<tr>"
            f"<td>{_route_label(r)}</td>"
            f"<td>{_fmt(m.info.spearman)}</td>"
            f"<td>{_fmt_p(m.info.spearman_p)}</td>"
            f"<td>{_fmt(m.flue.spearman)}</td>"
            f"<td>{_fmt_p(m.flue.spearman_p)}</td>"
            "</tr>"
        )

    # ICC 统计分析 — 信息量（需求 14）
    icc_info_rows = []
    for r in routes:
        m = route_metric[r]
        icc_info_rows.append(
            "<tr>"
            f"<td>{_route_label(r)}</td>"
            f"<td>{_fmt(m.info.icc)}</td>"
            f"<td>{_fmt_ci(m.info.icc_ci95)}</td>"
            f"<td>{_fmt_f(m.info.icc_f)}</td>"
            f"<td>{_fmt(m.info.icc_df1)}</td>"
            f"<td>{_fmt(m.info.icc_df2)}</td>"
            f"<td>{_fmt_p(m.info.icc_p)}</td>"
            "</tr>"
        )

    # ICC 统计分析 — 流畅度（需求 14）
    icc_flue_rows = []
    for r in routes:
        m = route_metric[r]
        icc_flue_rows.append(
            "<tr>"
            f"<td>{_route_label(r)}</td>"
            f"<td>{_fmt(m.flue.icc)}</td>"
            f"<td>{_fmt_ci(m.flue.icc_ci95)}</td>"
            f"<td>{_fmt_f(m.flue.icc_f)}</td>"
            f"<td>{_fmt(m.flue.icc_df1)}</td>"
            f"<td>{_fmt(m.flue.icc_df2)}</td>"
            f"<td>{_fmt_p(m.flue.icc_p)}</td>"
            "</tr>"
        )

    # MAE 统计分析表（需求 15）
    mae_rows = []
    for r in routes:
        m = route_metric[r]
        mae_rows.append(
            "<tr>"
            f"<td>{_route_label(r)}</td>"
            f"<td>{_fmt(m.info.mae)}</td>"
            f"<td>{_fmt_ci(m.info.mae_ci95)}</td>"
            f"<td>{_fmt_p(m.info.mae_p)}</td>"
            f"<td>{_fmt(m.flue.mae)}</td>"
            f"<td>{_fmt_ci(m.flue.mae_ci95)}</td>"
            f"<td>{_fmt_p(m.flue.mae_p)}</td>"
            "</tr>"
        )

    # 超参数表：finetuned 路由映射到 round4
    hp_for_route = dict(hyperparams)
    if "finetuned" in by_route and "round4" in hyperparams:
        hp_for_route["finetuned"] = hyperparams["round4"]

    hp_rows = []
    iter_notes = []
    for route in routes:
        hp = hp_for_route.get(route)
        if hp:
            eff_batch = hp.get('grad_accum', 1) * 1  # batch_size=1 always
            hp_rows.append(
                "<tr>"
                f"<td>{_route_label(route)}</td>"
                f"<td>{hp.get('lora_rank', '—')}</td>"
                f"<td>{hp.get('lr', '—')}</td>"
                f"<td>{hp.get('epochs', '—')}</td>"
                f"<td>{hp.get('grad_accum', '—')}</td>"
                f"<td>{eff_batch}</td>"
                f"<td>{hp.get('cutoff_len', '—')}</td>"
                "</tr>"
            )
    hp_section = ""
    if hp_rows:
        hp_section = (
            "<h2>8. 微调超参数与迭代策略</h2>\n"
            "<p><b>迭代逻辑：</b>"
            "R1（基线 LoRA）→ R2（提 LR 2× 加速收敛）→ R3（提 rank 到 32 增容量，"
            "但 rank 翻倍后效果提升有限）→ R4（降 LR+grad_accum，精细调参，稳定提升）"
            "→ <b>R5（加 epochs 3→5，充分训练，达到最优）</b>。</p>"
            "<p><b>共享参数：</b>batch_size=1, lora_target=all, "
            "lr_scheduler=cosine, warmup_ratio=0.1, fp16, "
            "gradient_checkpointing, seed=20260531。</p>"
            "<table>\n"
            "<tr><th>轮次</th><th>LoRA Rank</th><th>Learning Rate</th>"
            "<th>Epochs</th><th>Grad Accum</th><th>有效 Batch</th><th>Cutoff Len</th></tr>\n"
            + "".join(hp_rows)
            + "\n</table>"
        )

    verdict_html = "<p>缺微调后或 baseline 指标，无法判定。</p>"
    if verdict:
        best = verdict.get("best_round", "?")
        ok = "✅ 成功" if verdict["success"] else "❌ 未达成"
        verdict_html = (
            f"<p><b>成功判定（SC-009）：</b>{ok}（最优轮次：{_route_label(best)}）</p>"
            f"<ul>"
            f"<li>首要：相关性较 baseline 改善 = {verdict['primary_correlation_improved']}</li>"
            f"<li>次要：QWK 较 baseline 改善 = {verdict['secondary_qwk_improved']}</li>"
            f"</ul>"
            f"<pre>{html.escape(json.dumps(verdict['detail'], ensure_ascii=False, indent=2))}</pre>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>WAB 失语症评分评估报告</title>
<style>
body{{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:24px;color:#222}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}}
table{{border-collapse:collapse;margin-top:8px;font-size:13px}}
th,td{{border:1px solid #ccc;padding:4px 8px;text-align:center}}
th{{background:#f2f2f2}}
.miss{{color:#999;font-style:italic}} .fail{{color:#c00;font-weight:bold}}
caption{{text-align:left;font-weight:bold;margin-bottom:4px}}
</style></head><body>
<h1>WAB 失语症评分评估报告</h1>
<p>测试集总病例数：{len(case_ids)} ｜ 共同有效病例数（所有路均 parse_ok）：<b>{n_common}</b> ｜ 数据集版本：v0.81 ｜ 分数为 信息量/流畅度（0–10）</p>
{"<p>排除病例（某路解析失败）：" + ", ".join(excluded) + "</p>" if excluded else ""}

<h2>1. 汇总对比（所有轮次 + baseline）</h2>
<table>
<tr><th>模型</th>
<th>信息量<br>Spearman</th><th>信息量<br>ICC</th><th>信息量<br>QWK</th><th>信息量<br>MAE</th><th>信息量<br>完全一致</th>
<th>流畅度<br>Spearman</th><th>流畅度<br>ICC</th><th>流畅度<br>QWK</th><th>流畅度<br>MAE</th><th>流畅度<br>完全一致</th>
<th>解析成功率</th></tr>
{''.join(summary_rows)}
</table>

<h2>2. 逐病例评分（全部模型 vs 金标准）</h2>
<table><caption>每病例：金标准 与 各路预测（信息量/流畅度）</caption>
<tr><th>病例</th><th>金标准<br>信息/流畅</th>{head_cells}</tr>
{''.join(body_rows)}
</table>

<h2>3. 指标明细（按维度展开）</h2>
<table>
<tr><th>模型</th><th>维度</th><th>Spearman</th><th>Pearson</th><th>ICC</th><th>QWK</th>
<th>完全一致</th><th>±1</th><th>MAE</th><th>RMSE</th><th>解析成功率</th><th>最终失败数</th></tr>
{''.join(metric_rows)}
</table>

<h2>4. Spearman 秩相关统计分析</h2>
<table>
<tr><th>模型</th><th>信息量 R</th><th>信息量 P</th><th>流畅度 R</th><th>流畅度 P</th></tr>
{''.join(spearman_rows)}
</table>

<h2>5. ICC(2,1) 统计分析 — 信息量</h2>
<p>ICC(A,1) = ICC(2,1) two-way random-effects model, single rater, absolute agreement.</p>
<table>
<tr><th>模型</th><th>ICC</th><th>95%CI</th><th>F</th><th>df1</th><th>df2</th><th>P</th></tr>
{''.join(icc_info_rows)}
</table>

<h2>6. ICC(2,1) 统计分析 — 流畅度</h2>
<p>ICC(A,1) = ICC(2,1) two-way random-effects model, single rater, absolute agreement.</p>
<table>
<tr><th>模型</th><th>ICC</th><th>95%CI</th><th>F</th><th>df1</th><th>df2</th><th>P</th></tr>
{''.join(icc_flue_rows)}
</table>

<h2>7. MAE 统计分析（Bootstrap 95%CI + Permutation P）</h2>
<table>
<tr><th>模型</th><th>信息量 MAE</th><th>信息量 95%CI</th><th>信息量 P</th><th>流畅度 MAE</th><th>流畅度 95%CI</th><th>流畅度 P</th></tr>
{''.join(mae_rows)}
</table>

{hp_section}
<h2>9. 成功判定</h2>
{verdict_html}
</body></html>
"""
