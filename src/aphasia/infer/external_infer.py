"""统一外部模型推理（OpenAI 兼容 API）：deepseek-v4-pro / GLM 5.1 等。

通过 --provider 参数选择目标模型；凭据经环境变量注入。
网络/配额/凭据缺失 → unavailable=true，不阻塞其余路。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

from ..config import ExternalModelConfig, load_external_configs
from .parse import parse_scores


def _load_test(test_path: str) -> list[dict]:
    rows = []
    with open(test_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_prompt_messages(rec: dict) -> list[dict]:
    """从 ShareGPT messages 中取 system + user（不包括 assistant/gold）。"""
    msgs = rec["messages"]
    return [m for m in msgs if m["role"] in ("system", "user")]


def _chat(cfg: ExternalModelConfig, messages: list[dict], timeout: int = 300) -> str:
    """调用 OpenAI 兼容 /chat/completions。"""
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": 0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def _unavailable_record(r: dict, provider: str, qtype: str) -> dict:
    return {
        "case_id": r["case_id"],
        "model_route": provider,
        "round": None,
        "qtype": qtype,
        "raw_output": "",
        "info_pred": None,
        "flue_pred": None,
        "reason_text": None,
        "parse_ok": False,
        "retry_count": 0,
        "unavailable": True,
        "info_gold": r["info_gold"],
        "flue_gold": r["flue_gold"],
    }


def run_external_infer(args) -> int:
    """统一外部推理入口。从 args.provider 选择目标模型（"deepseek" | "glm"）。"""
    provider = args.provider
    configs = load_external_configs()
    cfg = configs.get(provider)

    if cfg is None or not cfg.available:
        print(
            f"[infer:external:{provider}] 凭据缺失（环境变量未设），标记该路不可用",
            flush=True,
        )
        rows = _load_test(args.test)
        results = [_unavailable_record(r, provider, r["qtype"]) for r in rows]
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write(out_path, results)
        return 0

    rows = _load_test(args.test)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for r in rows:
        prompt_msgs = _extract_prompt_messages(r)
        if r["qtype"] == "score":
            results.append(_infer_score(cfg, provider, r, prompt_msgs, args.max_retry))
        else:
            results.append(_infer_reason(cfg, provider, r, prompt_msgs))
        # 每处理完一条就输出进度
        last = results[-1]
        if last["qtype"] == "score":
            status = "parse_ok" if last["parse_ok"] else ("unavailable" if last["unavailable"] else "parse_fail")
            print(
                f"[infer:external:{provider}] {last['case_id']} "
                f"info={last['info_pred']} flue={last['flue_pred']} {status} "
                f"({len(results)}/{len(rows)})",
                flush=True,
            )

    _write(out_path, results)
    unavail = sum(1 for x in results if x["unavailable"])
    failed = sum(
        1 for x in results
        if x["qtype"] == "score" and not x["parse_ok"] and not x["unavailable"]
    )
    print(
        f"[infer:external:{provider}] {len(results)} 条，不可用 {unavail}，"
        f"score 最终解析失败 {failed}",
        flush=True,
    )
    return 0


def _infer_score(
    cfg: ExternalModelConfig, provider: str, r: dict,
    prompt_msgs: list[dict], max_retry: int,
) -> dict:
    raw, pr, retries = "", None, 0
    for attempt in range(max_retry + 1):
        try:
            raw = _chat(cfg, prompt_msgs)
        except (urllib.error.URLError, KeyError, TimeoutError, OSError) as e:
            print(
                f"[infer:external:{provider}] {r['case_id']} 请求失败: {type(e).__name__}",
                flush=True,
            )
            return _unavailable_record(r, provider, "score")
        pr = parse_scores(raw)
        if pr.ok:
            break
        retries = attempt + 1
    return {
        "case_id": r["case_id"],
        "model_route": provider,
        "round": None,
        "qtype": "score",
        "raw_output": raw,
        "info_pred": pr.info if pr and pr.ok else None,
        "flue_pred": pr.flue if pr and pr.ok else None,
        "reason_text": None,
        "parse_ok": bool(pr and pr.ok),
        "retry_count": retries,
        "unavailable": False,
        "info_gold": r["info_gold"],
        "flue_gold": r["flue_gold"],
    }


def _infer_reason(
    cfg: ExternalModelConfig, provider: str, r: dict, prompt_msgs: list[dict],
) -> dict:
    try:
        raw = _chat(cfg, prompt_msgs)
    except (urllib.error.URLError, KeyError, TimeoutError, OSError) as e:
        print(
            f"[infer:external:{provider}] {r['case_id']} reason 请求失败: {type(e).__name__}",
            flush=True,
        )
        return _unavailable_record(r, provider, "reason")
    return {
        "case_id": r["case_id"],
        "model_route": provider,
        "round": None,
        "qtype": "reason",
        "raw_output": raw,
        "info_pred": None,
        "flue_pred": None,
        "reason_text": raw,
        "parse_ok": True,
        "retry_count": 0,
        "unavailable": False,
        "info_gold": r["info_gold"],
        "flue_gold": r["flue_gold"],
    }


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
