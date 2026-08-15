"""本地推理：baseline（AWQ 基座）与 finetuned（基座+LoRA）（FR-009/010, research R4）。

读取 ShareGPT 格式 test.jsonl（messages 字段含 system/user/assistant），
取 system+user 生成，score 解析失败按 max-retry 重跑。
"""

from __future__ import annotations

import json
from pathlib import Path

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


def _load_model(base: str, adapter: str | None):
    """加载 AWQ 基座（+可选 LoRA 适配器）。延迟导入重依赖。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model, torch


def _generate(tok, model, torch, messages: list[dict]) -> str:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tok.decode(gen, skip_special_tokens=True).strip()


def run_local_infer(args) -> int:
    route = args.route  # finetuned | baseline
    rows = _load_test(args.test)
    adapter = args.adapter if route == "finetuned" else None
    round_n = None
    if adapter:
        m = Path(adapter).name
        if m.startswith("round") and m[5:].isdigit():
            round_n = int(m[5:])

    tok, model, torch = _load_model(
        args.base if hasattr(args, "base") else _default_base(), adapter
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for r in rows:
        prompt_msgs = _extract_prompt_messages(r)
        if r["qtype"] == "score":
            results.append(
                _infer_score(tok, model, torch, r, prompt_msgs, route, round_n, args.max_retry)
            )
        else:
            raw = _generate(tok, model, torch, prompt_msgs)
            results.append(_reason_record(r, route, round_n, raw))

    _write(out_path, results)
    failed = sum(1 for x in results if x["qtype"] == "score" and not x["parse_ok"])
    print(f"[infer:{route}] {len(results)} 条，score 最终解析失败 {failed} 条", flush=True)
    return 0


def _default_base() -> str:
    from .. import config

    return str(config.BASE_MODEL)


def _infer_score(tok, model, torch, r, prompt_msgs, route, round_n, max_retry) -> dict:
    raw = ""
    pr = None
    retries = 0
    for attempt in range(max_retry + 1):
        raw = _generate(tok, model, torch, prompt_msgs)
        pr = parse_scores(raw)
        if pr.ok:
            break
        retries = attempt + 1
    return {
        "case_id": r["case_id"],
        "model_route": route,
        "round": round_n,
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


def _reason_record(r, route, round_n, raw) -> dict:
    return {
        "case_id": r["case_id"],
        "model_route": route,
        "round": round_n,
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
