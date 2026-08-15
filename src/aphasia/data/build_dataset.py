"""构建训练/测试数据集（FR-001~005，data-model 实体 3）。

划分严格按"用途"列；排除无效/缺金标准/缺转写；每病例产出 score 与 reason 两类记录
（real 无理由时仅产 score）。每行 ShareGPT 格式（messages 字段含 system/user/assistant）。
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import config
from . import prompts
from .gold_reader import FakeCase, GoldCase, GoldData, read_gold


def _record(
    case_id: str,
    split: str,
    source: str,
    qtype: str,
    criterion,
    dialogue: str,
    assistant_content: str,
    info_gold: int,
    flue_gold: int,
) -> dict:
    system_content = prompts.build_system_content(criterion)
    user_content = prompts.build_user_content(criterion, dialogue, qtype)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    return {
        "messages": messages,
        "case_id": case_id,
        "split": split,
        "source": source,
        "qtype": qtype,
        "info_gold": info_gold,
        "flue_gold": flue_gold,
    }


def _usage_to_split(usage: str) -> str | None:
    if usage in config.USAGE_TRAIN:
        return "train"
    if usage in config.USAGE_TEST:
        return "test"
    return None  # 无效或未知 → 排除


def _read_transcript(conversation_dir: Path, case_id: str) -> str | None:
    f = conversation_dir / f"{case_id}.txt"
    if not f.exists():
        return None
    text = f.read_text(encoding="utf-8").strip()
    return text or None


def build_records(
    gold: GoldData, conversation_dir: Path
) -> tuple[list[dict], list[str]]:
    """返回 (records, log_lines)。"""
    records: list[dict] = []
    log: list[str] = []
    crit = gold.criterion

    # --- 真实病例（来自 conversation/ + 打分汇总金标准） ---
    for c in gold.cases:
        split = _usage_to_split(c.usage)
        if split is None:
            log.append(f"DROP {c.case_id}: 用途={c.usage!r}（排除）")
            continue
        if c.info_gold < 0 or c.flue_gold < 0:
            log.append(f"DROP {c.case_id}: 缺信息量/流畅度金标准")
            continue
        dialogue = _read_transcript(conversation_dir, c.case_id)
        if dialogue is None:
            log.append(f"DROP {c.case_id}: 缺对话转写 {c.case_id}.txt")
            continue
        # score 类必产
        records.append(
            _record(
                c.case_id, split, "real", "score", crit, dialogue,
                prompts.format_score_output(c.info_gold, c.flue_gold),
                c.info_gold, c.flue_gold,
            )
        )
        # real 无可靠理由文本 → 仅产 score（FR-005 已澄清）
        log.append(f"KEEP {c.case_id}: split={split} source=real (score only)")

    # --- 合成病例（fakeasr：含理由，产 score + reason） ---
    for i, f in enumerate(gold.fakes):
        fake_id = f"FA{i:03d}"
        # score
        records.append(
            _record(
                fake_id, "train", "fake", "score", crit, f.dialogue,
                prompts.format_score_output(f.info_gold, f.flue_gold),
                f.info_gold, f.flue_gold,
            )
        )
        # reason（有可靠理由来源）
        if f.reason.strip():
            records.append(
                _record(
                    fake_id, "train", "fake", "reason", crit, f.dialogue,
                    f.reason, f.info_gold, f.flue_gold,
                )
            )
        log.append(f"KEEP {fake_id}: split=train source=fake")

    return records, log


def run_build(gold_path: str, conversation_dir: str, out_dir: str) -> int:
    """CLI 入口。保真校验失败时由 read_gold 抛错，返回非 0。"""
    try:
        gold = read_gold(gold_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[build] 保真校验失败: {e}", flush=True)
        return 1

    conv_dir = Path(conversation_dir)
    records, log = build_records(gold, conv_dir)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = [r for r in records if r["split"] == "train"]
    test = [r for r in records if r["split"] == "test"]

    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "test.jsonl", test)
    (out / "build_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")

    n_test_cases = len({r["case_id"] for r in test})
    n_train_cases = len({r["case_id"] for r in train})
    print(
        f"[build] train={len(train)} 条 ({n_train_cases} 病例) "
        f"test={len(test)} 条 ({n_test_cases} 病例) "
        f"dataset_version={config.DATASET_VERSION}",
        flush=True,
    )
    return 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
