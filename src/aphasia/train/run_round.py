"""单轮 LORA 微调驱动（data-model 实体 4）。

调用 `llamafactory-cli train <round{N}.yaml>`，落 adapter，记录超参与 swanlab run 到
artifacts/iterations/round{N}.json。OOM/失败时返回非 0（原则 III）。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .. import config
from .make_lf_config import make_config, write_config_yaml


def run_round(args) -> int:
    round_n = args.round
    config_out = args.config_out or str(config.CONFIGS_DIR / f"round{round_n}.yaml")
    adapter_out = args.adapter_out or str(config.ARTIFACTS_DIR / "adapters" / f"round{round_n}")

    cfg = make_config(
        round_n=round_n,
        base_model=args.base,
        train_jsonl=args.data,
        adapter_out=adapter_out,
        lora_rank=args.lora_rank,
        lr=args.lr,
        grad_accum=args.grad_accum,
        num_epochs=args.epochs,
    )
    yaml_path = write_config_yaml(cfg, config_out)
    print(f"[train] round {round_n} 配置写入 {yaml_path}", flush=True)

    # 记录超参与数据集版本（原则 II：可追溯）
    iter_dir = config.ARTIFACTS_DIR / "iterations"
    iter_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "round": round_n,
        "lf_config_path": str(yaml_path),
        "hyperparams": {
            "lora_rank": args.lora_rank,
            "lr": args.lr,
            "grad_accum": args.grad_accum,
            "cutoff_len": config.TRAIN_CUTOFF_LEN,
        },
        "dataset_version": config.DATASET_VERSION,
        "seed": config.SEED,
        "adapter_path": adapter_out,
        "swanlab_run": cfg["swanlab_run_name"],
    }
    (iter_dir / f"round{round_n}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 执行微调（SWANLAB_MODE=local 绕过 API key 检查）
    env = {**dict(subprocess.os.environ), "SWANLAB_MODE": "local"}
    cmd = ["llamafactory-cli", "train", str(yaml_path)]
    print(f"[train] 执行: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        print(
            f"[train] round {round_n} 失败 (returncode={proc.returncode})；"
            f"若为 OOM 请下调 lora_rank/grad_accum 后重跑该轮（原则 III）",
            flush=True,
        )
        return proc.returncode
    print(f"[train] round {round_n} 完成，adapter -> {adapter_out}", flush=True)
    return 0
