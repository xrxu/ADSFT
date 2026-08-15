"""生成 LLaMA-Factory LORA 微调配置（FR-006~008/013，原则 II/III）。

要点：
- AWQ 基座走 PTQ，**不设** quantization_bit（research R1）。
- cutoff_len 固定 8192（≥下限）；gradient_checkpointing + batch1 + grad_accum 控显存。
- swanlab local 跟踪；固定 seed；数据集用 alpaca 格式（system/instruction/input/output）。
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .. import config

# 数据集名（注册进生成的 dataset_info.json）
DATASET_NAME = "wab_aphasia"


def write_dataset_info(train_jsonl: Path, dest_dir: Path) -> Path:
    """生成 LLaMA-Factory 可识别的 dataset_info.json（alpaca 格式）。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    info = {
        DATASET_NAME: {
            "file_name": str(Path(train_jsonl).resolve()),
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    p = dest_dir / "dataset_info.json"
    p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def make_config(
    round_n: int,
    base_model: str,
    train_jsonl: str,
    adapter_out: str,
    *,
    lora_rank: int = 16,
    lr: float = 1e-4,
    grad_accum: int = 8,
    num_epochs: float = 3.0,
) -> dict:
    """返回 LLaMA-Factory YAML 配置 dict。"""
    dataset_dir = config.CONFIGS_DIR / f"round{round_n}_data"
    write_dataset_info(Path(train_jsonl), dataset_dir)

    return {
        # model
        "model_name_or_path": base_model,
        "trust_remote_code": True,
        # method（AWQ 基座：不设 quantization_bit，走 PTQ；LoRA 挂在量化权重上）
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": lora_rank,
        "lora_target": "all",
        # dataset
        "dataset": DATASET_NAME,
        "dataset_dir": str(dataset_dir),
        "template": "qwen",
        "cutoff_len": config.TRAIN_CUTOFF_LEN,  # 8192，硬约束
        "overwrite_cache": True,
        "preprocessing_num_workers": 8,
        # output
        "output_dir": str(adapter_out),
        "logging_steps": 5,
        "save_steps": 200,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": True,
        # swanlab local（原则 II）
        "report_to": "swanlab",
        "use_swanlab": True,
        "swanlab_mode": "local",
        "swanlab_project": "wab-aphasia",
        "swanlab_run_name": f"round{round_n}-rank{lora_rank}-lr{lr:g}",
        "swanlab_logdir": str(config.ARTIFACTS_DIR / "swanlab"),
        # train（显存控制：batch1 + grad_accum + 梯度检查点）
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": grad_accum,
        "gradient_checkpointing": True,
        "learning_rate": lr,
        "num_train_epochs": num_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "fp16": True,  # AWQ 仅支持 fp16（不支持 bf16）
        "seed": config.SEED,  # 可复现
        "ddp_timeout": 180000000,
    }


def write_config_yaml(cfg: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    return path
