"""deepseek-v4-pro 外部推理 — 向后兼容 thin wrapper。

新代码请用 aphasia.infer.external_infer.run_external_infer(provider="deepseek")。
"""

from __future__ import annotations

from .external_infer import run_external_infer


def run_deepseek_infer(args) -> int:
    """委托给统一外部推理模块（provider=deepseek）。"""
    args.provider = "deepseek"
    return run_external_infer(args)
