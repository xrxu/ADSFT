"""统一 CLI 入口：build / train / infer / report（契约见 contracts/cli.md）。

用法：conda run -n llama_factory python -m aphasia.cli <subcommand> ...
"""

from __future__ import annotations

import argparse
import sys

from . import config


def _add_build(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("build", help="构建训练/测试数据集")
    p.add_argument("--gold", default=str(config.GOLD_XLSX))
    p.add_argument("--conversation-dir", default=str(config.CONVERSATION_DIR))
    p.add_argument("--out", default=str(config.ARTIFACTS_DIR / "dataset"))


def _add_train(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="单轮 LORA 微调")
    p.add_argument("--round", type=int, required=True, choices=range(1, 11))
    p.add_argument("--base", default=str(config.BASE_MODEL))
    p.add_argument("--data", default=str(config.ARTIFACTS_DIR / "dataset" / "train.jsonl"))
    p.add_argument("--config-out", default=None)
    p.add_argument("--adapter-out", default=None)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=float, default=3.0)


def _add_infer(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("infer", help="单路推理")
    p.add_argument("--route", required=True,
                   choices=["finetuned", "baseline", "deepseek", "external"])
    p.add_argument("--provider", default=None, choices=["deepseek", "glm"],
                   help="外部模型 provider（仅 --route external 需要）")
    p.add_argument("--test", default=str(config.ARTIFACTS_DIR / "dataset" / "test.jsonl"))
    p.add_argument("--adapter", default=None, help="finetuned 路的 LoRA 适配器目录")
    p.add_argument("--out", required=True)
    p.add_argument("--max-retry", type=int, default=3)


def _add_report(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("report", help="生成 HTML 评估报告")
    p.add_argument("--gold", default=str(config.ARTIFACTS_DIR / "dataset" / "test.jsonl"))
    p.add_argument("--infer", nargs="+", required=True, help="各路推理 jsonl")
    p.add_argument("--out", default=str(config.REPORT_DIR / "wab_eval_report.html"))
    p.add_argument("--clean", action="store_true", help="仅输出共同有效病例（排除解析失败病例）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aphasia", description="WAB 失语症评分 LORA 微调流水线")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_build(sub)
    _add_train(sub)
    _add_infer(sub)
    _add_report(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "build":
        from .data.build_dataset import run_build

        return run_build(args.gold, args.conversation_dir, args.out)

    if args.command == "train":
        from .train.run_round import run_round

        return run_round(args)

    if args.command == "infer":
        if args.route == "deepseek":
            from .infer.deepseek_infer import run_deepseek_infer
            return run_deepseek_infer(args)
        if args.route == "external":
            if not args.provider:
                print("[infer] --route external 需要 --provider {deepseek|glm}", flush=True)
                return 2
            from .infer.external_infer import run_external_infer
            return run_external_infer(args)
        from .infer.local_infer import run_local_infer
        return run_local_infer(args)

    if args.command == "report":
        from .eval.report import run_report

        return run_report(args.gold, args.infer, args.out, clean_only=args.clean)

    return 2


if __name__ == "__main__":
    sys.exit(main())
