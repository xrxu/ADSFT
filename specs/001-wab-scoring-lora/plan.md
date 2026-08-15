# Implementation Plan: WAB 失语症评分 LORA 微调流水线

**Branch**: `001-wab-scoring-lora` | **Date**: 2026-05-31 | **Updated**: 2026-06-07 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-wab-scoring-lora/spec.md`

## Summary

构建一条端到端流水线：从 gold xlsx 与对话转写构建带金标准的训练/测试数据集（严格按"用途"列
划分，排除"无效"），用 LLaMA-Factory 对 `Qwen3-32B-AWQ` 做 LORA 微调并迭代 5 轮（swanlab local
跟踪），在测试集上对微调后模型 / 未微调 baseline / deepseek-v4-pro / GLM 5.1 **四路**推理，解析
评分（评分与理由分两问），计算两维（信息量、流畅度）吻合度指标（相关性为首要判据、QWK 次要），
最终产出逐病例的 HTML 报告并放入工作目录。

数据规模（已核实"打分汇总"页）：训练 144 + 训练FA0 35 = **179** 训练病例，**26** 测试病例，
17 个"无效"排除。

## Technical Context

**Language/Version**: Python 3.12（conda env `llama_factory`，路径
`/home/xinxu/local/.conda/envs/llama_factory`）。注意系统 base Python 为 3.14 且缺 ML 栈——
所有训练/推理/数据脚本 MUST 在该 conda env 下运行。

**Primary Dependencies**（env 已实测可用）: llamafactory 0.9.3.dev0、torch 2.7.1+cu126、
transformers 4.52.4、peft 0.15.2、swanlab 0.6.8、autoawq 0.2.9、openpyxl 3.1.5、pandas 2.3.0、
scipy 1.15.3、scikit-learn 1.7.0。**vllm 未安装**（见 research：baseline/微调推理用 LLaMA-Factory
/transformers，不强依赖 vllm）。

**Storage**: 纯文件。源数据 `datasets/`（只读）；产出写入 `specs/001-wab-scoring-lora/` 下的
数据集 jsonl、LLaMA-Factory 配置、推理结果 jsonl 与最终 HTML 报告（HTML 报告同时放工作目录根）。

**Testing**: pytest（断言数据集划分/金标准一致性/解析器鲁棒性/指标计算）。env 含 sklearn/scipy
用于指标，无需额外安装。

**Target Platform**: Linux 单机，单卡 NVIDIA A100-PCIE-40GB（已确认 40960 MiB，空载占用 ~41 MiB）。

**Project Type**: 单项目、研究用批处理流水线（CLI/脚本驱动，无 Web/UI 面）。

**Performance Goals**: 非吞吐导向。约束是"装得下、跑得完"：5 轮微调每轮可在单卡完成且不 OOM；
测试集仅 26 例，推理可串行。

**Constraints**:
- 单卡 40GB 显存**不得溢出**；训练 sequence length **≥ 8192**；推理 seq len 不设上限（受显存约束）。
- 基座为 AWQ 4-bit（gemm, group_size=128）量化权重——LORA 必须以 QLoRA/量化适配方式挂载。
- 评分标准/问题/图画描述/system prompt **实时从** gold xlsx "信息量和流畅度"页读取，禁止硬编码。
- 评分与理由**分两次提问**。
- 解析失败的病例**必须重跑**，不得静默填默认值。

**Scale/Scope**: 179 训练 + 26 测试病例；每病例 ×2 问法（评分/理由）→ 训练样本量约数百条级别
（含 fakeasr 合成数据）；4 路推理 × 26 测试 × 2 问 = 208 次推理（外部模型可能缺失）。

**External Models**:
- deepseek-v4-pro + GLM 5.1：均通过阿里云百炼 DashScope OpenAI 兼容 API
  （`https://dashscope.aliyuncs.com/compatible-mode/v1`），共享 `DASHSCOPE_API_KEY`；
  token 计划付费。deepseek 默认模型 `deepseek-v4-pro`，GLM 默认模型 `glm-5.1`
- 两路外部模型共用一个统一推理模块（`external_infer.py`），通过 `--provider` 区分

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

依据 `.specify/memory/constitution.md` v1.0.0 的 5 条原则逐条核对：

| 原则 | 设计如何满足 | 判定 |
|------|--------------|------|
| I. 金标准保真 | 数据构建模块从 xlsx 实时读取问题/标准/prompt；划分严格按"用途"列；排除"无效"；金标准取自"打分汇总"页 | PASS |
| II. 可复现与实验跟踪 | swanlab local 记录每轮超参/数据集版本(v0.80)/过程；固定随机种子；5 轮迭代逐轮评估 | PASS |
| III. 硬件边界资源纪律 | QLoRA on AWQ + 梯度检查点 + 受控 batch/grad-accum；训练 seq len 8192；配置前先估显存 | PASS |
| IV. 鲁棒推理与解析 | 评分/理由分两问；解析器失败触发重跑；越界分视为解析失败 | PASS |
| V. 评估透明 | HTML 报告逐病例列三路评分；含相关性/QWK/完全一致率/±1/MAE/RMSE 与运营指标 | PASS |

**Gate 结论（Phase 0 前）**: PASS，无违规，Complexity Tracking 留空。

**Re-check（Phase 1 设计后）**: 见本文件末尾"Post-Design Constitution Re-check"——PASS。

## Project Structure

### Documentation (this feature)

```text
specs/001-wab-scoring-lora/
├── plan.md              # 本文件
├── spec.md              # 功能规范（已含 Clarifications）
├── research.md          # Phase 0 输出
├── data-model.md        # Phase 1 输出
├── quickstart.md        # Phase 1 输出
├── contracts/           # Phase 1 输出（接口/数据契约）
│   ├── dataset-record.schema.json   # 训练/测试 jsonl 记录契约
│   ├── inference-output.schema.json # 推理结果 jsonl 记录契约
│   └── cli.md                       # 各阶段 CLI 命令契约
└── checklists/
    └── requirements.md  # 规范质量检查单（/speckit-specify 输出）
```

### Source Code (repository root)

```text
src/aphasia/
├── data/
│   ├── gold_reader.py      # 读 xlsx：打分汇总 / 信息量和流畅度 / fakeasr
│   ├── build_dataset.py    # 构建 train/test jsonl（评分+理由两类样本），按用途划分
│   └── prompts.py          # 从 xlsx 字段拼装 system/question prompt（不硬编码）
├── train/
│   ├── make_lf_config.py   # 生成 LLaMA-Factory YAML（QLoRA on AWQ, seq 8192, swanlab）
│   └── run_round.py        # 驱动单轮微调（5 轮迭代的一轮）
├── infer/
│   ├── local_infer.py      # 微调后 & baseline（transformers/LLaMA-Factory）推理
│   ├── external_infer.py   # 统一外部模型推理（deepseek-v4-pro / GLM 5.1，OpenAI 兼容 API）
│   ├── deepseek_infer.py   # 向后兼容 thin wrapper（调用 external_infer）
│   └── parse.py            # 输出→(info, flue) 解析；失败标记触发重跑
├── eval/
│   ├── metrics.py          # Spearman/Pearson/QWK/ExactMatch/±1/MAE/RMSE + 运营指标
│   └── report.py           # 生成 HTML 报告（逐病例 + 汇总）
└── cli.py                  # 子命令入口：build / train / infer / report

configs/                    # 生成的 LLaMA-Factory YAML（5 轮）落盘于此
artifacts/                  # 数据集 jsonl、推理 jsonl、checkpoints 指针、报告
report/                     # 最终 HTML 报告（工作目录可见）

tests/
├── unit/                   # prompts/parse/metrics 单测
└── integration/            # 数据集划分一致性、金标准一致性、端到端小样本

datasets/                   # 既有：conversation/ + gold/ + diag/（只读输入）
```

**Structure Decision**: 选用单项目布局（研究批处理工具，无前后端之分）。代码按流水线阶段
（data → train → infer → eval）分包，每阶段一个可独立测试的模块，并由 `cli.py` 统一编排，
对应 spec 的三个用户故事（US1=data、US2=train、US3=infer+eval）。

## Complexity Tracking

> 无 Constitution 违规，无需填写。

## Post-Design Constitution Re-check

Phase 1 设计（data-model / contracts / quickstart）完成后复核：

- 契约 `dataset-record.schema.json` 强制记录携带 `source`(real/fake)、`split`、金标准两维，
  支撑原则 I（保真）与可核对性。
- 契约 `inference-output.schema.json` 含 `parse_ok` 与 `retry_count` 字段，落实原则 IV（重跑、
  不静默填充）。
- `make_lf_config.py` 模板固定 `cutoff_len=8192`、`quantization_method`/QLoRA、梯度检查点与
  swanlab，落实原则 II、III。
- `report.py` 输出逐病例三路明细 + 全部指标，落实原则 V。

**Re-check 结论**: PASS，无新增违规。
