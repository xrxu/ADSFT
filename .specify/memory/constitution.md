<!--
Sync Impact Report
- Version change: (template, unversioned) → 1.0.0
- Bump rationale: Initial ratification — all placeholders replaced with concrete
  principles derived from project context (WAB 失语症评分 LORA 微调项目).
- Modified principles: (none — first concrete version)
- Added sections:
  - Core Principles (5): Gold-Standard Fidelity / Reproducibility & Experiment
    Tracking / Hardware-Bounded Resource Discipline / Robust Inference & Parsing /
    Evaluation Transparency
  - Additional Constraints (technology stack & data governance)
  - Development Workflow (iteration & reporting gates)
  - Governance
- Removed sections: (none)
- Templates requiring updates:
  - .specify/templates/plan-template.md ✅ (Constitution Check gate is generic,
    auto-derives from principles — no edit needed)
  - .specify/templates/spec-template.md ✅ (no principle-specific references)
  - .specify/templates/tasks-template.md ✅ (no principle-specific references)
- Deferred TODOs: (none)
-->

# AphasiaCLD Constitution

本项目对大模型做 LORA 微调，使其依据 **WAB（西方失语症成套测验）** 标准，根据"医生—病人"
对话给病人的**信息量**与**流畅度**两个维度打分（各 0–10 分）。本章程为该临床 NLP 微调项目
的不可协商原则。

## Core Principles

### I. 金标准保真 (Gold-Standard Fidelity)

评分逻辑、问题文本、图画描述、system prompt、信息量/流畅度评分标准，**必须**从
`datasets/gold/WAB_summary_v0.80.xlsx` 的"信息量和流畅度"页实时读取，禁止凭记忆复制或硬编码
摘要。训练/测试集划分**必须**严格遵循"打分汇总"页的"用途"列（`训练`/`训练FA0` 为训练集，
`测试` 为测试集，`无效` 病例禁止使用）。金标准取自"信息量"与"流畅度"两列。
*理由：这是临床任务，评分标准与数据划分的任何偏差都会直接污染微调信号与评估结论。*

### II. 可复现与实验跟踪 (Reproducibility & Experiment Tracking)

每一轮微调**必须**用 swanlab（local mode）记录过程与结果；超参数、随机种子、数据集版本
（如 v0.80）**必须**随实验一并记录，使任意一轮结果可被重新追溯与复现。微调**必须**迭代 5 轮，
且逐轮记录参数与对应效果，不得跳过中间轮次的评估。
*理由：5 轮迭代的价值在于横向对比；缺失跟踪信息会使轮次间的效果归因无法成立。*

### III. 硬件边界资源纪律 (Hardware-Bounded Resource Discipline)

所有训练与推理配置**必须**保证在单卡 A100 40GB 上不发生显存溢出（OOM）。训练 sequence
length **必须** ≥ 8192；推理 sequence length 不设上限但仍受显存约束。任何提高 batch size、
序列长度或并行度的改动，**必须**先评估显存占用再落地。
*理由：单卡 40GB 是硬约束，OOM 会直接中断长耗时的微调任务，浪费算力与时间。*

### IV. 鲁棒推理与解析 (Robust Inference & Parsing)

推理输出**必须**可被稳定解析为结构化的信息量分与流畅度分。解析失败的病例**必须**重跑推理，
不得以缺失值或默认分静默填充。理由与打分**必须**分两个问题独立提问（一个问评分，一个问理由），
不得合并。
*理由：解析失败若被静默吞掉，会使测试报告的样本数与分布失真，得出错误结论。*

### V. 评估透明 (Evaluation Transparency)

最终**必须**用 HTML 输出完整测试报告并 check-in 到工作目录，逐项列出测试集每个病例的详细评分。
报告**必须**包含三路对比：微调后模型、未微调模型（baseline）、以及 deepseek v4 pro（网络部署
模型）的推理输出。
*理由：临床评分的可信度依赖逐病例可审查与多基线对比，聚合指标不足以支撑结论。*

## Additional Constraints

技术栈与数据治理约束：

- **模型**：`Qwen3-32B-AWQ`（位于 `/proj/models/Qwen/Qwen3-32B-AWQ`）。
- **微调工具**：LLaMA-Factory（LORA）。
- **实验跟踪**：swanlab（local mode）。
- **对比基线**：未微调的同一模型；deepseek v4 pro（网络部署，非本地）。
- **数据来源**：真实对话在 `datasets/conversation/P*.txt`；AI 合成对话在 xlsx 的 `fakeasr` 页。
- **数据排除**："用途"标为 `无效` 的病例（方言重/ASR 错误多/质量差）一律不得进入任何数据集。
- **环境**：本工作目录即仓库；"check-in 到仓库"指放入本工作目录。

## Development Workflow

- 微调按 5 轮迭代推进；每轮结束**必须**评估该轮参数与效果，作为是否进入下一轮的依据。
- 测试**必须**在 5 轮微调完成后执行，并产出 HTML 报告。
- 任何改动评分相关 prompt 或数据划分的工作，**必须**回溯至原则 I 校验取数来源。
- 显存相关配置改动**必须**经原则 III 的显存评估后方可合入。

## Governance

本章程优先于其他临时实践。修订**必须**记录于本文件并更新版本号与日期；版本遵循语义化版本：

- **MAJOR**：原则的移除或不兼容的重定义。
- **MINOR**：新增原则或实质性扩展指导。
- **PATCH**：措辞澄清、错字、非语义性细化。

所有计划（plan）与实现（implement）阶段**必须**对照本章程的 Constitution Check 门禁；违反原则
而无正当理由的，门禁判定为失败。运行时开发指导以 `CLAUDE.md` 为准。

**Version**: 1.0.0 | **Ratified**: 2026-05-31 | **Last Amended**: 2026-05-31
