# WAB Aphasia Scoring — LORA Fine-tuning Pipeline

> 基于 Qwen3-32B-AWQ 的 WAB 失语症评分 LORA 微调流水线

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[English](#english) | [中文](#chinese)

---

## English

### Overview

This project fine-tunes **Qwen3-32B-AWQ** using **LORA** (via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)) to score aphasic patients' **Information Content** (info, 0–10) and **Fluency** (flue, 0–10) according to the **Western Aphasia Battery (WAB)** standard. The input is a doctor–patient dialogue transcript (7 questions about a picnic scene picture), and the model outputs two-dimensional clinical scores with reasoning.

### Project Structure

```text
aphasiacld/
├── datasets/                    # Raw data (read-only)
│   ├── conversation/            # 187 ASR dialogue transcripts (P001–P220)
│   ├── gold/                    # WAB_summary_v0.80.xlsx — gold standard & prompts
│   └── diag/                    # WAB reference doc
├── src/aphasia/                 # Pipeline source code
│   ├── cli.py                   # CLI entry point
│   ├── config.py                # Configuration management
│   ├── data/                    # Dataset construction
│   │   ├── gold_reader.py       # Read gold xlsx sheets
│   │   ├── prompts.py           # Prompt templates (from xlsx)
│   │   └── build_dataset.py     # Build train/test JSONL datasets
│   ├── train/                   # LORA fine-tuning
│   │   ├── make_lf_config.py    # Generate LLaMA-Factory configs
│   │   └── run_round.py         # Execute one training round
│   ├── infer/                   # Inference
│   │   ├── local_infer.py       # Local model inference (baseline + fine-tuned)
│   │   ├── external_infer.py    # External model via DashScope API
│   │   └── parse.py             # Robust score parsing with retry
│   └── eval/                    # Evaluation
│       ├── metrics.py           # Correlation, QWK, MAE, RMSE, agreement
│       └── report.py            # HTML report generation
├── tests/                       # Unit tests (pytest)
├── specs/001-wab-scoring-lora/  # Design docs, plans, contracts
├── configs/                     # Training config templates
├── report/                      # Generated HTML reports
├── scripts/                     # Utility scripts
├── swanlog/                     # SwanLab experiment tracking logs
└── artifacts/                   # Output artifacts
```

### Key Features

- **End-to-end pipeline**: `data → train (5 rounds) → infer (4-way) → eval → HTML report`
- **4-way inference comparison**: Fine-tuned model vs. Untuned baseline vs. DeepSeek-V4-Pro vs. GLM 5.1
- **Constitution-compliant**: Gold-standard fidelity, reproducible experiment tracking (SwanLab), hardware-aware resource discipline (single A100 40GB), robust inference with retry, transparent evaluation
- **Training/Test split**: Strictly follows the "用途" (usage) column in gold xlsx — 179 train, 26 test, 17 invalid (excluded)
- **Score + reasoning separation**: Two separate queries per case (one for score, one for reasoning)

### Quick Start

```bash
# Activate the conda environment
conda activate llama_factory

# Run the full pipeline
python -m aphasia.cli full-pipeline

# Or run individual stages
python -m aphasia.cli build-dataset
python -m aphasia.cli train-round --round 1
python -m aphasia.cli infer --model fine-tuned
python -m aphasia.cli evaluate
python -m aphasia.cli generate-report

# Run tests
python -m pytest tests/ -v
```

### Requirements

- **Hardware**: Single NVIDIA A100-PCIE-40GB (40960 MiB)
- **Environment**: Conda env `llama_factory` (Python 3.12)
- **Base Model**: `Qwen3-32B-AWQ` (AWQ 4-bit quantized, at `/proj/models/Qwen/Qwen3-32B-AWQ`)
- **Key Dependencies**: LLaMA-Factory 0.9.3, torch 2.7.1+cu126, transformers 4.52.4, peft 0.15.2, swanlab 0.6.8, autoawq 0.2.9
- **Fine-tuned Adapter**: [xrxu/aphasia_adapter](https://huggingface.co/xrxu/aphasia_adapter) on Hugging Face

### Evaluation Metrics

- **Primary**: Pearson/Spearman correlation (score vs. gold)
- **Secondary**: Quadratic Weighted Kappa (QWK), exact agreement rate, ±1 tolerance, MAE, RMSE
- **ICC**: Intraclass Correlation Coefficient (3,1) absolute agreement

### Data

- 187 real aphasic patient dialogue transcripts (Chinese, ASR)
- Gold standard scores in `datasets/gold/WAB_summary_v0.80.xlsx` (5 sheets)
- AI-generated synthetic dialogues (fakeasr) for data augmentation

---

## 中文

### 概述

本项目对 **Qwen3-32B-AWQ** 进行 **LORA** 微调（基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)），使其能够依据 **WAB（西方失语症成套测验）** 标准，根据医生与病人的对话转录文本，对病人的**信息量**（Information Content, 0–10 分）和**流畅度**（Fluency, 0–10 分）两个维度进行自动评分。

### 诊断场景

病人面前有一幅"湖边一家人户外活动"的图画，医生依次问 7 个问题（前 6 题为问答，第 7 题为看图说话），模型需根据病人的回答综合评估其信息量和流畅度。

### 项目结构

```text
aphasiacld/
├── datasets/                    # 原始数据（只读）
│   ├── conversation/            # 187 份真实对话语音转写（P001–P220）
│   ├── gold/                    # WAB_summary_v0.80.xlsx — 金标准与 prompt 模板
│   └── diag/                    # WAB 诊断量表参考文档
├── src/aphasia/                 # 流水线源代码
│   ├── cli.py                   # CLI 入口
│   ├── config.py                # 配置管理
│   ├── data/                    # 数据集构建
│   │   ├── gold_reader.py       # 读取 xlsx 金标准
│   │   ├── prompts.py           # 从 xlsx 加载 prompt 模板
│   │   └── build_dataset.py     # 构建训练/测试 JSONL 数据集
│   ├── train/                   # LORA 微调
│   │   ├── make_lf_config.py    # 生成 LLaMA-Factory 训练配置
│   │   └── run_round.py         # 执行单轮训练
│   ├── infer/                   # 推理
│   │   ├── local_infer.py       # 本地模型推理（baseline + 微调后）
│   │   ├── external_infer.py    # 外部模型推理（DashScope API）
│   │   └── parse.py             # 鲁棒评分解析（含重试机制）
│   └── eval/                    # 评估
│       ├── metrics.py           # 相关性、QWK、MAE、RMSE、一致率
│       └── report.py            # HTML 报告生成
├── tests/                       # 单元测试（pytest）
├── specs/001-wab-scoring-lora/  # 设计文档、计划、合约
├── configs/                     # 训练配置模板
├── report/                      # 生成的 HTML 报告
├── scripts/                     # 工具脚本
├── swanlog/                     # SwanLab 实验跟踪日志
└── artifacts/                   # 输出产物
```

### 核心特性

- **端到端流水线**：`数据构建 → 训练（5 轮迭代）→ 推理（四路对比）→ 评估 → HTML 报告`
- **四路推理对比**：微调模型 vs. 未微调 baseline vs. DeepSeek-V4-Pro vs. GLM 5.1
- **约法三章合规**：金标准保真、可复现实验跟踪（SwanLab）、硬件边界资源纪律（单卡 A100 40GB）、鲁棒推理与重试、透明评估
- **训练/测试集划分**：严格按 xlsx "用途"列划分——训练 179 例、测试 26 例、无效 17 例（排除）
- **评分与理由分离**：每个病例分两次提问（一次问评分，一次问理由）
- **训练约束**：单卡 A100 40GB 不 OOM；sequence length ≥ 8192；QLoRA on AWQ 4-bit

### 快速开始

```bash
# 激活 conda 环境
conda activate llama_factory

# 运行完整流水线
python -m aphasia.cli full-pipeline

# 或分阶段运行
python -m aphasia.cli build-dataset       # 构建数据集
python -m aphasia.cli train-round --round 1  # 第 N 轮训练
python -m aphasia.cli infer --model fine-tuned  # 推理
python -m aphasia.cli evaluate             # 评估
python -m aphasia.cli generate-report      # 生成报告

# 运行测试
python -m pytest tests/ -v
```

### 环境要求

- **硬件**：单卡 NVIDIA A100-PCIE-40GB（40960 MiB 显存）
- **环境**：Conda env `llama_factory`（Python 3.12，路径 `/home/xinxu/local/.conda/envs/llama_factory`）
- **基座模型**：`Qwen3-32B-AWQ`（AWQ 4-bit 量化，路径 `/proj/models/Qwen/Qwen3-32B-AWQ`）
- **主要依赖**：LLaMA-Factory 0.9.3、torch 2.7.1+cu126、transformers 4.52.4、peft 0.15.2、swanlab 0.6.8、autoawq 0.2.9
- **微调适配器**：[xrxu/aphasia_adapter](https://huggingface.co/xrxu/aphasia_adapter)（Hugging Face）

### 评估指标

- **首要指标**：Pearson/Spearman 相关系数（评分与金标准）
- **次要指标**：二次加权 Kappa（QWK）、完全一致率、±1 容差、MAE、RMSE
- **ICC**：组内相关系数 (3,1) 绝对一致性

### 数据集

- 187 份真实失语症患者对话转录（中文，ASR 文本）
- 金标准评分在 `datasets/gold/WAB_summary_v0.80.xlsx`（含 5 个工作表）
- AI 生成的合成对话（fakeasr）用于数据增强

### 参考

- 实现计划：`specs/001-wab-scoring-lora/plan.md`
- 功能规格：`specs/001-wab-scoring-lora/spec.md`
- 项目宪法：`.specify/memory/constitution.md`
- Claude Code 指引：`CLAUDE.md`