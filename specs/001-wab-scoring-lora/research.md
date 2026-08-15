# Phase 0 Research: WAB 失语症评分 LORA 微调流水线

所有技术未知项已通过环境实测 + 源码核对解决，无遗留 NEEDS CLARIFICATION。

---

## R1. 在 AWQ 量化基座上做 LORA 微调

**Decision**: 直接加载 `/proj/models/Qwen/Qwen3-32B-AWQ`（已是 AWQ 4-bit PTQ 权重），用
LLaMA-Factory `finetuning_type: lora` 挂 LoRA 适配器；**不**设 `quantization_bit`（那是给
全精度基座做 on-the-fly bnb 量化用的）。

**Rationale**: 实测 LLaMA-Factory 0.9.3 的 `model/model_utils/quantization.py` 注释明确优先级
"PTQ-quantized (train/infer) > AutoGPTQ (export) > On-the-fly quantization"，且当
`config.quantization_config` 已存在（AWQ gemm, bits=4, group_size=128）时走 PTQ 分支直接使用。
基座 config 已确认含该 AWQ 量化配置；autoawq 0.2.9 在 env 中可用。LoRA 加在量化权重之上即
QLoRA 形态，显著省显存。

**Alternatives considered**:
- 用全精度 `Qwen3-32B` + bnb 4-bit on-the-fly QLoRA：显存与 AWQ 接近，但需求点名 AWQ 基座，
  且会引入额外量化步骤，放弃。
- 全参数微调：32B 单卡 40GB 必 OOM，放弃。

---

## R2. 32B + seq_len 8192 在单卡 A100 40GB 内不 OOM

**Decision**: QLoRA(on AWQ) + `cutoff_len: 8192` + 梯度检查点(`gradient_checkpointing: true`) +
`per_device_train_batch_size: 1` + `gradient_accumulation_steps`（如 8）+ `bf16`/AWQ compute dtype +
LoRA 仅挂注意力/MLP 线性层（`lora_target: all`，`lora_rank` 起始 16）。flash-attention 若可用则启用。

**Rationale**: AWQ 4-bit 基座权重 ~ 20GB 级，激活在 seq 8192 下靠梯度检查点压制；batch=1 +
梯度累积维持有效 batch 而不抬高峰值显存。约束来自 constitution 原则 III（≥8192、不 OOM）。
首轮以保守配置验证显存水位，再逐轮调 `lora_rank`/`grad_accum`。

**Alternatives considered**:
- seq_len < 8192：违反约束，放弃。
- 多卡：环境仅单卡 A100，放弃。
- DeepSpeed ZeRO-3/FSDP：单卡收益有限且与某些量化路径不兼容，首轮不引入。

**Open risk**: 实际峰值显存以首轮跑通为准；若逼近上限，先降 `lora_rank`、再降 `cutoff_len`
的训练样本占比（保留 ≥8192 能力但裁剪超长样本数量），最后才考虑 offload。

---

## R3. swanlab local mode 实验跟踪

**Decision**: LLaMA-Factory 训练配置加 `use_swanlab: true` + `swanlab_mode: local`（或等价
env），每轮一个独立 run，run 名编码轮次与关键超参（如 `r{round}-rank{R}-lr{LR}`）；日志目录落在
`artifacts/swanlab/`。

**Rationale**: swanlab 0.6.8 在 env 中可用，支持 local 模式离线记录，满足 constitution 原则 II
（每轮可追溯、记录超参/数据集版本）。数据集版本(v0.80)与种子写入 run 的 config/notes。

**Alternatives considered**: wandb/tensorboard——需求点名 swanlab，放弃。

---

## R4. 微调后模型 & baseline 的推理方式

**Decision**: 用 transformers + peft（baseline 直接加载 AWQ 基座；微调后加载基座 + LoRA 适配器）
做批推理；不依赖 vllm（env 未装）。推理 seq len 不设上限（受显存约束），测试集仅 26 例可串行。

**Rationale**: vllm 缺失且非必需——26 例 ×2 问 ×2 本地路 = 104 次生成，串行可接受。autoawq 推理
路径成熟。这样推理与训练共用同一 env、同一加载逻辑，降低不一致风险。

**Alternatives considered**: 装 vllm 提速——收益对 26 例不明显，且引入安装风险，放弃（如后续
扩到大测试集再评估）。

---

## R5. 外部网络模型接入（deepseek-v4-pro + GLM 5.1）

**Decision**: 采用统一外部推理模块（`external_infer.py`），通过 OpenAI 兼容 HTTP API 调用，
支持多 provider 注册表。各模型凭据独立经环境变量注入（**不写死**、**不入库**）。网络/配额
失败时该路标记"缺失"，记录原因，不阻塞其余报告（spec Edge Case + Assumption）。

**Provider 配置**（均走阿里云百炼 DashScope，共享 `DASHSCOPE_API_KEY`）:
- **deepseek-v4-pro**: `DEEPSEEK_BASE_URL`（默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`）
  / `DASHSCOPE_API_KEY` / `DEEPSEEK_MODEL`（默认 `deepseek-v4-pro`）
- **GLM 5.1**（新增，v0.80 需求）: `GLM_BASE_URL`（默认同百炼端点）
  / `DASHSCOPE_API_KEY` / `GLM_MODEL`（默认 `glm-5.1`），token 计划付费

**Rationale**: 统一模块避免代码重复（方案 B），与 spec 澄清一致。GLM 5.1 与 deepseek 采用相同的
优雅降级模式，独立 env var 前缀互不干扰。密钥经 env 注入符合安全惯例。

**Alternatives considered**: 为 GLM 独立建模块（方案 A）——代码重复，已放弃；硬编码端点/密钥——
安全与可移植性差，放弃。

**Open item**: 确切 base_url 与模型 ID 需用户在运行前提供（quickstart 中以 env 变量说明）。

---

## R6. 评分输出解析与重跑（评分/理由分两问）

**Decision**: 两问法——
- 问 A（评分）：要求模型只回两个 0–10 整数（信息量、流畅度），用正则 + 容错（中文数字、
  "信息量X分流畅度Y分"等模式）解析为 `(info, flue)`。
- 问 B（理由）：自由文本，原样保留。
解析判定 `parse_ok`：两分均在 0–10 整数域内才算成功；越界/缺失/多解→失败。失败触发**重跑**
（带计数上限，如 3 次），最终仍失败则 `parse_ok=false` 并在报告显式标注（不填默认分）。

**Rationale**: 落实原则 IV 与 spec FR-010/SC-006。两问分离避免理由文本污染分数解析。重跑上限
防止无限循环同时保留"显式标注失败"的透明性。

**Alternatives considered**:
- 让模型直接输出 JSON：可作为 prompt 内的输出格式约束以提升解析率，但仍需正则兜底，二者结合采用。
- 单问同时要分数+理由：违反 spec"分两问"约束，放弃。

---

## R7. 指标计算（相关性首要、QWK 次要）

**Decision**: 按信息量/流畅度两维分别计算并给总体：Spearman、Pearson（scipy.stats）、QWK
（sklearn `cohen_kappa_score(weights='quadratic')`）、完全一致率、±1 容忍准确率、MAE、RMSE；
运营类：首次解析成功率、重跑后最终失败数。成功判据：相关性 > baseline 为首要，QWK > baseline
为次要（spec SC-009）。

**Rationale**: env 含 scipy 1.15.3 / sklearn 1.7.0，全部指标可直接算。0–10 有序整数评分用相关性
+ QWK 比纯准确率更贴合（spec 已采纳）。

**Alternatives considered**: 仅 MAE/准确率——对有序评分信息量不足，已被 spec 升级，放弃为唯一判据。

---

## R8. 环境与既有文档校正

**Decision**: 所有脚本在 conda env `llama_factory`（Python 3.12）下运行；CLI 命令显式用该 env 的
解释器/`conda run -n llama_factory`。

**Rationale**: 实测系统 base Python 为 3.14 且无 torch/transformers/llamafactory；而
`/home/xinxu/local/.conda/envs/llama_factory` 已完整provisioned（llamafactory 0.9.3、torch
2.7.1+cu126 等）。这修正了 CLAUDE.md "llamafactory 尚未安装" 的旧说法——**已安装于该 env**。
LLaMA-Factory 源码位于 `/home/xinxu/workarea/finetune/LLaMA-Factory`（editable 安装）。

**Alternatives considered**: 在 base 重新安装全套——多余且易冲突，放弃。
