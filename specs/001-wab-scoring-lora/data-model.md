# Phase 1 Data Model: WAB 失语症评分 LORA 微调流水线

实体来自 spec 的 Key Entities + Functional Requirements。本流水线无数据库，实体落地为
jsonl / 配置 / 报告文件；下文给出字段、来源、校验规则与状态流转。

---

## 1. Case（病例）

一次 WAB 评估的原始单位。

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `case_id` | string | 文件名 / "打分汇总".编号 | 如 `P001`，主键 |
| `dialogue` | string | `datasets/conversation/{id}.txt` 或 `fakeasr` 页 | 无标点连续转写或合成对话 |
| `info_gold` | int(0–10) | "打分汇总".信息量 | 信息量金标准 |
| `flue_gold` | int(0–10) | "打分汇总".流畅度 | 流畅度金标准 |
| `usage` | enum | "打分汇总".用途 | `训练`/`训练FA0`/`测试`/`无效` |
| `language` | string\|null | "打分汇总".语言 | 如 "有"/方言备注 |
| `source` | enum | 派生 | `real`(转写) / `fake`(fakeasr) |

**校验规则**（对应 FR-002/003、SC-001/002）：
- `usage == 无效` → 该 Case 不得进入任何 split（直接丢弃）。
- 缺 `info_gold` 或 `flue_gold` → 不得进入任何 split，记入构建日志。
- 真实 Case 必须能在 `datasets/conversation/` 找到对应 `.txt`；找不到则记日志并跳过。
- `info_gold`/`flue_gold` 必须为 0–10 整数。

**State / 划分流转**：
```
原始 Case ──(usage)──▶ 训练/训练FA0 ─▶ split=train
                   └──▶ 测试        ─▶ split=test
                   └──▶ 无效        ─▶ dropped（不产出样本）
```

---

## 2. ScoringCriterion（评分标准）

权威来源：gold xlsx **"信息量和流畅度"** 页。**运行时实时读取，禁止硬编码**（FR-001，原则 I）。

| 字段 | 来源列 | 用途 |
|------|--------|------|
| `system` | system | system prompt（失语症诊断医生设定） |
| `field` | field | 图画场景描述（湖边户外活动） |
| `dialogue_scope` | dialogue_scope | 7 个问题文本 |
| `diag_info_criterion` | diag_info_criterion | 信息量 0–10 评分标准 |
| `diag_flue_criterion` | diag_flue_criterion | 流畅度 0–10 评分标准 |
| `question_surfix` | question_surfix | 拼到提问后的固定后缀 |

**校验**：构建样本前必须成功读到上述非空字段；任一缺失则中止构建并报错（保真优先于继续）。

---

## 3. TrainingSample / EvalSample（样本，jsonl 一行）

由 Case + ScoringCriterion 组装；每个进入 split 的 Case 产出**两条**样本（FR-005）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `case_id` | string | 溯源 |
| `split` | enum | `train` / `test` |
| `source` | enum | `real` / `fake` |
| `qtype` | enum | `score`（问评分） / `reason`（问理由） |
| `system` | string | 取自 ScoringCriterion.system（+ 评分标准注入） |
| `instruction` | string | 提问：score 问"信息量/流畅度各几分"；reason 问理由 |
| `input` | string | 对话内容 + question_surfix |
| `output` | string | score: 规范化分数串；reason: 金标准理由文本（fake 有；real 若无则按规则生成/留空策略见下） |
| `info_gold` | int | 冗余存金标准，便于评估对齐 |
| `flue_gold` | int | 同上 |

**说明 / 校验**（FR-003/004/005、SC-002）：
- `qtype=score` 的 `output` 必须可被 R6 解析器解析回 `(info_gold, flue_gold)`，否则样本构建失败。
- `real` Case 通常无现成"理由"文本——`reason` 训练样本仅在有可靠理由来源时构造；无则该 Case
  只产出 `score` 样本（在 data-model 层允许 0 或 1 条 reason 样本，构建日志标注）。`fake`(fakeasr)
  页含评分理由，可直接用作 `reason.output`。
- 训练集 = 所有 `split=train` 样本；测试集评估以 `score` 类为分数来源、`reason` 类为理由展示。

契约见 `contracts/dataset-record.schema.json`。

---

## 4. TuningIteration（微调轮次，共 5 轮）

| 字段 | 类型 | 说明 |
|------|------|------|
| `round` | int(1–5) | 轮次 |
| `lf_config_path` | path | 该轮 LLaMA-Factory YAML（configs/round{N}.yaml） |
| `hyperparams` | object | lora_rank/lr/grad_accum/cutoff_len 等 |
| `dataset_version` | string | `v0.80` |
| `seed` | int | 固定随机种子 |
| `adapter_path` | path | 输出 LoRA 适配器目录 |
| `swanlab_run` | string | 对应 swanlab run 名 |
| `eval_metrics` | object | 该轮在测试集上的指标（见实体 6） |

**校验**（FR-006/007/012/013、SC-003/004、原则 II/III）：
- 必须存在 round 1..5，每轮在 swanlab 留独立 run。
- `cutoff_len ≥ 8192`。
- 训练过程不得 OOM（失败即该轮无效，需调参重跑该轮）。
- 评估直接在 test split 上（无独立验证集，FR-013）。

**State 流转**：`configured → trained(adapter 落盘) → evaluated(eval_metrics 写入)`；五轮串行，
逐轮据 `eval_metrics` 决定下一轮超参。

---

## 5. InferenceResult（推理结果，jsonl 一行）

测试集每个 Case × 每路模型（finetuned/baseline/deepseek/glm）× 每问型一条记录。

| 字段 | 类型 | 说明 |
|------|------|------|
| `case_id` | string | |
| `model_route` | enum | `finetuned` / `baseline` / `deepseek` / `glm` |
| `round` | int\|null | finetuned 对应轮次；其余 null |
| `qtype` | enum | `score` / `reason` |
| `raw_output` | string | 模型原始输出 |
| `info_pred` | int\|null | score 解析出的信息量分 |
| `flue_pred` | int\|null | score 解析出的流畅度分 |
| `reason_text` | string\|null | reason 输出 |
| `parse_ok` | bool | 两分均在 0–10 整数域才 true |
| `retry_count` | int | 重跑次数（≤ 上限） |
| `unavailable` | bool | 该路不可用（如 deepseek 网络失败） |

**校验**（FR-009/010、SC-005/006、原则 IV）：
- `qtype=score` 且 `parse_ok=false` → 必须重跑直至成功或达上限；达上限仍失败保留
  `parse_ok=false`，报告显式标注，**不**填默认分。
- `model_route=deepseek` 不可用 → `unavailable=true`，不计入"未列出"。

契约见 `contracts/inference-output.schema.json`。

---

## 6. EvaluationReport（评估报告）

测试集**四路**推理的逐病例明细 + 指标汇总，HTML。

| 组成 | 内容 |
|------|------|
| 逐病例表 | 每 Case：金标准(info/flue) + 四路 (info_pred/flue_pred/parse_ok) + reason 摘要 |
| 指标汇总 | 四路各自、按信息量/流畅度两维 + 总体：Spearman/Pearson/QWK/ExactMatch/±1/MAE/RMSE |
| 运营汇总 | 各路解析成功率、重跑后最终失败数 |
| 成功判定 | 相关性(首要)与 QWK(次要) 微调后 vs baseline 的对比结论（SC-009） |
| 缺失标注 | deepseek 不可用、解析最终失败的 Case 显式标注 |

**校验**（FR-011、SC-005/006/007/008/009、原则 V）：HTML 落 `report/` 且可在浏览器打开；
测试集每个 Case 都出现在逐病例表；指标齐全。
