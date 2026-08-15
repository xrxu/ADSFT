---
description: "Task list for WAB 失语症评分 LORA 微调流水线"
---

# Tasks: WAB 失语症评分 LORA 微调流水线

**Input**: Design documents from `/specs/001-wab-scoring-lora/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: 仅对高风险纯逻辑单元（评分解析、指标、数据集完整性）生成针对性测试任务；训练/推理
胶水层不做完整 TDD。

**Organization**: 按用户故事分组——US1=数据集构建、US2=迭代微调、US3=推理+评估报告。

**环境前提**: 所有命令在 conda env `llama_factory`（Python 3.12）下运行，例如
`conda run -n llama_factory python -m aphasia.cli ...`。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件、无未完成依赖）
- **[Story]**: US1/US2/US3；Setup/Foundational/Polish 阶段无标签

## Path Conventions

单项目布局（plan.md）：源码 `src/aphasia/`、测试 `tests/`、产物 `artifacts/`/`configs/`/`report/`，
只读输入 `datasets/`。

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 项目骨架与依赖确认

- [X] T001 创建项目结构：`src/aphasia/{data,train,infer,eval}/__init__.py`、`src/aphasia/__init__.py`、`tests/{unit,integration}/`、`artifacts/`、`configs/`、`report/` 目录（按 plan.md 结构）
- [X] T002 创建 `pyproject.toml`（或 `setup.cfg`），将包 `aphasia` 注册为可 `python -m aphasia.cli` 调用，声明依赖以 conda env `llama_factory` 已装版本为准（torch 2.7.1/transformers 4.52.4/peft 0.15.2/llamafactory 0.9.3/swanlab 0.6.8/openpyxl/pandas/scipy/scikit-learn），不重装
- [X] T003 [P] 配置 `pytest.ini`（testpaths=tests）与 `.gitignore`（忽略 `artifacts/`、`configs/*.yaml` 产物、swanlab 日志、模型权重，保留 `report/*.html`）

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 所有故事共享的取数与 CLI 骨架，必须先完成

**⚠️ CRITICAL**: 用户故事任务在本阶段完成前不能开始

- [X] T004 实现 gold xlsx 读取器 `src/aphasia/data/gold_reader.py`：用 openpyxl 读取"打分汇总"（编号/信息量/流畅度/用途/语言）、"信息量和流畅度"（system/field/dialogue_scope/diag_info_criterion/diag_flue_criterion/question_surfix）、"fakeasr"三页，返回结构化对象；缺关键字段时抛错（原则 I，data-model 实体 1/2）
- [X] T005 [P] 实现 CLI 骨架 `src/aphasia/cli.py`：argparse 子命令 `build`/`train`/`infer`/`report`，参数签名严格对齐 `contracts/cli.md`，仅占位转发（不含逻辑）
- [X] T006 [P] 实现配置与日志基础 `src/aphasia/config.py`：固定随机种子常量、`DATASET_VERSION="v0.80"`、路径常量（基座模型、xlsx、conversation 目录）、从环境变量读取 deepseek 凭据（`DEEPSEEK_BASE_URL/API_KEY/MODEL`，缺失则标记不可用，密钥不写日志）

**Checkpoint**: 取数与 CLI 骨架就绪，三个故事可并行开工

---

## Phase 3: User Story 1 - 构建可训练的数据集 (Priority: P1) 🎯 MVP

**Goal**: 从 xlsx + 转写构建按"用途"划分、含金标准、prompt 取自 xlsx 的 train/test jsonl。

**Independent Test**: 运行 `build` 后，test.jsonl=26 病例、train 仅含 训练/训练FA0、无 无效；
抽查金标准与"打分汇总"一致；prompt 文本与"信息量和流畅度"页逐字一致。

### Tests for User Story 1 ⚠️

- [X] T007 [P] [US1] 数据集完整性测试 `tests/integration/test_dataset_split.py`：构建后断言 test=26、train 病例 usage∈{训练,训练FA0}、0 个 无效、每条 info_gold/flue_gold 与"打分汇总"一致（SC-001/002）
- [X] T008 [P] [US1] prompt 保真测试 `tests/unit/test_prompts.py`：断言生成的 system/question/评分标准子串与 gold_reader 读出的 xlsx 字段逐字一致（不含硬编码摘要）

### Implementation for User Story 1

- [X] T009 [P] [US1] 实现 prompt 拼装 `src/aphasia/data/prompts.py`：从 ScoringCriterion 字段拼 system（含 diag_info/flue_criterion 注入）、score 提问与 reason 提问、input（对话 + question_surfix）；评分输出格式约束（要求两整数，便于解析）
- [X] T010 [US1] 实现数据集构建 `src/aphasia/data/build_dataset.py`：合并真实(conversation/*.txt)与 fake(fakeasr)病例 → 按 usage 划分 → 排除无效/缺金标准/缺转写并写 build_log.txt → 每病例产出 score 与 reason 两类记录（real 无理由时仅产 score）→ 写 train.jsonl/test.jsonl，每行符合 `contracts/dataset-record.schema.json`（FR-001~005，data-model 实体 3）
- [X] T011 [US1] 在 `src/aphasia/cli.py` 接通 `build` 子命令到 build_dataset，参数对齐 `contracts/cli.md`，保真校验失败时非 0 退出
- [X] T012 [US1] 运行 `build` 生成 `artifacts/dataset/{train,test}.jsonl`+build_log，并跑 T007/T008 验证通过

**Checkpoint**: US1 独立可用——干净、可核对的数据集已产出（MVP）

---

## Phase 4: User Story 2 - 迭代微调模型并跟踪实验 (Priority: P1)

**Goal**: 对 AWQ 基座做 LoRA，迭代 5 轮，swanlab local 跟踪，单卡不 OOM，逐轮在测试集评估。

**Independent Test**: swanlab 有 5 个独立 run（含超参/数据集版本）；每轮产出 adapter；训练
seq_len≥8192 且无 OOM；每轮有测试集评估结果。

### Implementation for User Story 2

- [X] T013 [P] [US2] 实现 LLaMA-Factory 配置生成 `src/aphasia/train/make_lf_config.py`：输出 round{N}.yaml，固定 `cutoff_len: 8192`、`finetuning_type: lora`、**不设** quantization_bit（AWQ 走 PTQ，research R1）、`gradient_checkpointing: true`、`per_device_train_batch_size: 1`、`gradient_accumulation_steps`、`lora_rank`/`lora_target: all`、`use_swanlab: true`+local、固定 seed、数据集指向 train.jsonl（FR-006~008/013，原则 II/III）
- [X] T014 [US2] 实现单轮微调驱动 `src/aphasia/train/run_round.py`：调用 `llamafactory-cli train round{N}.yaml`，落 adapter 到 `artifacts/adapters/round{N}`，记录 swanlab run 名与超参到 `artifacts/iterations/round{N}.json`（data-model 实体 4）
- [X] T015 [US2] 在 `src/aphasia/cli.py` 接通 `train --round` 子命令到 make_lf_config+run_round，参数对齐 `contracts/cli.md`；OOM 即非 0 退出
- [X] T016 [US2] 执行 round 1（保守配置 lora_rank16/batch1/grad_accum8）跑通确认显存不溢出（SC-004），记录峰值显存到 round1.json
- [X] T017 [US2] 依次执行 round 2/3/4/5，每轮据上一轮测试集评估（依赖 US3 的 infer+metrics，见 Dependencies）按相关性(首要)/QWK(次要)趋势人工调 lora_rank/lr/grad_accum（显存允许内）；汇总五轮参数与效果以供横向对比；确保 swanlab 留 5 个独立 run（FR-012、SC-003）

**Checkpoint**: US2 完成——5 轮 LoRA 适配器 + 可追溯实验记录

---

## Phase 5: User Story 3 - 产出多基线评估报告 (Priority: P2)

**Goal**: 测试集三路推理（微调后/baseline/deepseek），解析评分（评分与理由分两问，失败重跑），
计算两维指标，产出逐病例 HTML 报告。

**Independent Test**: 报告可在浏览器打开；测试集每病例三路评分齐全；指标含 Spearman/Pearson/
QWK/ExactMatch/±1/MAE/RMSE 与运营指标；含"微调后 vs baseline 相关性(首要)+QWK(次要)"结论。

### Tests for User Story 3 ⚠️

- [X] T018 [P] [US3] 评分解析测试 `tests/unit/test_parse.py`：覆盖正常("信息量X流畅度Y")、中文数字、越界(>10)、缺失、多解等用例，断言 parse_ok 与 (info,flue) 正确（FR-010，research R6）
- [X] T019 [P] [US3] 指标测试 `tests/unit/test_metrics.py`：用已知小数组断言 Spearman/Pearson/QWK/ExactMatch/±1/MAE/RMSE 数值正确（FR-014，research R7）

### Implementation for User Story 3

- [X] T020 [P] [US3] 实现评分解析 `src/aphasia/infer/parse.py`：正则+容错将 score 输出还原为 (info,flue)，两分均 0–10 整数才 parse_ok=true，否则 false（供重跑判定）
- [X] T021 [P] [US3] 实现指标计算 `src/aphasia/eval/metrics.py`：按信息量/流畅度两维+总体算 Spearman/Pearson(scipy)、QWK(sklearn quadratic)、ExactMatch、±1、MAE、RMSE，及解析成功率/重跑后失败数
- [X] T022 [US3] 实现本地推理 `src/aphasia/infer/local_infer.py`：baseline 加载 AWQ 基座；finetuned 加载基座+LoRA adapter；对 test.jsonl 分 score/reason 两问生成；score 失败按 max-retry 重跑；写符合 `contracts/inference-output.schema.json` 的 jsonl（FR-009/010，data-model 实体 5，research R4）
- [X] T023 [P] [US3] 实现 deepseek 推理 `src/aphasia/infer/deepseek_infer.py`：OpenAI 兼容 HTTP 调用（env 注入端点/密钥），网络/配额失败则该批 unavailable=true 不阻塞；输出同一 jsonl 契约（research R5）
- [X] T024 [US3] 在 `src/aphasia/cli.py` 接通 `infer --route` 子命令到 local/deepseek 推理，参数对齐 `contracts/cli.md`
- [X] T025 [US3] 实现 HTML 报告 `src/aphasia/eval/report.py`：逐病例三路评分表 + 两维/总体指标汇总 + 运营指标 + 成功判定区块(相关性首要、QWK次要 vs baseline) + 缺失/解析失败显式标注；输出到 `report/`（FR-011，SC-005~009，data-model 实体 6，原则 V）
- [X] T026 [US3] 在 `src/aphasia/cli.py` 接通 `report` 子命令到 report.py，参数对齐 `contracts/cli.md`
- [X] T027 [US3] 执行三路 infer + report，生成 `report/wab_eval_report.html`，浏览器打开验证逐病例与指标齐全（SC-008）

**Checkpoint**: 三路对比 HTML 报告产出，全部用户故事独立可用

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: 跨故事收尾

- [X] T028 [P] 运行 `quickstart.md` 全流程做端到端验证，修正命令/路径偏差
- [X] T029 [P] 校正 `CLAUDE.md`：将"llamafactory 尚未安装""不是 git 仓库"等过期描述更新为 research 实测结论（env 已装、已是 git 仓库）
- [X] T030 解析鲁棒性兜底复查：确认所有 score 推理记录无最终未标注的 parse_ok=false 残留（SC-006）

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: 无依赖，立即开始
- **Foundational (Phase 2)**: 依赖 Setup；阻塞所有用户故事
- **User Stories (Phase 3+)**: 均依赖 Foundational
- **Polish (Phase 6)**: 依赖期望的用户故事完成

### User Story Dependencies

- **US1（P1，数据集）**: Foundational 后即可，独立。是 US2/US3 的数据前提。
- **US2（P1，微调）**: 需 US1 产出的 train.jsonl；其逐轮评估(T017)依赖 US3 的 infer+metrics。
- **US3（P2，推理+报告）**: 需 US1 的 test.jsonl；finetuned 路需 US2 的 adapter，但 baseline/
  deepseek 路与指标/报告骨架可在 US2 完成前先行实现与测试。

> 注：US2 与 US3 存在交叉——逐轮调参(T017)用 US3 的评估能力。建议顺序：US1 → US3 的解析/指标/
> baseline 推理(T018~T024 可先做 baseline 部分) → US2 的 5 轮 → US3 的 finetuned 推理+报告。

### Within Each User Story

- 测试（如有）先写并失败再实现
- prompts/parse/metrics（纯逻辑）先于依赖它们的 build/infer/report
- CLI 接通在对应实现之后

### Parallel Opportunities

- Setup：T003 与 T001/T002 串行后可并行收尾
- Foundational：T005、T006 可并行（T004 完成后）
- US1：T007、T008（测试）并行；T009 与测试并行
- US3：T018、T019（测试）并行；T020、T021、T023 不同文件可并行

---

## Parallel Example: User Story 3

```bash
# 先并行写测试（应失败）：
Task: "评分解析测试 tests/unit/test_parse.py"
Task: "指标测试 tests/unit/test_metrics.py"

# 再并行实现不同文件：
Task: "评分解析 src/aphasia/infer/parse.py"
Task: "指标计算 src/aphasia/eval/metrics.py"
Task: "deepseek 推理 src/aphasia/infer/deepseek_infer.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 1 Setup → 2. Phase 2 Foundational → 3. Phase 3 US1 → 4. **STOP & VALIDATE**：
   独立核对数据集（T007/T008）→ 干净数据集即首个可交付增量。

### Incremental Delivery

1. Setup + Foundational → 地基就绪
2. US1 → 验证数据集 → 交付（MVP）
3. US3 的解析/指标/baseline 部分 → 可先评估 baseline
4. US2 四轮微调（用 US3 评估能力逐轮调参）
5. US3 finetuned 推理 + 三路 HTML 报告 → 最终交付

### 单人顺序建议

US1 → US3(解析/指标/baseline) → US2(5 轮) → US3(finetuned+报告) → Polish。

---

## Notes

- [P] = 不同文件、无未完成依赖
- [Story] 标签用于可追溯性
- 所有命令在 conda env `llama_factory` 下运行
- 验证测试在实现前失败
- 每个任务或逻辑组后提交
- 避免：模糊任务、同文件冲突、破坏故事独立性的跨故事依赖
