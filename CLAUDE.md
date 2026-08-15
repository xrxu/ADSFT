# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

使用简体中文。

# currentDate
Today's date is 2026/05/31.

## 项目目标

本项目对大模型做 LORA 微调，使其能够依据 **WAB（西方失语症成套测验 / Western Aphasia Battery）** 标准，根据"医生—病人"对话给病人的**信息量（info）**和**流畅度（flue）**两个维度打分（各 0–10 分）。这是一个临床 NLP 微调项目，不是传统软件工程仓库——目前仓库中**只有数据集，没有训练/推理代码**，代码需按下述需求从零搭建。

## 评估任务结构

诊断场景固定：病人面前有一幅"湖边一家人户外活动"的图画，医生依次问 7 个问题（前 6 题为问答，第 7 题为看图说话）：
1. 你今天好吗？ 2. 你以前来过这里吗？ 3. 你叫什么名字？ 4. 你住在哪里？ 5. 你做什么工作？ 6. 你为什么到这里来？你有什么不舒服吗？ 7. 请你告诉我，你在这画中看见些什么？

**问答正确性判定规则**（打分时必须严格遵守）：
- 第 1 题：仅表达感受即算正确。
- 第 2、6 题：回答合理即算正确。
- 第 4 题（住址）：只说到城市也算正确。
- 同音不同字算作正确；工作/职业只需模糊匹配；自我纠正后的回答算正确；错误回答必须是完全错误，没有模棱两可的回答。
- 前 6 题不要遗漏，并结合图片描述综合打分。

完整的信息量与流畅度 0–10 分评分标准、问题文本、图画描述、system prompt 等，均存放在 `datasets/gold/WAB_summary_v0.80.xlsx` 的 **"信息量和流畅度"** 页（字段：`system` / `field` / `dialogue_scope` / `diag_info_criterion` / `diag_flue_criterion` / `question_surfix` 等）。**编写或修改 prompt 时必须从这里取最新文本，不要凭记忆复制本文件的摘要。**

## 数据集

- `datasets/conversation/P*.txt` — 187 个真实病例的对话语音转写（ASR 文本），文件名即病例编号（P001…P220，编号不连续）。这些是无标点连续转写文本。
- `datasets/gold/WAB_summary_v0.80.xlsx` — **金标准与项目控制中心**，含 5 个工作表：
  - `更新说明` — 版本变更日志（当前 v0.8）。
  - `打分汇总` — 每个病例的金标准分（列：编号 / 语音记录 / 信息量 / 流畅度 / 理解 / 复述 / 命名 / AQ / 失语类型 / **用途** / 语言）。
  - `fakeasr` — ChatGPT 生成的合成对话及其目标分（用于扩充训练集，每条含生成 prompt + 对话 + 评分理由）。
  - `信息量和流畅度` — 评分标准与 prompt 模板（见上）。
  - `需求` — **项目需求清单**（见下"微调需求"）。
- `datasets/diag/WAB.doc` — WAB 诊断量表参考文档。

**xlsx 无法用普通工具直接读**——用 Python 解压读取，例如：`python3 -c "import zipfile,re; z=zipfile.ZipFile('datasets/gold/WAB_summary_v0.80.xlsx'); print(re.findall(r'<t[^>]*>(.*?)</t>', z.read('xl/sharedStrings.xml').decode()))"`。建议优先用 `openpyxl`/`pandas` 按 sheet 读取以保留行列结构。

### 训练/测试集划分（来自"需求"页，必须遵守）
- **训练集** = `打分汇总` 页中"用途"列标为 `训练` 或 `训练FA0` 的病例。
- **测试集** = "用途"列标为 `测试` 的病例。
- 金标准在"信息量"和"流畅度"两列。"用途"标为 `无效` 的病例（质量差/方言重/ASR 错误多，单元格标黄）**不得使用**。
- 真实对话在 `datasets/conversation/`，AI 合成对话在 `fakeasr` 页。

## 微调需求（来自 xlsx "需求"页，按此执行）

- 模型：`Qwen3-32B-AWQ`（位于 `/proj/models/Qwen/Qwen3-32B-AWQ`；模型根目录 `/proj/models`）。
- 工具：用 **LLaMA-Factory** 做 LORA 微调（**已安装**于 conda env `llama_factory`，Python 3.12，路径 `/home/xinxu/local/.conda/envs/llama_factory`；源码 editable 安装于 `/home/xinxu/workarea/finetune/LLaMA-Factory`。所有流水线命令在该 env 下运行：`conda run -n llama_factory python -m aphasia.cli ...`）。
- 实验跟踪：用 **swanlab（local mode）** 保存微调过程，并保存微调结果。
- 硬件约束：单卡 **A100 40GB**，参数务必保证不显存溢出；训练 sequence length ≥ 8192，推理 sequence length 不设限。
- 流程：微调**迭代 5 轮**，逐轮评估参数与效果；完成后跑测试并出测试报告。
- 报告：用 **HTML** 输出完整报告并 check-in 到仓库；报告需列出测试集每一项推理的详细评分。
- 基准对比：测试集在**未微调**模型上的推理输出作为 baseline；同时加入测试集在 **deepseek v4 pro（网络部署模型，非本地）** 上的推理输出做对比。
- 鲁棒性：推理结果若解析失败的病例需重跑推理。
- 问答方式：理由与打分分两个问题问（一个问题问评分，另一个问题问理由）。

## 环境说明

- 仓库**已是 git 仓库**；"check-in 到仓库"指放入本工作目录。
- GPU：`nvidia-smi` 确认为 A100-PCIE-40GB。
- 数据划分（实测）：训练 179（训练 144 + 训练FA0 35）、测试 26、无效 17。
- 流水线代码在 `src/aphasia/`（data/train/infer/eval + cli.py）；规范在 `specs/001-wab-scoring-lora/`。

<!-- SPECKIT START -->
当前特性：WAB 失语症评分 LORA 微调流水线（分支 `001-wab-scoring-lora`）。
技术栈、项目结构、各阶段 CLI 命令等详见实现计划：
`specs/001-wab-scoring-lora/plan.md`
（配套：spec.md / research.md / data-model.md / contracts/ / quickstart.md）

环境校正（research 实测）：微调全栈已安装于 conda env `llama_factory`
（Python 3.12，路径 `/home/xinxu/local/.conda/envs/llama_factory`），
而非"尚未安装"；LLaMA-Factory 源码在 `/home/xinxu/workarea/finetune/LLaMA-Factory`。
数据划分（已核实）：训练 179（训练 144 + 训练FA0 35）、测试 26、无效 17。
<!-- SPECKIT END -->
