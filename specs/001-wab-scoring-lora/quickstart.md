# Quickstart：WAB 失语症评分 LORA 微调流水线

面向开发/复现者，端到端跑通的最短路径。所有命令在 conda env `llama_factory` 下执行。

## 0. 前置

```bash
# 环境（已 provisioned，勿在 base 重装）
conda activate llama_factory   # Python 3.12, llamafactory 0.9.3, torch 2.7.1+cu126, swanlab 0.6.8
nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader   # 期望 A100 40GB

# 外部对比模型凭据——经环境变量注入，勿写入代码或提交
# 两者均走阿里云百炼 DashScope OpenAI 兼容 API，共享 DASHSCOPE_API_KEY
export DASHSCOPE_API_KEY="<由你提供>"

# deepseek-v4-pro
export DEEPSEEK_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DEEPSEEK_MODEL="deepseek-v4-pro"

# GLM 5.1
export GLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export GLM_MODEL="glm-5.1"
```

关键路径：
- 基座模型：`/proj/models/Qwen/Qwen3-32B-AWQ`（AWQ 4-bit）
- 金标准：`datasets/gold/WAB_summary_v0.80.xlsx`
- 对话转写：`datasets/conversation/P*.txt`
- LLaMA-Factory 源码：`/home/xinxu/workarea/finetune/LLaMA-Factory`

## 1. 构建数据集（US1）

```bash
python -m aphasia.cli build \
  --gold datasets/gold/WAB_summary_v0.80.xlsx \
  --conversation-dir datasets/conversation \
  --out artifacts/dataset
```

**预期**：`train.jsonl`（约 179 病例来源样本）、`test.jsonl`（26 病例）、`build_log.txt`。
**自检**：`test.jsonl` 病例数 == 26；无 usage=无效 病例；抽查金标准与"打分汇总"一致。

## 2. 迭代微调 5 轮（US2）

```bash
for N in 1 2 3 4 5; do
  python -m aphasia.cli train --round $N \
    --base /proj/models/Qwen/Qwen3-32B-AWQ \
    --data artifacts/dataset/train.jsonl \
    --config-out configs/round$N.yaml \
    --adapter-out artifacts/adapters/round$N
  # 每轮结束：在 test 上评估、看 swanlab、据指标调下一轮超参
done
```

**显存提示**：首轮用保守配置（lora_rank 16, batch 1, grad_accum 8, gradient_checkpointing,
cutoff_len 8192）跑通确认不 OOM，再逐轮调。
**自检**：swanlab local 有 5 个独立 run；每轮产出 adapter 目录。

## 3. 四路推理（US3）

```bash
# 微调后（取效果最好的轮次 adapter）
python -m aphasia.cli infer --route finetuned --adapter artifacts/adapters/round4 \
  --test artifacts/dataset/test.jsonl --out artifacts/infer/finetuned.jsonl --max-retry 3
# 未微调 baseline
python -m aphasia.cli infer --route baseline \
  --test artifacts/dataset/test.jsonl --out artifacts/infer/baseline.jsonl --max-retry 3
# deepseek-v4-pro（向后兼容）
python -m aphasia.cli infer --route deepseek \
  --test artifacts/dataset/test.jsonl --out artifacts/infer/deepseek.jsonl --max-retry 3
# GLM 5.1（新增，阿里云百炼 DashScope）
python -m aphasia.cli infer --route external --provider glm \
  --test artifacts/dataset/test.jsonl --out artifacts/infer/glm.jsonl --max-retry 3
```

**自检**：每路 jsonl 行数 = 26×2（score+reason）；score 记录无 parse_ok=false 残留
（除非达重跑上限并已标注）。

## 4. 生成 HTML 报告（US3）

```bash
python -m aphasia.cli report \
  --gold artifacts/dataset/test.jsonl \
  --infer artifacts/infer/finetuned.jsonl artifacts/infer/baseline.jsonl \
    artifacts/infer/deepseek.jsonl artifacts/infer/glm.jsonl \
  --out report/wab_eval_report.html
```

**自检**：浏览器打开 `report/wab_eval_report.html`；测试集每个病例都在逐病例表；
四路指标（含 Spearman/Pearson/QWK/...）齐全；含"微调后 vs baseline 相关性(首要)+QWK(次要)"结论。

## 验收对照（spec Success Criteria）

| 步骤 | 对应 SC |
|------|---------|
| 1 数据集 | SC-001, SC-002 |
| 2 微调 | SC-003, SC-004 |
| 3 推理 | SC-005, SC-006 |
| 4 报告 | SC-007, SC-008, SC-009 |
