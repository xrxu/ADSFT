# CLI 契约：流水线子命令

所有命令在 conda env `llama_factory` 下运行（见 research R8）。统一入口 `src/aphasia/cli.py`，
通过 `conda run -n llama_factory python -m aphasia.cli <subcommand>` 调用。

约定：输入只读自 `datasets/`；产物写入 `artifacts/`、`configs/`、`report/`。

---

## `build` — 构建数据集（US1 / FR-001~005）

```
python -m aphasia.cli build \
  --gold datasets/gold/WAB_summary_v0.80.xlsx \
  --conversation-dir datasets/conversation \
  --out artifacts/dataset
```

**输出**：`artifacts/dataset/train.jsonl`、`artifacts/dataset/test.jsonl`（每行符合
`dataset-record.schema.json`）、`artifacts/dataset/build_log.txt`（列出被丢弃的无效/缺金标准/
缺转写的 Case）。

**契约**：
- train.jsonl 仅含 usage∈{训练,训练FA0} 的 Case；test.jsonl 仅含 usage=测试；无 usage=无效。
- 每条记录的 info_gold/flue_gold 等于"打分汇总"对应行。
- system/instruction/input 中的问题、图画描述、评分标准逐字来自"信息量和流畅度"页。
- 退出码 0=成功；非 0=保真校验失败（如读不到评分标准）。

---

## `train` — 单轮微调（US2 / FR-006~008,012,013）

```
python -m aphasia.cli train --round {1..4} \
  --base /proj/models/Qwen/Qwen3-32B-AWQ \
  --data artifacts/dataset/train.jsonl \
  --config-out configs/round{N}.yaml \
  --adapter-out artifacts/adapters/round{N}
```

**输出**：该轮 LLaMA-Factory YAML、LoRA 适配器、swanlab local run。

**契约**：
- 生成的 YAML 必含 `cutoff_len: 8192`（≥8192）、LoRA 设置、`use_swanlab: true`+local、固定 `seed`。
- 不设 `quantization_bit`（AWQ 基座走 PTQ 路径，见 research R1）。
- 运行不得 OOM；OOM 即该轮失败，需调参重跑（退出码非 0）。

---

## `infer` — 四路推理（US3 / FR-009,010,015）

```
# 本地推理（微调后 / baseline）
python -m aphasia.cli infer \
  --route {finetuned|baseline} \
  --test artifacts/dataset/test.jsonl \
  --adapter artifacts/adapters/round{N}   # 仅 finetuned
  --out artifacts/infer/{route}.jsonl \
  --max-retry 3

# 外部推理（统一入口，通过 --provider 区分）
python -m aphasia.cli infer \
  --route external --provider {deepseek|glm} \
  --test artifacts/dataset/test.jsonl \
  --out artifacts/infer/{provider}.jsonl \
  --max-retry 3

# 向后兼容（仍可用）
python -m aphasia.cli infer \
  --route deepseek \
  --test artifacts/dataset/test.jsonl \
  --out artifacts/infer/deepseek.jsonl \
  --max-retry 3
```

两路外部模型均走阿里云百炼 DashScope OpenAI 兼容 API（默认端点 `https://dashscope.aliyuncs.com/compatible-mode/v1`），共享 `DASHSCOPE_API_KEY`。deepseek 用 `DEEPSEEK_MODEL`（默认 `deepseek-v4-pro`），GLM 用 `GLM_MODEL`（默认 `glm-5.1`）。**密钥不入库、不写日志。**

**契约**（每行符合 `inference-output.schema.json`）：
- score 问型解析失败→重跑至成功或达 `--max-retry`；达上限仍失败保留 parse_ok=false，不填默认分。
- 外部模型网络/配额失败→该批记录 unavailable=true，命令仍以 0 退出（不阻塞其余路）。
- 评分与理由作为两次独立提问分别产生 qtype=score 与 qtype=reason 记录。

---

## `report` — 生成 HTML 报告（US3 / FR-011, 指标 FR-014）

```
python -m aphasia.cli report \
  --gold artifacts/dataset/test.jsonl \
  --infer artifacts/infer/finetuned.jsonl artifacts/infer/baseline.jsonl artifacts/infer/deepseek.jsonl artifacts/infer/glm.jsonl \
  --out report/wab_eval_report.html
```

**契约**：
- 报告逐病例列出四路 info/flue 评分 + 解析状态 + reason 摘要（测试集每个 Case 均出现）。
- 指标按信息量/流畅度两维 + 总体：Spearman、Pearson、QWK、ExactMatch、±1、MAE、RMSE；
  运营：解析成功率、重跑后最终失败数。
- 成功判定区块：相关性(首要)、QWK(次要) 微调后 vs baseline 的对比结论（外部模型不参与判定）。
- 外部模型不可用、解析最终失败的 Case 显式标注。
- HTML 落 `report/` 并可在浏览器打开。
