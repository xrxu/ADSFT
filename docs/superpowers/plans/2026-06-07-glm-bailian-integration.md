# GLM 5.1 百炼外部对比模型集成 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 GLM 5.1（阿里云百炼 DashScope）作为第 4 条外部对比推理路线，采用统一外部推理模块重构。

**Architecture:** 将 `deepseek_infer.py` 的单模型逻辑抽取为 `external_infer.py` 通用 OpenAI 兼容推理模块，通过 provider 注册表支持 deepseek + GLM 双路。保留 `deepseek_infer.py` 作为 thin wrapper 以保证向后兼容。配置层从单一 `DeepSeekConfig` 扩展为多 provider 的 `ExternalModelConfig`。

**Tech Stack:** Python 3.12, urllib (stdlib, 无额外 HTTP 依赖), 阿里云百炼 DashScope OpenAI 兼容 API (https://dashscope.aliyuncs.com/compatible-mode/v1)

---

### Task 1: Config — 统一外部模型配置

**Files:**
- Modify: `src/aphasia/config.py`

- [ ] **Step 1: 新增 `ExternalModelConfig` 和 `load_external_configs()`**

```python
# config.py — 在 DeepSeekConfig 之后增加

@dataclass(frozen=True)
class ExternalModelConfig:
    """外部模型配置（OpenAI 兼容 API），支持多 provider。"""
    provider: str          # "deepseek" | "glm"
    base_url: str | None
    api_key: str | None
    model: str
    display_name: str      # 报告用的显示名

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key)


def load_external_configs() -> dict[str, ExternalModelConfig]:
    """从环境变量加载所有外部模型配置。
    
    deepseek: DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY / DEEPSEEK_MODEL（默认 "deepseek v4 pro"）
    glm: GLM_BASE_URL（默认 https://dashscope.aliyuncs.com/compatible-mode/v1）/ DASHSCOPE_API_KEY / GLM_MODEL（默认 "glm-5.1"）
    """
    configs: dict[str, ExternalModelConfig] = {}
    
    # deepseek（保持向后兼容）
    deepseek_url = os.environ.get("DEEPSEEK_BASE_URL")
    if deepseek_url:
        configs["deepseek"] = ExternalModelConfig(
            provider="deepseek",
            base_url=deepseek_url,
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek v4 pro"),
            display_name="deepseek v4 pro",
        )
    
    # glm（新增）
    glm_url = os.environ.get("GLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    glm_key = os.environ.get("DASHSCOPE_API_KEY")
    if glm_key:
        configs["glm"] = ExternalModelConfig(
            provider="glm",
            base_url=glm_url,
            api_key=glm_key,
            model=os.environ.get("GLM_MODEL", "glm-5.1"),
            display_name="GLM 5.1",
        )
    
    return configs
```

- [ ] **Step 2: 标记 `load_deepseek_config()` 为 deprecated wrapper**

```python
# 在 load_deepseek_config() 上方添加注释，但保留函数体不变以保持向后兼容
# 注释：Deprecated: prefer load_external_configs()["deepseek"] for new code.
```

- [ ] **Step 3: 运行现有测试确保无回归**

```bash
conda run -n llama_factory python -m pytest tests/ -v --tb=short
```
Expected: 全部 PASS（现有测试不应受影响）

- [ ] **Step 4: Commit**

```bash
git add src/aphasia/config.py
git commit -m "feat: add ExternalModelConfig for multi-provider external model support

Add load_external_configs() returning provider registry.
Deprecate load_deepseek_config() in favor of load_external_configs()['deepseek'].
GLM 5.1 configured via GLM_BASE_URL/DASHSCOPE_API_KEY/GLM_MODEL env vars.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Infer — 统一外部推理模块

**Files:**
- Create: `src/aphasia/infer/external_infer.py`
- Modify: `src/aphasia/infer/deepseek_infer.py`

- [ ] **Step 1: 创建 `external_infer.py`**

```python
"""统一外部模型推理（OpenAI 兼容 API）：deepseek v4 pro / GLM 5.1。

通过 --provider 参数选择目标模型；凭据经环境变量注入。
网络/配额/凭据缺失 → unavailable=true，不阻塞其余路。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

from ..config import ExternalModelConfig, load_external_configs
from .parse import parse_scores


def _load_test(test_path: str) -> list[dict]:
    rows = []
    with open(test_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_prompt_messages(rec: dict) -> list[dict]:
    """从 ShareGPT messages 中取 system + user（不包括 assistant/gold）。"""
    msgs = rec["messages"]
    return [m for m in msgs if m["role"] in ("system", "user")]


def _chat(cfg: ExternalModelConfig, messages: list[dict], timeout: int = 60) -> str:
    """调用 OpenAI 兼容 /chat/completions。"""
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": 0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def _unavailable_record(r: dict, provider: str, qtype: str) -> dict:
    return {
        "case_id": r["case_id"],
        "model_route": provider,
        "round": None,
        "qtype": qtype,
        "raw_output": "",
        "info_pred": None,
        "flue_pred": None,
        "reason_text": None,
        "parse_ok": False,
        "retry_count": 0,
        "unavailable": True,
        "info_gold": r["info_gold"],
        "flue_gold": r["flue_gold"],
    }


def run_external_infer(args) -> int:
    """统一外部推理入口。从 args.provider 选择目标模型。"""
    provider = args.provider  # "deepseek" | "glm"
    configs = load_external_configs()
    cfg = configs.get(provider)
    
    if cfg is None or not cfg.available:
        print(
            f"[infer:external:{provider}] 凭据缺失（环境变量未设），标记该路不可用",
            flush=True,
        )
        rows = _load_test(args.test)
        results = [_unavailable_record(r, provider, r["qtype"]) for r in rows]
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write(out_path, results)
        return 0
    
    rows = _load_test(args.test)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    results: list[dict] = []
    for r in rows:
        prompt_msgs = _extract_prompt_messages(r)
        if r["qtype"] == "score":
            results.append(_infer_score(cfg, provider, r, prompt_msgs, args.max_retry))
        else:
            results.append(_infer_reason(cfg, provider, r, prompt_msgs))
    
    _write(out_path, results)
    unavail = sum(1 for x in results if x["unavailable"])
    failed = sum(
        1 for x in results
        if x["qtype"] == "score" and not x["parse_ok"] and not x["unavailable"]
    )
    print(
        f"[infer:external:{provider}] {len(results)} 条，不可用 {unavail}，"
        f"score 最终解析失败 {failed}",
        flush=True,
    )
    return 0


def _infer_score(cfg: ExternalModelConfig, provider: str, r: dict, prompt_msgs: list[dict], max_retry: int) -> dict:
    raw, pr, retries = "", None, 0
    for attempt in range(max_retry + 1):
        try:
            raw = _chat(cfg, prompt_msgs)
        except (urllib.error.URLError, KeyError, TimeoutError, OSError) as e:
            print(
                f"[infer:external:{provider}] {r['case_id']} 请求失败: {type(e).__name__}",
                flush=True,
            )
            return _unavailable_record(r, provider, "score")
        pr = parse_scores(raw)
        if pr.ok:
            break
        retries = attempt + 1
    return {
        "case_id": r["case_id"],
        "model_route": provider,
        "round": None,
        "qtype": "score",
        "raw_output": raw,
        "info_pred": pr.info if pr and pr.ok else None,
        "flue_pred": pr.flue if pr and pr.ok else None,
        "reason_text": None,
        "parse_ok": bool(pr and pr.ok),
        "retry_count": retries,
        "unavailable": False,
        "info_gold": r["info_gold"],
        "flue_gold": r["flue_gold"],
    }


def _infer_reason(cfg: ExternalModelConfig, provider: str, r: dict, prompt_msgs: list[dict]) -> dict:
    try:
        raw = _chat(cfg, prompt_msgs)
    except (urllib.error.URLError, KeyError, TimeoutError, OSError) as e:
        print(
            f"[infer:external:{provider}] {r['case_id']} reason 请求失败: {type(e).__name__}",
            flush=True,
        )
        return _unavailable_record(r, provider, "reason")
    return {
        "case_id": r["case_id"],
        "model_route": provider,
        "round": None,
        "qtype": "reason",
        "raw_output": raw,
        "info_pred": None,
        "flue_pred": None,
        "reason_text": raw,
        "parse_ok": True,
        "retry_count": 0,
        "unavailable": False,
        "info_gold": r["info_gold"],
        "flue_gold": r["flue_gold"],
    }


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
```

- [ ] **Step 2: 将 `deepseek_infer.py` 重写为 thin wrapper**

```python
"""deepseek v4 pro 外部推理 — 向后兼容 thin wrapper。

新代码请用 aphasia.infer.external_infer.run_external_infer(provider="deepseek")。
"""

from __future__ import annotations

from .external_infer import run_external_infer


def run_deepseek_infer(args) -> int:
    """委托给统一外部推理模块（provider=deepseek）。"""
    args.provider = "deepseek"
    return run_external_infer(args)
```

- [ ] **Step 3: 运行现有测试确保 deepseek 路向后兼容**

```bash
conda run -n llama_factory python -m pytest tests/ -v --tb=short
```
Expected: 全部 PASS

- [ ] **Step 4: Commit**

```bash
git add src/aphasia/infer/external_infer.py src/aphasia/infer/deepseek_infer.py
git commit -m "feat: add unified external inference module with GLM 5.1 support

Create external_infer.py — generic OpenAI-compatible HTTP inference for
multi-provider usage (deepseek + glm). Refactors deepseek_infer.py to a
thin backward-compatible wrapper.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: CLI — 新增 `--route external --provider` 入口

**Files:**
- Modify: `src/aphasia/cli.py`

- [ ] **Step 1: 更新 `_add_infer()` 与 dispatch 逻辑**

修改 `cli.py` 中的 `_add_infer` 函数——添加 `--route external` 选项和 `--provider` 参数：

```python
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
```

修改 `main()` 中的 infer dispatch：

```python
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
```

- [ ] **Step 2: 验证 CLI help**

```bash
conda run -n llama_factory python -m aphasia.cli infer --help
```
Expected: 显示 `--route {finetuned,baseline,deepseek,external}` 和 `--provider {deepseek,glm}`

- [ ] **Step 3: Commit**

```bash
git add src/aphasia/cli.py
git commit -m "feat: add --route external --provider for unified external inference

CLI now supports:
  --route external --provider deepseek  (new recommended way)
  --route deepseek                      (backward compatible)
  --route external --provider glm       (new GLM 5.1 route)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Report — 支持 GLM 路由显示

**Files:**
- Modify: `src/aphasia/eval/report.py`

- [ ] **Step 1: 更新 `_route_label()` 和 `_route_sort_key()`**

```python
def _route_label(route: str) -> str:
    """根据文件路径段生成显示名。"""
    if route == "baseline":
        return "baseline(未微调)"
    if route == "deepseek":
        return "deepseek v4 pro"
    if route == "glm":
        return "GLM 5.1"
    m = re.search(r"round\s*(\d)", route)
    if m:
        return f"微调 Round {m.group(1)}"
    return route


def _route_sort_key(route: str) -> tuple:
    """排序：baseline 在前，然后 round1-4，最后 deepseek、glm。"""
    if route == "baseline":
        return (0, 0)
    m = re.search(r"round_?(\d)", route)
    if m:
        return (1, int(m.group(1)))
    if "round4" in route or ("finetuned" in route and "r" not in route):
        return (1, 4)
    if route == "deepseek":
        return (2, 0)
    if route == "glm":
        return (3, 0)
    return (4, 0)
```

- [ ] **Step 2: 更新 `run_report()` 中 common_valid 排除逻辑**

将 `excluded_deepseek` 的排除逻辑改为排除所有外部模型（`deepseek` + `glm`）：

```python
# 计算共同有效病例（排除外部模型：deepseek / glm，其余路均 parse_ok 的交集）
common_valid: set[str] | None = None
for route, scores in by_route.items():
    if route in ("deepseek", "glm"):  # 外部模型不参与共同有效病例计算
        continue
    ok_ids = {cid for cid, r in scores.items() if r.get("parse_ok")}
    common_valid = ok_ids if common_valid is None else common_valid & ok_ids
common_valid = common_valid or set()
```

- [ ] **Step 3: 更新 verdict 查找逻辑确保只取本地路的 microtuning rounds**

```python
# 找最佳微调轮次 vs baseline（排除外部模型）
if "baseline" in route_metric:
    ft_routes = [r for r in by_route if r not in ("baseline", "deepseek", "glm")]
    # ... 其余不变
```

- [ ] **Step 4: 运行现有测试确保无回归**

```bash
conda run -n llama_factory python -m pytest tests/ -v --tb=short
```
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/aphasia/eval/report.py
git commit -m "feat: add GLM 5.1 route display in HTML report

- _route_label(): add 'GLM 5.1' for glm route
- _route_sort_key(): glm sorted after deepseek
- common_valid: exclude all external models (deepseek, glm)
- ft_routes filter: exclude external models from best-round search

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Quickstart — 更新文档

**Files:**
- Modify: `specs/001-wab-scoring-lora/quickstart.md`

- [ ] **Step 1: 更新前置环境变量和推理命令**

在前置部分增加 GLM 凭据配置，推理部分增加 GLM 路：

```markdown
## 0. 前置

```bash
# 环境
conda activate llama_factory
nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader

# 外部对比模型凭据——经环境变量注入，勿写入代码或提交
# deepseek v4 pro
export DEEPSEEK_BASE_URL="<由你提供>"
export DEEPSEEK_API_KEY="<由你提供>"
export DEEPSEEK_MODEL="deepseek v4 pro"

# GLM 5.1（阿里云百炼 DashScope）
export GLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY="<由你提供>"
export GLM_MODEL="glm-5.1"
```

## 3. 四路推理（US3）

```bash
# 微调后
python -m aphasia.cli infer --route finetuned --adapter artifacts/adapters/round4 \
  --test artifacts/dataset/test.jsonl --out artifacts/infer/finetuned.jsonl --max-retry 3
# 未微调 baseline
python -m aphasia.cli infer --route baseline \
  --test artifacts/dataset/test.jsonl --out artifacts/infer/baseline.jsonl --max-retry 3
# deepseek v4 pro（向后兼容）
python -m aphasia.cli infer --route deepseek \
  --test artifacts/dataset/test.jsonl --out artifacts/infer/deepseek.jsonl --max-retry 3
# GLM 5.1（新增）
python -m aphasia.cli infer --route external --provider glm \
  --test artifacts/dataset/test.jsonl --out artifacts/infer/glm.jsonl --max-retry 3
```

**自检**：每路 jsonl 行数 = 26×2（score+reason）；score 记录无 parse_ok=false 残留。

## 4. 生成 HTML 报告（US3）

```bash
python -m aphasia.cli report \
  --gold artifacts/dataset/test.jsonl \
  --infer artifacts/infer/finetuned.jsonl artifacts/infer/baseline.jsonl \
    artifacts/infer/deepseek.jsonl artifacts/infer/glm.jsonl \
  --out report/wab_eval_report.html
```

**自检**：浏览器打开 report；四路指标齐全。
```

- [ ] **Step 2: Commit**

```bash
git add specs/001-wab-scoring-lora/quickstart.md
git commit -m "docs: update quickstart for GLM 5.1 Bailian integration

Add GLM_* env vars, --route external --provider glm, and 四路 report command.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: 端到端验证

- [ ] **Step 1: 运行全量测试**

```bash
conda run -n llama_factory python -m pytest tests/ -v --tb=short
```
Expected: 全部 PASS

- [ ] **Step 2: 手动验证 CLI 新路由**

```bash
# 验证 help
conda run -n llama_factory python -m aphasia.cli infer --help

# 验证 GLM 路由（无凭据时预期优雅降级）
export DASHSCOPE_API_KEY=""
conda run -n llama_factory python -m aphasia.cli infer \
  --route external --provider glm \
  --test artifacts/dataset/test.jsonl \
  --out /tmp/glm_test.jsonl
# Expected: "凭据缺失...标记该路不可用"，退出码 0，输出文件每行 unavailable=true
```

- [ ] **Step 3: Commit（如有变更）**

```bash
git add -A
git commit -m "chore: final verification after GLM 5.1 integration"
```
