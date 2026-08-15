"""全局配置：路径常量、随机种子、数据集版本、deepseek 凭据（经环境变量注入）。

密钥只从环境变量读取，绝不写入日志或仓库。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- 固定常量（原则 II 可复现） ---
SEED = 20260531
DATASET_VERSION = "v0.81"

# --- 路径常量 ---
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "datasets"
CONVERSATION_DIR = DATASETS_DIR / "conversation"
GOLD_XLSX = DATASETS_DIR / "gold" / "WAB_summary_v0.81.xlsx"
BASE_MODEL = Path("/proj/models/Qwen/Qwen3-32B-AWQ")

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CONFIGS_DIR = REPO_ROOT / "configs"
REPORT_DIR = REPO_ROOT / "report"

# --- 训练硬约束（原则 III） ---
TRAIN_CUTOFF_LEN = 8192  # 训练 sequence length 下限

# --- 用途标签（原则 I：唯一权威划分依据） ---
USAGE_TRAIN = {"训练", "训练FA0"}
USAGE_TEST = {"测试"}
USAGE_INVALID = {"无效"}


@dataclass(frozen=True)
class DeepSeekConfig:
    """deepseek v4 pro（网络部署）配置；凭据缺失时 available=False。

    Deprecated: prefer ExternalModelConfig via load_external_configs()["deepseek"]。
    """

    base_url: str | None
    api_key: str | None
    model: str

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key)

    def __repr__(self) -> str:  # 不泄露密钥
        return (
            f"DeepSeekConfig(base_url={self.base_url!r}, "
            f"model={self.model!r}, api_key={'set' if self.api_key else 'unset'})"
        )


# Deprecated: prefer load_external_configs()["deepseek"] for new code.
def load_deepseek_config() -> DeepSeekConfig:
    """从环境变量读取 deepseek 凭据。"""
    return DeepSeekConfig(
        base_url=os.environ.get("DEEPSEEK_BASE_URL"),
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
    )


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

    def __repr__(self) -> str:  # 不泄露密钥
        return (
            f"ExternalModelConfig(provider={self.provider!r}, "
            f"base_url={self.base_url!r}, model={self.model!r}, "
            f"api_key={'set' if self.api_key else 'unset'})"
        )


def load_external_configs() -> dict[str, ExternalModelConfig]:
    """从环境变量加载所有外部模型配置。

    deepseek: DEEPSEEK_BASE_URL（默认 https://dashscope.aliyuncs.com/compatible-mode/v1）
         / DASHSCOPE_API_KEY / DEEPSEEK_MODEL（默认 "deepseek-v4-pro"）
    glm: GLM_BASE_URL（默认 https://dashscope.aliyuncs.com/compatible-mode/v1）
         / DASHSCOPE_API_KEY / GLM_MODEL（默认 "glm-5.1"）
    """
    configs: dict[str, ExternalModelConfig] = {}

    # deepseek（与 GLM 共享百炼 DashScope 端点 + DASHSCOPE_API_KEY）
    deepseek_url = os.environ.get(
        "DEEPSEEK_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    deepseek_key = os.environ.get("DASHSCOPE_API_KEY")
    if deepseek_key:
        configs["deepseek"] = ExternalModelConfig(
            provider="deepseek",
            base_url=deepseek_url,
            api_key=deepseek_key,
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            display_name="deepseek-v4-pro",
        )

    # glm（新增：阿里云百炼 DashScope）
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
