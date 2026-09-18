"""LLM 客户端：步骤 3 的大模型接口层。

默认走 OpenAI 兼容的 /chat/completions —— 阿里云 DashScope 兼容模式（Qwen 系列）、
vLLM、Ollama、OpenAI 本体等均可直接用，只需改 base_url / model / api_key_env。
要接私有协议的大模型：继承 BaseLLM 实现 chat()，在 get_llm() 里注册一行即可。
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests

logger = logging.getLogger("meetmin")


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # 本地估算值（服务端未返回 usage 时使用）
    estimated: bool = False

    def add(self, other: "Usage") -> "Usage":
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.estimated = self.estimated or other.estimated
        return self

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BaseLLM(ABC):
    name = "base"

    @abstractmethod
    def chat(self, messages: list[dict], *, temperature: float | None = None,
             max_tokens: int | None = None) -> tuple[str, Usage]:
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLM):
    name = "openai_compatible"

    def __init__(self, lcfg: dict):
        self.lcfg = lcfg
        self.base_url = str(lcfg.get("base_url", "")).rstrip("/")
        if not self.base_url:
            raise RuntimeError("llm.base_url 未配置")
        self.model = lcfg.get("model", "")
        self.api_key = os.environ.get(lcfg.get("api_key_env", "LLM_API_KEY"), "")
        if not self.api_key:
            logger.warning(
                "环境变量 %s 未设置，若接口需要鉴权请在生成纪要前配置。",
                lcfg.get("api_key_env", "LLM_API_KEY"),
            )

    def chat(self, messages: list[dict], *, temperature: float | None = None,
             max_tokens: int | None = None) -> tuple[str, Usage]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.lcfg.get("temperature", 0.3) if temperature is None else temperature,
            "max_tokens": int(max_tokens or self.lcfg.get("max_output_tokens", 2000)),
            "stream": False,
        }
        payload.update(self.lcfg.get("extra_request") or {})
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        timeout = float(self.lcfg.get("timeout_sec", 120))
        retries = max(1, int(self.lcfg.get("retries", 3)))
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code != 200:
                    raise RuntimeError(f"LLM 接口返回 HTTP {r.status_code}: {r.text[:400]}")
                j = r.json()
                text = j["choices"][0]["message"]["content"] or ""
                u = j.get("usage") or {}
                cpl = int(u.get("completion_tokens", 0) or 0)
                ppl = int(u.get("prompt_tokens", 0) or 0)
                if not (ppl or cpl):  # 部分服务不返回 usage，按字符估算
                    cpl = max(1, int(len(text) / 1.5))
                    return text, Usage(calls=1, prompt_tokens=ppl, completion_tokens=cpl, estimated=True)
                return text, Usage(calls=1, prompt_tokens=ppl, completion_tokens=cpl)
            except Exception as e:
                last_err = e
                wait = min(30.0, 2.0 ** attempt)
                logger.warning("LLM 调用失败（%d/%d）：%s；%.0fs 后重试", attempt + 1, retries, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"LLM 调用连续失败：{last_err}")


class MockLLM(BaseLLM):
    """离线联调：不调任何网络，按 prompt 结构返回可控文本。"""

    name = "mock"

    def __init__(self, lcfg: dict | None = None):
        self.lcfg = lcfg or {}

    def chat(self, messages: list[dict], *, temperature: float | None = None,
             max_tokens: int | None = None) -> tuple[str, Usage]:
        user = messages[-1]["content"] if messages else ""
        if "压缩提炼" in user:  # map 阶段
            text = "- 【片段要点】（mock）保留话题/决策/任务，压缩后的中间摘要"
        else:
            text = "### 会议主题\n（mock 纪要正文，未调用真实大模型）\n"
        ppl = max(1, int(len(user) / 1.5))
        cpl = max(1, int(len(text) / 1.5))
        return text, Usage(calls=1, prompt_tokens=ppl, completion_tokens=cpl, estimated=True)


def get_llm(cfg: dict) -> BaseLLM:
    lcfg = cfg.get("llm", {}) or {}
    provider = str(lcfg.get("provider", "openai_compatible")).lower()
    if provider in ("mock", "none", "offline", ""):
        return MockLLM(lcfg)
    if provider in ("openai_compatible", "openai", "dashscope", "qwen"):
        return OpenAICompatibleLLM(lcfg)
    raise ValueError(f"未知 LLM provider: {provider}（可选 openai_compatible | mock）")
