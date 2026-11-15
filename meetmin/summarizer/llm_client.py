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
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

logger = logging.getLogger("meetmin")

# Do not retry earlier than requested by the service; ask the user to retry later instead.
MAX_RETRY_AFTER_SECONDS = 300


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse HTTP Retry-After delta-seconds or HTTP-date; invalid values use backoff."""
    if not value:
        return None
    value = value.strip()
    if value.isascii() and value.isdecimal():
        # Avoid integer conversion limits and unnecessarily large sleeps.
        digits = value.lstrip("0") or "0"
        if len(digits) > 6:
            return float(MAX_RETRY_AFTER_SECONDS + 1)
        return float(int(digits))
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _http_error(response: requests.Response) -> RuntimeError:
    status = response.status_code
    if status in (401, 403):
        hint = "鉴权失败，请检查 API Key、访问权限及接口地址"
    elif status in (400, 404, 422):
        hint = "请求无效，请检查模型名、接口地址和请求参数"
    elif status == 429:
        hint = "请求过于频繁或额度受限"
    elif 500 <= status < 600:
        hint = "服务端暂时出错"
    else:
        hint = "接口请求失败"
    return RuntimeError(f"LLM 接口返回 HTTP {status}：{hint}。{response.text[:200]}")


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # 本地估算值（服务端未返回 usage 时使用）
    estimated: bool = False
    # complete: 接口明确返回 stop；truncated: 至少一段最终输出被截断；unknown: 未知。
    finish_status: str = "unknown"

    def add(self, other: "Usage") -> "Usage":
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.estimated = self.estimated or other.estimated
        if "truncated" in (self.finish_status, other.finish_status):
            self.finish_status = "truncated"
        elif "unknown" in (self.finish_status, other.finish_status):
            self.finish_status = "unknown"
        else:
            self.finish_status = "complete"
        return self

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BaseLLM(ABC):
    name = "base"

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, Usage]:
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

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, Usage]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.lcfg.get("temperature", 0.3)
            if temperature is None
            else temperature,
            "max_tokens": int(max_tokens or self.lcfg.get("max_output_tokens", 2000)),
            "stream": False,
        }
        payload.update(self.lcfg.get("extra_request") or {})
        text, usage, reason = self._request(payload)
        if reason == "length":
            current_limit = int(payload["max_tokens"])
            if current_limit <= 0:
                raise RuntimeError("LLM 输出上限必须为正数，无法重试截断结果。")
            payload = {**payload, "max_tokens": current_limit * 2}
            logger.warning(
                "LLM 输出因长度限制截断，将上限从 %d 提高到 %d 后重试一次。",
                current_limit,
                current_limit * 2,
            )
            text, retry_usage, reason = self._request(payload)
            # The discarded first response still consumed tokens, but its truncation
            # does not describe the replacement response.
            usage.calls += retry_usage.calls
            usage.prompt_tokens += retry_usage.prompt_tokens
            usage.completion_tokens += retry_usage.completion_tokens
            usage.estimated = usage.estimated or retry_usage.estimated
            if reason == "length":
                logger.warning("LLM 提高输出上限后仍被截断，结果将标记为不完整。")
        if not text.strip():
            raise RuntimeError("LLM 返回空结果，请检查模型和输出额度；未生成纪要。")
        usage.finish_status = (
            "complete" if reason == "stop" else "truncated" if reason == "length" else "unknown"
        )
        return text, usage

    def _request(self, payload: dict) -> tuple[str, Usage, str | None]:
        """一次模型调用，可按 F1 策略重试 HTTP 请求。"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        timeout = float(self.lcfg.get("timeout_sec", 120))
        retries = max(1, int(self.lcfg.get("retries", 3)))
        last_err: Exception | None = None
        for attempt in range(retries):
            retry_after: float | None = None
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            except requests.exceptions.SSLError as error:
                raise RuntimeError(f"LLM 连接证书错误，请检查证书配置：{error}") from error
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as error:
                last_err = error
            except requests.exceptions.RequestException as error:
                raise RuntimeError(f"LLM 请求失败，请检查接口配置：{error}") from error
            else:
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    last_err = _http_error(r)
                    if r.status_code == 429:
                        retry_after = _retry_after_seconds(r.headers.get("Retry-After"))
                elif r.status_code != 200:
                    raise _http_error(r)
                else:
                    try:
                        j = r.json()
                        choice = j["choices"][0]
                        text = choice["message"]["content"] or ""
                        reason = choice.get("finish_reason")
                        if not isinstance(text, str):
                            raise TypeError("content 必须是字符串")
                        u = j.get("usage") or {}
                        cpl = int(u.get("completion_tokens", 0) or 0)
                        ppl = int(u.get("prompt_tokens", 0) or 0)
                    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
                        raise RuntimeError(
                            "LLM 接口返回的数据格式无效，请检查接口和模型配置。"
                        ) from error
                    if not (ppl or cpl):  # 部分服务不返回 usage，按字符估算
                        cpl = max(1, int(len(text) / 1.5))
                        return (
                            text,
                            Usage(
                                calls=1, prompt_tokens=ppl, completion_tokens=cpl, estimated=True
                            ),
                            reason,
                        )
                    return text, Usage(calls=1, prompt_tokens=ppl, completion_tokens=cpl), reason
            if attempt + 1 == retries:
                break
            if retry_after is not None and retry_after > MAX_RETRY_AFTER_SECONDS:
                raise RuntimeError(
                    f"{last_err} 服务要求等待的时间超过自动重试上限 "
                    f"{MAX_RETRY_AFTER_SECONDS} 秒，请稍后重试。"
                ) from last_err
            wait = retry_after if retry_after is not None else min(30.0, 2.0**attempt)
            logger.warning(
                "LLM 调用失败（%d/%d）：%s；%.1f 秒后重试", attempt + 1, retries, last_err, wait
            )
            time.sleep(wait)
        raise RuntimeError(f"LLM 调用连续失败（共 {retries} 次）：{last_err}") from last_err


class MockLLM(BaseLLM):
    """离线联调：不调任何网络，按 prompt 结构返回可控文本。"""

    name = "mock"

    def __init__(self, lcfg: dict | None = None):
        self.lcfg = lcfg or {}

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, Usage]:
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
