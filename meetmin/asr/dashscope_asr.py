"""云端 ASR 参考实现：阿里云 DashScope Paraformer 录音文件转写（自带说话人分离）。

这是"步骤 2 预留接口"的云端示例：
  * 优点：转写 + 分离一步到位、无需本地模型；
  * 前提：音频必须有一个服务端可访问的 URL（如 OSS 签名链接），本工具不代管上传；
    用 `--file-url` 传入，或在 config.yaml 的 dashscope_asr 节加 file_url。
  * 换成其它云端 ASR（讯飞/腾讯云/OpenAI 兼容音频接口）时，照抄本类的
    「提交 → 轮询 → 解析成 Utterance」结构即可。
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import requests

from .base import ASRProvider, ASRResult, Utterance


class DashScopeParaformerASR(ASRProvider):
    name = "dashscope"

    def __init__(self, ccfg: dict):
        self.ccfg = ccfg
        self.base_url = str(ccfg.get("base_url", "https://dashscope.aliyuncs.com")).rstrip("/")

    def _api_key(self) -> str:
        key = os.environ.get(self.ccfg.get("api_key_env", "DASHSCOPE_API_KEY"), "")
        if not key:
            raise RuntimeError(
                f"未设置环境变量 {self.ccfg.get('api_key_env')}（云端 ASR 需要 API Key）。"
            )
        return key

    def transcribe(
        self,
        audio: np.ndarray | None,
        *,
        file_path: str | None = None,
        file_url: str | None = None,
        **kwargs: Any,
    ) -> ASRResult:
        url = file_url or self.ccfg.get("file_url")
        if not url:
            raise RuntimeError(
                "DashScope 文件转写要求音频有服务端可访问的 URL（例如 OSS 签名链接）。\n"
                "  用 --file-url 传入；或改用本地 provider：config.yaml 里 asr.provider: faster_whisper"
            )
        key = self._api_key()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        params: dict[str, Any] = {
            "channel_id": [0],
            "language_hints": self.ccfg.get("language_hints") or ["zh", "en"],
            "diarization_enabled": True,
        }
        if self.ccfg.get("speaker_count"):
            params["speaker_count"] = int(self.ccfg["speaker_count"])

        resp = requests.post(
            f"{self.base_url}/api/v1/services/audio/asr/transcription",
            headers=headers,
            json={
                "model": self.ccfg.get("model", "paraformer-v2"),
                "input": {"file_urls": [url]},
                "parameters": params,
            },
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"提交转写任务失败 HTTP {resp.status_code}: {resp.text[:400]}")
        task_id = resp.json().get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"提交转写任务异常: {resp.text[:400]}")

        # 轮询任务
        interval = float(self.ccfg.get("poll_interval_sec", 3))
        deadline = time.time() + float(self.ccfg.get("timeout_sec", 1800))
        result_url = None
        while time.time() < deadline:
            time.sleep(interval)
            r = requests.get(
                f"{self.base_url}/api/v1/tasks/{task_id}",
                headers={"Authorization": f"Bearer {key}"},
                timeout=30,
            )
            out = r.json().get("output", {}) or {}
            status = out.get("task_status")
            if status == "SUCCEEDED":
                results = out.get("results") or []
                if results:
                    result_url = results[0].get("transcription_url")
                break
            if status in ("FAILED", "CANCELED", "UNKNOWN"):
                raise RuntimeError(f"转写任务失败: {out}")
        if not result_url:
            raise RuntimeError("转写任务超时或未返回结果 URL")

        data = requests.get(result_url, timeout=60).json()
        utterances: list[Utterance] = []
        language = None
        for tr in data.get("transcripts", []) or []:
            hints = tr.get("language_hints") or []
            if hints and not language:
                language = hints[0]
            for s in tr.get("sentences", []) or []:
                utterances.append(
                    Utterance(
                        start=float(s.get("begin_time", 0)) / 1000.0,
                        end=float(s.get("end_time", 0)) / 1000.0,
                        text=(s.get("text") or "").strip(),
                        speaker=s.get("speaker_id"),
                    )
                )
        return ASRResult(
            utterances=utterances,
            language=language,
            audio_16k=None,
            extra={"asr_model": self.ccfg.get("model"), "task_id": task_id},
        )
