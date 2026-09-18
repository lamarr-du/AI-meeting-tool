"""ASR 抽象层：统一输出带时间戳的 Utterance 列表。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Utterance:
    """一句话（whisper segment 粒度，或按说话人切分后的子句）。"""

    start: float
    end: float
    text: str
    speaker: str | int | None = None
    # 词级时间戳：[{'w': 文本, 's': 开始秒, 'e': 结束秒}]
    words: list[dict] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "start": round(float(self.start), 2),
            "end": round(float(self.end), 2),
            "text": self.text,
        }
        if self.speaker is not None:
            d["speaker"] = self.speaker
        if self.words:
            d["words"] = self.words
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Utterance":
        return cls(
            start=float(d.get("start", 0)),
            end=float(d.get("end", 0)),
            text=d.get("text", ""),
            speaker=d.get("speaker"),
            words=d.get("words"),
        )


@dataclass
class ASRResult:
    utterances: list[Utterance] = field(default_factory=list)
    language: str | None = None
    # 本地 provider 可回传已解码的 16k 音频，供说话人分离复用，避免二次解码
    audio_16k: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ASRProvider(ABC):
    name = "base"

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray | None,
        *,
        file_path: str | None = None,
        **kwargs: Any,
    ) -> ASRResult:
        """audio: 16kHz 单声道 float32；file_path: 原始文件（云端/直读场景使用）。"""
        raise NotImplementedError
