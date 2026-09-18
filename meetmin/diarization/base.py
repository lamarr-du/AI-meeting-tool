"""Diarizer 抽象：输入 16k 单声道音频，输出带簇标签的时间段。"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Diarizer(ABC):
    name = "base"

    @abstractmethod
    def diarize(
        self,
        audio16k: np.ndarray,
        *,
        speech_spans: list[tuple[float, float]] | None = None,
    ) -> list[dict]:
        """返回 [{'start': 秒, 'end': 秒, 'label': 簇标签(int)}]，按时间升序。

        speech_spans: ASR 已识别的语音区间，用于跳过静音段、只对有人说话的窗提特征。
        """
        raise NotImplementedError
