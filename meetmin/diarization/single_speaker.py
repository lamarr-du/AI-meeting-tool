"""单说话人兜底（关闭分离 / 依赖缺失时使用）。"""

from __future__ import annotations

import numpy as np

from .base import Diarizer


class SingleSpeakerDiarizer(Diarizer):
    name = "single"

    def diarize(self, audio16k: np.ndarray, *, speech_spans=None) -> list[dict]:
        duration = len(audio16k) / 16000.0 if audio16k is not None else 0.0
        return [{"start": 0.0, "end": duration, "label": 0}]
