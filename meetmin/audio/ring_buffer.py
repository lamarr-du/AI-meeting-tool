"""实时转写的 PCM 事件与按录音样本定位的有界缓冲。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class PcmChunk:
    start: int
    pcm: bytes


@dataclass(frozen=True)
class IdleTick:
    sample: int


AudioEvent = PcmChunk | IdleTick


class AudioRingBuffer:
    """由单个消费者读写。索引与已发布 WAV 的 16 kHz 样本一致。"""

    def __init__(self, seconds: float):
        self._samples = np.empty(round(seconds * SAMPLE_RATE), dtype=np.float32)
        self.end = 0

    @property
    def start(self) -> int:
        return max(0, self.end - len(self._samples))

    def append(self, chunk: PcmChunk) -> None:
        if chunk.start != self.end:
            raise ValueError(f"实时音频缺失：预期样本 {self.end}，收到 {chunk.start}")
        pcm = np.frombuffer(chunk.pcm, dtype="<i2")
        capacity = len(self._samples)
        end = self.end + len(pcm)
        tail = pcm[-capacity:]
        index = (end - len(tail)) % capacity
        first = min(len(tail), capacity - index)
        self._samples[index : index + first] = tail[:first] / 32768.0
        self._samples[: len(tail) - first] = tail[first:] / 32768.0
        self.end = end

    def read(self, start: int, end: int) -> np.ndarray:
        if not self.start <= start <= end <= self.end:
            raise ValueError("实时音频已超出缓冲区，可用保存的录音重新转写")
        index = start % len(self._samples)
        count = end - start
        first = min(count, len(self._samples) - index)
        if first == count:
            return self._samples[index : index + count].copy()
        return np.concatenate((self._samples[index:], self._samples[: count - first]))
