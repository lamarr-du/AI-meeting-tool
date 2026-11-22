"""按轨检测语音，串行调度一个 Whisper 模型的短窗解码。"""

from __future__ import annotations

import re
import unicodedata
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..audio.ring_buffer import SAMPLE_RATE, AudioEvent, AudioRingBuffer, IdleTick
from .base import Utterance
from .quality import Decoded
from .zh_convert import to_simplified

_PRE_ROLL = SAMPLE_RATE // 4
_POST_ROLL = SAMPLE_RATE // 8
_SCAN_STEP = SAMPLE_RATE // 5
_SENTENCE_END = re.compile(r"""[。！？.!?][”’"')）]*$""")
_UNITS = re.compile(r"[a-z0-9]+|[^\W_]")


def _units(text: str) -> list[str]:
    return _UNITS.findall(unicodedata.normalize("NFKC", to_simplified(text)).casefold())


def _stable_sentence(previous: list[dict], current: list[dict]) -> int:
    if not previous or not current or abs(previous[0]["s"] - current[0]["s"]) > 0.8:
        return 0
    old = iter(_units("".join(word["w"] for word in previous)))
    stable = 0
    for index, word in enumerate(current, 1):
        if any(next(old, None) != unit for unit in _units(word["w"])):
            break
        if _SENTENCE_END.search(word["w"].rstrip()):
            stable = index
    return stable


@dataclass
class _Speech:
    start: int
    end: int
    floor: int
    decoded_to: int
    closed: bool = False
    previous: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    final: bool
    capped: bool
    due: int


class LiveTrack:
    """每轨独立保留 VAD 时间轴和草稿，共用外部解码器。"""

    def __init__(
        self,
        *,
        ring_sec: float,
        endpoint_ms: int,
        max_window_sec: float,
        vad: Callable[[np.ndarray], list[dict]],
    ):
        self.audio = AudioRingBuffer(ring_sec)
        self.pause = round(endpoint_ms * SAMPLE_RATE / 1000)
        self.max_window = round(max_window_sec * SAMPLE_RATE)
        self.vad = vad
        self._speech: deque[_Speech] = deque()
        self._scanned = 0
        self._speech_end = 0

    def feed(self, event: AudioEvent) -> None:
        if isinstance(event, IdleTick):
            if self.audio.end > self._scanned:
                self._scan()
            self._close_at(event.sample)
            return
        self.audio.append(event)
        if self.audio.end - self._scanned >= _SCAN_STEP:
            self._scan()

    def _scan(self) -> None:
        end = self.audio.end
        start = max(self.audio.start, self._scanned - SAMPLE_RATE)
        samples = self.audio.read(start, end)
        ranges = self.vad(samples) if np.any(samples) else []
        for region in ranges:
            onset, offset = start + region["start"], min(end, start + region["end"])
            if offset <= self._speech_end:
                continue
            self._close_at(onset)
            if not self._speech or self._speech[-1].closed:
                # Repeated VAD scans can rediscover old audio. Retain pre-roll only
                # from the gap preceding a genuinely new region.
                first = max(self.audio.start, self._speech_end, onset - _PRE_ROLL)
                self._speech.append(_Speech(first, offset, first, first))
            else:
                self._speech[-1].end = offset
            self._speech_end = offset
        self._scanned = end
        self._close_at(end)

    def _close_at(self, sample: int) -> None:
        if self._speech and sample - self._speech[-1].end >= self.pause:
            self._speech[-1].closed = True

    def close(self) -> None:
        if self.audio.end > self._scanned:
            self._scan()
        if self._speech:
            self._speech[-1].closed = True

    @property
    def has_speech(self) -> bool:
        return bool(self._speech)

    def next_window(self, hop: int) -> Window | None:
        if not self._speech:
            return None
        region = self._speech[0]
        start = max(region.start, region.floor - _PRE_ROLL)
        # If inference fell behind, decode queued short turns together. Keep
        # their intervening audio and let the punctuation model split them.
        while region.closed and len(self._speech) > 1:
            following = self._speech[1]
            if (
                not following.closed
                or following.end > start + self.max_window
                or following.start - region.end > 2 * SAMPLE_RATE
            ):
                break
            self._speech.popleft()
            following.start = region.start
            following.floor = region.floor
            following.decoded_to = region.decoded_to
            following.previous = region.previous
            region = following
        end = min(self.audio.end, region.end + _POST_ROLL)
        capped = region.end > start + self.max_window
        if not (region.closed or capped) and end - region.decoded_to < hop:
            return None
        return Window(
            start,
            min(end, start + self.max_window),
            region.closed and not capped,
            capped,
            region.decoded_to,
        )

    def decode(
        self, window: Window, decoder: Callable[[np.ndarray, float], list[Decoded]]
    ) -> tuple[Utterance | None, str]:
        region = self._speech[0]
        if window.final and region.decoded_to >= region.end and region.previous:
            words = region.previous
        else:
            samples = self.audio.read(window.start, window.end)
            decoded = decoder(samples, window.start / SAMPLE_RATE)
            floor, end = region.floor / SAMPLE_RATE, window.end / SAMPLE_RATE
            words = []
            for entry in decoded:
                u = entry.utterance
                for word in u.words or [{"w": u.text, "s": u.start, "e": u.end}]:
                    begin = max(window.start / SAMPLE_RATE, word["s"])
                    stop = min(end, word["e"])
                    if stop > floor and (begin + stop) / 2 >= floor and word["w"].strip():
                        words.append({"w": word["w"], "s": max(floor, begin), "e": stop})
            if window.final and not words:
                words = region.previous

        count = (
            len(words)
            if window.final or window.capped
            else _stable_sentence(region.previous, words)
        )
        committed, pending = words[:count], words[count:]
        utterance = None
        if committed:
            utterance = Utterance(
                committed[0]["s"],
                max(w["e"] for w in committed),
                "".join(w["w"] for w in committed).strip(),
                words=committed,
            )
            region.floor = round(utterance.end * SAMPLE_RATE)
        if window.capped:
            # Make progress even if the decoder returned no words near the window edge.
            region.floor = max(region.floor, window.end)
        region.previous = pending
        region.decoded_to = window.end
        if window.final:
            self._speech.popleft()
        return utterance, "".join(w["w"] for w in pending).strip()
