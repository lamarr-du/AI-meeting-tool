"""在可读转写中整理中文标点，保留原始转写和词级时间戳。"""

from __future__ import annotations

import re

from .base import Utterance

_CHINESE_COMMA = re.compile(r"(?<=[\u3400-\u9fff])[ \t]*,[ \t]*|[ \t]*,[ \t]*(?=[\u3400-\u9fff])")
_CLOSING_MARKS = "\"'”’」』）】》)]"
_SENTENCE_PAUSE_SECONDS = 0.8


def punctuate_chinese(utterances: list[Utterance], language: str | None) -> None:
    if language != "zh":
        return
    for index, utterance in enumerate(utterances):
        text = _CHINESE_COMMA.sub("，", utterance.text.strip())
        at_boundary = index == len(utterances) - 1 or (
            round(utterances[index + 1].start - utterance.end, 3) >= _SENTENCE_PAUSE_SECONDS
        )
        if at_boundary and text.rstrip(_CLOSING_MARKS).rstrip()[-1:].isalnum():
            text += "。"
        utterance.text = text
