"""零依赖繁转简（基于 OpenCC Apache-2.0 TSCharacters + TSPhrases）。

解决 Whisper 在中文转写中跨 30 秒窗口随机混入繁体字的问题：
  - 先按最大前向匹配应用 OpenCC TSPhrases（含乾坤等防误转保护词条）
  - 再按 TSCharacters 单字映射表转为标准简体中文
"""

from __future__ import annotations

from ._t2s_data import T2S_CHAR_DST, T2S_CHAR_SRC, T2S_PHRASES
from .base import Utterance

_CHAR_MAP: dict[str, str] = dict(zip(T2S_CHAR_SRC, T2S_CHAR_DST, strict=True))
_PHRASE_MAX_LEN_BY_FIRST: dict[str, int] = {}
for _phrase in T2S_PHRASES:
    _first = _phrase[0]
    _prev = _PHRASE_MAX_LEN_BY_FIRST.get(_first, 0)
    if len(_phrase) > _prev:
        _PHRASE_MAX_LEN_BY_FIRST[_first] = len(_phrase)


def to_simplified(text: str) -> str:
    """将文本中的繁体中文确定性地转换为简体中文；英文与数字原样保留。"""
    if not text:
        return text
    n = len(text)
    out: list[str] = []
    i = 0
    while i < n:
        ch = text[i]
        max_len = _PHRASE_MAX_LEN_BY_FIRST.get(ch)
        if max_len:
            matched = False
            for length in range(min(max_len, n - i), 1, -1):
                seg = text[i : i + length]
                repl = T2S_PHRASES.get(seg)
                if repl is not None:
                    out.append(repl)
                    i += length
                    matched = True
                    break
            if matched:
                continue
        out.append(_CHAR_MAP.get(ch, ch))
        i += 1
    return "".join(out)


def convert_utterances_to_simplified(utterances: list[Utterance]) -> None:
    """原地将一组 Utterance 的文本与词级文本转为简体中文。"""
    for u in utterances:
        u.text = to_simplified(u.text)
        if u.polished_text:
            u.polished_text = to_simplified(u.polished_text)
        if u.words:
            for w in u.words:
                if isinstance(w.get("w"), str):
                    w["w"] = to_simplified(w["w"])
