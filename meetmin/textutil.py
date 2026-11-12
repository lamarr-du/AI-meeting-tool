"""中文/英文混排文本拼接等小工具。"""

from __future__ import annotations


def _is_ascii_word_char(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def smart_join(parts) -> str:
    """拼接分词片段：仅当两侧都是 ASCII 字母数字时补空格，避免中文词间被加空格。"""
    out = ""
    for p in parts:
        p = (p or "").strip()
        if not p:
            continue
        if out and _is_ascii_word_char(out[-1]) and _is_ascii_word_char(p[0]):
            out += " "
        out += p
    return out


def ts(sec: float) -> str:
    """秒 → MM:SS / H:MM:SS。"""
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
