"""ASR 文本自然成句重排、标点恢复、轻量顺滑与会议段落聚合。

对应四阶段自然语句重排设计：
  1. 语速自适应同说话人预合并 + 禁尾词/中英码转换守卫
  2. 纯 ONNX 语义标点恢复（FireRedPunc Q8W / CT-Transformer INT8）与自然句切分
  3. 词级时间戳无损回溯 + 保真/轻量顺滑双视图
  4. 面向阅读与会议纪要的自然段落聚合与标点衔接
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .base import Utterance
from .punc_onnx import Punctuator

_CHINESE_COMMA = re.compile(r"(?<=[\u3400-\u9fff])[ \t]*,[ \t]*|[ \t]*,[ \t]*(?=[\u3400-\u9fff])")
_CLOSING_MARKS = "\"'”’」』）】》)]"
_SENTENCE_PAUSE_SECONDS = 0.8

TERMINAL_PUNCS = frozenset({"。", "？", "！", ".", "?", "!"})
PAUSE_PUNCS = frozenset({"，", ",", "；", ";", "：", ":", "、"})
ALL_PUNCS = TERMINAL_PUNCS | PAUSE_PUNCS

# 禁尾词（US8364485B2）：句末停顿若落在未完结的填充词、连词或结构助词上，放宽合并阈值防止误切。
IMPERMISSIBLE_TAIL_WORDS = frozenset(
    {
        "那个",
        "这个",
        "然后",
        "但是",
        "因为",
        "所以",
        "如果",
        "就是",
        "开始",
        "而且",
        "并且",
        "以及",
        "关于",
        "对于",
        "正在",
        "的",
        "地",
        "得",
        "在",
        "把",
        "被",
    }
)

# 保留数字千分位（1,024）、版本号（v1.2）、连字符术语（large-v3-turbo）、符号（@ / C++ / 100%）等完整 ASCII 词元。
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._,/+:-][A-Za-z0-9]+)*[+%#]*|[@#%&]|[\u3400-\u9fff]")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_ASCII_WORD_START = re.compile(r"^[A-Za-z0-9_@]")
_ASCII_WORD_END = re.compile(r"[A-Za-z0-9_+%#]$")


@dataclass
class TokenSpan:
    text: str
    start_ms: int
    end_ms: int
    speaker: str | int | None
    is_ascii: bool = False


def has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def is_zh_like(utterances: Sequence[Utterance], language: str | None) -> bool:
    if language:
        return language.lower().startswith("zh")
    sample = "".join(u.text for u in utterances[:20])
    cjk_count = len(_CJK_RE.findall(sample))
    return cjk_count >= 4 and cjk_count * 3 >= max(1, len(sample.strip()))


def _ends_with_terminal_punct(text: str) -> bool:
    cleaned = text.rstrip(_CLOSING_MARKS).rstrip()
    return bool(cleaned) and cleaned[-1] in TERMINAL_PUNCS


def _ends_with_any_punct(text: str) -> bool:
    cleaned = text.rstrip(_CLOSING_MARKS).rstrip()
    return bool(cleaned) and cleaned[-1] in ALL_PUNCS


def _is_ascii_word_char(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def punctuate_chinese(utterances: list[Utterance], language: str | None) -> None:
    """规则版中文标点整理（零模型下载，用于 rule 模式或 ONNX 不可用时的兜底）。"""
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


def _interpolate_tokens(
    raw_tokens: list[str],
    start_ms: int,
    end_ms: int,
    speaker: str | int | None,
) -> list[TokenSpan]:
    if not raw_tokens:
        return []
    weights = [max(1.0, len(t) * 0.45) if t.isascii() else 1.0 for t in raw_tokens]
    total_w = sum(weights) or 1.0
    dur = max(1, end_ms - start_ms)
    spans: list[TokenSpan] = []
    acc = 0.0
    for tok, w in zip(raw_tokens, weights):
        st = start_ms + int(dur * acc / total_w)
        acc += w
        ed = start_ms + int(dur * acc / total_w)
        spans.append(TokenSpan(tok, st, max(st + 1, ed), speaker, tok.isascii()))
    return spans


def _coalesce_whisper_words(words: list[dict], u: Utterance) -> list[tuple[str, int, int]]:
    """合并 Whisper BPE 拆分出来的同一英文词片（如 're' + 'view' -> 'review'，' patch' 保持独立）。"""
    merged: list[tuple[str, int, int]] = []
    for item in words:
        raw_w = str(item.get("w") or "")
        if not raw_w.strip():
            continue
        st = int(round(float(item.get("s", u.start)) * 1000))
        ed = max(st + 1, int(round(float(item.get("e", item.get("s", u.end))) * 1000)))
        has_leading_space = raw_w[0].isspace()
        stripped = raw_w.strip()
        if (
            merged
            and not has_leading_space
            and _ASCII_WORD_END.search(merged[-1][0])
            and _ASCII_WORD_START.search(stripped)
        ):
            prev_w, prev_st, prev_ed = merged[-1]
            merged[-1] = (prev_w + stripped, prev_st, max(prev_ed, ed))
        elif (
            merged
            and not has_leading_space
            and merged[-1][0][-1:].isascii()
            and stripped[:1] in ".,/-:+"
            and len(stripped) > 1
            and stripped[1:2].isalnum()
        ):
            prev_w, prev_st, prev_ed = merged[-1]
            merged[-1] = (prev_w + stripped, prev_st, max(prev_ed, ed))
        else:
            merged.append((stripped, st, ed))
    return merged


def tokenize_utterance(u: Utterance) -> list[TokenSpan]:
    """将 Utterance 转为带精确毫秒时间戳的 TokenSpan 列表。

    优先使用原生词级时间戳（先合并 Whisper BPE 英文碎片，再对多字词内部线性插值）；
    无词级时间戳时按中文 1.0 / 英文近似音节权重在句级起止内插值。
    """
    if u.words:
        coalesced = _coalesce_whisper_words(u.words, u)
        spans: list[TokenSpan] = []
        for word_text, st, ed in coalesced:
            sub_tokens = _TOKEN_RE.findall(word_text)
            if not sub_tokens:
                continue
            if len(sub_tokens) == 1:
                tok = sub_tokens[0]
                spans.append(TokenSpan(tok, st, ed, u.speaker, tok.isascii()))
            else:
                spans.extend(_interpolate_tokens(sub_tokens, st, ed, u.speaker))
        if spans:
            return spans

    s_ms = int(round(u.start * 1000))
    e_ms = max(s_ms + 1, int(round(u.end * 1000)))
    raw_tokens = _TOKEN_RE.findall(u.text)
    return _interpolate_tokens(raw_tokens, s_ms, e_ms, u.speaker)


def render_tokens(spans: Sequence[TokenSpan], tags: Sequence[str]) -> str:
    """将词元与预测标点渲染为整洁的中英混排自然句。"""
    out: list[str] = []
    for i, (sp, tag) in enumerate(zip(spans, tags)):
        if (
            i > 0
            and sp.is_ascii
            and out
            and not out[-1].endswith(("，", "。", "？", "！", "、", " "))
        ):
            out.append(" ")
        elif i > 0 and (not sp.is_ascii) and spans[i - 1].is_ascii and tags[i - 1] == "":
            out.append(" ")
        out.append(sp.text)
        if tag:
            out.append(tag)
    return "".join(out)


def lightweight_polish(text: str) -> str:
    """零幻觉轻量规则顺滑：仅剔除明显口癖填充词与直接口吃重复，绝不改写主干事实。"""
    s = text.strip()
    if not s:
        return s
    # 1. 句首或逗号后孤立的语气词（如 "呃，我这边" -> "我这边"）
    s = re.sub(r"(?:^|(?<=[，。？！]))(?:呃|嗯|啊)[，、]+", "", s)
    # 2. 连续重复的填充词（"那个那个" -> "那个"）或紧跟代词/常见动词前的无实义"那个"
    s = re.sub(r"(?:那个){2,}", "那个", s)
    s = re.sub(r"(?:那个|呃|嗯)(?=[，。？！]|我|他|她|它|今天|开始|对应)", "", s)
    # 3. 直接口吃重复词（如 "给给"、"开始慢慢的开始" -> "开始慢慢的"、"就是就是" -> "就是"）
    s = re.sub(r"([\u3400-\u9fff]{2,4})(慢慢的|就是)?\1", r"\1\2", s)
    s = re.sub(r"([给就那这把在和跟对])\1+", r"\1", s)
    # 4. 清理多余标点
    s = re.sub(r"[，、]{2,}", "，", s)
    s = re.sub(r"^[，、]+", "", s)
    s = re.sub(r"[，、]+([。？！])", r"\1", s)
    return s.strip() or text.strip()


def _adaptive_commit_pause_ms(
    buf: list[TokenSpan],
    next_tokens: list[TokenSpan],
    spk_rate_ema: dict[str | int | None, float],
) -> int:
    """语速自适应停顿阈值（CN110364145A + LiveKit EMA）+ 禁尾词/码转换宽限（US8364485B2）。"""
    speaker = buf[0].speaker
    rate = spk_rate_ema.get(speaker, 4.0)
    base_ms = max(350.0, min(1200.0, 700.0 * (4.0 / max(1.5, rate))))
    tail_2 = "".join(t.text for t in buf[-2:]) if len(buf) >= 2 else ""
    tail_1 = buf[-1].text
    code_switch = bool(next_tokens and (not buf[-1].is_ascii) and next_tokens[0].is_ascii)
    if tail_2 in IMPERMISSIBLE_TAIL_WORDS or tail_1 in IMPERMISSIBLE_TAIL_WORDS or code_switch:
        base_ms = min(1500.0, base_ms + 600.0)
    return int(min(2000.0, max(1100.0, base_ms * 1.6)))


def _update_rate_ema(tokens: list[TokenSpan], spk_rate_ema: dict[str | int | None, float]) -> None:
    if not tokens:
        return
    dur_s = max(0.2, (tokens[-1].end_ms - tokens[0].start_ms) / 1000.0)
    inst_rate = min(9.0, max(1.5, len(tokens) / dur_s))
    spk = tokens[0].speaker
    prev = spk_rate_ema.get(spk, 4.0)
    spk_rate_ema[spk] = 0.8 * prev + 0.2 * inst_rate


def _split_block_to_sentences(
    spans: list[TokenSpan],
    tags: list[str],
    *,
    max_sentence_tokens: int = 85,
    max_sentence_sec: float = 18.0,
    enable_polish: bool = True,
) -> list[Utterance]:
    """按终结标点与超长独白保护门控切出自然句，并无损回溯词级时间戳。"""
    if not spans:
        return []
    tags = list(tags)
    if tags and tags[-1] == "":
        tags[-1] = "。"
    max_dur_ms = int(round(max_sentence_sec * 1000))
    results: list[Utterance] = []
    seg_start = 0
    n = len(spans)
    idx = 0
    while idx < n:
        tok_count = idx - seg_start + 1
        seg_dur_ms = spans[idx].end_ms - spans[seg_start].start_ms
        is_end = tags[idx] in TERMINAL_PUNCS or idx == n - 1
        cut_idx = idx

        if not is_end and (tok_count >= max_sentence_tokens or seg_dur_ms >= max_dur_ms):
            search_lo = seg_start + min(50, max(10, tok_count // 2))
            soft_cut = -1
            for j in range(idx, search_lo - 1, -1):
                if tags[j] in ("，", ",", "。", "？", "！"):
                    soft_cut = j
                    break
            if soft_cut == -1 and idx > search_lo:
                gaps = [(spans[j + 1].start_ms - spans[j].end_ms, j) for j in range(search_lo, idx)]
                soft_cut = max(gaps)[1] if gaps else idx
            cut_idx = soft_cut if soft_cut >= seg_start else idx
            if tags[cut_idx] not in TERMINAL_PUNCS:
                tags[cut_idx] = "。"
            is_end = True

        if is_end:
            sub_spans = spans[seg_start : cut_idx + 1]
            sub_tags = tags[seg_start : cut_idx + 1]
            raw_text = render_tokens(sub_spans, sub_tags)
            polished = lightweight_polish(raw_text) if enable_polish else None
            results.append(
                Utterance(
                    start=round(sub_spans[0].start_ms / 1000.0, 2),
                    end=round(sub_spans[-1].end_ms / 1000.0, 2),
                    text=raw_text,
                    speaker=sub_spans[0].speaker,
                    polished_text=polished if (polished and polished != raw_text) else None,
                )
            )
            seg_start = cut_idx + 1
            idx = seg_start
        else:
            idx += 1
    return results


def reconstruct_sentences(
    utterances: list[Utterance],
    *,
    language: str | None,
    punctuator: Punctuator,
    max_sentence_tokens: int = 85,
    max_sentence_sec: float = 18.0,
    left_context_tokens: int = 15,
    enable_polish: bool = True,
) -> list[Utterance]:
    """执行四阶段中的阶段一至阶段三：语速自适应合并 → ONNX 语义标点切句 → 时间戳回溯与双视图。"""
    if not utterances or not is_zh_like(utterances, language):
        return utterances

    # 阶段一：同说话人短间隔自适应聚合（含禁尾词与中英码转换守卫）
    spk_rate_ema: dict[str | int | None, float] = {}
    blocks: list[list[TokenSpan]] = []
    cur: list[TokenSpan] = []

    for u in utterances:
        new_tokens = tokenize_utterance(u)
        if not new_tokens:
            continue
        _update_rate_ema(new_tokens, spk_rate_ema)
        if not cur:
            cur = new_tokens
            continue
        curr_spk = cur[0].speaker
        gap_ms = new_tokens[0].start_ms - cur[-1].end_ms
        commit_pause_ms = _adaptive_commit_pause_ms(cur, new_tokens, spk_rate_ema)
        if u.speaker != curr_spk or gap_ms >= commit_pause_ms:
            blocks.append(cur)
            cur = new_tokens
        else:
            cur.extend(new_tokens)
    if cur:
        blocks.append(cur)

    # 阶段二 & 三：携带前句只读左上下文运行标点模型，切出自然句并无损回溯时间戳
    reconstructed: list[Utterance] = []
    left_context: list[str] = []
    prev_speaker: str | int | None = None
    for block in blocks:
        blk_spk = block[0].speaker
        ctx = left_context if (left_context_tokens > 0 and blk_spk == prev_speaker) else []
        full_tokens = ctx + [t.text for t in block]
        full_tags = punctuator.predict_tags(full_tokens)
        blk_tags = full_tags[len(ctx) : len(ctx) + len(block)]
        if len(blk_tags) < len(block):
            blk_tags.extend([""] * (len(block) - len(blk_tags)))
        reconstructed.extend(
            _split_block_to_sentences(
                block,
                blk_tags,
                max_sentence_tokens=max_sentence_tokens,
                max_sentence_sec=max_sentence_sec,
                enable_polish=enable_polish,
            )
        )
        left_context = (
            [t.text for t in block[-left_context_tokens:]] if left_context_tokens > 0 else []
        )
        prev_speaker = blk_spk
    return reconstructed or utterances


def _utterance_field(item: Utterance | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _utterance_text(item: Utterance | dict[str, Any], *, use_polished: bool = False) -> str:
    if use_polished:
        polished = _utterance_field(item, "polished_text")
        if isinstance(polished, str) and polished.strip():
            return polished.strip()
    return str(_utterance_field(item, "text", "") or "").strip()


def join_utterance_texts(
    utterances: Sequence[Utterance | dict[str, Any]],
    *,
    language: str | None = None,
    use_polished: bool = False,
) -> str:
    """将同一段落内的多条发言平滑拼接；若片段未带标点则按停顿与禁尾词规则补标点。"""
    items = [u for u in utterances if _utterance_text(u, use_polished=use_polished)]
    if not items:
        return ""
    zh_mode = (language is not None and language.lower().startswith("zh")) or any(
        has_cjk(_utterance_text(u, use_polished=use_polished)) for u in items
    )
    out = ""
    for idx, item in enumerate(items):
        text = _utterance_text(item, use_polished=use_polished)
        if not out:
            out = text
            continue
        prev = items[idx - 1]
        gap = round(
            float(_utterance_field(item, "start", 0.0)) - float(_utterance_field(prev, "end", 0.0)),
            3,
        )
        if not _ends_with_any_punct(out):
            if zh_mode:
                tail_2 = out[-2:] if len(out) >= 2 else out
                tail_1 = out[-1:]
                code_switch = (not out[-1].isascii()) and text[0].isascii()
                is_hanging = (
                    tail_2 in IMPERMISSIBLE_TAIL_WORDS or tail_1 in IMPERMISSIBLE_TAIL_WORDS
                )
                if gap >= _SENTENCE_PAUSE_SECONDS:
                    out += "。"
                elif gap < 0.15 and (is_hanging or code_switch):
                    if _is_ascii_word_char(out[-1]) and _is_ascii_word_char(text[0]):
                        out += " "
                    elif code_switch:
                        out += " "
                else:
                    out += "，"
            else:
                out += " "
        else:
            if _is_ascii_word_char(out[-1]) and _is_ascii_word_char(text[0]):
                out += " "
        out += text
    return out


def group_paragraphs(
    utterances: Sequence[Utterance | dict[str, Any]],
    *,
    max_gap_sec: float = 2.2,
    max_chars: int = 220,
    max_duration_sec: float = 45.0,
) -> list[list[Any]]:
    """阶段四：将连续同说话人的自然句聚合成适合阅读与纪要的会议段落。

    分段条件：
      1. 说话人切换；
      2. 同说话人句间停顿 >= max_gap_sec（换话题/长停顿）；
      3. 单段累计字数 >= max_chars 或持续时长 >= max_duration_sec（防止单人长独白成文字墙）。
    """
    groups: list[list[Any]] = []
    cur: list[Any] = []
    for item in utterances:
        text = _utterance_text(item)
        if not text:
            continue
        if not cur:
            cur = [item]
            continue
        prev = cur[-1]
        same_speaker = _utterance_field(item, "speaker") == _utterance_field(prev, "speaker")
        gap = float(_utterance_field(item, "start", 0.0)) - float(
            _utterance_field(prev, "end", 0.0)
        )
        cur_chars = sum(len(_utterance_text(x)) for x in cur)
        cur_dur = float(_utterance_field(prev, "end", 0.0)) - float(
            _utterance_field(cur[0], "start", 0.0)
        )
        prev_terminal = _ends_with_terminal_punct(_utterance_text(prev))
        should_split = (
            not same_speaker
            or gap >= max_gap_sec
            or (cur_chars >= max_chars and (prev_terminal or gap >= 0.4))
            or (cur_dur >= max_duration_sec and (prev_terminal or gap >= 0.5))
        )
        if should_split:
            groups.append(cur)
            cur = [item]
        else:
            cur.append(item)
    if cur:
        groups.append(cur)
    return groups
