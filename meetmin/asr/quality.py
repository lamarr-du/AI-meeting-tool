"""检查本地转写，并按限额重跑可疑的局部音频。"""

from __future__ import annotations

import math
import re
import unicodedata
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import pairwise

from .base import Utterance

_MAX_CLIP_SEC = 20
_RETRY_MIN_LOGPROB = -0.8
_TIME_EPSILON = 0.0001  # WAV 切片的起止点取整到采样帧
_TEXT_LOOP = re.compile(r"(.{1,12}?)\1{5,}")
LIVE_REVIEW_TEXT = "〔该段音频待确认〕"

Span = tuple[float, float]


@dataclass(frozen=True)
class QualityOptions:
    mode: str = "detect"
    min_speech_gap_sec: float = 1.5
    compression_ratio_threshold: float = 2.4
    max_retries: int = 6

    @classmethod
    def from_config(cls, options: dict) -> QualityOptions:
        if not isinstance(options, dict):
            raise ValueError("asr.quality 必须是映射")
        defaults = asdict(cls())
        config = cls(**{key: options.get(key, value) for key, value in defaults.items()})
        if config.mode not in ("off", "detect", "retry"):
            raise ValueError("asr.quality.mode 必须是 off、detect 或 retry")
        for key in ("min_speech_gap_sec", "compression_ratio_threshold"):
            value = getattr(config, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not (math.isfinite(value) and value > 0)
            ):
                raise ValueError(f"asr.quality.{key} 必须是有限的正数")
        if type(config.max_retries) is not int or config.max_retries < 0:
            raise ValueError("asr.quality.max_retries 必须是非负整数")
        return config


@dataclass(frozen=True)
class Decoded:
    utterance: Utterance
    compression_ratio: float = 0.0
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass(frozen=True)
class _Issue:
    start: float
    end: float
    reasons: tuple[str, ...]
    original: Decoded | None = None


def _text_key(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def live_repeat_reasons(
    text: str, words: list[dict], threshold: float, *, compression_ratio: float = 0.0
) -> tuple[str, ...]:
    data = text.encode("utf-8")
    ratio = max(compression_ratio, len(data) / len(zlib.compress(data)))
    if ratio <= threshold:
        return ()
    reasons = []
    # A high compression ratio alone can also come from valid repeated terms.
    if _TEXT_LOOP.search(_text_key(text)):
        reasons.append("text_loop")
    if len(words) >= 12 and 4 * sum(w["s"] == w["e"] for w in words) >= len(words):
        reasons.append("collapsed_timestamps")
    return ("high_compression", *reasons) if reasons else ()


def _is_repeat(left: Utterance, right: Utterance) -> bool:
    if right.start - left.end > 2:
        return False
    key = _text_key(right.text)
    return len(key) >= 6 and key == _text_key(left.text)


def _uncovered(rows: list[Decoded], duration: float, minimum: float) -> list[Span]:
    end = 0.0
    gaps = []
    for row in rows:
        start = min(duration, row.utterance.start)
        if start > end and start - end + _TIME_EPSILON >= minimum:
            gaps.append((end, start))
        end = max(end, row.utterance.end)
    if duration > end and duration - end + _TIME_EPSILON >= minimum:
        gaps.append((end, duration))
    return gaps


def _speech_issues(gaps: list[Span], speech: list[Span], minimum: float) -> list[_Issue]:
    issues = []
    gap_index = speech_index = 0
    while gap_index < len(gaps) and speech_index < len(speech):
        gap_start, gap_end = gaps[gap_index]
        speech_start, speech_end = speech[speech_index]
        start, end = max(gap_start, speech_start), min(gap_end, speech_end)
        if end > start and end - start + _TIME_EPSILON >= minimum:
            # 同一长缺口分片，逐片使用同样的重试限额。
            count = math.ceil((end - start) / _MAX_CLIP_SEC)
            size = (end - start) / count
            for index in range(count):
                issues.append(
                    _Issue(start + index * size, start + (index + 1) * size, ("speech_gap",))
                )
        if gap_end <= speech_end:
            gap_index += 1
        else:
            speech_index += 1
    return issues


def reject_replacement(
    span: Span,
    candidate: list[Decoded],
    kept: list[Decoded],
    options: QualityOptions,
    *,
    min_coverage: float = 0,
) -> str | None:
    start, stop = span
    replacement = [row.utterance for row in candidate]
    if sum(len(_text_key(u.text)) for u in replacement) < 2:
        return "too_little_text"
    if any(
        not (
            math.isfinite(row.compression_ratio)
            and row.compression_ratio <= options.compression_ratio_threshold
            and math.isfinite(row.avg_logprob)
            and row.avg_logprob >= _RETRY_MIN_LOGPROB
            and math.isfinite(row.no_speech_prob)
            and row.no_speech_prob <= 0.6
        )
        for row in candidate
    ):
        return "low_confidence"
    end = start
    for u in replacement:
        if not (end - _TIME_EPSILON <= u.start < u.end <= stop + _TIME_EPSILON):
            return "invalid_timing"
        if any(
            not (u.start - 0.001 <= word["s"] <= word["e"] <= u.end + 0.001)
            for word in u.words or []
        ):
            return "invalid_timing"
        end = u.end
    kept_utterances = [row.utterance for row in kept]
    if any(
        max(a.start, b.start) + _TIME_EPSILON < min(a.end, b.end)
        for a in replacement
        for b in kept_utterances
    ):
        return "overlaps_neighbor"
    if sum(u.end - u.start for u in replacement) < (stop - start) * min_coverage:
        return "reduced_coverage"
    changed = {id(u) for u in replacement}
    combined = sorted(kept_utterances + replacement, key=lambda u: u.start)
    if any(
        (id(a) in changed or id(b) in changed) and _is_repeat(a, b) for a, b in pairwise(combined)
    ):
        return "still_repeated"
    return None


def review(
    decoded: list[Decoded],
    duration: float,
    options: QualityOptions,
    speech_spans: Callable[[], list[Span]],
    retry: Callable[[float, float], list[Decoded]],
) -> tuple[list[Utterance], dict]:
    if options.mode == "off":
        return [row.utterance for row in decoded], {"mode": "off"}

    rows = sorted(
        (row for row in decoded if row.utterance.text.strip()), key=lambda r: r.utterance.start
    )
    issues = []
    previous = None
    for row in rows:
        u = row.utterance
        reasons = []
        if previous is not None and _is_repeat(previous, u):
            reasons.append("adjacent_repeat")
        if row.compression_ratio > options.compression_ratio_threshold:
            reasons.append("high_compression")
        if reasons:
            issues.append(_Issue(u.start, u.end, tuple(reasons), row))
        previous = u
    gaps = _uncovered(rows, duration, options.min_speech_gap_sec)
    report: dict = {"mode": options.mode, "attempted": 0, "accepted": 0, "issues": []}
    if gaps:
        try:
            speech = speech_spans()
        except Exception as error:
            report["vad_error"] = f"{type(error).__name__}: {error}"
        else:
            issues.extend(_speech_issues(gaps, speech, options.min_speech_gap_sec))

    # 先尝试修复已输出的可疑文本，再补缺口。最终诊断仍按时间排序。
    for issue in issues:
        entry = {
            "start": round(issue.start, 3),
            "end": round(issue.end, 3),
            "reasons": list(issue.reasons),
            "status": "detected",
        }
        report["issues"].append(entry)
        if options.mode != "retry":
            continue
        if (
            issue.start < 0
            or issue.end > duration + _TIME_EPSILON
            or not (0 < issue.end - issue.start <= _MAX_CLIP_SEC + _TIME_EPSILON)
        ):
            entry.update(status="skipped", detail="clip_bounds")
            continue
        if report["attempted"] >= options.max_retries:
            entry.update(status="skipped", detail="retry_limit")
            continue
        report["attempted"] += 1
        try:
            candidate = retry(issue.start, issue.end)
        except Exception as error:
            entry.update(status="failed", detail=f"{type(error).__name__}: {error}")
            continue
        kept = [row for row in rows if row is not issue.original]
        if reason := reject_replacement(
            (issue.start, issue.end),
            candidate,
            kept,
            options,
            min_coverage=0.5 if issue.original else 0,
        ):
            entry.update(status="rejected", detail=reason)
            continue
        rows = sorted(kept + candidate, key=lambda row: row.utterance.start)
        entry["status"] = "accepted"
        if issue.original is not None:
            entry["original_utterance"] = issue.original.utterance.to_dict()
        report["accepted"] += 1
    report["issues"].sort(key=lambda entry: entry["start"])
    return [row.utterance for row in (rows if report["accepted"] else decoded)], report
