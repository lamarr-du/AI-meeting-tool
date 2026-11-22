"""把说话人时间段对齐到 ASR 句子：优先用词级时间戳逐词判人，并在换人处把长句切开。"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import replace
from math import isclose

from ..asr.base import Utterance
from ..textutil import smart_join

UNKNOWN = -999
UNKNOWN_NAME = "未知"
_MAX_NEAREST_GAP_SEC = 0.75


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _is_sorted_non_overlapping(segments: list[dict]) -> bool:
    for i in range(1, len(segments)):
        if segments[i]["start"] < segments[i - 1]["end"] - 1e-9:
            return False
    return True


def _speaker_at_point_sorted(t: float, segments: list[dict], starts: list[float]):
    idx = bisect_right(starts, t) - 1
    if idx >= 0 and segments[idx]["start"] <= t <= segments[idx]["end"]:
        return segments[idx]["label"]
    return None


def speaker_at_point(t: float, segments: list[dict]):
    best_label = None
    best_dist = float("inf")
    for s in segments:
        if s["start"] <= t <= s["end"]:
            dist = abs((s["start"] + s["end"]) / 2.0 - t)
            if dist < best_dist:
                best_dist = dist
                best_label = s["label"]
    return best_label


def majority_label(start: float, end: float, segments: list[dict]):
    score: dict = {}
    for s in segments:
        ov = _overlap(start, end, s["start"], s["end"])
        if ov > 0:
            score[s["label"]] = score.get(s["label"], 0.0) + ov
    if not score:
        return None
    return max(score.items(), key=lambda kv: kv[1])[0]


def _nearest_label(start: float, end: float, segments: list[dict]) -> str | int:
    distance = _MAX_NEAREST_GAP_SEC
    nearest: set[str | int] = set()
    for segment in segments:
        label = segment["label"]
        if label in (None, UNKNOWN, UNKNOWN_NAME, "说话人?"):
            continue
        gap = max(segment["start"] - end, start - segment["end"], 0.0)
        if isclose(gap, distance, rel_tol=0, abs_tol=1e-6):
            nearest.add(label)
        elif gap < distance:
            distance = gap
            nearest = {label}
    return next(iter(nearest)) if len(nearest) == 1 else UNKNOWN


def _word_span(w: dict, u: Utterance) -> tuple[float, float]:
    s = float(w.get("s", u.start))
    e = float(w.get("e", w.get("s", u.end)))
    return s, max(s, e)


def assign_speakers(
    utterances: list[Utterance],
    segments: list[dict],
    *,
    min_group_sec: float = 0.4,
) -> list[Utterance]:
    """为每条 utterance 赋 speaker；有词级时间戳时按换人点拆句。"""
    sorted_fast = bool(segments) and _is_sorted_non_overlapping(segments)
    seg_starts = [float(s["start"]) for s in segments] if sorted_fast else []
    out: list[Utterance] = []
    for u in utterances:
        if u.speaker is not None:
            out.append(u)
            continue
        if not segments:
            out.append(replace(u, speaker=UNKNOWN))
            continue
        base = majority_label(u.start, u.end, segments)
        if base is None:
            base = _nearest_label(u.start, u.end, segments)

        if not u.words or len(u.words) < 2:
            out.append(replace(u, speaker=base))
            continue

        # 逐词取说话人，落在两段之间的空隙词沿用句级多数
        word_labels = []
        for w in u.words:
            s, e = _word_span(w, u)
            mid = (s + e) / 2
            lab = (
                _speaker_at_point_sorted(mid, segments, seg_starts)
                if sorted_fast
                else speaker_at_point(mid, segments)
            )
            word_labels.append(base if lab is None else lab)

        groups: list[list] = []
        for w, lab in zip(u.words, word_labels):
            if groups and groups[-1][0] == lab:
                groups[-1][1].append(w)
            else:
                groups.append([lab, [w]])

        # 平滑：过短的说话人片段并入相邻片段（优先并入前段）
        changed = True
        while changed and len(groups) > 1:
            changed = False
            for i, (lab, ws) in enumerate(groups):
                dur = _word_span(ws[-1], u)[1] - _word_span(ws[0], u)[0]
                if dur < min_group_sec:
                    if i > 0:
                        groups[i - 1][1].extend(ws)
                        groups.pop(i)
                    else:
                        groups[1][1] = ws + groups[1][1]
                        groups.pop(0)
                    changed = True
                    break
        # 并入后可能出现相邻同标签组 → 二次合并
        merged_groups: list[list] = []
        for lab, ws in groups:
            if merged_groups and merged_groups[-1][0] == lab:
                merged_groups[-1][1].extend(ws)
            else:
                merged_groups.append([lab, list(ws)])
        groups = merged_groups

        if len(groups) == 1:
            out.append(replace(u, speaker=groups[0][0]))
            continue

        for lab, ws in groups:
            text = smart_join([w.get("w", "") for w in ws])
            if not text:
                continue
            out.append(
                Utterance(
                    start=_word_span(ws[0], u)[0],
                    end=_word_span(ws[-1], u)[1],
                    text=text,
                    speaker=lab,
                    words=ws,
                )
            )
    return out


def speaker_name(index: int) -> str:
    return "说话人" + (chr(ord("A") + index) if index < 26 else str(index + 1))


def relabel(utterances: list[Utterance]) -> list[str]:
    """把内部簇标签/云端 speaker_id 按首次发言顺序映射为 '说话人A/B/...'，原地改写并返回名单。"""
    mapping: dict = {}
    speakers: list[str] = []

    for u in utterances:
        key = u.speaker
        if key in (None, UNKNOWN, UNKNOWN_NAME, "说话人?"):
            u.speaker = UNKNOWN_NAME
            continue
        if key not in mapping:
            mapping[key] = speaker_name(len(speakers))
            speakers.append(mapping[key])
        u.speaker = mapping[key]
    return speakers
