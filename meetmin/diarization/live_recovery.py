"""Restore live speaker edits by time after retranscribing recovered audio."""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import replace
from pathlib import Path

from ..asr.base import Utterance
from .align import assign_speakers


def _root(speaker: int, aliases: dict[int, int]) -> int:
    visited = set()
    while speaker in aliases and speaker not in visited:
        visited.add(speaker)
        speaker = aliases[speaker]
    return speaker


def _named_intervals(path: Path) -> list[dict]:
    commits: dict[str, tuple[int, int]] = {}
    assignments: dict[str, list] = {}
    names: dict[int, str] = {}
    aliases: dict[int, int] = {}
    try:
        stream = path.open(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    with stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                # Following a crash the last JSONL record may be incomplete.
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if kind == "asr.commit" and event.get("track") == "loopback":
                segment, start, end = (event.get(k) for k in ("segment_id", "start_ms", "end_ms"))
                if (
                    isinstance(segment, str)
                    and type(start) is int
                    and type(end) is int
                    and 0 <= start < end
                ):
                    commits[segment] = (start, end)
            elif kind == "speaker.rename":
                speaker, name = event.get("speaker_id"), event.get("display_name")
                if (
                    type(speaker) is int
                    and isinstance(name, str)
                    and name.strip()
                    and name.isprintable()
                ):
                    names[speaker] = name
            elif kind in ("diarization.assign", "diarization.reconcile"):
                updates = [event] if kind == "diarization.assign" else event.get("updates")
                for update in updates if isinstance(updates, list) else ():
                    if isinstance(update, dict) and isinstance(
                        segment := update.get("segment_id"), str
                    ):
                        parts = update.get("parts")
                        if isinstance(parts, list):
                            assignments[segment] = parts
                merges = event.get("merges")
                for merge in merges if isinstance(merges, list) else ():
                    if isinstance(merge, dict):
                        old, target = merge.get("from_id"), merge.get("to_id")
                        if (
                            type(old) is int
                            and type(target) is int
                            and _root(target, aliases) != old
                        ):
                            aliases[old] = target

    if not names:
        return []
    names = {_root(speaker, aliases): name for speaker, name in names.items()}
    spans: list[tuple[int, int, str]] = []
    for segment, (start, end) in commits.items():
        points = sorted(
            (
                (max(start, part["start_ms"]), part.get("speaker_id"))
                for part in assignments.get(segment, [])
                if isinstance(part, dict)
                and type(part.get("start_ms")) is int
                and part["start_ms"] < end
            ),
            key=lambda item: item[0],
        )
        for index, (first, speaker) in enumerate(points):
            last = points[index + 1][0] if index + 1 < len(points) else end
            if (
                first < last
                and type(speaker) is int
                and (name := names.get(_root(speaker, aliases)))
            ):
                spans.append((first, last, name))
    intervals: list[dict] = []
    for start, end, name in sorted(spans):
        first = max(start / 1000, intervals[-1]["end"]) if intervals else start / 1000
        last = end / 1000
        if first < last:
            intervals.append({"start": first, "end": last, "label": name})
    return intervals


def restore_live_speaker_names(utterances: list[Utterance], path: Path) -> list[Utterance]:
    """Overlay named intervals while conserving words and the uncommitted audio tail."""
    named = _named_intervals(path)
    if not named:
        return utterances
    ends = [interval["end"] for interval in named]
    restored = []
    for utterance in utterances:
        index = bisect_right(ends, utterance.start)
        spans = []
        start = utterance.start
        while index < len(named) and named[index]["start"] < utterance.end:
            interval = named[index]
            a = max(start, interval["start"])
            b = min(utterance.end, interval["end"])
            if a > start:
                spans.append({"start": start, "end": a, "label": utterance.speaker})
            if b > a:
                spans.append({"start": a, "end": b, "label": interval["label"]})
            start = b
            index += 1
        if not spans:
            restored.append(utterance)
            continue
        if start < utterance.end:
            spans.append({"start": start, "end": utterance.end, "label": utterance.speaker})
        restored.extend(
            assign_speakers([replace(utterance, speaker=None)], spans, min_group_sec=0.25)
        )
    return restored
