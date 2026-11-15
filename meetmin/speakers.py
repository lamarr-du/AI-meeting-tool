"""Preview and rename speakers in an aligned transcript."""

from __future__ import annotations

import re

from .diarization.align import UNKNOWN_NAME
from .textutil import ts

_GENERATED_NAME = re.compile(r"说话人([A-Z]|[0-9]+)")
_UNKNOWN_NAMES = {UNKNOWN_NAME, "说话人?"}


def _roster(transcript: dict) -> dict[str, str]:
    names = dict.fromkeys(
        name
        for name in [
            *(transcript.get("speakers") or []),
            *(row.get("speaker") for row in transcript["utterances"]),
        ]
        if isinstance(name, str) and name and name not in _UNKNOWN_NAMES
    )
    saved = transcript.get("speaker_names") or {}
    previous_labels = {name: label for label, name in saved.items() if name in names}
    return {previous_labels.get(name, name): name for name in names}


def speaker_preview(transcript: dict) -> list[str]:
    roster = _roster(transcript)
    if not roster:
        return ["暂无可改名的说话人。"]
    by_speaker: dict[str, list[dict]] = {name: [] for name in roster.values()}
    for row in transcript["utterances"]:
        name = row.get("speaker")
        if name in by_speaker and row["text"].strip():
            by_speaker[name].append(row)
    lines: list[str] = []
    for label, name in roster.items():
        code = _GENERATED_NAME.fullmatch(label)
        selector = code[1] if code else label
        lines.append(f"[{selector}] {name}")
        longest = sorted(
            by_speaker[name], key=lambda row: (-len(row["text"].strip()), row["start"])
        )
        for row in longest[:3]:
            text = " ".join(row["text"].split())
            lines.append(f"  [{ts(row['start'])}] {text}")
    return lines


def rename_speakers(transcript: dict, assignments: list[str]) -> dict:
    roster = _roster(transcript)
    updated = dict(roster)
    changed: set[str] = set()
    for assignment in assignments:
        selector, sep, name = assignment.partition("=")
        selector, name = selector.strip(), name.strip()
        if not sep or not selector or not name or not name.isprintable() or name in _UNKNOWN_NAMES:
            raise ValueError(f"无效的改名项 {assignment!r}，请使用 A=姓名")
        matches = []
        for label, current in roster.items():
            code = _GENERATED_NAME.fullmatch(label)
            if selector in (label, current) or (code and selector.upper() == code[1]):
                matches.append(label)
        if len(matches) != 1:
            raise ValueError(f"说话人 {selector!r} 不存在或有歧义，请先运行 speakers list")
        label = matches[0]
        if label in changed:
            raise ValueError(f"重复指定说话人 {selector!r}")
        changed.add(label)
        updated[label] = name
    if len(set(updated.values())) != len(updated):
        raise ValueError("改名后有重名，请为每位说话人指定不同姓名")

    replacements = {roster[label]: updated[label] for label in changed}
    rows = [
        {**row, "speaker": replacements[row["speaker"]]}
        if row.get("speaker") in replacements
        else row
        for row in transcript["utterances"]
    ]
    return {
        **transcript,
        "utterances": rows,
        "speakers": list(updated.values()),
        "speaker_names": updated,
    }
