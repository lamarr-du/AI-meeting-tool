"""Apply auditable glossary replacements to an already aligned transcript."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .glossary import load_glossary

_HAN = re.compile(r"[\u3400-\u9fff]")
_ASCII_WORD = re.compile(r"[a-zA-Z0-9_]")
_ASCII_LETTER = re.compile(r"[a-zA-Z]")


@dataclass(frozen=True)
class _Rule:
    source: str
    target: str
    kind: str
    needs_context: bool = False


def _strings(entry: dict, key: str) -> list[str]:
    values = entry.get(key, [])
    if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"术语 {entry['term']!r} 的 {key} 必须是非空字符串的列表")
    return [value.strip() for value in values]


def _rules(entries: list[dict]) -> list[_Rule]:
    rules: dict[str, _Rule] = {}

    def register(rule: _Rule) -> None:
        key = rule.source.casefold()
        if existing := rules.get(key):
            if existing.target != rule.target:
                raise ValueError(
                    f"术语表中 {rule.source!r} 同时指向 {existing.target!r} 和 {rule.target!r}"
                )
        else:
            rules[key] = rule

    # The accepted spelling of an alias is retained; case normalization keeps it an alias.
    for entry in entries:
        term = entry["term"]
        register(_Rule(term, term, "case"))
        for alias in _strings(entry, "aliases"):
            if alias.casefold() != term.casefold():
                register(_Rule(alias, alias, "case"))
    for entry in entries:
        needs_context = entry.get("needs_context", False)
        if type(needs_context) is not bool:
            raise ValueError(f"术语 {entry['term']!r} 的 needs_context 必须是布尔值")
        for source in _strings(entry, "mishearings"):
            short_han = bool(_HAN.search(source)) and sum(c.isalnum() for c in source) <= 3
            register(_Rule(source, entry["term"], "mishearing", needs_context or short_han))
    return sorted(rules.values(), key=lambda rule: len(rule.source), reverse=True)


def _pattern(rules: list[_Rule]) -> re.Pattern[str]:
    alternatives = []
    for index, rule in enumerate(rules):
        token = re.escape(rule.source)
        left = r"(?<![a-zA-Z0-9_])" if _ASCII_WORD.fullmatch(rule.source[0]) else ""
        right = r"(?![a-zA-Z0-9_])" if _ASCII_WORD.fullmatch(rule.source[-1]) else ""
        alternatives.append(f"(?P<r{index}>{left}{token}{right})")
    # Lookahead finds overlapping starts; length priority is applied before rewriting.
    return re.compile("(?=" + "|".join(alternatives) + ")", re.IGNORECASE | re.ASCII)


def _matches(
    text: str, pattern: re.Pattern[str], rules: list[_Rule]
) -> list[tuple[int, int, _Rule]]:
    candidates = []
    for match in pattern.finditer(text):
        group = match.lastgroup
        assert group is not None
        start, end = match.span(group)
        candidates.append((start, end, rules[int(group[1:])]))
    occupied = bytearray(len(text))
    selected = []
    for start, end, rule in sorted(candidates, key=lambda m: (m[0] - m[1], m[0])):
        if not any(occupied[start:end]):
            occupied[start:end] = b"\1" * (end - start)
            selected.append((start, end, rule))
    return sorted(selected, key=lambda m: m[0])


def correct_transcript(transcript: dict, glossary_path: Path) -> dict:
    rules = _rules(load_glossary(glossary_path))
    pattern = _pattern(rules) if rules else None
    corrected = []
    changes = []
    context_review = []
    for index, utterance in enumerate(transcript["utterances"]):
        text = utterance["text"]
        parts: list[str] = []
        cursor = offset = 0
        for start, end, rule in _matches(text, pattern, rules) if pattern else []:
            before = text[start:end]
            if before == rule.target or (
                rule.kind == "case" and not _ASCII_LETTER.search(rule.source)
            ):
                continue
            record = {
                "utterance_index": index,
                "start_sec": utterance.get("start"),
                "end_sec": utterance.get("end"),
                "speaker": utterance.get("speaker"),
                "source_span": [start, end],
                "before": before,
                "after": rule.target,
                "rule": rule.kind,
            }
            if rule.needs_context:
                context_review.append(record)
                continue
            corrected_start = start + offset
            record["corrected_span"] = [corrected_start, corrected_start + len(rule.target)]
            changes.append(record)
            parts.extend((text[cursor:start], rule.target))
            cursor = end
            offset += len(rule.target) - (end - start)
        corrected.append({**utterance, "text": "".join([*parts, text[cursor:]])})
    return {
        **transcript,
        "utterances": corrected,
        "glossary_correction": {
            "source": "transcript.json",
            "glossary_path": str(glossary_path.resolve()),
            "changes": changes,
            "context_review": context_review,
        },
    }
