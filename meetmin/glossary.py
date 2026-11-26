"""Load the D1 glossary and select confirmed terms for ASR."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import yaml


def _load_single_glossary(path: Path) -> list[dict]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取术语表 {path}：{error}") from error
    if not isinstance(document, dict) or not isinstance(document.get("terms"), list):
        raise ValueError(f"术语表 {path} 需要提供 terms 列表")
    entries = []
    for index, entry in enumerate(document["terms"], 1):
        term = entry.get("term") if isinstance(entry, dict) else None
        if not isinstance(term, str) or not term.strip():
            raise ValueError(f"术语表 {path} 的第 {index} 条需要提供非空 term")
        entries.append({**entry, "term": term.strip()})
    return entries


def load_glossary(path: Path) -> list[dict]:
    entries = _load_single_glossary(path)
    if path.name == "glossary.yaml":
        local_path = path.with_name("glossary.local.yaml")
        if local_path.is_file():
            entries.extend(_load_single_glossary(local_path))
    return entries


def load_terms(path: Path) -> list[str]:
    terms: dict[str, str] = {}
    for entry in load_glossary(path):
        term = entry["term"]
        terms.setdefault(term.casefold(), term)
    return list(terms.values())


def select_terms(terms: list[str], priority: list[str] | None) -> list[str]:
    if priority is None:
        return terms
    by_name = {term.casefold(): term for term in terms}
    selected: dict[str, None] = {}
    for name in priority:
        term = by_name.get(name.strip().casefold())
        if term is None:
            raise ValueError(f"术语表中找不到标准写法 {name!r}")
        selected[term] = None
    return list(selected)


def glossary_hotwords(
    terms: list[str],
    *,
    existing: str | None,
    language: str | None,
    token_count: Callable[[str], int],
    token_limit: int,
    max_terms: int,
) -> tuple[str | None, list[str]]:
    selected: list[str] = []
    prompt = existing
    for term in terms:
        words = [*selected, term]
        if language == "zh":
            glossary = "本次会议涉及 " + "、".join(words) + "。"
        else:
            glossary = ", ".join(words)
        candidate = " ".join(part for part in (existing, glossary) if part)
        # Whisper counts the leading space and truncates any excess tokens.
        if token_count(" " + candidate) <= token_limit:
            prompt = candidate
            selected.append(term)
            if len(selected) >= max_terms:
                break
    return prompt, selected
