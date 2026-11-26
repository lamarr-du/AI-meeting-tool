"""Local speaker profiles and an opt-in voiceprint library shared across meetings."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np

from ..asr.base import Utterance
from ..audio.segments import SessionLock
from ..session import write_json_safely
from ..speakers import is_person_name
from .align import UNKNOWN_NAME
from .clustering import SR

_MIN_SIMILARITY = 0.72
_MIN_MARGIN = 0.08
_EMBEDDING_DIM = 192
_Speaker = TypeVar("_Speaker", str, int)


@dataclass(frozen=True)
class Enrollment:
    added: list[str]
    existing: list[str]


def audio_fingerprint(audio: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(audio))).hexdigest()


def _unit_vector(value) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, OverflowError) as exc:
        raise ValueError("声纹向量格式无效") from exc
    if vector.shape != (_EMBEDDING_DIM,) or not np.isfinite(vector).all():
        raise ValueError("声纹向量格式无效")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-9:
        raise ValueError("声纹向量长度无效")
    return vector / norm


def local_profiles(path: Path, utterances: list[Utterance], audio_sha256: str) -> dict | None:
    """Derive a profile from windows with strong support for a single transcript speaker."""
    try:
        with np.load(path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata"].item()))
            model = metadata.get("model") if isinstance(metadata, dict) else None
            if (
                not isinstance(model, str)
                or not model
                or metadata.get("schema_version") != 2
                or metadata.get("sample_rate") != SR
                or metadata.get("audio_sha256") != audio_sha256
            ):
                return None
            vectors, starts, ends = cached["embeddings"], cached["starts"], cached["ends"]
            if (
                starts.ndim != 1
                or ends.shape != starts.shape
                or vectors.shape != (len(starts), _EMBEDDING_DIM)
                or starts.dtype.kind not in "iu"
                or ends.dtype.kind not in "iu"
            ):
                return None
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        return None

    # Merge overlapping ASR intervals per speaker so their coverage is counted once.
    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for utterance in sorted(utterances, key=lambda item: item.start):
        speaker = utterance.speaker
        if (
            not isinstance(speaker, str)
            or speaker == "我"
            or not math.isfinite(utterance.start)
            or not math.isfinite(utterance.end)
            or utterance.end <= utterance.start
        ):
            continue
        start, end = round(utterance.start * SR), round(utterance.end * SR)
        items = spans[speaker]
        if items and start <= items[-1][1]:
            items[-1] = (items[-1][0], max(end, items[-1][1]))
        else:
            items.append((start, end))
    span_ends = {speaker: [end for _, end in items] for speaker, items in spans.items()}
    windows: dict[str, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
    for start, end, vector in sorted(zip(starts, ends, vectors), key=lambda row: row[0]):
        start, end = int(start), int(end)
        duration = end - start
        if start < 0 or duration < 1.5 * SR:
            continue
        coverage = {}
        for speaker, items in spans.items():
            covered = 0
            for index in range(bisect_right(span_ends[speaker], start), len(items)):
                a, b = items[index]
                if a >= end:
                    break
                covered += min(b, end) - max(a, start)
            if covered:
                coverage[speaker] = covered
        if not coverage:
            continue
        speaker = max(coverage, key=lambda key: coverage[key])
        own = coverage[speaker]
        if (
            speaker == UNKNOWN_NAME
            or own < 0.8 * duration
            or sum(coverage.values()) - own > 0.1 * duration
        ):
            continue
        try:
            unit = _unit_vector(vector)
        except ValueError:
            continue
        windows[speaker].append((start, end, unit))

    profiles = {}
    for speaker, speaker_windows in windows.items():
        matrix = np.stack([vector for _, _, vector in speaker_windows])
        center = matrix.mean(axis=0)
        norm = float(np.linalg.norm(center))
        if norm <= 1e-9:
            continue
        good = matrix @ (center / norm) >= 0.6
        if np.count_nonzero(good) < 2:
            continue
        total = previous_end = 0
        for (start, end, _), keep in zip(speaker_windows, good, strict=True):
            if keep:
                total += max(0, end - max(start, previous_end))
                previous_end = max(previous_end, end)
        if total >= 3 * SR:
            profiles[speaker] = _unit_vector(matrix[good].mean(axis=0)).tolist()
    return {"model": model, "speakers": profiles} if profiles else None


class VoiceprintMatcher:
    """A snapshot of the registered voices for one embedding model."""

    def __init__(self, people: dict[str, list[float]]):
        self.names = list(people)
        self.registered = np.stack([_unit_vector(people[name]) for name in self.names])

    def match(
        self, profiles: Mapping[_Speaker, object], *, reserved: set[str]
    ) -> dict[_Speaker, str]:
        if not profiles:
            return {}
        labels = list(profiles)
        local = np.stack([_unit_vector(profiles[label]) for label in labels])
        scores = local @ self.registered.T
        matches = {}
        for row, label in enumerate(labels):
            order = np.argsort(scores[row])[::-1]
            best = int(order[0])
            score = float(scores[row, best])
            runner_up = float(scores[row, order[1]]) if len(order) > 1 else -1.0
            competing = max(
                (float(scores[i, best]) for i in range(len(labels)) if i != row), default=-1.0
            )
            if (
                self.names[best] not in reserved
                and score >= _MIN_SIMILARITY
                and score - max(runner_up, competing) >= _MIN_MARGIN
            ):
                matches[label] = self.names[best]
        return matches


class VoiceprintLibrary:
    def __init__(self, directory: Path):
        self.path = directory / "speakers.json"

    def _load(self) -> dict[str, dict[str, list[float]]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("公共声纹库版本无效")
        models = data.get("models")
        if not isinstance(models, dict):
            raise ValueError("公共声纹库格式无效")
        for people in models.values():
            if not isinstance(people, dict) or any(not is_person_name(name) for name in people):
                raise ValueError("公共声纹库姓名格式无效")
            for vector in people.values():
                _unit_vector(vector)
        return models

    def matcher(self, model: str) -> VoiceprintMatcher | None:
        people = self._load().get(model)
        return VoiceprintMatcher(people) if people else None

    def match(
        self, model: str, profiles: Mapping[str, object], *, reserved: set[str]
    ) -> dict[str, str]:
        matcher = self.matcher(model)
        return matcher.match(profiles, reserved=reserved) if matcher is not None else {}

    def enroll(self, model: str, profiles: dict) -> Enrollment:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with SessionLock(
            self.path.parent, name=".speakers.lock", busy_message="公共声纹库正在由另一个任务更新"
        ):
            models = self._load()
            people = models.setdefault(model, {})
            added, existing = [], []
            for name, profile in profiles.items():
                if name in people:
                    existing.append(name)
                else:
                    people[name] = _unit_vector(profile).tolist()
                    added.append(name)
            if added:
                write_json_safely(self.path, {"schema_version": 1, "models": models})
        return Enrollment(added, existing)
