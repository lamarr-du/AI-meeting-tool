"""CPU speaker embeddings, streaming assignments, and stable session speaker IDs."""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path

import numpy as np

from ..asr.base import Utterance
from ..asr.zh_convert import to_simplified
from .align import UNKNOWN_NAME, assign_speakers, speaker_name
from .campplus_onnx import DEFAULT_CLUSTER_THRESHOLD, CampplusOnnxDiarizer
from .clustering import (
    SR,
    StreamingRmsTracker,
    build_cache_signature,
    cluster_embeddings,
    extract_wave_slice,
    frame_active_ratio,
    linear_sum_assignment_max,
    merge_non_overlapping,
    rms,
    save_embedding_cache,
)
from .voiceprints import VoiceprintMatcher


@dataclass
class VoiceTurn:
    segment_id: str
    utterance: Utterance
    display_text: str
    audio: np.ndarray | None
    audio_start: int


@dataclass
class EmbeddingRecord:
    start: int
    end: int
    vector: np.ndarray
    core: bool
    turn_id: str
    speaker: int | None = None

    @property
    def duration(self) -> float:
        return (self.end - self.start) / SR


@dataclass
class _Turn:
    utterance: Utterance
    display_text: str
    records: list[EmbeddingRecord]
    pieces: list[Utterance] = field(default_factory=list)
    provisional: bool = True
    locked: bool = False


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


class ProgressiveDiarizer:
    """Owns mutable speaker state on one worker thread."""

    def __init__(self, cfg: dict, *, model_cache_dir=None):
        self.cfg = cfg["diarization"]
        live = (cfg.get("live") or {}).get("diarization") or {}
        encoder_cfg = dict(self.cfg)
        if threads := encoder_cfg.get("num_threads"):
            encoder_cfg["num_threads"] = min(4, int(threads))
        self.embedder = CampplusOnnxDiarizer(encoder_cfg, model_cache_dir=model_cache_dir)
        self.min_rms = float(self.cfg.get("min_window_rms", 0.003))
        self.window_sec = float(self.cfg.get("window_sec", 2.5))
        self.min_embed_sec = float(live.get("min_embed_sec", 1.5))
        self.assign_similarity = float(live.get("online_assign_sim", 0.58))
        self.new_similarity = float(live.get("online_new_cluster_sim", 0.40))
        self.ema = float(live.get("online_ema_alpha", 0.15))
        self.warmup = deque(
            float(value) for value in live.get("warmup_recluster_sec", [20, 45, 90])
        )
        self.steady = float(live.get("steady_recluster_sec", 45))
        self.max_embeddings = int(live.get("bounded_max_embeddings", 1000))
        self.window_history = float(live.get("bounded_window_sec", 900))
        self.prototype_count = int(live.get("prototypes_per_speaker", 15))
        self.freeze_after = float(live.get("freeze_after_sec", 120))
        self.hysteresis = float(live.get("relabel_hysteresis_margin", 0.08))
        if not (
            0.4 <= self.min_embed_sec <= self.window_sec
            and -1 <= self.new_similarity < self.assign_similarity <= 1
            and 0 < self.ema <= 1
            and list(self.warmup) == sorted(set(self.warmup))
            and all(math.isfinite(value) and value > 0 for value in self.warmup)
            and all(
                math.isfinite(value) and value > 0
                for value in (self.steady, self.window_history, self.freeze_after, self.window_sec)
            )
            and 2 <= self.max_embeddings <= 1100
            and self.prototype_count >= 1
            and 0 <= self.hysteresis <= 2
        ):
            raise ValueError("live.diarization 声纹分离参数无效")
        self.rms_tracker = StreamingRmsTracker(self.min_rms)
        self.records: list[EmbeddingRecord] = []
        self.turns: dict[str, _Turn] = {}
        self.centroids: dict[int, np.ndarray] = {}
        self.names: dict[int, str] = {}
        self.voiceprint_matcher: VoiceprintMatcher | None = None
        self._voiceprint_names: dict[int, str] = {}
        self.aliases: dict[int, int] = {}
        self._next_id = 0
        self._speech_seconds = 0.0
        self._position = 0.0
        self._last_recluster = 0.0
        self._new_core = 0
        self._last_epoch_count = 0
        self._epoch = 0

    def _new_speaker(self, vector: np.ndarray) -> int:
        speaker = self._next_id
        self._next_id += 1
        self.centroids[speaker] = vector
        return speaker

    def _extract(self, job: VoiceTurn) -> list[EmbeddingRecord]:
        if job.audio is None:
            return []
        start = max(job.audio_start, round(job.utterance.start * SR))
        end = min(job.audio_start + len(job.audio), round(job.utterance.end * SR))
        if end <= start:
            return []
        threshold = self.rms_tracker.update(
            job.audio[start - job.audio_start : end - job.audio_start]
        )
        count = math.ceil((end - start) / (self.window_sec * SR))
        edges = [start + round(i * (end - start) / count) for i in range(count + 1)]
        windows = []
        wavs = []
        for a, b in zip(edges, edges[1:]):
            local_a, local_b = a - job.audio_start, b - job.audio_start
            voiced = job.audio[local_a:local_b]
            if len(voiced) < 400 or rms(voiced) < self.min_rms:
                continue
            core = (b - a) / SR >= float(self.cfg.get("min_cluster_dur_sec", 1.2)) and (
                frame_active_ratio(voiced, threshold)
                >= float(self.cfg.get("min_active_ratio", 0.45))
            )
            windows.append((a, b, core))
            wavs.append(extract_wave_slice(job.audio, local_a, local_b))
        if not wavs:
            return []
        vectors = self.embedder.embed_waveforms(wavs)
        return [
            EmbeddingRecord(a, b, vector, core, job.segment_id)
            for (a, b, core), vector in zip(windows, vectors, strict=True)
            if np.isfinite(vector).all() and np.any(vector)
        ]

    def _rough_label(self, turn: _Turn) -> tuple[int | None, bool]:
        core = [item for item in turn.records if item.core]
        duration = sum(item.duration for item in core)
        if duration < self.min_embed_sec:
            if self.turns:
                previous = next(reversed(self.turns.values())).pieces[-1]
                if 0 <= turn.utterance.start - previous.end <= 2 and isinstance(
                    previous.speaker, int
                ):
                    return previous.speaker, True
            return None, True
        vector = _unit(
            np.average(
                [item.vector for item in core], weights=[item.duration for item in core], axis=0
            )
        )
        if not self.centroids:
            return self._new_speaker(vector), True
        speaker, similarity = max(
            ((speaker, float(vector @ center)) for speaker, center in self.centroids.items()),
            key=lambda match: match[1],
        )
        if similarity < self.new_similarity and duration >= 1.8:
            return self._new_speaker(vector), True
        homogeneous = all(
            float(item.vector @ self.centroids[speaker]) >= self.assign_similarity for item in core
        )
        if (
            similarity >= self.assign_similarity
            and homogeneous
            and duration >= 2
            and speaker not in self.names
        ):
            self.centroids[speaker] = _unit(
                (1 - self.ema) * self.centroids[speaker] + self.ema * vector
            )
        return (
            speaker if similarity >= self.new_similarity else None
        ), similarity < self.assign_similarity

    def add(self, job: VoiceTurn) -> list[dict]:
        records = self._extract(job)
        turn = _Turn(job.utterance, job.display_text, records)
        speaker, turn.provisional = self._rough_label(turn)
        for item in records:
            item.speaker = speaker
        turn.pieces = [replace(job.utterance, speaker=speaker)]
        self.turns[job.segment_id] = turn
        self.records.extend(records)
        self._position = max(self._position, job.utterance.end)
        self._speech_seconds += sum(item.duration for item in records)
        self._new_core += sum(item.core for item in records)
        update = self._update(job.segment_id, turn)
        events = [{"type": "diarization.assign", **update}]
        due = bool(self.warmup and self._speech_seconds >= self.warmup[0])
        while self.warmup and self._speech_seconds >= self.warmup[0]:
            self.warmup.popleft()
        if due or (
            not self.warmup
            and (self._position - self._last_recluster >= self.steady or self._new_core >= 30)
        ):
            if event := self.reconcile():
                events.append(event)
        return events

    def _cluster_input(self) -> list[EmbeddingRecord]:
        if len(self.records) <= self.max_embeddings:
            return self.records
        cutoff = (self._position - self.window_history) * SR
        recent = [record for record in self.records if record.end >= cutoff]
        prototypes: dict[int, list[EmbeddingRecord]] = defaultdict(list)
        for record in self.records:
            if record.end < cutoff and record.core and record.speaker in self.centroids:
                assert record.speaker is not None
                prototypes[record.speaker].append(record)
        for speaker, records in prototypes.items():
            center = self.centroids[speaker]
            records.sort(key=lambda item: float(item.vector @ center), reverse=True)
            del records[self.prototype_count :]
        selected = [
            records[index]
            for index in range(self.prototype_count)
            for records in prototypes.values()
            if index < len(records)
        ][: self.max_embeddings // 4]
        budget = self.max_embeddings - len(selected)
        return selected + recent[-budget:]

    def _match(self, records, labels, centers) -> dict[int, int]:
        ids = list(self.centroids)
        count = len(centers)
        if not ids:
            return {index: self._new_speaker(center) for index, center in enumerate(centers)}
        similarity = np.stack([self.centroids[speaker] for speaker in ids]) @ centers.T
        overlap = np.zeros_like(similarity)
        indices = {speaker: row for row, speaker in enumerate(ids)}
        for item, label in zip(records, labels, strict=True):
            if item.speaker in indices:
                overlap[indices[item.speaker], label] += item.duration
        # Zero weights leave identities unmatched. A renamed identity can win a
        # close match, while pinned utterances remain locked even on a cluster collision.
        weights = 0.65 * overlap + 0.35 * np.maximum(similarity, 0)
        weights[(overlap == 0) & (similarity < self.assign_similarity)] = 0
        for row, speaker in enumerate(ids):
            if speaker in self.names:
                weights[row] *= 4
        old, new = linear_sum_assignment_max(weights)
        result = {
            int(col): ids[row]
            for row, col in zip(old, new, strict=True)
            if col < count and weights[row, col] > 0
        }
        for index, center in enumerate(centers):
            if index not in result:
                result[index] = self._new_speaker(center)
        return result

    def reconcile(self) -> dict | None:
        self._last_recluster = self._position
        self._new_core = 0
        if self._last_epoch_count == len(self.records):
            return None
        self._last_epoch_count = len(self.records)
        records = self._cluster_input()
        if not any(item.core for item in records):
            return None
        vectors = np.stack([item.vector for item in records])
        durations = np.array([item.duration for item in records])
        core = np.array([item.core for item in records])
        raw_labels = cluster_embeddings(
            vectors,
            self.cfg,
            default_threshold=DEFAULT_CLUSTER_THRESHOLD,
            core_mask=core,
            durations=durations,
        )
        _, labels = np.unique(raw_labels, return_inverse=True)
        centers = np.stack(
            [_unit(vectors[labels == label].mean(axis=0)) for label in np.unique(labels)]
        )
        matching = self._match(records, labels, centers)
        candidates = dict(
            zip(matching.values(), (centers[label] for label in matching), strict=True)
        )
        touched: set[str] = set()
        merges = self._merge_aliases(records, labels, matching, candidates, touched)
        for record, label in zip(records, labels, strict=True):
            turn = self.turns[record.turn_id]
            frozen = self._position - turn.utterance.end > self.freeze_after
            if turn.locked or frozen:
                continue
            old = record.speaker
            proposed = matching[int(label)]
            new_center = candidates[proposed]
            if old is not None and old != proposed and not turn.provisional:
                if float(record.vector @ (new_center - self.centroids[old])) < self.hysteresis:
                    continue
            record.speaker = proposed
            touched.add(record.turn_id)
        for speaker, center in candidates.items():
            if speaker not in self.names:
                self.centroids[speaker] = center
        updates = []
        for segment_id in self.turns:
            if segment_id not in touched:
                continue
            turn = self.turns[segment_id]
            before = self._update(segment_id, turn)
            self._assign_pieces(turn)
            updated = self._update(segment_id, turn)
            if updated != before or merges:
                updates.append(updated)
        if not updates and not merges:
            return None
        self._epoch += 1
        return {
            "type": "diarization.reconcile",
            "epoch": self._epoch,
            "updates": updates,
            "merges": merges,
        }

    def _merge_aliases(self, records, labels, matching, candidates, touched) -> list[dict]:
        unmatched: dict[int, list[tuple[EmbeddingRecord, int]]] = defaultdict(list)
        for record, label in zip(records, labels, strict=True):
            if record.speaker is not None and record.speaker not in candidates and record.core:
                unmatched[record.speaker].append((record, matching[int(label)]))
        merges: dict[int, int] = {}
        for old, items in unmatched.items():
            targets = {target for _, target in items}
            if old in self.names or len(targets) != 1 or len(items) < 2:
                continue
            target = next(iter(targets))
            if sum(item.duration for item, _ in items) >= 4 and all(
                float(item.vector @ candidates[target]) >= self.assign_similarity
                for item, _ in items
            ):
                merges[old] = target
        if merges:
            for record in self.records:
                if record.speaker in merges:
                    record.speaker = merges[record.speaker]
                    touched.add(record.turn_id)
            for segment_id, turn in self.turns.items():
                for piece in turn.pieces:
                    if isinstance(piece.speaker, int) and piece.speaker in merges:
                        piece.speaker = merges[piece.speaker]
                        touched.add(segment_id)
            for old, target in merges.items():
                self.centroids.pop(old)
                self.aliases[old] = target
        return [{"from_id": old, "to_id": target} for old, target in merges.items()]

    def _assign_pieces(self, turn: _Turn) -> None:
        if turn.locked:
            return
        items = [item for item in turn.records if item.speaker is not None]
        if not items:
            return
        spans = merge_non_overlapping(
            [item.start for item in items],
            [item.end for item in items],
            np.array([item.speaker for item in items], dtype=int),
            merge_gap_sec=0.3,
        )
        turn.pieces = assign_speakers([turn.utterance], spans, min_group_sec=1.0)
        turn.provisional = not any(item.core for item in items)

    def _update(self, segment_id: str, turn: _Turn) -> dict:
        return {
            "segment_id": segment_id,
            "parts": [
                {
                    "start_ms": round(piece.start * 1000),
                    "speaker_id": piece.speaker,
                    "display_name": (
                        self.display_name(piece.speaker)
                        if isinstance(piece.speaker, int)
                        else "说话人待定"
                    ),
                    "provisional": turn.provisional and not turn.locked,
                    "text": turn.display_text
                    if len(turn.pieces) == 1
                    else to_simplified(piece.text),
                }
                for piece in turn.pieces
            ],
        }

    def display_name(self, speaker) -> str:
        if not isinstance(speaker, int):
            return UNKNOWN_NAME
        return self.names.get(speaker) or self._voiceprint_names.get(speaker, speaker_name(speaker))

    @cached_property
    def model_signature(self) -> str:
        return self.embedder._model_signature()

    @property
    def speaker_names(self) -> dict[int, str]:
        return {**self._voiceprint_names, **self.names}

    def match_voiceprints(self) -> list[dict]:
        matches = (
            self.voiceprint_matcher.match(
                {
                    speaker: center
                    for speaker, center in self.centroids.items()
                    if speaker not in self.names and np.any(center)
                },
                reserved=set(self.names.values()),
            )
            if self.voiceprint_matcher is not None
            else {}
        )
        previous = self._voiceprint_names
        self._voiceprint_names = matches
        return [
            {
                "type": "speaker.rename",
                "speaker_id": speaker,
                "display_name": self.display_name(speaker),
                "pin_anchor": False,
            }
            for speaker in sorted(previous.keys() | matches.keys())
            if speaker not in self.names and previous.get(speaker) != matches.get(speaker)
        ]

    def rename(self, speaker: int, name: str) -> dict:
        while speaker in self.aliases:
            speaker = self.aliases[speaker]
        name = name.strip()
        if speaker not in self.centroids:
            raise ValueError("该说话人已不在当前录音中")
        if (
            not name
            or len(name) > 60
            or not name.isprintable()
            or name in ("我", UNKNOWN_NAME, "说话人待定")
            or (re.fullmatch(r"说话人(?:[A-Z]|\d+)", name) and name != speaker_name(speaker))
            or any(other != speaker and used == name for other, used in self.names.items())
        ):
            raise ValueError("请输入不重复的说话人姓名，最多 60 字")
        self.names[speaker] = name
        for turn in self.turns.values():
            if any(piece.speaker == speaker for piece in turn.pieces):
                turn.locked = True
        return {
            "type": "speaker.rename",
            "speaker_id": speaker,
            "display_name": name,
            "pin_anchor": True,
        }

    def archive(self, replacements: dict[str, Utterance] | None = None) -> list[Utterance]:
        result: list[Utterance] = []
        for segment_id, turn in self.turns.items():
            if replacements and segment_id in replacements:
                utterance = replacements[segment_id]
                turn = replace(
                    turn,
                    utterance=utterance,
                    pieces=[replace(utterance, speaker=turn.pieces[0].speaker)],
                )
                self._assign_pieces(turn)
            result.extend(
                replace(piece, speaker=self.display_name(piece.speaker)) for piece in turn.pieces
            )
        return result

    def save_cache(self, path: Path, audio_sha256: str) -> None:
        if self.records:
            signature = build_cache_signature(
                model=self.model_signature,
                mode="live_commit",
                win_sec=self.window_sec,
                hop_sec=0.0,
                merge_gap_sec=0.0,
                min_rms=self.min_rms,
                speech_spans=None,
                audio_sha256=audio_sha256,
            )
            save_embedding_cache(
                path,
                np.stack([record.vector for record in self.records]),
                [record.start for record in self.records],
                [record.end for record in self.records],
                signature,
            )
