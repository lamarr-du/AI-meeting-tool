"""实时转写会话，录音写盘与本地模型推理解耦。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import wave
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from queue import Empty, Full, Queue, SimpleQueue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any

import numpy as np

from .asr.base import ASRResult, Utterance
from .asr.faster_whisper_asr import FasterWhisperASR
from .asr.live_whisper import LiveTrack, ReviewSpan, review_redecode
from .asr.quality import LIVE_REVIEW_TEXT, QualityOptions
from .asr.zh_convert import to_simplified
from .audio.dsp import highpass_speech
from .audio.ring_buffer import SAMPLE_RATE, AudioEvent, PcmChunk
from .config import deep_merge
from .diarization.align import UNKNOWN_NAME, speaker_name
from .diarization.live_diarizer import ProgressiveDiarizer, VoiceTurn
from .pipeline import Pipeline
from .session import Session

LiveEventCallback = Callable[[dict[str, Any]], None]
SpeakerRenames = SimpleQueue[tuple[int, str]]


class LiveSessionCoordinator:
    """采集线程仅投递事件。后台线程持有模型、VAD 状态和定稿日志。"""

    def __init__(
        self,
        cfg: dict,
        session: Session,
        emit: LiveEventCallback,
        *,
        dual_track: bool,
        renames: SpeakerRenames | None = None,
    ):
        live = cfg.get("live") or {}
        self.hop = float(live.get("hop_sec", 1.8))
        self.slow_hop = float(live.get("backpressure_hop_sec", 4))
        self.max_window = float(live.get("max_window_sec", 15))
        self.ring_sec = float(live.get("ring_buffer_sec", 60))
        self.endpoint_ms = int(live.get("endpoint_silence_ms", 600))
        self.flush_timeout = float(live.get("flush_timeout_sec", 12))
        self.repair_max_clips = live.get("repair_max_clips", 6)
        if type(self.repair_max_clips) is not int or self.repair_max_clips < 0:
            raise ValueError("live.repair_max_clips 必须是非负整数")
        if not (
            0.3 <= self.hop <= self.slow_hop <= self.max_window <= 20
            and self.max_window + 2 <= self.ring_sec <= 300
            and 200 <= self.endpoint_ms <= 2000
            and math.isfinite(self.flush_timeout)
            and self.flush_timeout > 0
        ):
            raise ValueError("live 实时转写的步长、缓冲时长或收尾时限配置无效")

        offline_diarizer = str(
            (cfg.get("diarization") or {}).get("provider", "campplus_onnx")
        ).lower()
        diarizer = (
            offline_diarizer
            if offline_diarizer
            in (
                "campplus_onnx",
                "campplus",
                "onnx",
                "3dspeaker",
            )
            else "single"
        )
        self.pipe = Pipeline(
            deep_merge(
                cfg,
                {
                    "asr": {
                        "provider": "faster_whisper",
                        "profile": "auto",
                        "device": live.get("device", "auto"),
                        "model_size": live.get("model_size") or "large-v3-turbo",
                        "local_files_only": True,
                        "language": (cfg.get("asr") or {}).get("language") or "zh",
                        "beam_size": int(live.get("beam_size", 1)),
                        "word_timestamps": True,
                        "condition_on_previous_text": False,
                    },
                    "diarization": {"provider": diarizer, "local_files_only": True},
                    "punctuation": {"local_files_only": True},
                },
            ),
            progress=lambda message: self._status(message),
        )
        self.session = session
        self.emit = emit
        self.names = ("loopback", "microphone") if dual_track else ("loopback",)
        self._inbox: Queue[tuple[str, AudioEvent]] = Queue(maxsize=2048)
        self._input_done = Event()
        self._live_done = Event()
        self._cancelled = Event()
        self._overflow = Event()
        self._thread = Thread(target=self._run, name="live-whisper", daemon=True)
        self._tracks: dict[str, LiveTrack] = {}
        self._utterances: dict[str, list[Utterance]] = {name: [] for name in self.names}
        self._quality = QualityOptions.from_config(self.pipe.cfg["asr"].get("quality", {}))
        self._deferred: dict[tuple[str, int], ReviewSpan] = {}
        self._repair_reports: dict[str, list[dict]] = {name: [] for name in self.names}
        self._replacements: dict[str, Utterance] = {}
        self._drafts = dict.fromkeys(self.names, "")
        self._metadata: dict = {}
        self.error: str | None = None
        voice = live.get("diarization") or {}
        self._voice_enabled = (
            bool(voice.get("enabled", True))
            and diarizer != "single"
            and bool(self.pipe.cfg["diarization"].get("enabled", True))
        )
        self._speaker_renames = renames if renames is not None else SimpleQueue()
        self._voice_inbox: Queue[VoiceTurn] = Queue(maxsize=64)
        self._voice_done = Event()
        self._voice_cancelled = Event()
        self._voice_thread: Thread | None = None
        self._voice: ProgressiveDiarizer | None = None
        self._voice_warning: str | None = None
        self._events_lock = Lock()
        self._events_file = session.root / "events.jsonl"
        self._audio_hash = hashlib.sha256()
        bleed = live.get("bleed_suppression") or {}
        self._bleed_enabled = dual_track and bool(bleed.get("enabled", True))
        self._bleed_window = float(bleed.get("window_ms", 600)) / 1000
        self._bleed_similarity = float(bleed.get("min_bigram_similarity", 0.75))
        if not (0 < self._bleed_window <= 2 and 0 < self._bleed_similarity <= 1):
            raise ValueError("live.bleed_suppression 串音过滤参数无效")
        self._recent = {name: deque[tuple[int, float, set[str]]](maxlen=64) for name in self.names}
        self._suppressed: set[int] = set()

    def start(self) -> None:
        self._thread.start()

    def feed(self, track: str, event: AudioEvent) -> None:
        if self._cancelled.is_set() or self._overflow.is_set():
            return
        try:
            self._inbox.put_nowait((track, event))
        except Full:
            if isinstance(event, PcmChunk):
                self._overflow.set()

    def cancel(self) -> None:
        self._cancelled.set()
        self._voice_cancelled.set()

    def close(self) -> None:
        """Wait for the model owner and the voice worker after cancelling input."""
        self.cancel()
        self._thread.join()
        if self._voice_thread is not None:
            self._voice_thread.join()

    def _status(self, message: str) -> None:
        if not self._cancelled.is_set():
            self.emit({"type": "live.status", "message": message})

    def _run(self) -> None:
        provider = None
        try:
            from faster_whisper.vad import get_speech_timestamps

            self._status("正在加载本地实时转写模型…")
            self._tracks = {
                name: LiveTrack(
                    ring_sec=self.ring_sec,
                    endpoint_ms=self.endpoint_ms,
                    max_window_sec=self.max_window,
                    compression_ratio_threshold=self._quality.compression_ratio_threshold,
                    vad=lambda audio: get_speech_timestamps(
                        highpass_speech(audio, SAMPLE_RATE),
                        threshold=0.55,
                        min_speech_duration_ms=180,
                        min_silence_duration_ms=200,
                        speech_pad_ms=40,
                    ),
                )
                for name in self.names
            }
            provider = FasterWhisperASR(
                self.pipe.cfg["asr"], model_cache_dir=self.pipe.cfg.get("model_cache_dir")
            )
            options = provider.build_stream_options()
            self._metadata = provider.stream_metadata()
            self._status(f"实时转写 · {self._metadata['effective_device'].upper()}")
            self._events_file.touch(exist_ok=False)
            if self._voice_enabled and not self._cancelled.is_set():
                self._voice_thread = Thread(
                    target=self._run_voice, name="live-diar-cpu", daemon=True
                )
                self._voice_thread.start()
            self._consume(provider, options)
            self._metadata = provider.stream_metadata()
            self._live_done.set()
            if self._deferred and not self._cancelled.is_set():
                audio, microphone = self.session.recording_paths()
                self._replacements = self._repair_deferred(
                    provider, options, {"loopback": audio, "microphone": microphone}
                )
        except Exception as exc:
            self.error = str(exc)
            self._status(f"实时转写已暂停，录音继续保存 · {exc}")
            self.cancel()
        finally:
            try:
                if provider is not None:
                    provider.close()
            finally:
                self._live_done.set()

    def _consume(self, provider: FasterWhisperASR, options: dict) -> None:
        punctuator = self.pipe._resolve_punctuator()
        decode_seconds = 0.0
        closing = False
        while not self._cancelled.is_set():
            for _ in range(128):
                try:
                    name, event = self._inbox.get_nowait()
                except Empty:
                    break
                self._feed_track(name, event)
            if self._overflow.is_set():
                raise RuntimeError("实时音频已积压，保存的 WAV 可用于重新转写")
            if not closing and self._input_done.is_set() and self._inbox.empty():
                for track in self._tracks.values():
                    track.close()
                closing = True

            active = sum(track.has_speech for track in self._tracks.values())
            hop_seconds = min(self.slow_hop, max(self.hop, decode_seconds * active * 1.2))
            hop = round(hop_seconds * SAMPLE_RATE)
            windows = [
                (name, window)
                for name, track in self._tracks.items()
                if (window := track.next_window(hop)) is not None
            ]
            if not windows:
                if closing:
                    break
                try:
                    name, event = self._inbox.get(timeout=0.05)
                except Empty:
                    continue
                self._feed_track(name, event)
                continue
            name, window = min(windows, key=lambda item: (not item[1].final, item[1].due))
            started = monotonic()
            result, draft = self._tracks[name].decode(
                window, lambda audio, offset: provider.decode_window(audio, offset, options=options)
            )
            if self._cancelled.is_set():
                return
            elapsed = monotonic() - started
            if elapsed > 0.1:  # Finalizing an already decoded draft can bypass inference.
                decode_seconds = 0.7 * decode_seconds + 0.3 * elapsed if decode_seconds else elapsed
            device = provider.stream_metadata()["effective_device"]
            if device != self._metadata["effective_device"]:
                self._metadata = provider.stream_metadata()
                self._status(f"实时转写已切换至 {device.upper()}")
            if isinstance(result, ReviewSpan):
                self._deferred[name, len(self._utterances[name])] = result
                self._commit(
                    name,
                    Utterance(result.start, result.end, LIVE_REVIEW_TEXT),
                    punctuator,
                    review=result,
                )
            elif result is not None:
                self._commit(name, result, punctuator)
            if draft != self._drafts[name]:
                self._drafts[name] = draft
                self.emit({"type": "asr.interim", "track": name, "text": to_simplified(draft)})

    def _feed_track(self, name: str, event: AudioEvent) -> None:
        self._tracks[name].feed(event)
        if self._voice_enabled and name == "loopback" and isinstance(event, PcmChunk):
            pcm = np.frombuffer(event.pcm, dtype="<i2").astype("<f4") / 32768.0
            self._audio_hash.update(pcm)

    def _commit(
        self, name: str, utterance: Utterance, punctuator: Any, *, review: ReviewSpan | None = None
    ) -> None:
        if name == "microphone":
            utterance.speaker = "我"
        display = Utterance.from_dict(utterance.to_dict())
        display.words = [dict(word) for word in utterance.words or []]
        formatted = self.pipe._format_utterances(
            [display], self.pipe.cfg["asr"]["language"], punctuator=punctuator
        )
        display_document = self.pipe._glossary_correction(
            {"utterances": [segment.to_dict() for segment in formatted]}
        )
        index = len(self._utterances[name])
        segment_id = f"{name}-{index + 1}"
        matched_microphones = [] if review else self._bleed_matches(name, index, utterance)
        self._suppressed.update(matched_microphones)
        preview = " ".join(row["text"] for row in display_document["utterances"])
        event = {
            "type": "asr.commit",
            "track": name,
            "segment_id": segment_id,
            "start_ms": round(utterance.start * 1000),
            "end_ms": round(utterance.end * 1000),
            "text": preview,
            "utterance": utterance.to_dict(),
            "suppressed_bleed": name == "microphone" and index in self._suppressed,
            "speaker_pending": name == "loopback" and self._voice_enabled,
        }
        if review:
            event["quality"] = {"status": "pending", "reasons": list(review.reasons)}
        self._publish_event(event)
        self._utterances[name].append(utterance)
        if name == "loopback":
            for microphone in matched_microphones:
                self._publish_event(
                    {
                        "type": "asr.suppress_bleed",
                        "segment_id": f"microphone-{microphone + 1}",
                    }
                )
            self._queue_voice(segment_id, utterance, preview)

    def _bleed_matches(
        self,
        track: str,
        index: int,
        utterance: Utterance,
        *,
        repaired: set[str] | None = None,
    ) -> list[int]:
        if not self._bleed_enabled:
            return []
        text = "".join(char for char in to_simplified(utterance.text).casefold() if char.isalnum())
        if len(text) < 8:
            return []
        grams = {a + b for a, b in zip(text, text[1:])}
        if len(grams) < 4:
            return []
        self._recent[track].append((index, utterance.start, grams))
        other = "microphone" if track == "loopback" else "loopback"
        matches = []
        for prior_index, start, prior_grams in self._recent[other]:
            if repaired is not None and not (
                f"{track}-{index + 1}" in repaired or f"{other}-{prior_index + 1}" in repaired
            ):
                continue
            delay = utterance.start - start if track == "microphone" else start - utterance.start
            mic_index = index if track == "microphone" else prior_index
            if (
                0 <= delay <= self._bleed_window
                and mic_index not in self._suppressed
                and len(grams & prior_grams) / len(grams | prior_grams) >= self._bleed_similarity
            ):
                matches.append(mic_index)
        return list(dict.fromkeys(matches))

    def _queue_voice(self, segment_id: str, utterance: Utterance, text: str) -> None:
        if self._voice_thread is None or self._voice_cancelled.is_set():
            return
        ring = self._tracks["loopback"].audio
        start = max(ring.start, round(utterance.start * SAMPLE_RATE) - SAMPLE_RATE // 5)
        end = min(ring.end, round(utterance.end * SAMPLE_RATE) + SAMPLE_RATE // 5)
        available = start <= round(utterance.start * SAMPLE_RATE) and start < end
        job = VoiceTurn(
            segment_id, utterance, text, ring.read(start, end) if available else None, start
        )
        try:
            self._voice_inbox.put_nowait(job)
        except Full:
            if self._voice_warning is None:
                self._voice_warning = "声纹队列积压，部分发言暂未分配说话人"
                self._status(self._voice_warning)

    def _publish_event(self, event: dict) -> None:
        with self._events_lock:
            if self._cancelled.is_set():
                return
            with self._events_file.open("a", encoding="utf-8") as log:
                log.write(json.dumps(event, ensure_ascii=False) + "\n")
                log.flush()
                os.fsync(log.fileno())
            self.emit(event)

    def _run_voice(self) -> None:
        try:
            voice = ProgressiveDiarizer(
                self.pipe.cfg, model_cache_dir=self.pipe.cfg.get("model_cache_dir")
            )
            self._voice = voice
            try:
                voice.voiceprint_matcher = self.pipe.voiceprint_library.matcher(
                    voice.model_signature
                )
            except (OSError, ValueError, TypeError, OverflowError) as exc:
                self._status(f"无法读取公共声纹库，继续录中说话人分离 · {exc}")
            while not self._voice_cancelled.is_set():
                while not self._speaker_renames.empty():
                    speaker, name = self._speaker_renames.get_nowait()
                    try:
                        self._publish_event(voice.rename(speaker, name))
                    except ValueError as exc:
                        self._status(str(exc))
                    else:
                        self._publish_voiceprint_matches(voice)
                try:
                    job = self._voice_inbox.get(timeout=0.05)
                except Empty:
                    if self._voice_done.is_set() and self._speaker_renames.empty():
                        if event := voice.reconcile():
                            self._publish_event(event)
                        self._publish_voiceprint_matches(voice)
                        return
                    continue
                for event in voice.add(job):
                    if not self._voice_cancelled.is_set():
                        self._publish_event(event)
                self._publish_voiceprint_matches(voice)
        except Exception as exc:
            self._voice_warning = f"录中说话人分离已暂停 · {exc}"
            self._status(self._voice_warning)
        finally:
            self._voice_cancelled.set()

    def _publish_voiceprint_matches(self, voice: ProgressiveDiarizer) -> None:
        if self._voice_cancelled.is_set():
            return
        try:
            matches = voice.match_voiceprints()
        except (ValueError, TypeError, OverflowError) as exc:
            self._status(f"无法匹配公共声纹，继续录中说话人分离 · {exc}")
            voice.voiceprint_matcher = None
            matches = voice.match_voiceprints()
        for event in matches:
            if self._voice_cancelled.is_set():
                break
            self._publish_event(event)

    def finish(self) -> dict | None:
        """录音成功封存后冲刷尾窗，再交给标准离线产物整理。"""
        self._input_done.set()
        deadline = monotonic() + self.flush_timeout
        if not self._live_done.wait(self.flush_timeout):
            self.error = "实时转写收尾超时"
            self.cancel()
        if self.error is not None:
            return None
        self._voice_done.set()
        voice = None
        if self._voice_thread is not None:
            self._voice_thread.join(max(0, deadline - monotonic()))
            if self._voice_thread.is_alive():
                self._voice_cancelled.set()
                self._status("录中声纹收尾超时，正在使用会后分离")
            else:
                voice = self._voice
        self._thread.join()  # 录后补转独立于实时尾窗时限，继续由同一个 CUDA 工作线程执行。
        if self.error is not None:
            return None
        try:
            audio, microphone = self.session.recording_paths()
            loopback_result = self._track_result("loopback", audio)
            mic_result = self._track_result("microphone", microphone) if microphone else None
            raw = self.pipe._save_raw_asr(
                self.session, "faster_whisper", loopback_result, mic_result
            )
            if voice is None or (self._voice_warning and not voice.turns):
                return self.pipe.diarize(self.session, suppressed_microphone=self._suppressed)
            audio_sha256: str | None
            audio_sha256 = self._audio_hash.hexdigest()
            try:
                voice.save_cache(self.session.root / "embeddings.npz", audio_sha256)
            except OSError as exc:
                audio_sha256 = None
                self._status(f"无法保存录中声纹缓存 · {exc}")
            utterances = voice.archive(self._replacements)
            utterances.extend(
                replace(utterance, speaker=UNKNOWN_NAME)
                for index, utterance in enumerate(self._utterances["loopback"], 1)
                if f"loopback-{index}" not in voice.turns
            )
            utterances.sort(key=lambda utterance: (utterance.start, utterance.end))
            speakers = list(
                dict.fromkeys(
                    entry.speaker
                    for entry in utterances
                    if isinstance(entry.speaker, str) and entry.speaker != UNKNOWN_NAME
                )
            )
            return self.pipe._publish_aligned_transcript(
                self.session,
                raw,
                utterances,
                speakers,
                suppressed_microphone=self._suppressed,
                speaker_names={
                    speaker_name(key): name for key, name in voice.speaker_names.items()
                },
                voiceprint_audio_sha256=audio_sha256,
            )
        except Exception as exc:
            self.error = str(exc)
            return None

    def _repair_deferred(
        self, provider: FasterWhisperASR, options: dict, paths: dict[str, Path | None]
    ) -> dict[str, Utterance]:
        replacements: dict[str, Utterance] = {}
        if self.repair_max_clips:
            self._status("正在补转可疑音频片段…")
        options = {
            **options,
            "no_repeat_ngram_size": max(options.get("no_repeat_ngram_size", 0), 4),
        }
        for attempt, ((name, index), span) in enumerate(self._deferred.items()):
            if self._cancelled.is_set():
                break
            segment_id = f"{name}-{index + 1}"
            report = {
                "segment_id": segment_id,
                "start": round(span.start, 3),
                "end": round(span.end, 3),
                "reasons": list(span.reasons),
                "status": "pending",
            }
            path = paths[name]
            if attempt >= self.repair_max_clips:
                report["detail"] = "retry_limit"
            elif path is not None:
                started = monotonic()
                try:
                    audio, offset = _read_repair_clip(path, span)
                    decoded = provider.decode_window(audio, offset, options=options)
                    result, reason = review_redecode(
                        span,
                        decoded,
                        self._quality,
                        language=self.pipe.cfg["asr"].get("language"),
                    )
                    if result is None:
                        report["detail"] = reason
                    else:
                        result.speaker = self._utterances[name][index].speaker
                        self._utterances[name][index] = replacements[segment_id] = result
                        report["status"] = "accepted"
                except Exception as exc:
                    report["detail"] = f"{type(exc).__name__}: {exc}"
                report["elapsed_sec"] = round(monotonic() - started, 3)
            else:
                report["detail"] = "audio_unavailable"
            self._repair_reports[name].append(report)
            self._publish_event({"type": "asr.repair", "track": name, **report})

        if replacements and self._bleed_enabled:
            # 对补转段补做串音比对，保留录中已确认的结果和原索引。
            repaired = set(replacements)
            for recent in self._recent.values():
                recent.clear()
            turns = sorted(
                (u.start, name, index)
                for name, rows in self._utterances.items()
                for index, u in enumerate(rows)
                if u.text != LIVE_REVIEW_TEXT
            )
            for _, name, index in turns:
                self._suppressed.update(
                    self._bleed_matches(
                        name, index, self._utterances[name][index], repaired=repaired
                    )
                )
        return replacements

    def _track_result(self, name: str, path: Path) -> tuple[Path, ASRResult, float]:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
        if frames != self._tracks[name].audio.end:
            raise RuntimeError(f"{name} 的实时样本数与已保存 WAV 不一致")
        asr = self.pipe.cfg["asr"]
        result = ASRResult(
            utterances=self._utterances[name],
            language=asr["language"],
            extra={**self._metadata, "live": True},
        )
        if self._repair_reports[name]:
            result.extra["live_quality"] = {"issues": self._repair_reports[name]}
        return path, result, frames / SAMPLE_RATE


def _read_repair_clip(path: Path, span: ReviewSpan) -> tuple[np.ndarray, float]:
    context = round(0.75 * SAMPLE_RATE)
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError("补转需要 16 kHz、16 位单声道 WAV")
        start = max(0, round(span.start * SAMPLE_RATE) - context)
        end = min(wav.getnframes(), round(span.end * SAMPLE_RATE) + context)
        wav.setpos(start)
        audio = np.frombuffer(wav.readframes(end - start), dtype="<i2").astype(np.float32)
    return audio / 32768.0, start / SAMPLE_RATE
