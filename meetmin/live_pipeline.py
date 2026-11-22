"""实时转写会话，录音写盘与本地模型推理解耦。"""

from __future__ import annotations

import json
import math
import os
import wave
from collections.abc import Callable
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic
from typing import Any, TextIO

from .asr.base import ASRResult, Utterance
from .asr.faster_whisper_asr import FasterWhisperASR
from .asr.live_whisper import LiveTrack
from .asr.zh_convert import to_simplified
from .audio.ring_buffer import SAMPLE_RATE, AudioEvent, PcmChunk
from .config import deep_merge
from .pipeline import Pipeline
from .session import Session

LiveEventCallback = Callable[[dict[str, Any]], None]


class LiveSessionCoordinator:
    """采集线程仅投递事件。后台线程持有模型、VAD 状态和定稿日志。"""

    def __init__(self, cfg: dict, session: Session, emit: LiveEventCallback, *, dual_track: bool):
        live = cfg.get("live") or {}
        self.hop = float(live.get("hop_sec", 1.8))
        self.slow_hop = float(live.get("backpressure_hop_sec", 4))
        self.max_window = float(live.get("max_window_sec", 15))
        self.ring_sec = float(live.get("ring_buffer_sec", 60))
        self.endpoint_ms = int(live.get("endpoint_silence_ms", 600))
        self.flush_timeout = float(live.get("flush_timeout_sec", 12))
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
                        "compute_type": None,
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
        self._cancelled = Event()
        self._overflow = Event()
        self._thread = Thread(target=self._run, name="live-whisper", daemon=True)
        self._tracks: dict[str, LiveTrack] = {}
        self._utterances: dict[str, list[Utterance]] = {name: [] for name in self.names}
        self._drafts = dict.fromkeys(self.names, "")
        self._metadata: dict = {}
        self.error: str | None = None

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
                    vad=lambda audio: get_speech_timestamps(
                        audio,
                        min_speech_duration_ms=64,
                        min_silence_duration_ms=160,
                        speech_pad_ms=0,
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
            with (self.session.root / "events.jsonl").open("x", encoding="utf-8") as log:
                self._consume(provider, options, log)
            self._metadata = provider.stream_metadata()
        except Exception as exc:
            self.error = str(exc)
            self._status(f"实时转写已暂停，录音继续保存 · {exc}")
        finally:
            self.cancel()
            if provider is not None:
                provider.close()

    def _consume(self, provider: FasterWhisperASR, options: dict, log: TextIO) -> None:
        punctuator = self.pipe._resolve_punctuator()
        decode_seconds = 0.0
        closing = False
        while not self._cancelled.is_set():
            for _ in range(128):
                try:
                    name, event = self._inbox.get_nowait()
                except Empty:
                    break
                self._tracks[name].feed(event)
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
                self._tracks[name].feed(event)
                continue
            name, window = min(windows, key=lambda item: (not item[1].final, item[1].due))
            started = monotonic()
            utterance, draft = self._tracks[name].decode(
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
            if utterance is not None:
                self._commit(log, name, utterance, punctuator)
            if draft != self._drafts[name]:
                self._drafts[name] = draft
                self.emit({"type": "asr.interim", "track": name, "text": to_simplified(draft)})

    def _commit(self, log: TextIO, name: str, utterance: Utterance, punctuator: Any) -> None:
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
        event = {
            "type": "asr.commit",
            "track": name,
            "segment_id": f"{name}-{len(self._utterances[name]) + 1}",
            "start_ms": round(utterance.start * 1000),
            "end_ms": round(utterance.end * 1000),
            "text": " ".join(row["text"] for row in display_document["utterances"]),
            "utterance": utterance.to_dict(),
        }
        log.write(json.dumps(event, ensure_ascii=False) + "\n")
        log.flush()
        os.fsync(log.fileno())
        self._utterances[name].append(utterance)
        if not self._cancelled.is_set():
            self.emit(event)

    def finish(self) -> dict | None:
        """录音成功封存后冲刷尾窗，再交给标准离线产物整理。"""
        self._input_done.set()
        self._thread.join(self.flush_timeout)
        if self._thread.is_alive():
            self.error = "实时转写收尾超时"
            self.cancel()
        if self.error is not None:
            return None
        try:
            audio, microphone = self.session.recording_paths()
            loopback_result = self._track_result("loopback", audio)
            mic_result = self._track_result("microphone", microphone) if microphone else None
            self.pipe._save_raw_asr(self.session, "faster_whisper", loopback_result, mic_result)
            return self.pipe.diarize(self.session)
        except Exception as exc:
            self.error = str(exc)
            return None

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
        return path, result, frames / SAMPLE_RATE
