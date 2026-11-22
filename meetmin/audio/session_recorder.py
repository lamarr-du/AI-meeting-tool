"""同一会话的录音与恢复，双轨只有在两个 WAV 就绪后才发布。"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path

from ..session import Session
from .loopback_recorder import LoopbackRecorder, RecordingError
from .ring_buffer import AudioEvent
from .segments import SessionLock, merge_segments

TRACKS = (("loopback", "audio.wav"), ("microphone", "microphone.wav"))
RECORDING_DIAGNOSTICS = "recording_diagnostics.jsonl"


class SessionRecorder:
    def __init__(
        self,
        loopback: LoopbackRecorder,
        microphone: LoopbackRecorder | None = None,
    ):
        self._recorders = [loopback, microphone] if microphone is not None else [loopback]
        self._lock: SessionLock | None = None
        self._session: Session
        self._take: Path | None = None
        self._prepared: list[LoopbackRecorder] = []
        self._last_diagnostic = 0.0
        self._last_status_key: tuple[bool, int] = (False, 0)

    @property
    def dual_track(self) -> bool:
        return len(self._recorders) > 1

    def set_audio_tap(self, callback: Callable[[str, AudioEvent], None] | None) -> None:
        if self._lock is not None:
            raise RuntimeError("只能在录音启动前设置实时音频分流")
        for recorder, (track, _) in zip(self._recorders, TRACKS):
            recorder.audio_tap = partial(callback, track) if callback is not None else None

    def start(self, session: Session, *, overwrite: bool = False) -> None:
        if self._lock is not None:
            raise RuntimeError("录音正在进行或上次清理尚未完成")
        self._session = session
        self._lock = SessionLock(session.ensure().root)
        self._prepared = []
        self._take = None
        self._last_diagnostic = time.monotonic()
        self._last_status_key = (False, 0)
        recorder: LoopbackRecorder | None = None
        try:
            if not overwrite and (
                session.audio_path.exists() or session.recording_manifest.exists()
            ):
                raise FileExistsError("会话录音已存在；请使用 --overwrite")
            if self.dual_track:
                self._take = session.root / f"audio_parts-{uuid.uuid4().hex}"
                self._take.mkdir()
            for recorder, (track, filename) in zip(self._recorders, TRACKS):
                self._prepared.append(recorder)
                take = self._take
                recorder._prepare(
                    take / filename if take else session.audio_path,
                    overwrite=overwrite,
                    directory=take / track if take else None,
                    session_locked=True,
                )
            start = self._recorders[0]._now()
            for recorder in self._prepared:
                recorder._begin(start)
            self._diagnose("recording.started", self._status(allow_restart=True))
        except Exception as error:
            if recorder is not None:
                recorder._fail(error)
                self.stop()  # seals the tracks and reports the startup error
            else:
                self._release()
            raise

    def status(self) -> dict:
        info = self._status(allow_restart=True)
        if self._lock is None:
            return info
        status_key = (bool(info["failed"]), int(info["stream_restart_attempts"]))
        now = time.monotonic()
        if status_key != self._last_status_key:
            self._diagnose("recording.status_changed", info)
        elif now - self._last_diagnostic >= 30:
            self._diagnose("recording.checkpoint", info)
        else:
            return info
        self._last_diagnostic = now
        self._last_status_key = status_key
        return info

    def _status(self, *, allow_restart: bool) -> dict:
        info = self._recorders[0].status(allow_restart=allow_restart)
        if self.dual_track:
            info["microphone"] = self._recorders[1].status()
            info["failed"] |= info["microphone"]["failed"]
        return info

    def _diagnose(self, event: str, info: dict) -> None:
        tracks = {}
        for recorder, (track, _) in zip(self._prepared, TRACKS):
            values = info.get("microphone") if track == "microphone" else info
            if values is None:
                continue
            tracks[track] = {
                "device": recorder.device_name,
                "sample_rate": recorder.sample_rate_actual,
                "channels": recorder.channels_actual,
                "chunk_ms": recorder.chunk_ms,
                **{
                    key: values[key]
                    for key in (
                        "state",
                        "duration_sec",
                        "received_sec",
                        "padded_sec",
                        "no_data_sec",
                        "timing_mode",
                        "repaired_timestamp_blocks",
                        "stream_restart_attempts",
                        "stream_restarts",
                        "error",
                        "cleanup_errors",
                    )
                    if key in values
                },
            }
        entry = {"time": datetime.now().astimezone().isoformat(), "event": event, "tracks": tracks}
        try:
            with (self._session.root / RECORDING_DIAGNOSTICS).open("a", encoding="utf-8") as log:
                log.write(json.dumps(entry, ensure_ascii=False) + "\n")
                log.flush()
                os.fsync(log.fileno())
        except OSError:
            logging.getLogger("meetmin").warning("无法写入会话录音诊断日志", exc_info=True)

    def stop(self) -> dict:
        if self._lock is None:
            raise RuntimeError("未在录音")
        end = self._recorders[0]._now()
        for recorder in self._prepared:
            if recorder._stop_at is None:
                recorder._stop_at = end
            recorder.status()

        # Stop both producers before draining the queues and merging either WAV.
        closed = [recorder._close_stream() for recorder in self._prepared]
        errors = []
        tracks = []
        for recorder, can_finish, (track, _) in zip(self._prepared, closed, TRACKS):
            try:
                if not can_finish:
                    raise RecordingError("无法关闭音频流", recorder._info())
                info = recorder._finish(stream_already_closed=True)
            except RecordingError as error:
                info = error.info
                errors.append(f"{track}：{error}")
            tracks.append(info)
        info = dict(tracks[0])
        if len(tracks) > 1:
            info["microphone"] = tracks[1]
        info["cleanup_pending"] = any(track["cleanup_pending"] for track in tracks)
        info["failed"] = bool(errors)
        info["finalized"] = all(track["finalized"] for track in tracks)
        info["cleanup_errors"] = [error for track in tracks for error in track["cleanup_errors"]]
        try:
            if errors:
                raise RecordingError("；".join(errors), info)
            if self._take is not None:
                self._session.publish_dual_recording(self._take)
            else:
                self._session.recording_manifest.unlink(missing_ok=True)
        except OSError as error:
            info["failed"] = True
            info["finalized"] = False
            raise RecordingError(f"发布录音失败：{error}", info) from error
        finally:
            self._diagnose("recording.failed" if info["failed"] else "recording.stopped", info)
            if not info["cleanup_pending"]:
                self._release()
        return info

    def _release(self) -> None:
        assert self._lock is not None
        self._lock.close()
        self._lock = None


def recover_recording(session: Session, take: Path, output: Path | None) -> list[tuple[Path, dict]]:
    """调用方持有会话锁；手动导出的双轨写入一个新目录。"""
    if not any((take / track).exists() for track, _ in TRACKS):
        target = output or session.audio_path
        return [(target, merge_segments(take, target))]
    if not re.fullmatch(r"audio_parts-[0-9a-f]{32}", take.name):
        raise ValueError("双轨录音的分段目录名无效")
    for track, _ in TRACKS:
        source = take / track
        if source.is_symlink() or not source.is_dir():
            raise ValueError(f"缺少双轨录音分段：{source}")
    if output is None:
        recovered = [
            (take / name, merge_segments(take / track, take / name, overwrite=True))
            for track, name in TRACKS
        ]
        session.publish_dual_recording(take)
        return recovered

    output.mkdir()
    recovered = []
    try:
        for track, name in TRACKS:
            target = output / name
            recovered.append((target, merge_segments(take / track, target)))
    except Exception:
        for target, _ in recovered:
            target.unlink()
        output.rmdir()
        raise
    return recovered
