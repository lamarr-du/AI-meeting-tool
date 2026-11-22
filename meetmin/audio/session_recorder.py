"""同一会话的录音与恢复，双轨只有在两个 WAV 就绪后才发布。"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from functools import partial
from pathlib import Path

from ..session import Session
from .loopback_recorder import LoopbackRecorder, RecordingError
from .ring_buffer import AudioEvent
from .segments import SessionLock, merge_segments

TRACKS = (("loopback", "audio.wav"), ("microphone", "microphone.wav"))


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
        except Exception as error:
            if recorder is not None:
                recorder._fail(error)
                self.stop()  # seals the tracks and reports the startup error
            else:
                self._release()
            raise

    def status(self) -> dict:
        info = self._recorders[0].status()
        if self.dual_track:
            info["microphone"] = self._recorders[1].status()
            info["failed"] |= info["microphone"]["failed"]
        return info

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
