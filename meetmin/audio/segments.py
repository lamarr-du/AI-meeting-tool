"""Crash-recoverable PCM segments, with a session-wide recording/recovery lock."""

from __future__ import annotations

import importlib
import math
import os
import re
import struct
import tempfile
import time
import uuid
import wave
from pathlib import Path
from typing import Any, BinaryIO


class SessionLock:
    def __init__(self, root: Path):
        self.file = (root / ".audio.lock").open("a+b")
        try:
            self.file.seek(0, 2)
            if not self.file.tell():
                self.file.write(b"0")
                self.file.flush()
            self.file.seek(0)
            if os.name == "nt":
                msvcrt: Any = importlib.import_module("msvcrt")

                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl: Any = importlib.import_module("fcntl")
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise RuntimeError("此会话正在录音或恢复") from exc

    def close(self):
        if self.file.closed:
            return
        self.file.seek(0)
        if os.name == "nt":
            msvcrt: Any = importlib.import_module("msvcrt")

            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl: Any = importlib.import_module("fcntl")
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()


def _parts(directory: Path):
    files = sorted(directory.iterdir())
    if not files:
        raise ValueError("没有可恢复的分段")
    for index, path in enumerate(files, 1):
        if (
            path.is_symlink()
            or not path.is_file()
            or not re.fullmatch(r"\d{6}\.wav(?:\.part)?", path.name)
        ):
            raise ValueError(f"未知分段文件：{path}")
        if int(path.name[:6]) != index or (path.name.endswith(".part") and index != len(files)):
            raise ValueError("分段编号缺失、重复或未完成分段不在末尾")
    return files


def merge_segments(directory: Path, destination: Path, *, overwrite: bool = False) -> dict:
    """Merge only our canonical PCM WAVs. Never modify input files."""
    files = _parts(directory)
    temporary = None
    params = None
    total = 0
    discarded = 0
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".audio-", suffix=".tmp", delete=False
        ) as out:
            temporary = Path(out.name)
            with wave.open(out, "wb") as target:
                for path in files:
                    with path.open("rb") as src:
                        header = src.read(44)
                        if len(header) != 44:
                            raise ValueError(f"WAV 头不完整：{path}")
                        (
                            riff,
                            riff_size,
                            wav,
                            fmt,
                            fmt_size,
                            encoding,
                            channels,
                            rate,
                            byte_rate,
                            align,
                            bits,
                            data,
                            declared,
                        ) = struct.unpack("<4sI4s4sIHHIIHH4sI", header)
                        if (
                            (riff, wav, fmt, fmt_size, encoding, data, bits)
                            != (b"RIFF", b"WAVE", b"fmt ", 16, 1, b"data", 16)
                            or channels < 1
                            or rate < 1
                            or align != channels * 2
                            or byte_rate != rate * align
                        ):
                            raise ValueError(f"不支持的分段格式：{path}")
                        current = (channels, 2, rate)
                        if params is None:
                            params = current
                            target.setnchannels(channels)
                            target.setsampwidth(2)
                            target.setframerate(rate)
                        elif current != params:
                            raise ValueError("分段音频格式不一致")
                        size = path.stat().st_size - 44
                        if not path.name.endswith(".part") and (
                            size != declared or riff_size != size + 36 or size % align
                        ):
                            raise ValueError(f"已完成分段长度不一致：{path}")
                        usable = size - size % align
                        discarded += size - usable
                        total += usable
                        if total + 36 > 0xFFFFFFFF:
                            raise ValueError("录音超过 WAV 4 GiB 上限，分段已保留")
                        remaining = usable
                        while remaining:
                            chunk = src.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise OSError(f"读取分段失败：{path}")
                            target.writeframesraw(chunk)
                            remaining -= len(chunk)
            out.flush()
            os.fsync(out.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)  # Atomic no-overwrite publication, same filesystem.
            temporary.unlink()
        temporary = None
        assert params is not None
        return {
            "duration_sec": total / (params[0] * params[1] * params[2]),
            "parts": len(files),
            "discarded_bytes": discarded,
            "partial": files[-1].name.endswith(".part"),
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class SegmentedWriter:
    """Writer-thread-only WAV sink. The last segment remains repairable after a crash."""

    def __init__(
        self,
        path: Path,
        rate: int,
        channels: int,
        *,
        overwrite: bool,
        segment_seconds: float,
        sync_seconds: float,
    ):
        if (
            not math.isfinite(segment_seconds)
            or not math.isfinite(sync_seconds)
            or segment_seconds <= 0
            or sync_seconds <= 0
        ):
            raise ValueError("分段长度和落盘间隔必须为正数")
        self.path = path
        self.lock = SessionLock(path.parent)
        self.closed = False
        self.overwrite = overwrite
        self.rate = rate
        self.channels = channels
        self.limit = max(1, round(rate * segment_seconds))
        self.sync_seconds = sync_seconds
        self.last_sync = time.monotonic()
        self.frames = 0
        self.index = 0
        self.file: BinaryIO | None = None
        self.wav: wave.Wave_write | None = None
        try:
            if path.exists() and not overwrite:
                raise FileExistsError(f"录音已存在：{path}")
            self.directory = path.parent / ("audio_parts-" + uuid.uuid4().hex)
            self.directory.mkdir()
            self._open()
        except BaseException:
            if self.file is not None:
                self.file.close()
            self.lock.close()
            raise

    def _open(self):
        self.index += 1
        self.part = self.directory / f"{self.index:06}.wav.part"
        self.file = self.part.open("xb")
        self.wav = wave.open(self.file, "wb")
        self.wav.setnchannels(self.channels)
        self.wav.setsampwidth(2)
        self.wav.setframerate(self.rate)
        self.wav.writeframesraw(b"")
        self.frames = 0
        self.checkpoint(force=True)

    def checkpoint(self, *, force=False):
        if self.file is not None and (
            force or time.monotonic() - self.last_sync >= self.sync_seconds
        ):
            self.file.flush()
            os.fsync(self.file.fileno())
            self.last_sync = time.monotonic()

    def _seal(self):
        assert self.wav is not None and self.file is not None
        self.wav.close()
        self.wav = None
        self.checkpoint(force=True)
        self.file.close()
        self.file = None
        self.part.rename(self.part.with_suffix(""))

    def writeframesraw(self, data: bytes):
        stride = self.channels * 2
        if len(data) % stride:
            raise ValueError("音频数据不是完整帧")
        offset = 0
        while offset < len(data):
            if self.frames == self.limit:
                self._seal()
                self._open()
            count = min((len(data) - offset) // stride, self.limit - self.frames)
            assert self.wav is not None
            self.wav.writeframesraw(data[offset : offset + count * stride])
            self.frames += count
            offset += count * stride
            self.checkpoint()

    def close(self):
        if self.closed:
            return
        try:
            if self.wav is not None:
                self._seal()
        finally:
            if self.file is not None:
                self.file.close()
            self.closed = True

    def release(self):
        self.lock.close()
