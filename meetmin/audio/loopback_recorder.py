"""Windows WASAPI loopback 内录。

录的是"电脑正在播放的声音"（会议远端发言、以及会议软件混入输出的任何声音）。
注意：本地麦克风说话默认不会出现在电脑输出里，因此不会进内录（README 有解决办法）。
"""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


def _import_pyaudio():
    try:
        import pyaudiowpatch as pyaudio

        return pyaudio
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("缺少依赖 pyaudiowpatch，请先执行: pip install pyaudiowpatch") from e


def list_loopback_devices() -> list[dict]:
    """列出所有 loopback（内录）设备。"""
    pyaudio = _import_pyaudio()
    pa = pyaudio.PyAudio()
    out = []
    try:
        for d in pa.get_loopback_device_info_generator():
            out.append(
                {
                    "index": int(d["index"]),
                    "name": str(d["name"]),
                    "sample_rate": int(d["defaultSampleRate"]),
                    "max_channels": int(d["maxInputChannels"]),
                }
            )
    finally:
        pa.terminate()
    return out


class LoopbackRecorder:
    """后台线程持续把 loopback 输入写入 WAV（int16）。"""

    def __init__(
        self,
        device: int | str | None = None,
        channels: int | None = None,
        sample_rate: int | None = None,
        chunk_ms: int = 100,
        silence_threshold: float = 0.003,
    ):
        self.device = device
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk_ms = max(20, int(chunk_ms))
        self.silence_threshold = float(silence_threshold)
        if not 0 <= self.silence_threshold <= 1:
            raise ValueError("silence_threshold 必须在 0 到 1 之间")
        self._level_lock = threading.Lock()
        self._frames = 0
        self._silent_frames = 0
        self._active_frames = 0
        self._level = 0.0
        self._last_data_time: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._pa: Any = None
        self._stream: Any = None
        self._wf: wave.Wave_write | None = None
        self._audio_file: BinaryIO | None = None
        self._chunks = 0
        self._chunk_frames = 0
        self.device_name = ""
        self.sample_rate_actual = 0
        self.channels_actual = 0
        self.path: Path | None = None

    # ---------------- 设备解析 ----------------
    def _resolve_device(self, pa):
        if isinstance(self.device, int):
            return pa.get_device_info_by_index(self.device)
        if isinstance(self.device, str) and self.device.strip():
            key = self.device.strip().lower()
            for d in pa.get_loopback_device_info_generator():
                if key in str(d["name"]).lower():
                    return d
            raise RuntimeError(
                f"未找到名称包含 {self.device!r} 的 loopback 设备，可用: python main.py devices"
            )
        pyaudio = _import_pyaudio()
        try:
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
            if default_out.get("is_loopback_dev"):
                return default_out
            for d in pa.get_loopback_device_info_generator():
                if str(default_out["name"]) in str(d["name"]) or str(d["name"]) in str(
                    default_out["name"]
                ):
                    return d
        except OSError:
            pass
        for d in pa.get_loopback_device_info_generator():  # 兜底：第一个 loopback
            return d
        raise RuntimeError("未找到 WASAPI loopback 设备（需要 Windows，且系统有可用输出设备）")

    # ---------------- 生命周期 ----------------
    def start(self, wav_path: str | Path, *, overwrite: bool = False) -> None:
        if self._thread is not None:
            raise RuntimeError("录音已在进行中")
        pyaudio = _import_pyaudio()
        self._pa = pyaudio.PyAudio()
        try:
            info = self._resolve_device(self._pa)
            sr = int(self.sample_rate or info["defaultSampleRate"])
            ch = int(self.channels or max(1, min(2, int(info["maxInputChannels"]))))
            path = Path(wav_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            self._audio_file = path.open("wb" if overwrite else "xb")
            self._wf = wave.open(self._audio_file, "wb")
            self._wf.setnchannels(ch)
            self._wf.setsampwidth(2)
            self._wf.setframerate(sr)

            self._chunk_frames = max(16, int(sr * self.chunk_ms / 1000))
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=ch,
                rate=sr,
                input=True,
                input_device_index=int(info["index"]),
                frames_per_buffer=self._chunk_frames,
            )
            self.device_name = str(info["name"])
            self.sample_rate_actual = sr
            self.channels_actual = ch
            self.path = path
            self._chunks = 0
            self._error = None
            self._stop.clear()
            with self._level_lock:
                self._last_data_time = time.monotonic()
            self._thread = threading.Thread(target=self._run, daemon=True, name="loopback-rec")
            self._thread.start()
        except Exception:
            self._cleanup()
            raise

    def _run(self) -> None:
        try:
            assert self._wf is not None
            while not self._stop.is_set():
                data = self._stream.read(self._chunk_frames, exception_on_overflow=False)
                self._wf.writeframes(data)
                self._chunks += 1
                self._update_level(data)
        except Exception as e:  # 记录错误，由 stop() 抛出
            self._error = e
        finally:
            try:
                if self._wf is not None:
                    self._wf.close()
            except Exception:
                pass

    def _update_level(self, data: bytes) -> None:
        samples = np.frombuffer(data, dtype="<i2").astype(np.float64)
        frames = len(samples) // self.channels_actual
        if not frames:
            return
        level = float(np.sqrt(np.mean((samples / 32768.0) ** 2)))
        with self._level_lock:
            self._last_data_time = time.monotonic()
            self._level = level
            self._frames += frames
            if level < self.silence_threshold:
                self._silent_frames += frames
            else:
                self._silent_frames = 0
                self._active_frames += frames

    def status(self) -> dict:
        with self._level_lock:
            sr = self.sample_rate_actual or 1
            return {
                "duration_sec": self._frames / sr,
                "level": self._level,
                "no_data_sec": max(0.0, time.monotonic() - self._last_data_time)
                if self._last_data_time is not None
                else 0.0,
                "silent_sec": self._silent_frames / sr,
                "active_fraction": self._active_frames / self._frames if self._frames else 0.0,
                "failed": self._error is not None,
            }

    def stop(self) -> dict:
        if self._thread is None:
            raise RuntimeError("未在录音")
        self._stop.set()
        self._thread.join(timeout=5)
        stats = self.status()
        info = self._cleanup()
        info.update(stats)
        if self._error is not None:
            raise self._error
        return info

    def _cleanup(self) -> dict:
        duration = 0.0
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._wf is not None:
            try:
                self._wf.close()
            except Exception:
                pass
            self._wf = None
        if self._audio_file is not None:
            self._audio_file.close()
            self._audio_file = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
        if self.sample_rate_actual and self._chunk_frames:
            duration = self._chunks * self._chunk_frames / self.sample_rate_actual
        return {
            "path": str(self.path) if self.path else None,
            "duration_sec": round(duration, 1),
            "sample_rate": self.sample_rate_actual,
            "channels": self.channels_actual,
            "device": self.device_name,
        }

    @property
    def is_recording(self) -> bool:
        return self._thread is not None and not self._stop.is_set()
