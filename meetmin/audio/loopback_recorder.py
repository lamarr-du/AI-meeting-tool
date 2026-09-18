"""Windows WASAPI loopback 内录。

录的是"电脑正在播放的声音"（会议远端发言、以及会议软件混入输出的任何声音）。
注意：本地麦克风说话默认不会出现在电脑输出里，因此不会进内录（README 有解决办法）。
"""
from __future__ import annotations

import threading
import wave
from pathlib import Path


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
    ):
        self.device = device
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk_ms = max(20, int(chunk_ms))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._pa = None
        self._stream = None
        self._wf = None
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
            raise RuntimeError(f"未找到名称包含 {self.device!r} 的 loopback 设备，可用: python main.py devices")
        pyaudio = _import_pyaudio()
        try:
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
            if default_out.get("is_loopback_dev"):
                return default_out
            for d in pa.get_loopback_device_info_generator():
                if str(default_out["name"]) in str(d["name"]) or str(d["name"]) in str(default_out["name"]):
                    return d
        except OSError:
            pass
        for d in pa.get_loopback_device_info_generator():  # 兜底：第一个 loopback
            return d
        raise RuntimeError("未找到 WASAPI loopback 设备（需要 Windows，且系统有可用输出设备）")

    # ---------------- 生命周期 ----------------
    def start(self, wav_path: str | Path) -> None:
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

            self._wf = wave.open(str(path), "wb")
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
            self._thread = threading.Thread(target=self._run, daemon=True, name="loopback-rec")
            self._thread.start()
        except Exception:
            self._cleanup()
            raise

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                data = self._stream.read(self._chunk_frames, exception_on_overflow=False)
                self._wf.writeframes(data)
                self._chunks += 1
        except Exception as e:  # 记录错误，由 stop() 抛出
            self._error = e
        finally:
            try:
                self._wf.close()
            except Exception:
                pass

    def stop(self) -> dict:
        if self._thread is None:
            raise RuntimeError("未在录音")
        self._stop.set()
        self._thread.join(timeout=5)
        info = self._cleanup()
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
