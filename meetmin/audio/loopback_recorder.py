"""Windows WASAPI loopback 内录。

录的是"电脑正在播放的声音"（会议远端发言、以及会议软件混入输出的任何声音）。
麦克风设备可使用同一套时间轴与分段写盘逻辑单独采集。
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dsp import rms
from .segments import SegmentedWriter, merge_segments


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


def list_microphone_devices() -> list[dict]:
    pyaudio = _import_pyaudio()
    pa = pyaudio.PyAudio()
    try:
        return [
            {
                "index": int(d["index"]),
                "name": str(d["name"]),
                "sample_rate": int(d["defaultSampleRate"]),
                "max_channels": int(d["maxInputChannels"]),
            }
            for d in pa.get_device_info_generator_by_host_api(host_api_type=pyaudio.paWASAPI)
            if d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice")
        ]
    finally:
        pa.terminate()


class RecordingError(RuntimeError):
    """录音失败；info 包含已写时长、文件位置和清理结果。"""

    def __init__(self, message: str, info: dict):
        super().__init__(message)
        self.info = info


@dataclass(frozen=True)
class _AudioBlock:
    data: bytes
    frames: int
    timing: dict
    received_at: float


class _AdaptiveLevel:
    """在量化为 16 位 PCM 前补偿低电平。由写线程持有，每次录音重新初始化。"""

    TARGET_RMS = 0.03
    RELEASE_SECONDS = 0.08
    RESET_AFTER_SILENCE_SECONDS = 0.2

    def __init__(self, rate: int, *, max_gain: float, noise_floor: float = 0.0):
        self.rate = rate
        self.max_gain = max_gain
        self.noise_floor = noise_floor
        self.reference = 0.0
        self.gain: float | None = None
        self.silent_frames = 0

    def add_silence(self, frames: int) -> None:
        self.silent_frames += frames
        if self.silent_frames >= self.rate * self.RESET_AFTER_SILENCE_SECONDS:
            self.reference = 0.0
            self.gain = None

    def to_pcm(self, data: bytes, frames: int) -> bytes:
        samples = np.frombuffer(data, dtype="<f4")
        peak = float(np.max(np.abs(samples)))
        if not math.isfinite(peak):
            raise RuntimeError("录音采样包含无效浮点值")
        if peak == 0:
            self.add_silence(frames)
            return bytes(len(data) // 2)

        self.silent_frames = 0
        level = rms(samples)
        # Release the previous level gradually when quieter speech follows without a pause.
        release = math.exp(-frames / (self.rate * self.RELEASE_SECONDS))
        self.reference = max(level, self.reference * release)
        gain_limit = self.max_gain
        if self.noise_floor:
            # Taper off boost near the microphone noise floor without gating the signal.
            boost_fraction = max(0.0, min(1.0, (level - self.noise_floor) / self.noise_floor))
            gain_limit = 1.0 + (self.max_gain - 1.0) * boost_fraction
        # Leave peak headroom when boosting and pass normal-level audio through at unity gain.
        gain = max(1.0, min(self.TARGET_RMS / self.reference, gain_limit, 0.95 / peak))
        if self.gain is not None and gain > self.gain:
            ramp = np.linspace(self.gain, gain, frames, dtype=np.float32)
            samples = samples.reshape(frames, -1) * ramp[:, None]
        else:
            samples = samples * gain
        self.gain = gain
        return np.rint(np.clip(samples * 32768, -32768, 32767)).astype("<i2").tobytes()


class _Timeline:
    """只在写线程使用；采集时间决定位置，消费队列的时间不参与计算。"""

    def __init__(self, rate: int, chunk_frames: int, start: float, stream_origin: float | None):
        self.rate = rate
        self.start = start
        self.stream_origin = stream_origin
        self.mode = "pending"
        self.previous_adc: float | None = None
        self.tolerance = max(round(rate * 0.02), chunk_frames // 2)

    def position(self, block: _AudioBlock, cursor: int) -> int:
        adc = block.timing.get("input_buffer_adc_time", 0.0)
        current = block.timing.get("current_time", 0.0)
        valid = (
            isinstance(adc, (int, float))
            and isinstance(current, (int, float))
            and math.isfinite(adc)
            and math.isfinite(current)
            and 0 < adc <= current
        )
        if self.mode == "pending":
            self.mode = "device" if valid and self.stream_origin is not None else "arrival"
        if self.mode == "device":
            if not valid or (self.previous_adc is not None and adc <= self.previous_adc):
                raise RuntimeError("音频采集时间戳无效或倒退，无法可靠定位音频")
            assert self.stream_origin is not None
            seconds = adc - self.stream_origin
            # Reject a clock reset / bogus timestamp before it can generate a huge zero-filled file.
            if seconds > block.received_at - self.start + 0.5 or seconds < -0.5:
                raise RuntimeError("音频采集时钟与录音时钟不一致")
            self.previous_adc = adc
        else:
            # Arrival timing is approximate: scheduler delays are indistinguishable from gaps.
            seconds = block.received_at - self.start - block.frames / self.rate
        target = max(0, round(seconds * self.rate))
        delta = target - cursor
        if delta < -self.tolerance:
            raise RuntimeError("音频块严重重叠或迟到，无法可靠保持时间轴")
        # WASAPI timestamps have jitter; contiguous blocks must not acquire tiny gaps each time.
        return target if delta > self.tolerance or cursor == 0 else cursor


class LoopbackRecorder:
    """回调采集、有界队列、单写线程；start/stop 由控制线程顺序调用。"""

    def __init__(
        self,
        device: int | str | None = None,
        channels: int | None = None,
        sample_rate: int | None = None,
        chunk_ms: int = 100,
        silence_threshold: float = 0.003,
        *,
        queue_seconds: float = 5.0,
        stop_timeout: float = 5.0,
        clock: Callable[[], float] | None = None,
        segment_seconds: float = 300,
        sync_seconds: float = 10,
        output_sample_rate: int | None = None,
        output_channels: int | None = None,
    ):
        self.device = device
        self.channels = channels
        self.sample_rate = sample_rate
        self.output_sample_rate = output_sample_rate
        self.output_channels = output_channels
        self._resampler: Any = None
        self.chunk_ms = max(20, int(chunk_ms))
        self.silence_threshold = float(silence_threshold)
        if not 0 <= self.silence_threshold <= 1:
            raise ValueError("silence_threshold 必须在 0 到 1 之间")
        if not math.isfinite(queue_seconds) or queue_seconds <= 0:
            raise ValueError("queue_seconds 必须是正数")
        if not math.isfinite(stop_timeout) or stop_timeout <= 0:
            raise ValueError("stop_timeout 必须是正数")
        self._clock = clock
        self._segment_seconds = segment_seconds
        self._sync_seconds = sync_seconds
        self._queue_size = max(1, math.ceil(queue_seconds * 1000 / self.chunk_ms))
        self._stop_timeout = stop_timeout
        self._level_lock = threading.Lock()
        self._frames = 0
        self._silent_frames = 0
        self._active_frames = 0
        self._level = 0.0
        self._last_data_time: float | None = None
        self._producer_done = threading.Event()
        self._failed = threading.Event()
        self._errors: queue.SimpleQueue[Exception] = queue.SimpleQueue()
        self._queue: queue.Queue[_AudioBlock] = queue.Queue(self._queue_size)
        self._state = "IDLE"
        self._started_at = 0.0
        self._stop_at: float | None = None
        self._timeline: _Timeline | None = None
        self._padded_frames = 0
        self._received_frames = 0
        self._finalized = False
        self._secondary_errors: list[str] = []
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._pa: Any = None
        self._stream: Any = None
        self._wf: SegmentedWriter | None = None
        self._audio_file: SegmentedWriter | None = None
        self._sealed_writer: SegmentedWriter | None = None
        self._chunk_frames = 0
        self._capture_sample_width = 2
        self._level_control: _AdaptiveLevel | None = None
        self.device_name = ""
        self.sample_rate_actual = 0
        self.channels_actual = 0
        self.path: Path | None = None

    def _now(self) -> float:
        return self._clock() if self._clock is not None else time.monotonic()

    def _fail(self, error: Exception) -> None:
        # SimpleQueue has no bounded-capacity wait. Keep exceptions inside the callback,
        # rather than letting PyAudio inject them asynchronously into the main thread.
        self._errors.put(error)
        self._failed.set()

    def _collect_errors(self) -> None:
        while True:
            try:
                error = self._errors.get_nowait()
            except queue.Empty:
                break
            if self._error is None:
                self._error = error
            else:
                self._secondary_errors.append(str(error))

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

    def _create_level_control(self, rate: int, info: dict) -> _AdaptiveLevel | None:
        if info.get("isLoopbackDevice"):
            return _AdaptiveLevel(rate, max_gain=10_000.0)  # Up to 80 dB for playback attenuation.
        return None

    # ---------------- 生命周期 ----------------
    def start(self, wav_path: str | Path, *, overwrite: bool = False) -> None:
        try:
            self._prepare(wav_path, overwrite=overwrite)
            self._begin(self._now())
        except Exception as error:
            if self._state != "STARTING":
                raise
            self._fail(error)
            self._state = "FAILED"
            self._stop_at = self._now()
            complete = self._shutdown()
            self._collect_errors()
            if not complete or self._secondary_errors:
                raise RecordingError(str(self._error), self._info()) from error
            raise

    def _prepare(
        self,
        wav_path: str | Path,
        *,
        overwrite: bool,
        directory: Path | None = None,
        session_locked: bool = False,
    ) -> None:
        if self._thread is not None or self._stream is not None or self._pa is not None:
            raise RuntimeError("录音正在进行或上次清理尚未完成")
        self._state = "STARTING"
        self._producer_done.clear()
        self._failed.clear()
        self._errors = queue.SimpleQueue()
        self._queue = queue.Queue(self._queue_size)
        self._error = None
        self._secondary_errors = []
        self._stop_at = None
        self._timeline = None
        self._finalized = False
        self._received_frames = 0
        self._padded_frames = 0
        with self._level_lock:
            self._frames = self._silent_frames = self._active_frames = 0
            self._level = 0.0
            self._last_data_time = None
        self.path = Path(wav_path)
        self.device_name = ""
        self.sample_rate_actual = self.channels_actual = 0
        pyaudio = _import_pyaudio()
        self._pa = pyaudio.PyAudio()
        info = self._resolve_device(self._pa)
        sr = int(self.sample_rate or info["defaultSampleRate"])
        ch = int(self.channels or max(1, min(2, int(info["maxInputChannels"]))))
        output_rate = self.output_sample_rate or sr
        output_ch = self.output_channels or ch
        if output_rate <= 0 or output_ch not in (1, ch):
            raise ValueError("录音保存格式的采样率或声道数无效")
        self._resampler = None
        if output_rate != sr:
            import soxr

            self._resampler = soxr.ResampleStream(sr, output_rate, output_ch, dtype="int16")
        self._level_control = self._create_level_control(sr, info)
        float_capture = self._level_control is not None
        self._capture_sample_width = 4 if float_capture else 2
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)

        self._wf = SegmentedWriter(
            path,
            output_rate,
            output_ch,
            overwrite=overwrite,
            segment_seconds=self._segment_seconds,
            sync_seconds=self._sync_seconds,
            directory=directory,
            session_locked=session_locked,
        )
        self._audio_file = self._wf

        self._chunk_frames = max(16, int(sr * self.chunk_ms / 1000))
        self.device_name = str(info["name"])
        self.sample_rate_actual = sr
        self.channels_actual = ch
        self._pa_continue = pyaudio.paContinue
        self._pa_abort = pyaudio.paAbort
        self._stream = self._pa.open(
            format=pyaudio.paFloat32 if float_capture else pyaudio.paInt16,
            channels=ch,
            rate=sr,
            input=True,
            input_device_index=int(info["index"]),
            frames_per_buffer=self._chunk_frames,
            stream_callback=self._callback,
            start=False,
        )

    def _begin(self, started_at: float) -> None:
        # Pa_GetStreamTime and monotonic have different epochs on Windows.
        before = self._now()
        offset: float | None
        try:
            stream_time = float(self._stream.get_time())
            offset = stream_time - (before + self._now()) / 2
            if not math.isfinite(stream_time) or stream_time <= 0:
                offset = None
        except (OSError, ValueError):
            offset = None
        self._started_at = started_at
        self._last_data_time = self._now()
        self._timeline = _Timeline(
            self.sample_rate_actual,
            self._chunk_frames,
            started_at,
            started_at + offset if offset is not None else None,
        )
        self._thread = threading.Thread(target=self._run, daemon=True, name="audio-writer")
        self._thread.start()
        self._stream.start_stream()
        self._state = "RECORDING"

    def _callback(self, data, frame_count, time_info, status_flags):
        if self._failed.is_set():
            return None, self._pa_abort
        try:
            if status_flags:
                raise RuntimeError(f"音频采集异常（PortAudio flags={status_flags}），可能丢帧")
            if frame_count:
                if not 0 < frame_count <= self._chunk_frames:
                    raise RuntimeError("音频回调帧数超过配置的缓冲区大小")
                expected_bytes = frame_count * self.channels_actual * self._capture_sample_width
                if data is None or len(data) != expected_bytes:
                    raise RuntimeError("音频回调数据长度与帧数不符")
                received = self._now()
                self._queue.put_nowait(_AudioBlock(data, frame_count, time_info, received))
                self._last_data_time = received
                self._received_frames += frame_count
        except queue.Full:
            self._fail(RuntimeError("录音队列已满，写盘速度不足；已停止采集以避免静默丢帧"))
            return None, self._pa_abort
        except Exception as error:
            self._fail(error)
            return None, self._pa_abort
        return None, self._pa_continue

    def _run(self) -> None:
        pending: _AudioBlock | None = None
        try:
            while not self._producer_done.is_set() or not self._queue.empty():
                try:
                    block = self._queue.get(timeout=0.05)
                except queue.Empty:
                    assert self._wf is not None
                    self._wf.checkpoint()
                    continue
                # Hold the last block so stop() can trim samples past the stop boundary.
                if pending is not None:
                    self._write_block(pending)
                pending = block
            if pending is not None:
                self._write_block(pending)
            if not self._failed.is_set():
                self._pad_to(self._stop_frame())
        except Exception as error:
            self._fail(error)
        finally:
            self._close_audio()

    def _stop_frame(self) -> int:
        assert self._stop_at is not None
        return max(0, round((self._stop_at - self._started_at) * self.sample_rate_actual))

    def _pad_to(self, target: int) -> None:
        assert self._wf is not None
        gap_frames = target - self._frames
        if gap_frames > 0 and self._level_control is not None:
            self._level_control.add_silence(gap_frames)
        while self._frames < target:
            frames = min(target - self._frames, self.sample_rate_actual)
            self._write_output(bytes(frames * self.channels_actual * 2))
            with self._level_lock:
                self._frames += frames
                self._padded_frames += frames

    def _write_block(self, block: _AudioBlock) -> None:
        assert self._timeline is not None and self._wf is not None
        position = self._timeline.position(block, self._frames)
        frames = block.frames
        if self._stop_at is not None:
            end = self._stop_frame()
            position = min(position, end)
            frames = min(frames, max(0, end - position))
        self._pad_to(position)
        data = block.data[: frames * self.channels_actual * self._capture_sample_width]
        if data:
            if self._level_control is not None:
                data = self._level_control.to_pcm(data, frames)
            self._write_output(data)
            self._update_level(data, record_arrival=False)

    def _write_output(self, data: bytes) -> None:
        assert self._wf is not None
        samples = np.frombuffer(data, dtype="<i2").reshape(-1, self.channels_actual)
        if self._wf.channels == 1 and self.channels_actual > 1:
            samples = np.rint(samples.astype(np.float32).mean(axis=1)).astype(np.int16)
        elif self._wf.channels == 1:
            samples = samples[:, 0]
        if self._resampler is not None:
            samples = self._resampler.resample_chunk(samples)
        self._wf.writeframesraw(samples.astype("<i2", copy=False).tobytes())

    def _close_audio(self) -> None:
        # Called by the writer, or by start() only if no writer was ever started.
        closed = True
        writer = self._wf
        if writer is not None:
            try:
                try:
                    if self._resampler is not None:
                        shape = (0,) if writer.channels == 1 else (0, writer.channels)
                        tail = self._resampler.resample_chunk(
                            np.empty(shape, dtype=np.int16), last=True
                        )
                        writer.writeframesraw(tail.astype("<i2", copy=False).tobytes())
                        self._resampler = None
                finally:
                    writer.close()
            except Exception as error:
                closed = False
                self._fail(error)
            self._sealed_writer = writer
        self._wf = None
        self._audio_file = None
        self._finalized = closed

    def _update_level(self, data: bytes, *, record_arrival: bool = True) -> None:
        samples = np.frombuffer(data, dtype="<i2").astype(np.float64)
        frames = len(samples) // self.channels_actual
        if not frames:
            return
        level = float(np.sqrt(np.mean((samples / 32768.0) ** 2)))
        with self._level_lock:
            if record_arrival:
                self._last_data_time = self._now()
            self._level = level
            self._frames += frames
            if level < self.silence_threshold:
                self._silent_frames += frames
            else:
                self._silent_frames = 0
                self._active_frames += frames

    def status(self) -> dict:
        if self._state == "RECORDING" and not self._failed.is_set():
            try:
                if not self._stream.is_active():
                    self._fail(RuntimeError("录音流意外停止"))
                elif self._thread is None or not self._thread.is_alive():
                    self._fail(RuntimeError("录音写线程意外退出"))
            except Exception as error:
                self._fail(error)
        self._collect_errors()
        with self._level_lock:
            sr = self.sample_rate_actual or 1
            now = self._stop_at if self._stop_at is not None else self._now()
            return {
                "duration_sec": self._frames / sr,
                "level": self._level,
                "no_data_sec": max(0.0, now - self._last_data_time)
                if self._last_data_time is not None
                else 0.0,
                "silent_sec": self._silent_frames / sr,
                "active_fraction": self._active_frames / self._frames if self._frames else 0.0,
                "failed": self._failed.is_set(),
                "state": "FAILED" if self._failed.is_set() else self._state,
                "error": str(self._error) if self._error is not None else None,
                "cleanup_errors": list(self._secondary_errors),
                "received_sec": self._received_frames / sr,
                "padded_sec": self._padded_frames / sr,
                "timing_mode": self._timeline.mode if self._timeline is not None else "pending",
                "finalized": self._finalized,
                "cleanup_pending": self._thread is not None
                or self._stream is not None
                or self._pa is not None,
            }

    def stop(self) -> dict:
        if self._state == "IDLE":
            raise RuntimeError("未在录音")
        # Detect a stream that died just before the user requested stop, even if no
        # status polling occurred since the last successful callback.
        if self._state == "RECORDING":
            self.status()
        if self._stop_at is None:
            self._stop_at = self._now()
        return self._finish()

    def _finish(self, *, stream_already_closed: bool = False) -> dict:
        self._state = "STOPPING"
        complete = self._shutdown(stream_already_closed=stream_already_closed)
        self._state = "STOPPED" if complete and not self._failed.is_set() else "FAILED"
        info = self._info()
        if self._error is not None:
            raise RecordingError(str(self._error), info) from self._error
        return info

    def _close_stream(self) -> bool:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception as error:
                self._fail(error)
            try:
                # Closing is the barrier after which callbacks can no longer enqueue data.
                self._stream.close()
            except Exception as error:
                self._fail(error)
                # Keep resources alive for a later cleanup attempt.
                return False
            self._stream = None
        self._producer_done.set()
        return True

    def _shutdown(self, *, stream_already_closed: bool = False) -> bool:
        if not stream_already_closed and not self._close_stream():
            return False
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=self._stop_timeout)
            if self._thread.is_alive():
                self._fail(RuntimeError("等待录音写线程退出超时；文件仍由该线程持有，禁止复用"))
                return False
        elif self._wf is not None or self._audio_file is not None:
            self._close_audio()
        self._thread = None
        if self._sealed_writer is not None:
            writer = self._sealed_writer
            try:
                if not self._failed.is_set():
                    merge_segments(writer.directory, writer.path, overwrite=writer.overwrite)
            except Exception as error:
                self._finalized = False
                self._fail(error)
            finally:
                try:
                    writer.release()
                except Exception as error:
                    self._fail(error)
                self._sealed_writer = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception as error:
                self._fail(error)
                return False
            self._pa = None
        return True

    def _info(self) -> dict:
        return {
            "path": str(self.path) if self.path else None,
            "sample_rate": self.output_sample_rate or self.sample_rate_actual,
            "channels": self.output_channels or self.channels_actual,
            "capture_sample_rate": self.sample_rate_actual,
            "capture_channels": self.channels_actual,
            "device": self.device_name,
            **self.status(),
        }

    @property
    def is_recording(self) -> bool:
        return self._state == "RECORDING" and not self.status()["failed"]


class MicrophoneRecorder(LoopbackRecorder):
    def _create_level_control(self, rate: int, info: dict) -> _AdaptiveLevel:
        return _AdaptiveLevel(rate, max_gain=16.0, noise_floor=0.0003)

    def _resolve_device(self, pa):
        pyaudio = _import_pyaudio()
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        index = self.device if self.device is not None else wasapi["defaultInputDevice"]
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("麦克风设备索引必须是整数，请运行 meetmin devices")
        if index < 0:
            raise RuntimeError("未找到默认 WASAPI 麦克风，请检查 Windows 输入设置")
        info = pa.get_device_info_by_index(index)
        if (
            info["hostApi"] != wasapi["index"]
            or info["maxInputChannels"] < 1
            or info.get("isLoopbackDevice")
        ):
            raise ValueError(f"设备 [{index}] 不是 WASAPI 麦克风，请运行 meetmin devices")
        return info
