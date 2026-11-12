"""轻量音频处理：WAV 读写、单声道混合、重采样（ASR 统一用 16kHz 单声道 float32）。"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

TARGET_SR = 16000


def load_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """读 16bit PCM WAV 并混合为单声道 float32 [-1,1]。"""
    with wave.open(str(path), "rb") as wf:
        n_ch, sr, n = wf.getnchannels(), wf.getframerate(), wf.getnframes()
        sw = wf.getsampwidth()
        raw = wf.readframes(n)
    if sw != 2:
        raise ValueError(f"仅支持 16bit PCM WAV（当前 {sw * 8}bit）：{path}")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return x, sr


def save_wav_mono(path: str | Path, x: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """优先 soxr（高质量多相）；缺失时退化为线性插值（语音可接受）。"""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    try:
        import soxr

        return np.asarray(soxr.resample(x, sr_in, sr_out), dtype=np.float32)
    except ImportError:
        pass
    n_out = int(round(len(x) * sr_out / sr_in))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    t_out = np.linspace(0.0, len(x) - 1, n_out)
    return np.interp(t_out, np.arange(len(x)), x).astype(np.float32)


def rms(x: np.ndarray) -> float:
    if x is None or len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def audio_duration(path: str | Path) -> float:
    """尽力获取音频时长（秒）；无法判断时返回 -1。"""
    try:
        from faster_whisper.audio import decode_audio

        a = decode_audio(str(path), sampling_rate=TARGET_SR)
        return float(len(a) / TARGET_SR)
    except ImportError:
        pass
    except Exception:
        pass
    try:
        x, sr = load_wav_mono(path)
        return float(len(x) / sr)
    except Exception:
        return -1.0


def to_mono_16k(path: str | Path) -> np.ndarray:
    """任意受支持音频 → 16kHz 单声道 float32。优先用 faster-whisper 的解码器（支持 mp3 等）。"""
    try:
        from faster_whisper.audio import decode_audio

        return np.asarray(decode_audio(str(path), sampling_rate=TARGET_SR), dtype=np.float32)
    except ImportError:
        pass
    x, sr = load_wav_mono(path)
    return resample(x, sr, TARGET_SR)
