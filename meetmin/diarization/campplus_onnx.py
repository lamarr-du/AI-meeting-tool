"""说话人分离：3D-Speaker CAM++（ONNX Runtime）中英双语声纹嵌入 + 纯 NumPy 稳健聚类。

零 PyTorch / SpeechBrain / scikit-learn 依赖，直接复用 faster-whisper 自带的 onnxruntime 与 numpy。
首次运行自动下载约 27 MB 的单文件 ONNX 声纹模型（Apache-2.0，无需 HuggingFace Token）。
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..model_cache import configure_xet_cache, hub_cache_dir
from .base import Diarizer
from .clustering import (
    SR,
    build_cache_signature,
    build_windows,
    cluster_embeddings,
    extract_wave_slice,
    load_embedding_cache,
    merge_non_overlapping,
    save_embedding_cache,
)

CAMPPLUS_REPO = "csukuangfj/speaker-embedding-models"
CAMPPLUS_FILE = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
CAMPPLUS_MODEL_ID = f"{CAMPPLUS_REPO}/{CAMPPLUS_FILE}"
DEFAULT_CLUSTER_THRESHOLD = 0.54


def _mel_scale(freq: np.ndarray | float) -> np.ndarray:
    return 1127.0 * np.log(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def _build_kaldi_mel_banks(
    *,
    num_bins: int = 80,
    n_fft: int = 512,
    sample_rate: int = SR,
    low_freq: float = 20.0,
    high_freq: float = 8000.0,
) -> np.ndarray:
    num_fft_bins = n_fft // 2
    fft_bin_width = sample_rate / n_fft
    mel_low = float(_mel_scale(low_freq))
    mel_high = float(_mel_scale(high_freq))
    mel_delta = (mel_high - mel_low) / (num_bins + 1)
    banks = np.zeros((num_bins, num_fft_bins + 1), dtype=np.float32)
    freqs = np.arange(num_fft_bins + 1, dtype=np.float64) * fft_bin_width
    mels = _mel_scale(freqs)
    for b in range(num_bins):
        left = mel_low + b * mel_delta
        center = mel_low + (b + 1) * mel_delta
        right = mel_low + (b + 2) * mel_delta
        up = (mels - left) / (center - left)
        down = (right - mels) / (right - center)
        banks[b] = np.maximum(0.0, np.minimum(up, down)).astype(np.float32)
    return banks


_MEL_BANKS = _build_kaldi_mel_banks()
_POVEY_WINDOW = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(400) / 399.0)) ** 0.85


def kaldi_fbank(wav: np.ndarray) -> np.ndarray:
    """提取与 3D-Speaker CAM++ 一致的 80 维 Kaldi FBank + 句级均值归一化（CMN）。"""
    x = np.asarray(wav, dtype=np.float64)
    if len(x) < 400:
        x = np.pad(x, (0, 400 - len(x)))
    n_frames = 1 + (len(x) - 400) // 160
    idx = np.arange(400)[None, :] + np.arange(n_frames)[:, None] * 160
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)  # remove_dc_offset
    frames[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]  # preemphasis
    frames[:, 0] *= 1.0 - 0.97
    frames *= _POVEY_WINDOW[None, :]
    spec = np.fft.rfft(frames, n=512, axis=1)
    power = (spec.real**2 + spec.imag**2).astype(np.float32)
    mel = np.matmul(power, _MEL_BANKS.T)
    feat = np.log(np.maximum(mel, 1.1920929e-7))
    feat = feat - feat.mean(axis=0, keepdims=True)
    return feat.astype(np.float32)


class CampplusOnnxDiarizer(Diarizer):
    name = "campplus_onnx"
    supports_cache = True

    def __init__(
        self,
        dcfg: dict,
        *,
        model_cache_dir: str | Path | None = None,
        on_model_download: Callable[[bool], None] | None = None,
    ):
        configure_xet_cache(model_cache_dir)
        self._hub_cache = hub_cache_dir(model_cache_dir)
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("需要 onnxruntime：pip install onnxruntime") from e
        self.dcfg = dcfg
        self._on_model_download = on_model_download
        self._ort = ort
        self.session: Any = None
        self._input_name: str = ""

    def _resolve_model_path(self) -> Path:
        configured = self.dcfg.get("model_path")
        if configured:
            path = Path(configured).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"找不到指定的声纹 ONNX 模型文件：{path}")
            return path

        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        cached = try_to_load_from_cache(CAMPPLUS_REPO, CAMPPLUS_FILE, cache_dir=self._hub_cache)
        if isinstance(cached, str) and Path(cached).is_file():
            return Path(cached)

        if self._on_model_download is not None:
            self._on_model_download(True)
        try:
            downloaded = hf_hub_download(
                CAMPPLUS_REPO, filename=CAMPPLUS_FILE, cache_dir=self._hub_cache
            )
            return Path(downloaded)
        finally:
            if self._on_model_download is not None:
                self._on_model_download(False)

    def _load_session(self) -> None:
        if self.session is not None:
            return
        model_path = self._resolve_model_path()
        opts = self._ort.SessionOptions()
        threads = int(self.dcfg.get("num_threads") or min(4, os.cpu_count() or 2))
        opts.intra_op_num_threads = max(1, threads)
        opts.inter_op_num_threads = 1
        self.session = self._ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name

    def _embed_waveforms(self, wavs: list[np.ndarray]) -> np.ndarray:
        if not wavs:
            return np.empty((0, 0), dtype=np.float32)
        self._load_session()
        vecs: list[np.ndarray] = []
        for wav in wavs:
            feat = kaldi_fbank(wav)[None, :, :]
            out = self.session.run(None, {self._input_name: feat})[0]
            vec = np.asarray(out[0], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            vecs.append(vec / max(norm, 1e-9))
        return np.stack(vecs, axis=0)

    def _model_signature(self) -> str:
        configured = self.dcfg.get("model_path")
        if configured:
            return f"local:{Path(configured).expanduser().resolve()}"
        return CAMPPLUS_MODEL_ID

    def diarize(self, audio16k: np.ndarray, *, speech_spans=None, cache_path=None) -> list[dict]:
        if audio16k is None or len(audio16k) < SR:
            return []
        min_rms = float(self.dcfg.get("min_window_rms", 0.003))
        starts, ends, core_flags, mode, win_sec, hop_sec, merge_gap_sec = build_windows(
            audio16k, speech_spans, self.dcfg
        )
        audio_sha256 = hashlib.sha256(memoryview(np.ascontiguousarray(audio16k))).hexdigest()
        signature = build_cache_signature(
            model=self._model_signature(),
            mode=mode,
            win_sec=win_sec,
            hop_sec=hop_sec,
            merge_gap_sec=merge_gap_sec,
            min_rms=min_rms,
            speech_spans=speech_spans,
            audio_sha256=audio_sha256,
        )
        emb = load_embedding_cache(cache_path, signature, starts, ends)
        if emb is None:
            wavs = [extract_wave_slice(audio16k, a, b) for a, b in zip(starts, ends)]
            emb = self._embed_waveforms(wavs)
            save_embedding_cache(cache_path, emb, starts, ends, signature)
        if not starts:
            return []

        core_mask = np.asarray(core_flags, dtype=bool)
        durations = np.asarray([(e - s) / SR for s, e in zip(starts, ends)], dtype=np.float64)
        if "_cluster" in self.__dict__:
            labels = self._cluster(emb)
        else:
            labels = self._cluster(emb, core_mask=core_mask, durations=durations)

        seg_merge_gap = (
            max(hop_sec, merge_gap_sec) if mode == "sliding" else max(merge_gap_sec, 0.5)
        )
        return merge_non_overlapping(starts, ends, labels, merge_gap_sec=seg_merge_gap)

    def _cluster(
        self,
        emb: np.ndarray,
        *,
        core_mask: np.ndarray | None = None,
        durations: np.ndarray | None = None,
    ) -> np.ndarray:
        return cluster_embeddings(
            emb,
            self.dcfg,
            default_threshold=DEFAULT_CLUSTER_THRESHOLD,
            core_mask=core_mask,
            durations=durations,
        )
