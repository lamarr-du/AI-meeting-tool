"""本地 ASR：faster-whisper（CTranslate2 后端，CPU int8 即可流畅运行，零 token 成本）。

首次运行会自动从 HuggingFace 下载模型权重到本地缓存。
"""
from __future__ import annotations

import os
import logging
from typing import Any

import numpy as np

from .base import ASRProvider, ASRResult, Utterance


def _configure_huggingface_download(acfg: dict) -> None:
    """Apply download settings before Hugging Face client modules are imported."""
    endpoint = acfg.get("hf_endpoint")
    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", str(endpoint))
    if bool(acfg.get("hf_disable_xet", True)):
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


class FasterWhisperASR(ASRProvider):
    name = "faster_whisper"

    def __init__(self, acfg: dict):
        _configure_huggingface_download(acfg)
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("缺少依赖 faster-whisper，请先: pip install faster-whisper") from e
        self.acfg = acfg
        size = acfg.get("model_size", "small")
        self._model_class = WhisperModel
        self._size = size
        self._device = acfg.get("device", "auto")
        self._compute = acfg.get("compute_type", "int8")
        if self._device == "auto":
            from ..doctor import gpu_status
            gpu = gpu_status()
            # 保守预算；未知模型一律走 CPU。保留显式 compute_type。
            budget = {"tiny": 2048, "base": 2048, "small": 3072, "medium": 5120,
                      "large-v3-turbo": 6144, "large-v3": 8192}.get(str(size))
            self._device = "cuda" if gpu and not gpu[3] and budget and gpu[1] >= budget else "cpu"
            if self._device == "cuda" and self._compute == "int8":
                self._compute = "int8_float16"
        try:
            try:
                self.model = WhisperModel(size, device=self._device, compute_type=self._compute)
            except Exception as e:
                if self._device != "cuda" or not self._is_oom(e):
                    raise
                logging.getLogger("meetmin").warning("GPU 显存不足，回退 CPU int8：%s", e)
                self._device, self._compute = "cpu", "int8"
                self.model = WhisperModel(size, device="cpu", compute_type="int8")
        except Exception as e:
            raise RuntimeError(
                f"加载/下载 whisper 模型失败（{size}）：{e}\n"
                "  · 首次运行需联网下载权重；国内网络可在 config.yaml 设 asr.hf_endpoint: https://hf-mirror.com\n"
                "  · 已默认关闭 Hugging Face Xet 下载；如需启用可设 asr.hf_disable_xet: false\n"
                "  · 或先手动下载模型，把 asr.model_size 指向本地模型目录"
            ) from e

    @staticmethod
    def _is_oom(error: Exception) -> bool:
        message = str(error).lower()
        return "out of memory" in message or "cuda_error_out_of_memory" in message

    def transcribe(self, audio, *, file_path=None, progress_cb=None, **kwargs):
        try:
            return self._transcribe(audio, file_path=file_path, progress_cb=progress_cb, **kwargs)
        except Exception as e:
            if self._device != "cuda" or not self._is_oom(e):
                raise
            logging.getLogger("meetmin").warning("GPU 显存不足，回退 CPU int8 重试：%s", e)
            del self.model
            self._device, self._compute = "cpu", "int8"
            self.model = self._model_class(self._size, device="cpu", compute_type="int8")
            return self._transcribe(audio, file_path=file_path, progress_cb=progress_cb, **kwargs)

    def _transcribe(
        self,
        audio: np.ndarray | None,
        *,
        file_path: str | None = None,
        progress_cb: Any = None,
        **kwargs: Any,
    ) -> ASRResult:
        if audio is None:
            if not file_path:
                raise ValueError("需要 audio（16k float32 数组）或 file_path 之一")
            from ..audio.dsp import to_mono_16k

            audio = to_mono_16k(file_path)

        segments, info = self.model.transcribe(
            audio,
            language=self.acfg.get("language") or None,
            beam_size=int(self.acfg.get("beam_size", 1)),
            word_timestamps=bool(self.acfg.get("word_timestamps", True)),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
        )
        utterances: list[Utterance] = []
        total = float(getattr(info, "duration", 0.0) or (len(audio) / 16000.0))
        for seg in segments:
            words = [
                {"w": w.word, "s": round(float(w.start), 3), "e": round(float(w.end), 3)}
                for w in (seg.words or [])
            ]
            utterances.append(
                Utterance(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text.strip(),
                    words=words or None,
                )
            )
            if progress_cb:
                progress_cb(float(seg.end), total)
        return ASRResult(
            utterances=utterances,
            language=getattr(info, "language", None),
            audio_16k=audio,
            extra={"asr_model": self.acfg.get("model_size", "small"), "effective_device": self._device, "effective_compute_type": self._compute},
        )
