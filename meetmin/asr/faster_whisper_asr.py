"""本地 ASR：faster-whisper（CTranslate2 后端，CPU int8 即可流畅运行，零 token 成本）。

首次运行会自动从 HuggingFace 下载模型权重到本地缓存。
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import asdict
from typing import Any

import numpy as np

from ..audio.dsp import TARGET_SR
from .base import ASRProvider, ASRResult, Utterance
from .quality import Decoded, QualityOptions, Span, review

_VAD_PARAMETERS = {"min_silence_duration_ms": 400}


def _speech_spans(audio: np.ndarray) -> list[Span]:
    from faster_whisper.vad import get_speech_timestamps

    # 不把 Silero 的边缘补齐样本计成有声缺口。
    chunks = get_speech_timestamps(audio, **_VAD_PARAMETERS, speech_pad_ms=0)
    return [(chunk["start"] / TARGET_SR, chunk["end"] / TARGET_SR) for chunk in chunks]


def _decoded(segment: Any, offset: float = 0.0) -> Decoded:
    words = [
        {"w": w.word, "s": round(float(w.start) + offset, 3), "e": round(float(w.end) + offset, 3)}
        for w in (segment.words or [])
    ]
    return Decoded(
        Utterance(
            start=float(segment.start) + offset,
            end=float(segment.end) + offset,
            text=segment.text.strip(),
            words=words or None,
        ),
        compression_ratio=float(getattr(segment, "compression_ratio", 0.0)),
        avg_logprob=float(getattr(segment, "avg_logprob", 0.0)),
        no_speech_prob=float(getattr(segment, "no_speech_prob", 0.0)),
    )


def _decode_guards(config: dict, word_timestamps: bool) -> dict[str, Any]:
    ngram = config.get("no_repeat_ngram_size", 0)
    if type(ngram) is not int or ngram < 0:
        raise ValueError("asr.no_repeat_ngram_size 必须是非负整数")
    guards: dict[str, Any] = {"no_repeat_ngram_size": ngram}
    for key, default in (("repetition_penalty", 1.0), ("hallucination_silence_threshold", None)):
        value = config.get(key, default)
        if value is None and default is None:
            guards[key] = None
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not (math.isfinite(value) and value > 0)
        ):
            raise ValueError(f"asr.{key} 必须是有限的正数")
        guards[key] = value
    if guards["hallucination_silence_threshold"] is not None and not word_timestamps:
        raise ValueError("asr.hallucination_silence_threshold 需要 asr.word_timestamps: true")
    return guards


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
            budget = {
                "tiny": 2048,
                "base": 2048,
                "small": 3072,
                "medium": 5120,
                "large-v3-turbo": 6144,
                "large-v3": 8192,
            }.get(str(size))
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

        condition_on_previous_text = self.acfg.get("condition_on_previous_text", False)
        if not isinstance(condition_on_previous_text, bool):
            raise ValueError("asr.condition_on_previous_text 必须为 true 或 false")
        language = self.acfg.get("language") or None
        punctuation_prompt = self.acfg.get("punctuation_prompt")
        hotwords = self.acfg.get("hotwords")
        for key, value in (("punctuation_prompt", punctuation_prompt), ("hotwords", hotwords)):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"asr.{key} 必须是字符串或 null")
        prompts = (punctuation_prompt if language == "zh" else None, hotwords)
        effective_hotwords = (
            " ".join(prompt.strip() for prompt in prompts if prompt and prompt.strip()) or None
        )
        words_enabled = bool(self.acfg.get("word_timestamps", True))
        decode_guards = _decode_guards(self.acfg, words_enabled)
        quality_options = QualityOptions.from_config(self.acfg.get("quality", {}))
        options: dict[str, Any] = {
            "language": language,
            "beam_size": int(self.acfg.get("beam_size", 1)),
            "word_timestamps": words_enabled,
            "condition_on_previous_text": condition_on_previous_text,
            "hotwords": effective_hotwords,
            "vad_filter": True,
            "vad_parameters": _VAD_PARAMETERS,
            **decode_guards,
        }
        segments, info = self.model.transcribe(audio, **options)
        detected_language = getattr(info, "language", None)
        decoded = []
        duration = len(audio) / TARGET_SR
        total = float(getattr(info, "duration", 0.0) or duration)
        for seg in segments:
            decoded.append(_decoded(seg))
            if progress_cb:
                progress_cb(float(seg.end), total)

        retry_options = {
            **options,
            "language": detected_language or language,
            "beam_size": max(options["beam_size"], 3),
            "condition_on_previous_text": False,
            "no_repeat_ngram_size": max(decode_guards["no_repeat_ngram_size"], 4),
            "temperature": 0.0,
        }

        def retry(start: float, end: float) -> list[Decoded]:
            first, last = round(start * TARGET_SR), round(end * TARGET_SR)
            clips, _ = self.model.transcribe(audio[first:last], **retry_options)
            return [_decoded(seg, first / TARGET_SR) for seg in clips if seg.text.strip()]

        review_started = time.monotonic()
        utterances, quality_report = review(
            decoded, duration, quality_options, lambda: _speech_spans(audio), retry
        )
        if quality_options.mode != "off":
            quality_report["elapsed_sec"] = round(time.monotonic() - review_started, 3)
        if quality_options.mode == "retry":
            quality_report["retry_parameters"] = {
                key: retry_options[key]
                for key in (
                    "language",
                    "beam_size",
                    "condition_on_previous_text",
                    "no_repeat_ngram_size",
                    "temperature",
                )
            }
        return ASRResult(
            utterances=utterances,
            language=detected_language,
            audio_16k=audio,
            extra={
                "asr_model": self.acfg.get("model_size", "small"),
                "effective_device": self._device,
                "effective_compute_type": self._compute,
                "quality": quality_report,
                "parameters": {
                    "condition_on_previous_text": condition_on_previous_text,
                    "hotwords": effective_hotwords,
                    **decode_guards,
                    "quality": asdict(quality_options),
                },
            },
        )
