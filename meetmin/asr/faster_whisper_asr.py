"""本地 ASR：faster-whisper（CTranslate2 后端，CPU int8 即可流畅运行，零 token 成本）。

首次运行会自动从 HuggingFace 下载模型权重到本地缓存。
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.dsp import TARGET_SR
from ..glossary import glossary_hotwords, load_terms, select_terms
from ..model_cache import configure_xet_cache, hub_cache_dir
from .base import ASRProvider, ASRResult, Utterance
from .model_download import ModelDownloadProgress, cached_model, download_model
from .profile import memory_status, select_profile
from .quality import Decoded, QualityOptions, Span, review

_VAD_PARAMETERS = {
    "min_silence_duration_ms": 600,
    "min_speech_duration_ms": 250,
    "speech_pad_ms": 100,
}
_QUALITY_VAD_PARAMETERS = {"min_silence_duration_ms": 400}


def _speech_spans(audio: np.ndarray) -> list[Span]:
    from faster_whisper.vad import get_speech_timestamps

    # 不把 Silero 的边缘补齐样本计成有声缺口。
    chunks = get_speech_timestamps(audio, **_QUALITY_VAD_PARAMETERS, speech_pad_ms=0)
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

    def __init__(
        self,
        acfg: dict,
        *,
        model_cache_dir: str | Path | None = None,
        on_model_download: Callable[[ModelDownloadProgress], None] | None = None,
    ):
        _configure_huggingface_download(acfg)
        configure_xet_cache(model_cache_dir)
        self._hub_cache = hub_cache_dir(model_cache_dir)
        from ..doctor import gpu_status

        device = acfg.get("device", "auto")
        try_cuda = device == "cuda" or (
            device == "auto" and acfg.get("profile") not in ("cpu_light", "cpu_standard")
        )
        gpu = gpu_status(acfg.get("compute_type") or "int8_float16") if try_cuda else None
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("缺少依赖 faster-whisper，请先: pip install faster-whisper") from e
        self.acfg = acfg
        self._model_class = WhisperModel
        self._on_model_download = on_model_download
        self._profile = select_profile(
            acfg,
            memory=memory_status(),
            gpu=gpu,
        )
        self._size = self._profile.model
        self._device = self._profile.device
        self._compute = self._profile.compute_type
        self._used_fallback = False
        self._fallbacks: list[dict[str, str]] = []
        self._load_with_fallback()

    def _load_with_fallback(self) -> None:
        try:
            while True:
                source = self._model_source()
                try:
                    self.model = self._model_class(
                        source,
                        device=self._device,
                        compute_type=self._compute,
                        download_root=str(self._hub_cache),
                        **({"local_files_only": True} if self.acfg.get("local_files_only") else {}),
                    )
                    return
                except (OSError, RuntimeError) as error:
                    failure = self._cuda_failure(error)
                    if failure is None:
                        raise
                self._advance_fallback(*failure)
        except Exception as e:
            if self.acfg.get("local_files_only"):
                raise RuntimeError(f"无法加载本地 Whisper 模型（{self._size}）：{e}") from e
            raise RuntimeError(
                f"加载/下载 whisper 模型失败（{self._size}）：{e}\n"
                "  · 首次运行需联网下载权重；国内网络可在 config.yaml 设 asr.hf_endpoint: https://hf-mirror.com\n"
                "  · 已默认关闭 Hugging Face Xet 下载；如需启用可设 asr.hf_disable_xet: false\n"
                "  · 或先手动下载模型，把 asr.model_size 指向本地模型目录"
            ) from e

    def _model_source(self) -> str:
        source = self._size
        if self.acfg.get("local_files_only"):
            path = cached_model(source, cache_dir=self._hub_cache)
            if path is None:
                raise FileNotFoundError("请在录音前准备完整模型，含本地 tokenizer.json")
            return str(path)
        path = Path(source).expanduser()
        if path.is_dir():
            return str(path.resolve())
        if self._on_model_download is not None:
            return download_model(source, self._on_model_download, cache_dir=self._hub_cache)
        return source

    def _cuda_failure(self, error: Exception) -> tuple[str, bool] | None:
        if self._device != "cuda":
            return None
        message = str(error).lower()
        if any(
            marker in message
            for marker in ("out of memory", "cuda_error_out_of_memory", "status_alloc_failed")
        ):
            return str(error), True
        if self.acfg.get("device", "auto") == "auto" and any(
            marker in message
            for marker in (
                "cuda failed with error",
                "cuda driver version",
                "cuda initialization",
                "cudnn_status_",
                "cublas_status_",
                "cuinit",
                "no cuda-capable device",
                "requested int8_float16 compute type",
                "cublas64_",
                "cublaslt64_",
                "cudnn64_",
                "libcublas",
                "libcudnn",
            )
        ):
            return str(error), False
        return None

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model

    def _advance_fallback(self, reason: str, oom: bool) -> None:
        previous = f"{self._size} / {self._device} / {self._compute}"
        self.close()
        if (
            oom
            and self.acfg.get("device", "auto") == "auto"
            and not self.acfg.get("model_size")
            and self._size != "small"
        ):
            self._size = "small"
        else:
            self._size = self._profile.cpu_fallback_model
            self._device, self._compute = "cpu", "int8"
        self._used_fallback |= oom
        target = f"{self._size} / {self._device} / {self._compute}"
        self._fallbacks.append({"from": previous, "to": target, "reason": reason})
        logging.getLogger("meetmin").warning("%s → %s：%s", previous, target, reason)

    def _apply_glossary(
        self, hotwords: str | None, language: str | None
    ) -> tuple[str | None, dict]:
        config = self.acfg.get("glossary", {})
        if not isinstance(config, dict):
            raise ValueError("asr.glossary 必须是映射")
        path = config.get("path")
        if path is None:
            return hotwords, {}
        if not isinstance(path, str) or not path.strip():
            raise ValueError("asr.glossary.path 必须是非空路径或 null")
        priority = config.get("terms")
        if priority is not None and (
            not isinstance(priority, list) or any(not isinstance(name, str) for name in priority)
        ):
            raise ValueError("asr.glossary.terms 必须是字符串列表或 null")
        max_terms = config.get("max_terms", 20)
        if type(max_terms) is not int or max_terms <= 0:
            raise ValueError("asr.glossary.max_terms 必须是正整数")
        source = Path(path).expanduser()
        terms = select_terms(load_terms(source), priority)
        tokenizer = self.model.hf_tokenizer
        hotwords, selected = glossary_hotwords(
            terms,
            existing=hotwords,
            language=language,
            token_count=lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
            token_limit=self.model.max_length // 2 - 1,
            max_terms=max_terms,
        )
        return hotwords, {"path": str(source.resolve()), "terms": selected}

    def transcribe(self, audio, *, file_path=None, progress_cb=None, **kwargs):
        while True:
            try:
                return self._transcribe(
                    audio, file_path=file_path, progress_cb=progress_cb, **kwargs
                )
            except (OSError, RuntimeError) as error:
                failure = self._cuda_failure(error)
                if failure is None:
                    raise
            self._advance_fallback(*failure)
            self._load_with_fallback()

    def _decode_options(self) -> tuple[dict[str, Any], dict, dict[str, Any]]:
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
        effective_hotwords, glossary = self._apply_glossary(effective_hotwords, language)
        words_enabled = bool(self.acfg.get("word_timestamps", True))
        decode_guards = _decode_guards(self.acfg, words_enabled)
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
        return options, glossary, decode_guards

    def build_stream_options(self) -> dict[str, Any]:
        options, _, _ = self._decode_options()
        return {**options, "vad_filter": False, "temperature": 0.0}

    def decode_window(
        self, audio: np.ndarray, offset: float, *, options: dict[str, Any]
    ) -> list[Decoded]:
        while True:
            try:
                segments, _ = self.model.transcribe(audio, **options)
                return [_decoded(segment, offset) for segment in segments if segment.text.strip()]
            except (OSError, RuntimeError) as error:
                failure = self._cuda_failure(error)
                if failure is None:
                    raise
            self._advance_fallback(*failure)
            self._load_with_fallback()
            # 热词依赖新模型的 tokenizer，重跑参数继续沿用调用方的选择。
            options["hotwords"] = self.build_stream_options()["hotwords"]

    def stream_metadata(self) -> dict:
        return {
            "asr_model": self._size,
            "effective_device": self._device,
            "effective_compute_type": self._compute,
            "fallbacks": list(self._fallbacks),
            "profile": asdict(self._profile),
        }

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

        options, glossary, decode_guards = self._decode_options()
        quality_options = QualityOptions.from_config(self.acfg.get("quality", {}))
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
            "language": detected_language or options["language"],
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
                "asr_model": getattr(self, "_size", None) or self.acfg.get("model_size") or "small",
                **(
                    {
                        "profile": asdict(self._profile),
                        "cuda_oom_fallback": self._used_fallback,
                        "fallbacks": list(self._fallbacks),
                    }
                    if hasattr(self, "_profile")
                    else {}
                ),
                "effective_device": self._device,
                "effective_compute_type": self._compute,
                "quality": quality_report,
                "parameters": {
                    "condition_on_previous_text": options["condition_on_previous_text"],
                    "hotwords": options["hotwords"],
                    **({"glossary": glossary} if glossary else {}),
                    **decode_guards,
                    "quality": asdict(quality_options),
                },
            },
        )
