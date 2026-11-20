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
from .base import ASRProvider, ASRResult, Utterance
from .model_download import ModelDownloadProgress, download_model
from .profile import memory_status, select_profile
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

    def __init__(
        self,
        acfg: dict,
        *,
        on_model_download: Callable[[ModelDownloadProgress], None] | None = None,
    ):
        _configure_huggingface_download(acfg)
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("缺少依赖 faster-whisper，请先: pip install faster-whisper") from e
        self.acfg = acfg
        self._model_class = WhisperModel
        self._on_model_download = on_model_download
        from ..doctor import gpu_status

        self._profile = select_profile(
            acfg,
            memory=memory_status(),
            gpu=gpu_status() if acfg.get("device", "auto") == "auto" else None,
        )
        self._size = self._profile.model
        self._device = self._profile.device
        self._compute = self._profile.compute_type
        self._used_fallback = False
        try:
            try:
                self.model = self._load_model()
            except Exception as e:
                if self._device != "cuda" or not self._is_oom(e):
                    raise
                reason = str(e)
            else:
                return
            self._fallback_to_cpu(reason)
        except Exception as e:
            raise RuntimeError(
                f"加载/下载 whisper 模型失败（{self._size}）：{e}\n"
                "  · 首次运行需联网下载权重；国内网络可在 config.yaml 设 asr.hf_endpoint: https://hf-mirror.com\n"
                "  · 已默认关闭 Hugging Face Xet 下载；如需启用可设 asr.hf_disable_xet: false\n"
                "  · 或先手动下载模型，把 asr.model_size 指向本地模型目录"
            ) from e

    def _load_model(self):
        source = self._size
        if self._on_model_download is not None:
            path = Path(source).expanduser()
            source = str(path) if path.is_dir() else download_model(source, self._on_model_download)
        return self._model_class(source, device=self._device, compute_type=self._compute)

    @staticmethod
    def _is_oom(error: Exception) -> bool:
        message = str(error).lower()
        return "out of memory" in message or "cuda_error_out_of_memory" in message

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model

    def _fallback_to_cpu(self, reason: str) -> None:
        logging.getLogger("meetmin").warning("GPU 显存不足，回退 CPU int8：%s", reason)
        self.close()
        self._size = self._profile.cpu_fallback_model
        self._device, self._compute = "cpu", "int8"
        self._used_fallback = True
        self.model = self._load_model()

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
        try:
            return self._transcribe(audio, file_path=file_path, progress_cb=progress_cb, **kwargs)
        except Exception as e:
            if self._device != "cuda" or not self._is_oom(e):
                raise
            reason = str(e)
        self._fallback_to_cpu(reason)
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
        effective_hotwords, glossary = self._apply_glossary(effective_hotwords, language)
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
                "asr_model": getattr(self, "_size", None) or self.acfg.get("model_size") or "small",
                **(
                    {"profile": asdict(self._profile), "cuda_oom_fallback": self._used_fallback}
                    if hasattr(self, "_profile")
                    else {}
                ),
                "effective_device": self._device,
                "effective_compute_type": self._compute,
                "quality": quality_report,
                "parameters": {
                    "condition_on_previous_text": condition_on_previous_text,
                    "hotwords": effective_hotwords,
                    **({"glossary": glossary} if glossary else {}),
                    **decode_guards,
                    "quality": asdict(quality_options),
                },
            },
        )
