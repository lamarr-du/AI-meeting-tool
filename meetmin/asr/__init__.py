"""ASR provider 工厂。要接入自己的语音转文字服务：新建一个 ASRProvider 子类并在此注册。"""

from __future__ import annotations

from collections.abc import Callable

from .base import ASRProvider, ASRResult, Utterance  # noqa: F401
from .model_download import ModelDownloadProgress
from .whisper_scheduler import WhisperScheduler


def get_asr(
    cfg: dict,
    *,
    on_model_download: Callable[[ModelDownloadProgress], None] | None = None,
    scheduler: WhisperScheduler | None = None,
) -> ASRProvider:
    name = (cfg["asr"].get("provider") or "faster_whisper").lower()
    if name in ("faster_whisper", "faster-whisper", "whisper"):
        from .faster_whisper_asr import FasterWhisperASR

        return FasterWhisperASR(
            cfg["asr"],
            model_cache_dir=cfg.get("model_cache_dir"),
            on_model_download=on_model_download,
            scheduler=scheduler,
        )
    if name in ("dashscope", "paraformer"):
        from .dashscope_asr import DashScopeParaformerASR

        return DashScopeParaformerASR(cfg["dashscope_asr"])
    raise ValueError(f"未知 ASR provider: {name}（可选 faster_whisper | dashscope）")
