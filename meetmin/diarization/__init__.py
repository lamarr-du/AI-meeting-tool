"""说话人分离（diarization）工厂。"""

from __future__ import annotations

from collections.abc import Callable

from .base import Diarizer  # noqa: F401


def get_diarizer(cfg: dict, *, on_model_download: Callable[[bool], None] | None = None) -> Diarizer:
    d = cfg.get("diarization", {}) or {}
    provider = str(d.get("provider", "ecapa_speechbrain")).lower()
    if not d.get("enabled", True) or provider == "single":
        from .single_speaker import SingleSpeakerDiarizer

        return SingleSpeakerDiarizer()
    if provider in ("ecapa_speechbrain", "ecapa", "speechbrain"):
        from .ecapa_speechbrain import EcapaSpeechbrainDiarizer

        return EcapaSpeechbrainDiarizer(
            d, model_cache_dir=cfg.get("model_cache_dir"), on_model_download=on_model_download
        )
    raise ValueError(f"未知 diarization provider: {provider}（可选 ecapa_speechbrain | single）")
