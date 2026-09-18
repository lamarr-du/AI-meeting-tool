"""说话人分离（diarization）工厂：不可用时自动降级为单说话人。"""
from __future__ import annotations

import logging

from .base import Diarizer  # noqa: F401

logger = logging.getLogger("meetmin")


def get_diarizer(cfg: dict) -> Diarizer:
    d = cfg.get("diarization", {}) or {}
    provider = str(d.get("provider", "ecapa_speechbrain")).lower()
    if not d.get("enabled", True) or provider == "single":
        from .single_speaker import SingleSpeakerDiarizer

        return SingleSpeakerDiarizer()
    if provider in ("ecapa_speechbrain", "ecapa", "speechbrain"):
        try:
            from .ecapa_speechbrain import EcapaSpeechbrainDiarizer

            return EcapaSpeechbrainDiarizer(d)
        except ImportError as e:
            logger.warning(
                "说话人分离不可用（%s）。已降级为单说话人；"
                "启用请执行: pip install -r requirements-optional.txt",
                e,
            )
            from .single_speaker import SingleSpeakerDiarizer

            return SingleSpeakerDiarizer()
    raise ValueError(f"未知 diarization provider: {provider}（可选 ecapa_speechbrain | single）")
