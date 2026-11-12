"""ASR provider 工厂。要接入自己的语音转文字服务：新建一个 ASRProvider 子类并在此注册。"""

from __future__ import annotations

from .base import ASRProvider, ASRResult, Utterance  # noqa: F401


def get_asr(cfg: dict) -> ASRProvider:
    name = (cfg["asr"].get("provider") or "faster_whisper").lower()
    if name in ("faster_whisper", "faster-whisper", "whisper"):
        from .faster_whisper_asr import FasterWhisperASR

        return FasterWhisperASR(cfg["asr"])
    if name in ("dashscope", "paraformer"):
        from .dashscope_asr import DashScopeParaformerASR

        return DashScopeParaformerASR(cfg["dashscope_asr"])
    raise ValueError(f"未知 ASR provider: {name}（可选 faster_whisper | dashscope）")
