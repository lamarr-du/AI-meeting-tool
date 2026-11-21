"""桌面会话与后台工作，独立于 Qt，沿用 CLI 的会话产物。"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from threading import Event

from ..asr.model_download import ModelDownloadProgress
from ..audio.segments import SessionLock
from ..audio.session_recorder import SessionRecorder, recover_recording
from ..pipeline import Pipeline
from ..session import Session


def metadata(session: Session) -> dict:
    try:
        value = json.loads((session.root / "meeting.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def history(output_dir: str) -> list[Session]:
    root = Path(output_dir) / "sessions"
    if not root.exists():
        return []
    entries = [p for p in root.iterdir() if not p.is_symlink() and p.is_dir()]
    return [
        Session(output_dir, p.name)
        for p in sorted(entries, key=lambda p: p.stat().st_mtime, reverse=True)
    ]


def import_source(session: Session, source: Path) -> Path:
    """复制到会话目录，拷贝完成后再发布，原文件移动后仍可以重试。"""
    target = session.root / f"imported{source.suffix.lower()}"
    temp_path = None
    try:
        with (
            source.open("rb") as src,
            tempfile.NamedTemporaryFile(dir=session.root, delete=False) as dst,
        ):
            temp_path = Path(dst.name)
            shutil.copyfileobj(src, dst)
        temp_path.replace(target)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    session.save_json("meeting.json", {**metadata(session), "source": target.name})
    return target


def available_audio(session: Session) -> tuple[bool, Path | None]:
    """录音返回 (True, None) 让 Pipeline 自行读取包括麦克风在内的两轨。"""
    audio, _ = session.recording_paths()
    if audio.is_file():
        return True, None
    source = metadata(session).get("source")
    if isinstance(source, str) and Path(source).name == source and source not in (".", ".."):
        path = session.root / source
        if path.is_file():
            return True, path
    try:
        raw = json.loads((session.root / "asr_raw.json").read_text(encoding="utf-8"))
        path = Path(raw["audio_path"])
        if path.is_file():
            return True, path
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return False, None


def recovery_sources(session: Session) -> list[Path]:
    if session.audio_path.exists() or session.recording_manifest.exists():
        return []
    return sorted(
        p for p in session.root.glob("audio_parts-*") if p.is_dir() and not p.is_symlink()
    )


def restore_recording(session: Session, take: Path) -> None:
    with SessionLock(session.root):
        if take not in recovery_sources(session):
            raise ValueError("未找到可恢复的录音分段，或该会话已有录音")
        recover_recording(session, take, None)


def record(
    recorder: SessionRecorder,
    session: Session,
    stop_requested: Event,
    stage: Callable[[str], None],
    meter: Callable[[dict], None],
) -> dict:
    recorder.start(session)
    try:
        stage("正在录音")
        while not stop_requested.is_set():
            status = recorder.status()
            meter(status)
            if status["failed"] or stop_requested.wait(0.15):
                break
    finally:
        stage("正在保存录音…")
        result = recorder.stop()
    return result


def transcribe(
    pipe: Pipeline,
    session: Session,
    stage: Callable[[str], None],
    *,
    align_only: bool = False,
    on_model_download: Callable[[ModelDownloadProgress], None] | None = None,
) -> dict:
    if not align_only:
        provider = (pipe.cfg["asr"].get("provider") or "faster_whisper").lower()
        if provider not in ("faster_whisper", "faster-whisper", "whisper"):
            raise ValueError("桌面录音和导入使用本地转写，请将 asr.provider 设为 faster_whisper。")
        exists, source = available_audio(session)
        if not exists:
            raise FileNotFoundError("找不到该会话的音频文件。")
        stage("正在检查语音模型缓存和文件大小…")
        pipe.transcribe(
            session,
            audio_path=source,
            on_asr_ready=lambda: stage("正在转写音频…"),
            on_model_download=on_model_download,
        )
    stage("正在整理转写与说话人…")

    def speaker_download(active: bool) -> None:
        stage(
            "正在下载说话人识别模型，首次下载可能需要较长时间…"
            if active
            else "正在整理转写与说话人…"
        )

    return pipe.diarize(session, on_model_download=speaker_download)


def api_key_available(cfg: dict, api_key: str) -> bool:
    return bool(api_key or os.environ.get(cfg["llm"].get("api_key_env", "LLM_API_KEY")))
