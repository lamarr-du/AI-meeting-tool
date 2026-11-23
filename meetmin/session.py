"""一次会议 = 一个会话目录：data/sessions/<session_id>/。

产物：
  audio.wav        内录存档（16 kHz 单声道 16-bit PCM WAV）
  recording.json  双轨录音指向包含 audio.wav 与 microphone.wav 的同一批次
  asr_raw.json     原始转写（含词级时间戳与模型参数）
  embeddings.npz   ECAPA 窗口声纹嵌入及时间信息
  transcript.json  转写+分离结果（结构化，供纪要/增量处理）
  transcript.md    人类可读转写
  minutes.md       AI 会议纪要
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import secrets
import tempfile
from pathlib import Path


def clean_session_name(name: str) -> str:
    """将名称限制为跨平台安全的单个目录名前缀。"""
    name = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "-", str(name))
    name = re.sub(r"\.{2,}", "-", name).strip(" .-").rstrip(" .")
    return name or "session"


def new_session_id(prefix: str | None = None) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S%f") + "-" + secrets.token_hex(2)
    return (
        f"{clean_session_name(prefix)[:100].rstrip(' .-') or 'session'}-{stamp}"
        if prefix
        else stamp
    )


def write_json_safely(path: Path, obj) -> Path:
    temporary = path.with_name("." + path.name + "." + secrets.token_hex(8) + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class Session:
    def __init__(self, output_dir: str | Path, session_id: str):
        if not session_id or session_id in (".", "..") or "/" in session_id or "\\" in session_id:
            raise ValueError("无效的会话 ID")
        self.session_id = session_id
        self.root = Path(output_dir) / "sessions" / session_id

    def ensure(self) -> "Session":
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def audio_path(self) -> Path:
        return self.root / "audio.wav"

    @property
    def recording_manifest(self) -> Path:
        return self.root / "recording.json"

    def recording_paths(self) -> tuple[Path, Path | None]:
        try:
            data = json.loads(self.recording_manifest.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self.audio_path, None
        take = data.get("take") if isinstance(data, dict) else None
        if not isinstance(take, str) or not re.fullmatch(r"audio_parts-[0-9a-f]{32}", take):
            raise ValueError("双轨录音的分段目录名无效")
        folder = self.root / take
        audio, microphone = folder / "audio.wav", folder / "microphone.wav"
        if not audio.is_file() or not microphone.is_file():
            raise FileNotFoundError(f"双轨录音缺少 WAV 文件：{folder}")
        return audio, microphone

    def publish_dual_recording(self, take: Path) -> None:
        self.save_json("recording.json", {"take": take.name})

    def _require(self, name: str) -> Path:
        p = self.root / name
        if not p.exists():
            raise FileNotFoundError(
                f"会话 {self.session_id} 缺少 {name}；请先执行相应步骤（record/transcribe）。"
            )
        return p

    @property
    def transcript_json(self) -> Path:
        return self._require("transcript.json")

    @property
    def transcript_md(self) -> Path:
        return self.root / "transcript.md"

    @property
    def minutes_md(self) -> Path:
        return self.root / "minutes.md"

    # ---- 文本 ----
    def save_text(self, name: str, text: str) -> Path:
        self.ensure()
        p = self.root / name
        p.write_text(text, encoding="utf-8")
        return p

    def save_text_safely(self, name: str, text: str, *, overwrite: bool = False) -> Path:
        """先写临时文件，完成后发布；默认不替换已有文件。"""
        self.ensure()
        target = self.root / name
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.root, prefix=".meetmin-", delete=False
            ) as f:
                temporary = Path(f.name)
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            if overwrite:
                os.replace(temporary, target)
            else:
                os.link(temporary, target)
                temporary.unlink()
            temporary = None
            return target
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    # ---- JSON ----
    def save_json(self, name: str, obj) -> Path:
        self.ensure()
        return write_json_safely(self.root / name, obj)

    def save_transcript(self, transcript: dict) -> Path:
        return self.save_json("transcript.json", transcript)

    def load_transcript(self) -> dict:
        return json.loads(self.transcript_json.read_text(encoding="utf-8"))
