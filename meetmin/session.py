"""一次会议 = 一个会话目录：data/sessions/<session_id>/。

产物：
  audio.wav        内录原始存档（设备原生采样率）
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

    # ---- JSON ----
    def save_json(self, name: str, obj) -> Path:
        self.ensure()
        p = self.root / name
        temporary = self.root / ("." + name + "." + secrets.token_hex(8) + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=1)
                f.flush()
                os.fsync(f.fileno())
            temporary.replace(p)
        finally:
            temporary.unlink(missing_ok=True)
        return p

    def save_transcript(self, transcript: dict) -> Path:
        return self.save_json("transcript.json", transcript)

    def load_transcript(self) -> dict:
        return json.loads(self.transcript_json.read_text(encoding="utf-8"))
