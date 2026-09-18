"""一次会议 = 一个会话目录：data/sessions/<session_id>/。

产物：
  audio.wav        内录原始存档（设备原生采样率）
  transcript.json  转写+分离结果（结构化，供纪要/增量处理）
  transcript.md    人类可读转写
  minutes.md       AI 会议纪要
"""
from __future__ import annotations

import datetime as dt
import json
import secrets
from pathlib import Path


def new_session_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


class Session:
    def __init__(self, output_dir: str | Path, session_id: str):
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
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    def save_transcript(self, transcript: dict) -> Path:
        return self.save_json("transcript.json", transcript)

    def load_transcript(self) -> dict:
        return json.loads(self.transcript_json.read_text(encoding="utf-8"))
