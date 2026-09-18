"""三步流水线编排：录音(1) → 转写+分离(2) → 纪要(3)。各步可独立重跑。"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any

from .asr.base import ASRResult, Utterance
from .audio.dsp import audio_duration, to_mono_16k
from .diarization.align import assign_speakers, relabel
from .session import Session, new_session_id
from .summarizer.llm_client import BaseLLM, Usage, get_llm
from .summarizer.minutes import estimate_cost, generate_minutes
from .textutil import ts


class Pipeline:
    def __init__(self, cfg: dict, progress: Any = None):
        self.cfg = cfg
        self.out_dir = cfg.get("output_dir", "data")
        self._progress = progress or (lambda msg: print(msg, flush=True))
        self._last_report_time = 0.0

    def progress(self, msg: str, *, throttle_sec: float = 0.0) -> None:
        if throttle_sec:
            now = time.time()
            if now - self._last_report_time < throttle_sec:
                return
            self._last_report_time = now
        self._progress(msg)

    # ---------------- (0) 会话 ----------------
    def new_session(self, session_id: str | None = None) -> Session:
        return Session(self.out_dir, session_id or new_session_id()).ensure()

    # ---------------- (1) 录音 ----------------
    def recorder(self) -> "LoopbackRecorderLike":
        from .audio.loopback_recorder import LoopbackRecorder

        rcfg = self.cfg.get("recorder", {})
        return LoopbackRecorder(
            device=rcfg.get("device"),
            channels=rcfg.get("channels"),
            sample_rate=rcfg.get("sample_rate"),
            chunk_ms=int(rcfg.get("chunk_ms", 100)),
        )

    # ---------------- (2) 转写 + 说话人分离 ----------------
    def transcribe(self, session: Session, *, audio_path: str | Path | None = None,
                   file_url: str | None = None) -> dict:
        audio = Path(audio_path) if audio_path else session.audio_path
        if not Path(audio).exists() and not file_url:
            raise FileNotFoundError(f"找不到音频：{audio}（请先 record，或用 --audio 指定文件）")

        self.progress(f"[2/3] 转写+说话人分离：{audio}")
        t0 = time.time()
        from .asr import get_asr

        provider = get_asr(self.cfg)
        # 本地 ASR 传解码后的 16k 数组；云端（需 URL）传 None 由 provider 处理
        local_like = provider.name == "faster_whisper"
        audio16k = to_mono_16k(audio) if local_like else None

        def on_seg(pos: float, total: float) -> None:
            self.progress(f"  … {ts(pos)}/{ts(total)}（{pos / max(total, 1):.0%}）", throttle_sec=5.0)

        result: ASRResult = provider.transcribe(
            audio16k, file_path=str(audio), file_url=file_url,
            progress_cb=on_seg if local_like else None,
        )
        utterances = [u for u in result.utterances if u.text.strip()]
        self.progress(f"  {provider.name} 完成：{len(utterances)} 句，耗时 {time.time() - t0:.0f}s")
        audio16k = result.audio_16k if result.audio_16k is not None else audio16k

        speakers = self._diarize(result, utterances, audio16k)
        if audio16k is not None and len(audio16k):
            duration = len(audio16k) / 16000.0
        elif utterances:
            duration = utterances[-1].end
        else:
            duration = audio_duration(audio) if Path(audio).exists() else 0.0

        transcript = {
            "session_id": session.session_id,
            "audio_path": str(Path(audio).resolve()) if Path(audio).exists() else (file_url or str(audio)),
            "meeting_time": (
                dt.datetime.fromtimestamp(Path(audio).stat().st_mtime)
                if Path(audio).exists()
                else dt.datetime.now()
            ).strftime("%Y-%m-%d %H:%M"),
            "duration": round(duration, 1),
            "language": result.language,
            "speakers": speakers,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "asr": {"provider": provider.name, **({k: v for k, v in result.extra.items() if k != "duration"})},
            "utterances": [u.to_dict() for u in utterances],
        }
        # transcript.json 里剥离词级时间戳（省体积；纪要本来也不用）
        for d in transcript["utterances"]:
            d.pop("words", None)
        session.save_json("transcript.json", transcript)
        session.save_text("transcript.md", self._render_md(transcript))
        self.progress(f"  ✓ 会话目录：{session.root}")
        return transcript

    def _diarize(self, result: ASRResult, utterances: list[Utterance], audio16k) -> list[str]:
        """云端自带 speaker 时仅做名单映射；本地 ASR 按配置跑分离。"""
        dcfg = self.cfg.get("diarization", {}) or {}
        has_speaker = any(u.speaker is not None for u in utterances)
        if has_speaker:  # 云端（如 DashScope）自带分离
            return relabel(utterances)
        if not dcfg.get("enabled", True):
            for u in utterances:
                u.speaker = "全体"
            return ["全体"]
        if audio16k is None:
            self.progress("  ! 未获得本地音频波形，跳过说话人分离")
            for u in utterances:
                u.speaker = "说话人?"
            return ["说话人?"]
        self.progress("  说话人分离中…")
        from .diarization import get_diarizer

        segments = get_diarizer(self.cfg).diarize(
            audio16k, speech_spans=[(u.start, u.end) for u in utterances]
        )
        reassigned = assign_speakers(
            utterances, segments, min_group_sec=float(dcfg.get("min_group_sec", 0.4))
        )
        utterances[:] = reassigned
        return relabel(utterances)

    @staticmethod
    def _render_md(transcript: dict) -> str:
        lines = [
            f"# 会议转写 {transcript['session_id']}",
            "",
            f"- 音频：`{transcript.get('audio_path')}`",
            f"- 时间：{transcript.get('meeting_time')}　时长：{ts(float(transcript.get('duration', 0)))}"
            f"　语种：{transcript.get('language') or '未知'}",
            f"- ASR：{transcript.get('asr', {}).get('provider')}　说话人：{'、'.join(transcript.get('speakers') or [])}",
            "",
        ]
        for u in transcript["utterances"]:
            t0, t1 = float(u.get("start", 0)), float(u.get("end", 0))
            lines.append(f"**{u.get('speaker', '?')}** `[{ts(t0)}-{ts(t1)}]`")
            lines.append("")
            lines.append(u["text"])
            lines.append("")
        return "\n".join(lines)

    # ---------------- (3) AI 纪要 ----------------
    def summarize(self, session: Session, *, llm: BaseLLM | None = None) -> tuple[Path, Usage]:
        transcript = session.load_transcript()
        llm = llm or get_llm(self.cfg)
        self.progress(f"[3/3] 生成纪要（LLM：{llm.name}/{self.cfg['llm'].get('model')}）…")
        t0 = time.time()
        md, usage = generate_minutes(
            transcript, llm, config=self.cfg,
            progress_cb=lambda m: self.progress(m, throttle_sec=1.0),
        )
        scfg = self.cfg.get("summarizer", {})
        lines = [
            f"# {transcript['session_id']} 会议纪要",
            "",
            "> 由本地转写 + LLM 自动生成；行动项与数字请与转写原文核对后再分发。",
            "",
            md,
            "",
            "---",
            f"策略 token 用量：{usage.calls} 次调用 / 输入 {usage.prompt_tokens}"
            f" / 输出 {usage.completion_tokens}"
            + ("（估算值）" if usage.estimated else "")
            + f"｜耗时 {time.time() - t0:.0f}s",
        ]
        cost = estimate_cost(usage, scfg)
        if cost is not None:
            lines.append(f"估算成本：¥{cost:.4f}（按配置单价）")
        path = session.save_text("minutes.md", "\n".join(lines))
        self.progress(f"  ✓ 纪要已写入：{path}")
        return path, usage

    # ---------------- 组合 ----------------
    def process(self, audio_path: str | Path, *, session: Session | None = None,
                file_url: str | None = None) -> Session:
        session = session or self.new_session()
        self.transcribe(session, audio_path=audio_path, file_url=file_url)
        self.summarize(session)
        return session


LoopbackRecorderLike = Any  # 协议占位：start/stop/is_recording
