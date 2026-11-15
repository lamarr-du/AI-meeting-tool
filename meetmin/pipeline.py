"""三步流水线编排：录音(1) → 转写+分离(2) → 纪要(3)。各步可独立重跑。"""

from __future__ import annotations

import datetime as dt
import os
import time
import wave
from pathlib import Path
from typing import Any, Literal, overload

from .asr.base import ASRResult, Utterance
from .asr.punctuation import punctuate_chinese
from .audio.dsp import audio_duration, to_mono_16k
from .audio.loopback_recorder import LoopbackRecorder, MicrophoneRecorder
from .audio.session_recorder import SessionRecorder
from .diarization.align import UNKNOWN_NAME, assign_speakers, relabel
from .glossary_correction import correct_transcript
from .session import Session, new_session_id
from .summarizer.llm_client import BaseLLM, Usage, get_llm
from .summarizer.minutes import build_web_prompt, estimate_cost, generate_minutes
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
        root = Path(self.out_dir) / "sessions"
        root.mkdir(parents=True, exist_ok=True)
        for _ in range(100):
            session = Session(self.out_dir, new_session_id(session_id))
            try:
                session.root.mkdir()
                return session
            except FileExistsError:
                continue
        raise RuntimeError("无法生成唯一会话目录")

    # ---------------- (1) 录音 ----------------
    @overload
    def recorder(self, *, for_session: Literal[False] = False) -> LoopbackRecorder: ...

    @overload
    def recorder(self, *, for_session: Literal[True]) -> SessionRecorder: ...

    def recorder(self, *, for_session: bool = False) -> LoopbackRecorder | SessionRecorder:
        rcfg = self.cfg.get("recorder", {})
        options: dict[str, Any] = {
            "chunk_ms": int(rcfg.get("chunk_ms", 100)),
            "silence_threshold": float(rcfg.get("silence_threshold", 0.003)),
            "segment_seconds": float(rcfg.get("segment_seconds", 300)),
            "sync_seconds": float(rcfg.get("sync_seconds", 10)),
        }
        loopback = LoopbackRecorder(
            device=rcfg.get("device"),
            channels=rcfg.get("channels"),
            sample_rate=rcfg.get("sample_rate"),
            **options,
        )
        if not for_session:
            return loopback
        mic = None
        if rcfg.get("microphone", False):
            mic = MicrophoneRecorder(device=rcfg.get("microphone_device"), **options)
        return SessionRecorder(loopback, mic)

    # ---------------- (2) 转写 + 说话人分离 ----------------
    def transcribe(
        self, session: Session, *, audio_path: str | Path | None = None, file_url: str | None = None
    ) -> dict:
        audio, microphone = (Path(audio_path), None) if audio_path else session.recording_paths()
        if microphone is not None and self.cfg.get("asr", {}).get("provider") == "dashscope":
            raise ValueError("双轨会话需要本地 faster-whisper 分别转写两轨")
        if not audio.exists() and not file_url:
            raise FileNotFoundError(f"找不到音频：{audio}（请先 record，或用 --audio 指定文件）")

        self.progress(f"[2a] 转写：{audio}")
        t0 = time.time()
        from .asr import get_asr

        provider = get_asr(self.cfg)
        # 本地 ASR 传解码后的 16k 数组；云端（需 URL）传 None 由 provider 处理
        local_like = provider.name == "faster_whisper"

        def on_seg(pos: float, total: float) -> None:
            self.progress(
                f"  … {ts(pos)}/{ts(total)}（{pos / max(total, 1):.0%}）", throttle_sec=5.0
            )

        def transcribe_track(path: Path, url: str | None = None) -> tuple[ASRResult, float]:
            if microphone is not None:
                with wave.open(str(path), "rb") as wav:
                    if not wav.getnframes():
                        return ASRResult(), 0.0
            samples = to_mono_16k(path) if local_like else None
            result = provider.transcribe(
                samples,
                file_path=str(path),
                file_url=url,
                progress_cb=on_seg if local_like else None,
            )
            if local_like and (quality := result.extra.get("quality")):
                if issues := quality.get("issues"):
                    self.progress(
                        f"  转写体检标记 {len(issues)} 处，局部重试 {quality['attempted']} 次，"
                        f"采纳 {quality['accepted']} 处。"
                    )
                if error := quality.get("vad_error"):
                    self.progress(f"  ! 有声缺口检查失败：{error}")
            if result.audio_16k is not None:
                samples, result.audio_16k = result.audio_16k, None
            if samples is not None and len(samples):
                duration = len(samples) / 16000
            elif utterances := [u for u in result.utterances if u.text.strip()]:
                duration = utterances[-1].end
            else:
                duration = audio_duration(path) if path.exists() else 0.0
            return result, duration

        provider_name = provider.name
        try:
            result, duration = transcribe_track(audio, file_url)
            utterances = [u for u in result.utterances if u.text.strip()]
            self.progress(
                f"  {provider_name} 完成：{len(utterances)} 句，耗时 {time.time() - t0:.0f}s"
            )
            if microphone is not None:
                self.progress(f"[2a] 转写麦克风：{microphone}")
                mic_result, mic_duration = transcribe_track(microphone)
        finally:
            close = getattr(provider, "close", None)
            if close is not None:
                close()

        raw: dict[str, Any] = {
            "session_id": session.session_id,
            "audio_path": str(Path(audio).resolve())
            if Path(audio).exists()
            else (file_url or str(audio)),
            "meeting_time": (
                dt.datetime.fromtimestamp(Path(audio).stat().st_mtime)
                if Path(audio).exists()
                else dt.datetime.now()
            ).strftime("%Y-%m-%d %H:%M"),
            "duration": round(duration, 1),
            "language": result.language,
            "schema_version": 1,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "asr": {
                "provider": provider_name,
                **({k: v for k, v in result.extra.items() if k != "duration"}),
            },
            "utterances": [u.to_dict() for u in utterances],
        }
        acfg = self.cfg.get("asr", {})
        keys = (
            "profile",
            "model_size",
            "language",
            "device",
            "compute_type",
            "cuda_peak_mib",
            "beam_size",
            "word_timestamps",
        )
        raw["asr"]["parameters"] = {k: acfg[k] for k in keys if k in acfg}
        if provider_name == "faster_whisper":
            raw["asr"]["parameters"].update(result.extra.get("parameters", {}))
        if provider_name == "dashscope":
            dc = self.cfg.get("dashscope_asr", {})
            raw["asr"]["parameters"] = {
                k: dc[k] for k in ("model", "speaker_count", "language_hints") if k in dc
            }
        if microphone is not None:
            mic_utterances = [u for u in mic_result.utterances if u.text.strip()]
            for u in mic_utterances:
                u.speaker = "我"
            raw["microphone"] = {
                "audio_path": str(microphone.resolve()),
                "language": mic_result.language,
                "asr": {k: v for k, v in mic_result.extra.items() if k != "duration"},
                "utterances": [u.to_dict() for u in mic_utterances],
            }
            if not utterances and mic_result.language:
                raw["language"] = mic_result.language
            raw["duration"] = round(max(duration, mic_duration), 1)
        # 在任何分离/对齐操作前保存原始词级数据。
        session.save_json("asr_raw.json", raw)
        self.progress(f"  ✓ 原始转写已保存：{session.root / 'asr_raw.json'}")
        return raw

    def diarize(self, session: Session) -> dict:
        """从原始转写重建精简转写；绝不调用 ASR。"""
        import json

        raw = json.loads(session._require("asr_raw.json").read_text(encoding="utf-8"))
        utterances = [Utterance.from_dict(u) for u in raw["utterances"] if u["text"].strip()]
        result = ASRResult(utterances=utterances, language=raw.get("language"))
        dcfg = self.cfg.get("diarization", {}) or {}
        needs_audio = (
            dcfg.get("enabled", True)
            and dcfg.get("provider", "ecapa_speechbrain") != "single"
            and not any(u.speaker is not None for u in utterances)
            and bool(utterances)
        )
        error = None
        try:
            audio16k = None
            if needs_audio:
                source = Path(raw["audio_path"])
                if not source.is_file():
                    raise FileNotFoundError(f"分离需要原始本地音频：{source}")
                audio16k = to_mono_16k(source)
            speakers = self._diarize(result, utterances, audio16k, session=session)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.progress(f"  ! 说话人分离失败（{error}），已降级为单说话人；可稍后重跑 diarize。")
            # 对齐可能已修改句子，必须从原始数据恢复后再降级。
            utterances = [Utterance.from_dict(u) for u in raw["utterances"] if u["text"].strip()]
            for u in utterances:
                u.speaker = "说话人A"
            speakers = ["说话人A"] if utterances else []
        local_whisper = (raw.get("asr") or {}).get("provider") == "faster_whisper"
        if local_whisper:
            punctuate_chinese(utterances, raw.get("language"))
        transcript = {k: v for k, v in raw.items() if k != "schema_version"}
        if microphone := raw.get("microphone"):
            transcript["microphone"] = {k: v for k, v in microphone.items() if k != "utterances"}
            mic_utterances = [
                Utterance.from_dict(u) for u in microphone["utterances"] if u["text"].strip()
            ]
            for u in mic_utterances:
                u.speaker = "我"
            if local_whisper:
                punctuate_chinese(mic_utterances, microphone.get("language"))
            utterances.extend(mic_utterances)
            utterances.sort(key=lambda u: (u.start, u.end))
            speakers = list(
                dict.fromkeys(
                    u.speaker
                    for u in utterances
                    if isinstance(u.speaker, str) and u.speaker != UNKNOWN_NAME
                )
            )
        transcript["speakers"] = speakers
        transcript["diarization"] = dict(dcfg)
        if error is not None:
            transcript["diarization_status"] = {"status": "fallback", "error": error}
        transcript["utterances"] = [u.to_dict() for u in utterances]
        for u in transcript["utterances"]:
            u.pop("words", None)
        session.save_json("transcript.json", transcript)
        session.save_text("transcript.md", self._render_md(transcript))
        self._correct_glossary(session, transcript)
        self.progress(f"  ✓ 分离结果已保存：{session.root}")
        return transcript

    def _diarize(
        self, result: ASRResult, utterances: list[Utterance], audio16k, *, session: Session
    ) -> list[str]:
        """云端自带 speaker 时仅做名单映射；本地 ASR 按配置跑分离。"""
        dcfg = self.cfg.get("diarization", {}) or {}
        has_speaker = any(u.speaker is not None for u in utterances)
        if has_speaker:  # 云端（如 DashScope）自带分离
            return relabel(utterances)
        if not dcfg.get("enabled", True):
            for u in utterances:
                u.speaker = "全体"
            return ["全体"]
        if not utterances:
            return []
        if dcfg.get("provider") == "single":
            for u in utterances:
                u.speaker = "说话人A"
            return ["说话人A"]
        if audio16k is None:
            self.progress("  ! 未获得本地音频波形，跳过说话人分离")
            for u in utterances:
                u.speaker = UNKNOWN_NAME
            return []
        self.progress("[2b] 说话人分离中…")
        from .diarization import get_diarizer

        diarizer = get_diarizer(self.cfg)
        options: dict[str, Any] = {"speech_spans": [(u.start, u.end) for u in utterances]}
        if diarizer.name == "ecapa_speechbrain":
            options["cache_path"] = session.root / "embeddings.npz"
        segments = diarizer.diarize(audio16k, **options)
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
        status = transcript.get("diarization_status") or {}
        if correction := transcript.get("glossary_correction"):
            lines.extend(
                [
                    f"> 术语纠错 {len(correction['changes'])} 处，待核对 {len(correction['context_review'])} 处。"
                    " 原文数据见 `transcript.json`，替换记录见 `transcript_corrected.json`。",
                    "",
                ]
            )
        if microphone := transcript.get("microphone"):
            lines.insert(3, f"- 麦克风：`{microphone['audio_path']}`")
        if status.get("status") == "fallback":
            track = "内录轨" if microphone else ""
            lines.extend([f"> {track}说话人分离失败，已降级为单说话人：{status.get('error')}", ""])
        for u in transcript["utterances"]:
            t0, t1 = float(u.get("start", 0)), float(u.get("end", 0))
            lines.append(f"**{u.get('speaker', '?')}** `[{ts(t0)}-{ts(t1)}]`")
            lines.append("")
            lines.append(u["text"])
            lines.append("")
        return "\n".join(lines)

    def _correct_glossary(self, session: Session, transcript: dict) -> dict:
        glossary = (self.cfg.get("asr") or {}).get("glossary") or {}
        if not isinstance(glossary, dict):
            raise ValueError("asr.glossary 必须是映射")
        path = glossary.get("correction_path")
        if path is None:
            return transcript
        if not isinstance(path, str) or not path.strip():
            raise ValueError("asr.glossary.correction_path 必须是非空路径或 null")
        corrected = correct_transcript(transcript, Path(path).expanduser())
        session.save_json("transcript_corrected.json", corrected)
        session.save_text_safely(
            "transcript_corrected.md", self._render_md(corrected), overwrite=True
        )
        return corrected

    # ---------------- (3) AI 纪要 ----------------
    def export_web_prompt(self, session: Session, *, overwrite: bool = False) -> Path:
        transcript = self._correct_glossary(session, session.load_transcript())
        prompt = build_web_prompt(transcript, config=self.cfg)
        return session.save_text_safely("minutes_web_prompt.txt", prompt, overwrite=overwrite)

    def import_web_minutes(
        self, session: Session, source: str | Path, *, overwrite: bool = False
    ) -> Path:
        # Validate the session before accepting a result associated with it.
        session.load_transcript()
        text = Path(source).read_text(encoding="utf-8-sig").strip()
        if not text:
            raise ValueError("导入文件为空，未写入纪要。")
        content = "> 由用户从 AI 网页导入；请对照转写核对行动项和数字。\n\n" + text + "\n"
        return session.save_text_safely("minutes.md", content, overwrite=overwrite)

    def summarize(self, session: Session, *, llm: BaseLLM | None = None) -> tuple[Path, Usage]:
        transcript = self._correct_glossary(session, session.load_transcript())
        try:
            lcfg = self.cfg.get("llm", {}) or {}
            if (
                llm is None
                and lcfg.get("provider", "openai_compatible")
                in ("openai_compatible", "openai", "dashscope", "qwen")
                and not os.environ.get(lcfg.get("api_key_env", "LLM_API_KEY"))
            ):
                self.progress(
                    "未检测到 LLM API Key；无鉴权的本地接口仍可使用。"
                    f'如需使用 AI 网页：meetmin summarize -n "{session.session_id}" --export-web'
                )
            llm = llm or get_llm(self.cfg)
            self.progress(f"[3/3] 生成纪要（LLM：{llm.name}/{self.cfg['llm'].get('model')}）…")
            t0 = time.time()
            md, usage = generate_minutes(
                transcript,
                llm,
                config=self.cfg,
                progress_cb=lambda m: self.progress(m, throttle_sec=1.0),
            )
        except Exception:
            self.progress(
                f'可导出提示词手动使用 AI 网页：meetmin summarize -n "{session.session_id}" --export-web'
            )
            raise
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
        if usage.finish_status == "truncated":
            lines.append("完整性：不完整（至少一阶段输出被截断，请调整输出上限后重新生成）。")
        elif usage.finish_status == "complete":
            lines.append("完整性：接口未报告截断（内容仍需人工核对）。")
        else:
            lines.append("完整性：无法确认（接口未提供可识别的结束原因）。")
        path = session.save_text("minutes.md", "\n".join(lines))
        self.progress(f"  ✓ 纪要已写入：{path}")
        return path, usage

    # ---------------- 组合 ----------------
    def process(
        self, audio_path: str | Path, *, session: Session | None = None, file_url: str | None = None
    ) -> Session:
        session = session or self.new_session()
        self.transcribe(session, audio_path=audio_path, file_url=file_url)
        self.diarize(session)
        self.summarize(session)
        return session
