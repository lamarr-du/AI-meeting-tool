"""离线单元测试（无网络、无大模型）。

运行： python -m pytest tests/test_offline.py（先安装 .[dev]）
"""

from __future__ import annotations

import math
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

from meetmin.asr.base import ASRResult, Utterance
from meetmin.asr.faster_whisper_asr import _configure_huggingface_download
from meetmin.audio import dsp
from meetmin.config import DEFAULT_CONFIG, deep_merge, load_config
from meetmin.diarization.align import assign_speakers, relabel
from meetmin.session import Session
from meetmin.summarizer.llm_client import MockLLM
from meetmin.summarizer.minutes import generate_minutes, render_transcript, split_chunks


class _TmpDir:
    """仓库内 .tmp 下的临时目录（不依赖系统 temp，兼容受限沙箱）。"""

    def __init__(self):
        base = Path(__file__).resolve().parents[1] / ".tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"test-{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True)

    def __enter__(self) -> str:
        return str(self.path)

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)
        return False


def _utts(n, spk_cycle=("说话人A", "说话人B")):
    out = []
    for i in range(n):
        out.append(
            Utterance(
                start=i * 5.0,
                end=i * 5.0 + 4.0,
                text=f"这是第{i}句测试内容，讨论排期与方案。" * 2,
                speaker=spk_cycle[i % len(spk_cycle)],
            )
        )
    return out


# ---------------- dsp ----------------
def test_resample_length_and_energy():
    t = np.arange(48000, dtype=np.float32) / 48000.0
    x = np.sin(2 * math.pi * 440 * t)
    y = dsp.resample(x, 48000, 16000)
    assert abs(len(y) - 16000) <= 2, len(y)
    assert 0.3 < dsp.rms(y) < 0.9


def test_wav_roundtrip():
    with _TmpDir() as td:
        p = Path(td) / "a.wav"
        x = (np.sin(np.linspace(0, 200, 8000)) * 0.5).astype(np.float32)
        dsp.save_wav_mono(p, x, 16000)
        y, sr = dsp.load_wav_mono(p)
        assert sr == 16000 and len(y) == len(x)
        assert np.abs(x - y).max() < 1e-3


def test_faster_whisper_decodes_wav():
    # 覆盖 faster-whisper 与 PyAV 的实际解码接口，无需下载模型。
    from faster_whisper.audio import decode_audio

    with _TmpDir() as td:
        p = Path(td) / "decode.wav"
        x = (np.sin(np.linspace(0, 200, 8000)) * 0.5).astype(np.float32)
        dsp.save_wav_mono(p, x, 16000)
        decoded = decode_audio(str(p), sampling_rate=16000)
        assert len(decoded) == len(x)
        assert np.max(np.abs(decoded - x)) < 1e-3


def test_speechbrain_model_uses_copy_and_user_cache():
    from types import ModuleType, SimpleNamespace
    from unittest.mock import patch

    from meetmin.diarization.ecapa_speechbrain import EcapaSpeechbrainDiarizer

    calls = []
    copy_strategy = object()
    speaker = ModuleType("speechbrain.inference.speaker")
    speaker.EncoderClassifier = SimpleNamespace(from_hparams=lambda **kwargs: calls.append(kwargs))
    fetching = ModuleType("speechbrain.utils.fetching")
    fetching.LocalStrategy = SimpleNamespace(COPY=copy_strategy)
    modules = {
        name: ModuleType(name)
        for name in ("torch", "speechbrain", "speechbrain.inference", "speechbrain.utils")
    }
    modules.update(
        {"speechbrain.inference.speaker": speaker, "speechbrain.utils.fetching": fetching}
    )
    with (
        _TmpDir() as td,
        patch.dict(sys.modules, modules),
        patch.object(Path, "home", return_value=Path(td)),
    ):
        EcapaSpeechbrainDiarizer({})._load_encoder()
        assert calls[0]["local_strategy"] is copy_strategy
        assert calls[0]["savedir"] == str(Path(td) / ".cache" / "meetmin" / "speechbrain")


def test_whisper_gpu_oom_retries_cpu():
    from types import ModuleType, SimpleNamespace
    from unittest.mock import patch

    from meetmin.asr.faster_whisper_asr import FasterWhisperASR

    calls = []

    class FakeModel:
        def __init__(self, size, device, compute_type):
            calls.append((device, compute_type))
            self.device = device

        def transcribe(self, audio, **kwargs):
            if self.device == "cuda":

                def failed():
                    raise RuntimeError("CUDA out of memory")
                    yield

                return failed(), SimpleNamespace(duration=1, language="zh")
            return iter([]), SimpleNamespace(duration=1, language="zh")

    module = ModuleType("faster_whisper")
    module.WhisperModel = FakeModel
    with (
        patch.dict(sys.modules, {"faster_whisper": module}),
        patch("meetmin.doctor.gpu_status", return_value=("GPU", 8192, 8192, [])),
    ):
        asr = FasterWhisperASR({"device": "auto", "model_size": "small"})
        assert asr.transcribe(np.zeros(16000, dtype=np.float32)).utterances == []
    assert calls == [("cuda", "int8_float16"), ("cpu", "int8")]


def test_doctor_checks_recording_before_start():
    from unittest.mock import patch

    from meetmin.doctor import run_doctor

    with (
        patch(
            "meetmin.audio.loopback_recorder._import_pyaudio", side_effect=RuntimeError("no device")
        ),
        patch("meetmin.doctor.gpu_status", return_value=None),
    ):
        assert not run_doctor(
            {
                "asr": {"provider": "faster_whisper"},
                "diarization": {"enabled": False},
                "llm": {"provider": "mock"},
            },
            recording=True,
        )


def test_session_names_and_lookup():
    import meetmin.cli as main
    from meetmin.pipeline import Pipeline
    from meetmin.session import clean_session_name

    with _TmpDir() as td:
        pipe = Pipeline({"output_dir": td})
        one = pipe.new_session("weekly")
        two = pipe.new_session("weekly")
        assert one.root != two.root and one.root.is_dir() and two.root.is_dir()
        bad = pipe.new_session("../other\\evil:*?")
        assert bad.root.parent == Path(td) / "sessions"
        assert (
            ".." not in bad.session_id and "/" not in bad.session_id and "\\" not in bad.session_id
        )
        root = Path(td) / "sessions"
        for name in (
            "weekly-20200101-000000-0001",
            "weekly-20210101-000000-0001",
            "notweekly-20300101-000000-0001",
        ):
            (root / name).mkdir()
        assert (
            main._find_session(td, "weekly-20200101-000000-0001").session_id
            == "weekly-20200101-000000-0001"
        )
        assert main._find_session(td, "weekly").session_id == two.session_id
        assert main._find_session(td, "notweekly").session_id.startswith("notweekly-")
        assert main._find_session(td, "nonexistent") is None
        (root / "weekly").mkdir()
        assert main._find_session(td, "weekly").session_id == "weekly"
        assert clean_session_name("../../") == "session"


def test_record_protects_existing_audio():
    from types import SimpleNamespace
    from unittest.mock import Mock

    import meetmin.cli as main

    with _TmpDir() as td:
        session = Session(td, "old").ensure()
        session.audio_path.write_bytes(b"original")
        pipe = SimpleNamespace(recorder=Mock())
        try:
            main._do_record(pipe, session, SimpleNamespace(overwrite=False))
            assert False, "应拒绝覆盖"
        except SystemExit:
            pass
        assert session.audio_path.read_bytes() == b"original"
        pipe.recorder.assert_not_called()


def test_recorder_exclusive_creation():
    from types import SimpleNamespace
    from unittest.mock import patch

    from meetmin.audio.loopback_recorder import LoopbackRecorder

    class FakePA:
        def get_device_info_by_index(self, index):
            return {
                "defaultSampleRate": 16000,
                "maxInputChannels": 1,
                "index": index,
                "name": "test",
            }

        def terminate(self):
            pass

    with _TmpDir() as td:
        path = Path(td) / "audio.wav"
        path.write_bytes(b"original")
        with patch(
            "meetmin.audio.loopback_recorder._import_pyaudio",
            return_value=SimpleNamespace(PyAudio=FakePA),
        ):
            try:
                LoopbackRecorder(device=1).start(path)
                assert False, "应原子拒绝覆盖"
            except FileExistsError:
                pass
        assert path.read_bytes() == b"original"


def test_recording_level_and_silence_reset():
    from meetmin.audio.loopback_recorder import LoopbackRecorder

    rec = LoopbackRecorder(silence_threshold=0.003)
    rec.sample_rate_actual = 16000
    rec.channels_actual = 2
    silence = np.zeros(16000 * 2, dtype="<i2").tobytes()
    # One channel may carry sound; opposite phases must not cancel.
    loud = np.tile(np.array([10000, -10000], dtype="<i2"), 16000).tobytes()
    for _ in range(30):
        rec._update_level(silence)
    state = rec.status()
    assert state["silent_sec"] == 30 and state["level"] == 0
    rec._update_level(loud)
    state = rec.status()
    assert state["silent_sec"] == 0 and state["level"] > 0.3
    assert abs(state["active_fraction"] - 1 / 31) < 0.001
    rec._update_level(silence)
    assert rec.status()["silent_sec"] == 1


def test_recording_silence_alert_rearms():
    import io
    from types import SimpleNamespace
    from unittest.mock import patch

    import meetmin.cli as main

    class Recorder:
        def __init__(self):
            self.states = iter([30, 31, 0, 30])

        def start(self, *args, **kwargs):
            pass

        def status(self):
            return {
                "level": 0.0,
                "duration_sec": 1,
                "silent_sec": next(self.states),
                "failed": False,
            }

        def stop(self):
            return {"duration_sec": 0, "device": "test", "sample_rate": 16000, "channels": 1}

    with _TmpDir() as td:
        session = Session(td, "test").ensure()
        output = io.StringIO()
        with (
            patch("meetmin.cli.time.sleep"),
            patch("meetmin.cli.time.monotonic", side_effect=[0, 1, 2, 3, 4, 5]),
            patch("meetmin.cli.signal.signal"),
            patch("sys.stdout", output),
        ):
            main._do_record(
                SimpleNamespace(recorder=Recorder),
                session,
                SimpleNamespace(overwrite=False, max_seconds=4),
            )
        assert output.getvalue().count("连续 30 秒低电平") == 2
        assert "已录" in output.getvalue()


def test_recording_vad_silence():
    from faster_whisper.vad import get_speech_timestamps

    assert get_speech_timestamps(np.zeros(16000 * 2, dtype=np.float32), sampling_rate=16000) == []


def test_raw_asr_survives_failed_diarization():
    import json
    from types import SimpleNamespace
    from unittest.mock import patch

    from meetmin.pipeline import Pipeline

    words = [{"w": "甲", "s": 0.0, "e": 0.9}, {"w": "乙", "s": 1.1, "e": 1.9}]
    fake_asr = SimpleNamespace(
        name="faster_whisper",
        transcribe=lambda *a, **k: ASRResult(
            utterances=[Utterance(0, 2, "甲乙", words=words)],
            language="zh",
            extra={"asr_model": "fake"},
        ),
    )
    with _TmpDir() as td:
        pipe = Pipeline(
            {"output_dir": td, "asr": {"model_size": "fake"}, "diarization": {"min_group_sec": 0.1}}
        )
        session = pipe.new_session("retry")
        dsp.save_wav_mono(session.audio_path, np.ones(32000, dtype=np.float32) * 0.1, 16000)
        with patch("meetmin.asr.get_asr", return_value=fake_asr):
            raw = pipe.transcribe(session)
        path = session.root / "asr_raw.json"
        original = path.read_bytes()
        assert raw["utterances"][0]["words"] == words
        assert raw["asr"]["parameters"]["model_size"] == "fake"
        with patch("meetmin.diarization.get_diarizer", side_effect=RuntimeError("分离崩溃")):
            try:
                pipe.diarize(session)
                assert False
            except RuntimeError:
                pass
        assert path.read_bytes() == original
        assert not (session.root / "transcript.json").exists()
        fake = SimpleNamespace(
            name="fake",
            diarize=lambda *a, **k: [
                {"start": 0, "end": 1, "label": 0},
                {"start": 1, "end": 2, "label": 1},
            ],
        )
        with (
            patch("meetmin.asr.get_asr", side_effect=AssertionError("不应重跑 ASR")),
            patch("meetmin.diarization.get_diarizer", return_value=fake),
        ):
            transcript = pipe.diarize(session)
        assert len(transcript["utterances"]) == 2
        assert all("words" not in u for u in transcript["utterances"])
        assert json.loads(original)["utterances"][0]["words"] == words
        assert path.read_bytes() == original


def test_embeddings_saved_before_clustering_and_reused():
    from unittest.mock import Mock

    from meetmin.diarization.ecapa_speechbrain import EcapaSpeechbrainDiarizer

    diarizer = object.__new__(EcapaSpeechbrainDiarizer)
    diarizer.dcfg = {"window_sec": 1, "hop_sec": 0.5}
    diarizer._embed = Mock(side_effect=lambda w: np.ones((len(w), 3), dtype=np.float32))
    diarizer._cluster = Mock(side_effect=RuntimeError("聚类失败"))
    audio = np.ones(32000, dtype=np.float32) * 0.1
    with _TmpDir() as td:
        cache = Path(td) / "embeddings.npz"
        try:
            diarizer.diarize(audio, speech_spans=[(0, 2)], cache_path=cache)
            assert False
        except RuntimeError:
            pass
        with np.load(cache, allow_pickle=False) as stored:
            assert stored["embeddings"].shape == (3, 3)
            assert list(stored["starts"]) == [0, 8000, 16000]
        diarizer._cluster = lambda e: np.zeros(len(e), dtype=int)
        diarizer.dcfg["cluster_threshold"] = 0.9
        diarizer.diarize(audio, speech_spans=[(0, 2)], cache_path=cache)
        assert diarizer._embed.call_count == 1
        diarizer.diarize(audio * 2, speech_spans=[(0, 2)], cache_path=cache)
        assert diarizer._embed.call_count == 2
        diarizer.dcfg["hop_sec"] = 1
        diarizer.diarize(audio * 2, speech_spans=[(0, 2)], cache_path=cache)
        assert diarizer._embed.call_count == 3


# ---------------- config / session ----------------
def test_config_merge():
    cfg = load_config(str(Path(__file__).parents[1] / "config.yaml"))
    assert cfg["asr"]["provider"] == "faster_whisper"
    assert cfg["llm"]["model"] == "qwen-3.8-flash"
    merged = deep_merge(DEFAULT_CONFIG, {"summarizer": {"direct_threshold_chars": 10}})
    assert merged["summarizer"]["direct_threshold_chars"] == 10
    assert (
        merged["summarizer"]["max_chunk_chars"] == DEFAULT_CONFIG["summarizer"]["max_chunk_chars"]
    )


def test_huggingface_download_settings_set_before_model_import():
    import os

    previous_endpoint = os.environ.pop("HF_ENDPOINT", None)
    previous_xet = os.environ.pop("HF_HUB_DISABLE_XET", None)
    try:
        _configure_huggingface_download(
            {"hf_endpoint": "https://hf-mirror.com", "hf_disable_xet": True}
        )
        assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
        assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    finally:
        if previous_endpoint is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = previous_endpoint
        if previous_xet is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = previous_xet


def test_session_io():
    with _TmpDir() as td:
        s = Session(td, "sid-1").ensure()
        s.save_json("transcript.json", {"utterances": [{"start": 0, "end": 1, "text": "甲"}]})
        loaded = s.load_transcript()
        assert loaded["utterances"][0]["text"] == "甲"


# ---------------- diarization align ----------------
def test_assign_splits_at_speaker_change():
    segments = [
        {"start": 0.0, "end": 5.0, "label": 0},
        {"start": 5.0, "end": 10.0, "label": 1},
    ]
    words = [
        {"w": "我", "s": 0.5, "e": 1.0},
        {"w": "来说", "s": 1.0, "e": 2.5},
        {"w": "两句", "s": 2.5, "e": 4.0},
        {"w": "同", "s": 5.2, "e": 5.6},
        {"w": "意", "s": 5.6, "e": 6.2},
    ]
    u = Utterance(0.5, 6.2, "我来说两句同意", words=words)
    out = assign_speakers([u], segments, min_group_sec=0.4)
    assert len(out) == 2, [(o.speaker, o.text) for o in out]
    assert out[0].speaker == 0 and out[1].speaker == 1
    assert "同意" in out[1].text


def test_assign_smoothing_absorbs_blip():
    segments = [
        {"start": 0.0, "end": 8.0, "label": 0},
        {"start": 8.0, "end": 9.0, "label": 1},  # 极短的串扰窗
        {"start": 9.0, "end": 16.0, "label": 0},
    ]
    ws = [
        {"w": "开", "s": 1.0, "e": 1.6},
        {"w": "场", "s": 1.6, "e": 2.2},
        {"w": "串", "s": 8.1, "e": 8.35},  # 落入 label1，但组时长 0.25s < 0.4 → 应被吸收
        {"w": "结", "s": 10.0, "e": 10.6},
        {"w": "束", "s": 10.6, "e": 11.2},
    ]
    u = Utterance(1.0, 11.2, "开场串结束", words=ws)
    out = assign_speakers([u], segments, min_group_sec=0.4)
    assert len(out) == 1 and out[0].speaker == 0, [(o.speaker, o.text) for o in out]
    assert "开场" in out[0].text and "结束" in out[0].text


def test_relabel_order_of_appearance():
    us = [
        Utterance(0, 1, "a", speaker=3),
        Utterance(1, 2, "b", speaker=0),
        Utterance(2, 3, "c", speaker=3),
    ]
    speakers = relabel(us)
    assert us[0].speaker == "说话人A" and us[1].speaker == "说话人B"
    assert us[2].speaker == "说话人A" and speakers == ["说话人A", "说话人B"]


# ---------------- minutes（省 token 编排）----------------
def test_render_merge_saves_lines():
    us = [
        Utterance(0, 1, "第一句", speaker="说话人A"),
        Utterance(1, 2, "第二句", speaker="说话人A"),
        Utterance(2, 3, "第三句", speaker="说话人B"),
    ]
    merged = render_transcript(us, merge_same_speaker=True)
    assert merged.count("\n") == 1 and "第一句第二句" in merged.replace(" ", "")
    unmerged = render_transcript(us, merge_same_speaker=False)
    assert unmerged.count("\n") == 2


def test_split_chunks_bounds_and_coverage():
    text = "\n".join(f"第{i}行内容，长度大约二十个字符左右。" for i in range(200))
    chunks = split_chunks(text, 300)
    assert all(len(c) <= 300 for c in chunks)
    assert "\n".join(chunks).replace("\n", "") == text.replace("\n", "")
    long_line = "啊" * 1000
    chunks2 = split_chunks(long_line, 300)
    assert "".join(chunks2) == long_line


def test_generate_minutes_direct():
    tr = {"duration": 60, "speakers": ["说话人A"], "utterances": [u.to_dict() for u in _utts(4)]}
    cfg = deep_merge(
        DEFAULT_CONFIG, {"summarizer": {"strategy": "auto", "direct_threshold_chars": 100000}}
    )
    md, usage = generate_minutes(tr, MockLLM(), config=cfg)
    assert "mock 纪要" in md and usage.calls == 1


def test_generate_minutes_map_reduce():
    tr = {
        "duration": 3600,
        "speakers": ["说话人A", "说话人B"],
        "utterances": [u.to_dict() for u in _utts(40)],
    }
    cfg = deep_merge(
        DEFAULT_CONFIG,
        {
            "summarizer": {
                "strategy": "map_reduce",
                "max_chunk_chars": 400,
                "direct_threshold_chars": 300,
            }
        },
    )
    md, usage = generate_minutes(tr, MockLLM(), config=cfg)
    assert "mock 纪要" in md
    assert usage.calls >= 2  # map 多次 + reduce 一次
    assert usage.prompt_tokens > 0


@pytest.mark.parametrize("compact_size, expected_rounds", [(200, 1), (150, 3), (0, 1)])
def test_compaction_stops_and_preserves_points(compact_size, expected_rounds, caplog):
    from meetmin.summarizer.llm_client import Usage

    class LongLLM:
        def __init__(self):
            self.rounds = 0
            self.final_prompt = ""

        def chat(self, messages, **kwargs):
            prompt = messages[-1]["content"]
            if "【片段】" in prompt:
                result = "原" * 10000
            elif "【要点】" in prompt:
                # Each round starts with the first chunk of the preceding result.
                if "原" in prompt or "压" in prompt:
                    self.rounds += 1
                size = (
                    int(len(prompt.split("【要点】\n", 1)[1]) * 0.6)
                    if compact_size == 150
                    else compact_size
                )
                result = "压" * size
            else:
                self.final_prompt = prompt
                result = "纪要"
            return result, Usage(calls=1)

    llm = LongLLM()
    tr = {"utterances": [u.to_dict() for u in _utts(1)]}
    cfg = {
        "summarizer": {
            "strategy": "map_reduce",
            "max_chunk_chars": 200,
            "direct_threshold_chars": 200,
        }
    }
    md, usage = generate_minutes(tr, llm, config=cfg)
    assert md == "纪要"
    assert "原" in llm.final_prompt or "压" in llm.final_prompt
    assert usage.calls == llm.rounds + 2
    # Round count is measured by warnings; compact calls span multiple chunks.
    warning = (
        "3 轮上限"
        if expected_rounds == 3
        else ("未产生要点" if compact_size == 0 else "未至少缩短")
    )
    assert warning in caplog.text


def test_generate_minutes_empty_raises():
    cfg = deep_merge(DEFAULT_CONFIG, {})
    try:
        generate_minutes({"utterances": []}, MockLLM(), config=cfg)
        assert False, "应当抛错"
    except RuntimeError:
        pass
