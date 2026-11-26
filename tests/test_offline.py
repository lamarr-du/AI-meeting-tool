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


def test_highpass_speech_suppresses_sub_80hz_hum_and_preserves_voice_band():
    sr = 16000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    hum_50hz = 0.08 * np.sin(2 * math.pi * 50 * t)
    voice_600hz = 0.08 * np.sin(2 * math.pi * 600 * t)
    mixed = (hum_50hz + voice_600hz).astype(np.float32)

    assert dsp.speech_band_ratio(hum_50hz, sr) < 0.02
    assert dsp.speech_band_ratio(voice_600hz, sr) > 0.95

    filtered_hum = dsp.highpass_speech(hum_50hz, sr)
    filtered_mixed = dsp.highpass_speech(mixed, sr)
    assert dsp.rms(filtered_hum) < 0.01
    # 零相位滤波不应改变 600 Hz 主语音分量的相位与幅度
    assert np.max(np.abs(filtered_mixed[200:-200] - voice_600hz[200:-200])) < 0.005


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


def test_speechbrain_model_downloads_directly_to_configured_cache(monkeypatch):
    from types import ModuleType, SimpleNamespace
    from unittest.mock import patch

    import huggingface_hub

    from meetmin.diarization.ecapa_speechbrain import ECAPA_FILES, EcapaSpeechbrainDiarizer

    calls = []
    no_link = object()
    speaker = ModuleType("speechbrain.inference.speaker")
    speaker.EncoderClassifier = SimpleNamespace(from_hparams=lambda **kwargs: calls.append(kwargs))
    fetching = ModuleType("speechbrain.utils.fetching")
    fetching.LocalStrategy = SimpleNamespace(NO_LINK=no_link)
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
    ):
        monkeypatch.delenv("HF_XET_CACHE", raising=False)
        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *_, **kwargs: None)
        snapshot = Path(td) / "snapshot"
        requests = []

        def fake_download(repo, *, cache_dir, allow_patterns):
            requests.append((repo, cache_dir, allow_patterns))
            return str(snapshot)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
        events = []
        EcapaSpeechbrainDiarizer(
            {}, model_cache_dir=td, on_model_download=events.append
        )._load_encoder()
        assert events == [True, False]
        assert requests[0][0] == "speechbrain/spkrec-ecapa-voxceleb"
        assert requests[0][1] == Path(td).resolve() / "huggingface" / "hub"
        assert "embedding_model.ckpt" in requests[0][2]
        assert calls[0]["local_strategy"] is no_link
        assert calls[0]["source"] == str(snapshot)
        assert calls[0]["overrides"] == {"pretrained_path": str(snapshot)}
        snapshot.mkdir()
        for name in ECAPA_FILES:
            (snapshot / name).touch()
        monkeypatch.setattr(
            huggingface_hub,
            "try_to_load_from_cache",
            lambda *_, **kwargs: str(snapshot / ECAPA_FILES[0]),
        )
        events.clear()
        EcapaSpeechbrainDiarizer(
            {}, model_cache_dir=td, on_model_download=events.append
        )._load_encoder()
        assert events == []
        assert len(requests) == 1


def test_whisper_gpu_oom_retries_cpu():
    from types import ModuleType, SimpleNamespace
    from unittest.mock import patch

    from meetmin.asr.cuda import CudaStatus
    from meetmin.asr.faster_whisper_asr import FasterWhisperASR

    calls = []

    class FakeModel:
        def __init__(self, size, device, compute_type, download_root):
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
        patch("meetmin.doctor.gpu_status", return_value=CudaStatus(memory=("GPU", 8192, 8192))),
    ):
        asr = FasterWhisperASR({"device": "auto", "model_size": "small", "cuda_peak_mib": 2400})
        assert asr.transcribe(np.zeros(16000, dtype=np.float32)).utterances == []
    assert calls == [("cuda", "int8_float16"), ("cpu", "int8")]


@pytest.mark.parametrize("configured, expected", [(None, False), (False, False), (True, True)])
def test_whisper_condition_on_previous_text(configured, expected):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from meetmin.asr.faster_whisper_asr import FasterWhisperASR

    asr = object.__new__(FasterWhisperASR)
    asr.acfg = {} if configured is None else {"condition_on_previous_text": configured}
    asr._device, asr._compute = "cpu", "int8"
    asr.model = Mock()
    asr.model.transcribe.return_value = (iter([]), SimpleNamespace(duration=1, language="zh"))
    result = asr.transcribe(np.zeros(16000, dtype=np.float32))
    assert asr.model.transcribe.call_args.kwargs["condition_on_previous_text"] is expected
    assert result.extra["parameters"]["condition_on_previous_text"] is expected


def test_whisper_condition_on_previous_text_rejects_non_boolean():
    from meetmin.asr.faster_whisper_asr import FasterWhisperASR

    asr = object.__new__(FasterWhisperASR)
    asr.acfg = {"condition_on_previous_text": "false"}
    asr._device = "cpu"
    with pytest.raises(ValueError, match="condition_on_previous_text"):
        asr.transcribe(np.zeros(16000, dtype=np.float32))


@pytest.mark.parametrize(
    "settings, expected",
    [
        (
            {
                "language": "zh",
                "punctuation_prompt": "示例，带标点。",
                "hotwords": " patch、DRC Stat ",
            },
            "示例，带标点。 patch、DRC Stat",
        ),
        ({"language": "zh"}, None),
        ({"language": "zh", "punctuation_prompt": None}, None),
        ({"language": None, "punctuation_prompt": "示例。", "hotwords": "patch"}, "patch"),
        ({"language": "en", "punctuation_prompt": "示例。"}, None),
    ],
)
def test_whisper_punctuation_uses_hotwords(settings, expected):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from meetmin.asr.faster_whisper_asr import FasterWhisperASR

    asr = object.__new__(FasterWhisperASR)
    asr.acfg = settings
    asr._device, asr._compute = "cpu", "int8"
    asr.model = Mock()
    asr.model.transcribe.return_value = ([], SimpleNamespace(duration=1, language="zh"))
    result = asr.transcribe(np.zeros(16000, dtype=np.float32))
    options = asr.model.transcribe.call_args.kwargs
    assert options["hotwords"] == result.extra["parameters"]["hotwords"] == expected
    assert "initial_prompt" not in options


@pytest.mark.parametrize(
    "provider, language", [("faster_whisper", "zh"), ("dashscope", "zh"), ("faster_whisper", "en")]
)
def test_diarize_formats_chinese_whisper_after_pauses(tmp_path, provider, language):
    from meetmin.pipeline import Pipeline

    pipe = Pipeline({"output_dir": tmp_path, "diarization": {"provider": "single"}})
    session = pipe.new_session("punctuation")
    source = [
        Utterance(0, 1, "先看 patch, 再看统计"),
        Utterance(1.3, 2, "合计 1,024 个"),
        Utterance(2.8, 3.5, "新版是 v1.2"),
        Utterance(4.5, 5.5, "这样可以吗？"),
    ]
    path = session.save_json(
        "asr_raw.json",
        {
            "session_id": session.session_id,
            "asr": {"provider": provider},
            "language": language,
            "utterances": [u.to_dict() for u in source],
        },
    )
    original = path.read_bytes()
    result = pipe.diarize(session)["utterances"]
    assert path.read_bytes() == original
    assert [(u["start"], u["end"]) for u in result] == [(u.start, u.end) for u in source]
    expected = [u.text for u in source]
    if (provider, language) == ("faster_whisper", "zh"):
        expected = ["先看 patch，再看统计", "合计 1,024 个。", "新版是 v1.2。", "这样可以吗？"]
    assert [u["text"] for u in result] == expected


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
    from unittest.mock import patch

    import meetmin.cli as main
    from meetmin.pipeline import Pipeline
    from meetmin.session import clean_session_name

    with _TmpDir() as td:
        pipe = Pipeline({"output_dir": td})
        with patch(
            "meetmin.pipeline.new_session_id",
            side_effect=["weekly-20990101-000000000001-aaaa", "weekly-20990101-000000000002-bbbb"],
        ):
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


def test_session_lookup_rejects_equal_latest_timestamps():
    import meetmin.cli as main

    with _TmpDir() as td:
        root = Path(td) / "sessions"
        first = "weekly-20261114-173326099747-ffff"
        second = "weekly-20261114-173326099747-0000"
        for name in (first, second, "weekly-20200101-000000-0001"):
            (root / name).mkdir(parents=True)
        with pytest.raises(SystemExit, match="请使用完整会话 ID") as exc:
            main._find_session(td, "weekly")
        assert first in str(exc.value) and second in str(exc.value)
        assert main._find_session(td, second).session_id == second


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
                "no_data_sec": 0,
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
                SimpleNamespace(recorder=lambda **_: Recorder()),
                session,
                SimpleNamespace(overwrite=False, max_seconds=4),
            )
        assert output.getvalue().count("连续 30 秒低电平") == 2
        assert "已录" in output.getvalue()


def test_recorder_tracks_missing_data_and_recovery():
    from unittest.mock import patch

    from meetmin.audio.loopback_recorder import LoopbackRecorder

    rec = LoopbackRecorder()
    rec.sample_rate_actual = 16000
    rec.channels_actual = 1
    rec._last_data_time = 100.0
    with patch("meetmin.audio.loopback_recorder.time.monotonic", return_value=111.0):
        assert rec.status()["no_data_sec"] == 11
        rec._update_level(b"")
        assert rec.status()["no_data_sec"] == 11
        rec._update_level(np.zeros(16000, dtype="<i2").tobytes())
        assert rec.status()["no_data_sec"] == 0
        assert rec.status()["duration_sec"] == 1
    with patch("meetmin.audio.loopback_recorder.time.monotonic", return_value=122.0):
        assert rec.status()["no_data_sec"] == 11


def test_recording_no_data_alert_rearms():
    import io
    from types import SimpleNamespace
    from unittest.mock import patch

    import meetmin.cli as main

    class Recorder:
        def __init__(self):
            self.states = iter([10, 11, 0.2, 10])

        def start(self, *args, **kwargs):
            pass

        def status(self):
            return {
                "level": 0.5,
                "duration_sec": 1,
                "silent_sec": 0,
                "no_data_sec": next(self.states),
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
                SimpleNamespace(recorder=lambda **_: Recorder()),
                session,
                SimpleNamespace(overwrite=False, max_seconds=4),
            )
        text = output.getvalue()
        assert text.count("连续 10 秒未收到音频数据") == 2
        assert text.count("已恢复接收音频数据") == 1
        assert "经过 00:01  已录 00:01  电平 无新数据" in text


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
            extra={"asr_model": "fake", "parameters": {"condition_on_previous_text": False}},
        ),
    )
    with _TmpDir() as td:
        pipe = Pipeline(
            {
                "output_dir": td,
                "asr": {"model_size": "fake"},
                "diarization": {"min_group_sec": 0.1},
                "llm": {},
            }
        )
        session = pipe.new_session("retry")
        dsp.save_wav_mono(session.audio_path, np.ones(32000, dtype=np.float32) * 0.1, 16000)
        with patch("meetmin.asr.get_asr", return_value=fake_asr):
            raw = pipe.transcribe(session)
        path = session.root / "asr_raw.json"
        original = path.read_bytes()
        assert raw["utterances"][0]["words"] == words
        assert raw["asr"]["parameters"]["model_size"] == "fake"
        assert raw["asr"]["parameters"]["condition_on_previous_text"] is False
        with patch("meetmin.diarization.get_diarizer", side_effect=RuntimeError("分离崩溃")):
            fallback = pipe.diarize(session)
        assert path.read_bytes() == original
        assert fallback["speakers"] == ["说话人A"]
        assert fallback["utterances"][0]["text"] == "甲乙。"
        assert fallback["diarization_status"] == {
            "status": "fallback",
            "error": "RuntimeError: 分离崩溃",
        }
        assert "分离崩溃" in (session.root / "transcript.md").read_text(encoding="utf-8")
        assert (
            json.loads((session.root / "transcript.json").read_text(encoding="utf-8"))[
                "diarization_status"
            ]
            == fallback["diarization_status"]
        )
        minutes_path, _ = pipe.summarize(session, llm=MockLLM())
        assert minutes_path.exists()
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
        assert "diarization_status" not in transcript
        assert all("words" not in u for u in transcript["utterances"])
        assert json.loads(original)["utterances"][0]["words"] == words
        assert path.read_bytes() == original


@pytest.mark.parametrize("fallback", [False, True])
def test_dual_transcribe_diarizes_only_loopback_and_preserves_overlap(
    tmp_path, monkeypatch, fallback
):
    from types import SimpleNamespace

    from meetmin.pipeline import Pipeline

    pipe = Pipeline({"output_dir": tmp_path, "llm": {"provider": "mock"}})
    session = pipe.new_session("dual")
    take = session.root / ("audio_parts-" + "a" * 32)
    loopback, mic = take / "audio.wav", take / "microphone.wav"
    dsp.save_wav_mono(loopback, np.full(48000, 0.1, dtype=np.float32), 16000)
    dsp.save_wav_mono(mic, np.full(48000, 0.3, dtype=np.float32), 16000)
    session.publish_dual_recording(take)
    calls = []

    def transcribe(audio, *, file_path, **kwargs):
        calls.append(file_path)
        if file_path == str(loopback):
            return ASRResult(
                utterances=[Utterance(0.5, 2, "远端发言"), Utterance(2.1, 2.7, "另一位发言")],
                language="zh",
            )
        assert np.mean(audio) > 0.2
        return ASRResult(
            utterances=[Utterance(1, 1.5, "本地发言", words=[{"w": "本地发言", "s": 1, "e": 1.5}])],
            language="zh",
        )

    def diarize(audio, **kwargs):
        assert np.mean(audio) < 0.2
        assert kwargs["speech_spans"] == [(0.5, 2), (2.1, 2.7)]
        if fallback:
            raise RuntimeError("分离失败")
        return [{"start": 0.5, "end": 2, "label": 1}, {"start": 2.1, "end": 2.7, "label": 2}]

    monkeypatch.setattr(
        "meetmin.asr.get_asr",
        lambda _: SimpleNamespace(name="faster_whisper", transcribe=transcribe),
    )
    monkeypatch.setattr(
        "meetmin.diarization.get_diarizer",
        lambda _: SimpleNamespace(name="fake", diarize=diarize),
    )
    raw = pipe.transcribe(session)
    assert calls == [str(loopback), str(mic)]
    assert raw["microphone"]["utterances"][0]["speaker"] == "我"
    assert raw["microphone"]["utterances"][0]["words"][0]["s"] == 1
    original = (session.root / "asr_raw.json").read_bytes()
    transcript = pipe.diarize(session)
    assert (session.root / "asr_raw.json").read_bytes() == original
    assert [u["text"] for u in transcript["utterances"]] == [
        "远端发言",
        "本地发言。",
        "另一位发言。",
    ]
    assert [u["start"] for u in transcript["utterances"]] == [0.5, 1, 2.1]
    assert [u["speaker"] for u in transcript["utterances"]] == [
        "说话人A",
        "我",
        "说话人A" if fallback else "说话人B",
    ]
    assert "utterances" not in transcript["microphone"]
    assert "**我**" in session.transcript_md.read_text(encoding="utf-8")
    minutes, _ = pipe.summarize(session, llm=MockLLM())
    assert minutes.is_file()

    pipe.cfg["asr"] = {"provider": "dashscope"}
    with pytest.raises(ValueError, match="双轨会话需要本地"):
        pipe.transcribe(session)


@pytest.mark.parametrize("failure", ["audio", "model", "alignment"])
def test_diarization_fallback_recovers_raw_utterances(failure):
    from types import SimpleNamespace
    from unittest.mock import patch

    from meetmin.pipeline import Pipeline

    with _TmpDir() as td:
        messages = []
        pipe = Pipeline({"output_dir": td, "diarization": {}}, progress=messages.append)
        session = pipe.new_session("failure")
        session.audio_path.touch()
        raw = {
            "session_id": session.session_id,
            "audio_path": str(session.audio_path),
            "utterances": [{"start": 0, "end": 1, "text": "原始内容"}],
        }
        session.save_json("asr_raw.json", raw)
        original = (session.root / "asr_raw.json").read_bytes()
        model = SimpleNamespace(name="fake", diarize=lambda *args, **kwargs: [])

        def fail_alignment(utterances, *args, **kwargs):
            utterances[0].text = "损坏"
            raise ValueError("对齐失败")

        with (
            patch(
                "meetmin.pipeline.to_mono_16k",
                side_effect=OSError("音频失败") if failure == "audio" else None,
                return_value=np.ones(10),
            ),
            patch(
                "meetmin.diarization.get_diarizer",
                side_effect=RuntimeError("模型失败") if failure == "model" else None,
                return_value=model,
            ),
            patch(
                "meetmin.pipeline.assign_speakers",
                side_effect=fail_alignment if failure == "alignment" else None,
            ),
        ):
            transcript = pipe.diarize(session)
        assert transcript["utterances"][0]["text"] == "原始内容"
        assert transcript["utterances"][0]["speaker"] == "说话人A"
        assert transcript["diarization_status"]["status"] == "fallback"
        assert any("已降级为单说话人" in message for message in messages)
        assert (session.root / "asr_raw.json").read_bytes() == original


def test_ecapa_passes_mono_batch_as_two_dimensional_audio(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock

    from meetmin.diarization.ecapa_speechbrain import EcapaSpeechbrainDiarizer

    diarizer = object.__new__(EcapaSpeechbrainDiarizer)
    diarizer._torch = SimpleNamespace(no_grad=nullcontext, from_numpy=lambda array: array)
    diarizer.encoder = Mock()
    encoded = diarizer.encoder.encode_batch.return_value
    encoded.squeeze.return_value.cpu.return_value.numpy.return_value = np.ones((2, 3))
    monkeypatch.setattr(diarizer, "_load_encoder", lambda: diarizer.encoder)
    windows = np.ones((2, 24000), dtype=np.float32)
    embeddings = diarizer._embed(windows)
    assert diarizer.encoder.encode_batch.call_args.args[0].shape == (2, 24000)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1)


def test_embeddings_saved_before_clustering_and_reused():
    from unittest.mock import Mock

    from meetmin.diarization.ecapa_speechbrain import EcapaSpeechbrainDiarizer

    diarizer = object.__new__(EcapaSpeechbrainDiarizer)
    diarizer.dcfg = {"window_mode": "sliding", "window_sec": 1, "hop_sec": 0.5}
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


def test_ecapa_speech_spans_chunking_and_variable_length_embedding():
    from unittest.mock import Mock

    from meetmin.diarization.ecapa_speechbrain import EcapaSpeechbrainDiarizer

    diarizer = object.__new__(EcapaSpeechbrainDiarizer)
    diarizer.dcfg = {
        "window_mode": "speech_spans",
        "window_sec": 2.5,
        "merge_gap_sec": 0.3,
        "min_cluster_dur_sec": 1.2,
    }
    calls: list[tuple[tuple[int, ...], object]] = []

    def fake_embed(w, wav_lens=None):
        calls.append((w.shape, None if wav_lens is None else wav_lens.tolist()))
        out = np.zeros((len(w), 3), dtype=np.float32)
        out[:, 0] = 1.0
        return out

    diarizer._embed = Mock(side_effect=fake_embed)
    diarizer._cluster = lambda e: np.zeros(len(e), dtype=int)
    audio = np.ones(16000 * 8, dtype=np.float32) * 0.1
    # (0, 2.0) 和 (2.2, 4.0) 间隔 0.2s <= 0.3s 应合并成 4.0s 并均分为两段 2.0s；
    # (5.5, 6.3) 长度 0.8s < 1.2s 应被标为非核心段，但同样提取嵌入并参与输出
    with _TmpDir() as td:
        cache = Path(td) / "embeddings.npz"
        segs = diarizer.diarize(
            audio,
            speech_spans=[(0.0, 2.0), (2.2, 4.0), (5.5, 6.3)],
            cache_path=cache,
        )
        assert len(calls) == 1
        assert calls[0][0][0] == 3
        assert calls[0][1] is not None  # 变长批次传入了 wav_lens
        assert len(segs) == 2  # [0, 4.0] 和 [5.5, 6.3]
        assert segs[0]["label"] == segs[1]["label"]
        # 改聚类参数必须复用缓存
        diarizer.dcfg["cluster_threshold"] = 0.72
        diarizer.dcfg["min_cluster_sec"] = 2.0
        diarizer.diarize(
            audio,
            speech_spans=[(0.0, 2.0), (2.2, 4.0), (5.5, 6.3)],
            cache_path=cache,
        )
        assert diarizer._embed.call_count == 1


def test_ecapa_centroid_reassignment_and_knn_outlier_filter_without_sklearn():
    from meetmin.diarization.ecapa_speechbrain import (
        _filter_knn_outliers,
        _reassign_by_centroids,
    )

    spk_a = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (4, 1))
    spk_b = np.tile(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), (4, 1))
    outlier = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    short_b = np.array([[0.1, 0.995, 0.0, 0.0]], dtype=np.float32)
    emb = np.vstack([spk_a, spk_b, outlier, short_b])
    inliers = _filter_knn_outliers(emb[:9], min_keep=2)
    assert list(inliers) == list(range(8))

    core_idx = np.arange(9, dtype=int)
    core_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2], dtype=int)
    durations = np.array([2.0] * 8 + [1.5, 0.8])
    reassigned = _reassign_by_centroids(
        emb,
        core_idx,
        core_labels,
        durations,
        min_cluster_size=3,
        min_cluster_sec=4.0,
    )
    assert set(reassigned.tolist()) == {0, 1}
    assert reassigned[9] == 1


def test_ecapa_min_cluster_filter_and_centroid_reassignment():
    from meetmin.diarization.ecapa_speechbrain import EcapaSpeechbrainDiarizer

    diarizer = object.__new__(EcapaSpeechbrainDiarizer)
    diarizer.dcfg = {
        "cluster_method": "average",
        "cluster_threshold": 0.4,
        "min_cluster_size": 3,
        "min_cluster_sec": 4.0,
    }
    # 两个主说话人（各 4 段，各 8 秒）+ 1 个孤立离群段 + 1 个非核心短段
    spk_a = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (4, 1))
    spk_b = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (4, 1))
    outlier = np.array([[0.3, 0.0, 0.9539]], dtype=np.float32)  # 更靠近 A，但独立成小簇
    short_b = np.array([[0.1, 0.995, 0.0]], dtype=np.float32)  # 非核心段，更靠近 B
    emb = np.vstack([spk_a, spk_b, outlier, short_b])
    core_mask = np.array([True] * 9 + [False])
    durations = np.array([2.0] * 8 + [1.5, 0.8])

    labels = diarizer._cluster(emb, core_mask=core_mask, durations=durations)
    assert len(set(labels.tolist())) == 2
    assert len(set(labels[:4].tolist())) == 1
    assert len(set(labels[4:8].tolist())) == 1
    assert labels[0] != labels[4]
    assert labels[8] == labels[0]  # 离群微簇被滤除并重分配到最近主簇 A
    assert labels[9] == labels[4]  # 非核心短段重分配到最近主簇 B


def test_ecapa_fixed_num_speakers_resists_outliers():
    from meetmin.diarization.ecapa_speechbrain import EcapaSpeechbrainDiarizer

    diarizer = object.__new__(EcapaSpeechbrainDiarizer)
    diarizer.dcfg = {"num_speakers": 3, "cluster_method": "auto"}
    rng = np.random.default_rng(7)
    centers = np.eye(3, 6, dtype=np.float32)
    groups = []
    for c in centers:
        pts = c + rng.normal(scale=0.05, size=(6, 6)).astype(np.float32)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        groups.append(pts)
    # 注入 2 个远离所有人的离群点
    outliers = np.array(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32
    )
    emb = np.vstack([*groups, outliers])
    labels = diarizer._cluster(
        emb, core_mask=np.ones(len(emb), dtype=bool), durations=np.full(len(emb), 2.0)
    )
    # 3 位真实说话人必须被完整分成 3 个互不相同的簇，不能被离群点占坑
    assert len(set(labels[:18].tolist())) == 3
    assert len(set(labels[0:6].tolist())) == 1
    assert len(set(labels[6:12].tolist())) == 1
    assert len(set(labels[12:18].tolist())) == 1


def test_ecapa_spectral_clustering_auto_and_fixed():
    from meetmin.diarization.ecapa_speechbrain import EcapaSpeechbrainDiarizer

    rng = np.random.default_rng(11)
    c1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    c2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    g1 = c1 + rng.normal(scale=0.05, size=(8, 4)).astype(np.float32)
    g2 = c2 + rng.normal(scale=0.05, size=(8, 4)).astype(np.float32)
    g1 /= np.linalg.norm(g1, axis=1, keepdims=True)
    g2 /= np.linalg.norm(g2, axis=1, keepdims=True)
    emb = np.vstack([g1, g2])
    durations = np.full(len(emb), 2.0)

    for num_speakers in (None, 2):
        diarizer = object.__new__(EcapaSpeechbrainDiarizer)
        diarizer.dcfg = {
            "num_speakers": num_speakers,
            "cluster_method": "spectral",
            "min_cluster_size": 2,
            "min_cluster_sec": 4.0,
        }
        labels = diarizer._cluster(
            emb, core_mask=np.ones(len(emb), dtype=bool), durations=durations
        )
        assert len(set(labels.tolist())) == 2
        assert len(set(labels[:8].tolist())) == 1
        assert len(set(labels[8:].tolist())) == 1
        assert labels[0] != labels[8]


def test_ecapa_non_overlapping_midpoint_boundary():
    from meetmin.diarization.ecapa_speechbrain import SR, _merge_non_overlapping

    # 两个相邻不同簇的 1.5s 滑窗，步长 0.75s：
    # 窗0 [0.0, 1.5] 中心 0.75；窗1 [0.75, 2.25] 中心 1.50；中点应为 1.125
    starts = [0, int(0.75 * SR), int(1.5 * SR)]
    ends = [int(1.5 * SR), int(2.25 * SR), int(3.0 * SR)]
    labels = np.array([0, 1, 1], dtype=int)
    segs = _merge_non_overlapping(starts, ends, labels, merge_gap_sec=0.75)
    assert len(segs) == 2
    assert segs[0]["start"] == 0.0
    assert segs[0]["end"] == 1.125
    assert segs[1]["start"] == 1.125
    assert segs[1]["end"] == 3.0
    assert segs[0]["end"] <= segs[1]["start"]


def test_campplus_onnx_kaldi_fbank_shape_and_cmn():
    from meetmin.diarization.campplus_onnx import kaldi_fbank

    rng = np.random.default_rng(3)
    wav = (rng.normal(scale=0.1, size=16000)).astype(np.float32)
    feat = kaldi_fbank(wav)
    assert feat.ndim == 2 and feat.shape[1] == 80
    assert feat.shape[0] == 98
    assert np.isfinite(feat).all()
    assert np.allclose(feat.mean(axis=0), 0.0, atol=1e-5)

    short_feat = kaldi_fbank(np.zeros(100, dtype=np.float32))
    assert short_feat.shape == (1, 80)


def test_campplus_onnx_downloads_to_configured_cache_and_notifies(monkeypatch):
    from types import ModuleType, SimpleNamespace

    from meetmin.diarization.campplus_onnx import (
        CAMPPLUS_FILE,
        CAMPPLUS_REPO,
        CampplusOnnxDiarizer,
    )

    with _TmpDir() as td:
        cache_root = Path(td) / "model-cache"
        expected_hub = cache_root / "huggingface" / "hub"
        cached_file = expected_hub / "downloaded.onnx"
        requests: list[tuple[str, str, str]] = []
        states: list[bool] = []
        hub = ModuleType("huggingface_hub")

        def try_cache(repo_id, filename, *, cache_dir):
            return str(cached_file) if cached_file.is_file() else None

        def download(repo_id, *, filename, cache_dir):
            requests.append((repo_id, filename, str(cache_dir)))
            cached_file.parent.mkdir(parents=True, exist_ok=True)
            cached_file.write_bytes(b"fake-onnx")
            return str(cached_file)

        hub.try_to_load_from_cache = try_cache  # type: ignore[attr-defined]
        hub.hf_hub_download = download  # type: ignore[attr-defined]

        fake_ort = SimpleNamespace(
            SessionOptions=lambda: SimpleNamespace(intra_op_num_threads=1, inter_op_num_threads=1),
            InferenceSession=lambda path, **kwargs: SimpleNamespace(
                get_inputs=lambda: [SimpleNamespace(name="x")],
                path=path,
            ),
        )
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

        diarizer = CampplusOnnxDiarizer(
            {}, model_cache_dir=cache_root, on_model_download=states.append
        )
        diarizer._load_session()
        assert requests == [(CAMPPLUS_REPO, CAMPPLUS_FILE, str(expected_hub))]
        assert states == [True, False]
        assert diarizer.session.path == str(cached_file)

        # 第二次命中缓存，不再触发下载或通知
        diarizer2 = CampplusOnnxDiarizer(
            {}, model_cache_dir=cache_root, on_model_download=states.append
        )
        diarizer2._load_session()
        assert len(requests) == 1
        assert states == [True, False]


def test_campplus_onnx_diarize_caches_embeddings_and_clusters():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from meetmin.diarization.campplus_onnx import CampplusOnnxDiarizer

    diarizer = object.__new__(CampplusOnnxDiarizer)
    diarizer.dcfg = {"window_mode": "speech_spans", "window_sec": 2.5}
    diarizer._input_name = "x"
    call_count = {"n": 0}

    def fake_run(_outputs, feed):
        call_count["n"] += 1
        assert feed["x"].ndim == 3 and feed["x"].shape[2] == 80
        # 前 3 段给说话人 A，后 3 段给说话人 B
        vec = (
            np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
            if call_count["n"] <= 3
            else np.array([[0.0, 3.0, 0.0]], dtype=np.float32)
        )
        return [vec]

    diarizer.session = SimpleNamespace(run=Mock(side_effect=fake_run))
    audio = np.ones(16000 * 14, dtype=np.float32) * 0.1
    spans = [(0.0, 6.0), (8.0, 14.0)]  # 各均分为 3 段 2.0s
    with _TmpDir() as td:
        cache = Path(td) / "embeddings.npz"
        segs = diarizer.diarize(audio, speech_spans=spans, cache_path=cache)
        assert len(segs) == 2
        assert segs[0]["label"] != segs[1]["label"]
        assert call_count["n"] == 6
        with np.load(cache, allow_pickle=False) as stored:
            assert stored["embeddings"].shape == (6, 3)
            assert np.allclose(np.linalg.norm(stored["embeddings"], axis=1), 1.0)
        # 改聚类阈值必须复用缓存
        diarizer.dcfg["cluster_threshold"] = 0.6
        segs2 = diarizer.diarize(audio, speech_spans=spans, cache_path=cache)
        assert len(segs2) == 2
        assert call_count["n"] == 6


def test_get_diarizer_defaults_to_campplus_onnx(monkeypatch):
    from types import SimpleNamespace

    from meetmin.diarization import get_diarizer

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace())
    cb = lambda active: None  # noqa: E731
    diarizer = get_diarizer({"diarization": {}}, on_model_download=cb)
    assert diarizer.name == "campplus_onnx"
    assert diarizer.supports_cache is True
    assert diarizer._on_model_download is cb


def test_align_speaker_at_point_prefers_nearest_center_on_overlap():
    from meetmin.diarization.align import speaker_at_point

    segments = [
        {"start": 0.0, "end": 1.5, "label": 0},  # 中心 0.75
        {"start": 0.75, "end": 2.25, "label": 1},  # 中心 1.50
    ]
    assert speaker_at_point(0.9, segments) == 0  # 距 0.75 更近
    assert speaker_at_point(1.4, segments) == 1  # 距 1.50 更近（修复旧逻辑总返回 0 的偏置）


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


def test_align_nearby_speaker_and_exclude_tied_or_distant_unknowns():
    segments = [{"start": 1, "end": 2, "label": 1}, {"start": 3, "end": 4, "label": 2}]
    original = [
        Utterance(0, 0.5, "近第一位"),
        Utterance(2.4, 2.6, "两侧等距"),
        Utterance(3.05, 3.5, "直接覆盖第二位"),
        Utterance(8, 9, "距离太远"),
    ]
    assigned = assign_speakers(original, segments)
    assert relabel(assigned) == ["说话人A", "说话人B"]
    assert [u.speaker for u in assigned] == ["说话人A", "未知", "说话人B", "未知"]
    assert [u.text for u in assigned] == [u.text for u in original]
    assert all(u.speaker is None for u in original)


def test_no_diarization_segments_do_not_create_speakers():
    assigned = assign_speakers([Utterance(0, 1, "未知的发言")], [])
    assert relabel(assigned) == []
    assert assigned[0].speaker == "未知"


# ---------------- minutes（省 token 编排）----------------
def test_render_merge_saves_lines():
    us = [
        Utterance(0, 1, "第一句", speaker="说话人A"),
        Utterance(1, 2, "第二句", speaker="说话人A"),
        Utterance(2, 3, "第三句", speaker="说话人B"),
    ]
    merged = render_transcript(us, merge_same_speaker=True)
    assert merged.count("\n") == 1 and "第一句，第二句" in merged.replace(" ", "")
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


# ---------------- natural sentence reconstruction & paragraphing ----------------
def test_tokenize_utterance_coalesces_whisper_bpe_and_preserves_special_tokens():
    from meetmin.asr.punctuation import tokenize_utterance

    u = Utterance(
        start=5.86,
        end=9.04,
        text="开始review patch v1.2 共1,024个",
        speaker="说话人A",
        words=[
            {"w": "开始", "s": 7.5, "e": 7.7},
            {"w": "re", "s": 7.7, "e": 8.72},
            {"w": "view", "s": 8.72, "e": 8.78},
            {"w": " patch", "s": 8.78, "e": 9.04},
        ],
    )
    spans = tokenize_utterance(u)
    assert [s.text for s in spans] == ["开", "始", "review", "patch"]
    assert spans[2].start_ms == 7700 and spans[2].end_ms == 8780
    assert spans[3].start_ms == 8780 and spans[3].end_ms == 9040

    # 无 words 时走加权插值，保留 v1.2、1,024、@、large-v3-turbo 等词元
    u2 = Utterance(0.0, 4.0, "联系@张三看v1.2的1,024条large-v3-turbo日志", speaker="说话人A")
    spans2 = tokenize_utterance(u2)
    texts = [s.text for s in spans2]
    assert "@" in texts
    assert "v1.2" in texts
    assert "1,024" in texts
    assert "large-v3-turbo" in texts

    # Whisper 将小数与扩展名拆成 '0.' + '6' 或 'Vidu.' + 'j' + 'son' 时应无损合并
    u3 = Utterance(
        start=10.0,
        end=12.0,
        text="更新了Vidu.json且距离0.6米",
        speaker="说话人A",
        words=[
            {"w": "更新了", "s": 10.0, "e": 10.4},
            {"w": "V", "s": 10.4, "e": 10.5},
            {"w": "idu.", "s": 10.5, "e": 10.7},
            {"w": "j", "s": 10.7, "e": 10.8},
            {"w": "son", "s": 10.8, "e": 11.0},
            {"w": "且距离", "s": 11.0, "e": 11.3},
            {"w": "0.", "s": 11.3, "e": 11.5},
            {"w": "6", "s": 11.5, "e": 11.7},
            {"w": "米", "s": 11.7, "e": 12.0},
        ],
    )
    spans3 = tokenize_utterance(u3)
    texts3 = [s.text for s in spans3]
    assert "Vidu.json" in texts3
    assert "0.6" in texts3


def test_reconstruct_sentences_stitches_cross_window_decimals_and_keeps_words_atomic():
    from meetmin.asr.punctuation import reconstruct_sentences

    class AggressivePeriodPunc:
        name = "aggressive"

        def predict_tags(self, tokens: list[str]) -> list[str]:
            tags = [""] * len(tokens)
            if tags:
                tags[-1] = "。"
            return tags

    # 实时滑窗切在 '0.' 与 '6米' 之间时应自动缝合为 '0.6'
    u_a = Utterance(
        0.0,
        1.0,
        "它不在0.",
        speaker="说话人A",
        words=[{"w": "它不在", "s": 0.0, "e": 0.7}, {"w": "0.", "s": 0.7, "e": 1.0}],
    )
    u_b = Utterance(
        1.05,
        2.0,
        "6米范围内",
        speaker="说话人A",
        words=[{"w": "6", "s": 1.05, "e": 1.25}, {"w": "米范围内", "s": 1.25, "e": 2.0}],
    )
    reconstructed = reconstruct_sentences(
        [u_a, u_b], language="zh", punctuator=AggressivePeriodPunc()
    )
    assert len(reconstructed) == 1
    assert "0.6 米" in reconstructed[0].text

    # 说话人边界刚好落在单个英文单词 BPE 子词中间（如 ' com' + 'ment'）时，禁止把单词劈成两半
    segments = [{"start": 0.0, "end": 1.5, "label": 0}, {"start": 1.5, "end": 3.0, "label": 1}]
    u_split = Utterance(
        0.5,
        2.8,
        "看有什么 comment 没有",
        words=[
            {"w": "看有什么", "s": 0.5, "e": 1.35},
            {"w": " com", "s": 1.38, "e": 1.49},
            {"w": "ment", "s": 1.51, "e": 1.65},
            {"w": "没有", "s": 2.1, "e": 2.8},
        ],
    )
    assigned = assign_speakers([u_split], segments, min_group_sec=0.2)
    joined_texts = [item.text for item in assigned]
    assert any("comment" in t for t in joined_texts)
    assert not any(t.endswith("com") or t.startswith("ment") for t in joined_texts)


def test_reconstruct_sentences_merges_code_switch_and_tracks_timestamps():
    from meetmin.asr.punctuation import reconstruct_sentences

    class FakePunc:
        name = "fake"

        def __init__(self):
            self.calls: list[list[str]] = []

        def predict_tags(self, tokens: list[str]) -> list[str]:
            self.calls.append(list(tokens))
            tags = [""] * len(tokens)
            for i, tok in enumerate(tokens):
                if tok == "定":
                    tags[i] = "，"
                elif tok == "慢" and i < 15:
                    tags[i] = "。"
                elif tok == "patch":
                    tags[i] = "，"
                elif tok == "了" and i == len(tokens) - 1:
                    tags[i] = "。"
            return tags

    utts = [
        Utterance(0.0, 2.0, "今天刚刚搞定", speaker="说话人A"),
        Utterance(2.0, 3.5, "但是速度有点慢", speaker="说话人A"),
        # 禁尾词 "那个" + 中英码转换 "review patch"，即使有 1.2s 停顿也应合并而非硬拆
        Utterance(3.5, 5.0, "然后那个开始慢慢的开始那个", speaker="说话人A"),
        Utterance(6.2, 7.0, "review patch", speaker="说话人A"),
        Utterance(7.0, 9.0, "已经对齐过了", speaker="说话人A"),
    ]
    punc = FakePunc()
    out = reconstruct_sentences(utts, language="zh", punctuator=punc)
    assert len(out) == 2
    assert out[0].start == 0.0 and out[0].end == 3.5
    assert out[0].text == "今天刚刚搞定，但是速度有点慢。"
    assert out[1].start == 3.5 and out[1].end == 9.0
    assert "review patch" in out[1].text
    assert out[1].polished_text is not None
    assert "开始慢慢的开始" not in out[1].polished_text


def test_reconstruct_sentences_monologue_soft_split():
    from meetmin.asr.punctuation import reconstruct_sentences

    class CommaOnlyPunc:
        name = "comma_only"

        def predict_tags(self, tokens: list[str]) -> list[str]:
            tags = [""] * len(tokens)
            if len(tokens) > 60:
                tags[58] = "，"
            return tags

    utts = [Utterance(0.0, 20.0, "测" * 95, speaker="说话人A")]
    out = reconstruct_sentences(
        utts,
        language="zh",
        punctuator=CommaOnlyPunc(),
        max_sentence_tokens=85,
        max_sentence_sec=18.0,
    )
    assert len(out) == 2
    assert out[0].text.endswith("。")
    assert out[1].text.endswith("。")
    assert out[0].start == 0.0 and out[1].end == 20.0


def test_group_paragraphs_and_render_md_merges_fragments_and_splits_on_long_pause():
    from meetmin.pipeline import Pipeline

    transcript = {
        "session_id": "demo",
        "audio_path": "a.wav",
        "meeting_time": "2026-11-21 10:00",
        "duration": 30.0,
        "language": "zh",
        "asr": {"provider": "faster_whisper"},
        "speakers": ["说话人A", "说话人B"],
        "utterances": [
            {"start": 0.0, "end": 2.0, "speaker": "说话人A", "text": "前几天网络出了点问题"},
            {"start": 2.0, "end": 4.0, "speaker": "说话人A", "text": "然后开始那个"},
            {"start": 4.0, "end": 5.0, "speaker": "说话人A", "text": "review patch"},
            # 停顿 3.0s 触发自然换段
            {"start": 8.0, "end": 10.0, "speaker": "说话人A", "text": "另外我们继续下一个议题。"},
            {"start": 10.5, "end": 13.0, "speaker": "说话人B", "text": "好的我来同步进度。"},
        ],
    }
    md = Pipeline._render_md(transcript)
    assert "**说话人A** `[00:00-00:05]`" in md
    assert "前几天网络出了点问题，然后开始那个 review patch" in md
    assert "**说话人A** `[00:08-00:10]`" in md
    assert "**说话人B** `[00:10-00:13]`" in md
    assert md.count("**说话人A**") == 2


def test_render_transcript_adds_punctuation_and_supports_polished_text():
    utts = [
        Utterance(0.0, 2.0, "网络出了点问题", speaker="说话人A", polished_text="网络出了问题"),
        Utterance(2.1, 4.0, "今天刚刚搞定。", speaker="说话人A", polished_text="今天搞定。"),
    ]
    raw_rendered = render_transcript(utts, merge_same_speaker=True, language="zh")
    assert raw_rendered == "说话人A: 网络出了点问题，今天刚刚搞定。"
    polished_rendered = render_transcript(
        utts, merge_same_speaker=True, language="zh", use_polished=True
    )
    assert polished_rendered == "说话人A: 网络出了问题，今天搞定。"


def test_punc_onnx_downloads_to_cache_and_aligns_offsets(monkeypatch):
    from types import ModuleType, SimpleNamespace
    from unittest.mock import patch

    import huggingface_hub

    from meetmin.asr.punc_onnx import (
        CT_TRANSFORMER_MODEL_FILE,
        CT_TRANSFORMER_REPO,
        FIRERED_MODEL_FILE,
        FIRERED_REPO,
        FIRERED_TOKENIZER_FILE,
        CtTransformerOnnx,
        FireRedPuncOnnx,
    )

    with _TmpDir() as td:
        hub_root = Path(td) / "cache"
        fr_onnx = Path(td) / FIRERED_MODEL_FILE
        fr_tok = Path(td) / FIRERED_TOKENIZER_FILE
        ct_onnx = Path(td) / CT_TRANSFORMER_MODEL_FILE
        fr_onnx.write_bytes(b"fake")
        fr_tok.write_text("{}", encoding="utf-8")
        ct_onnx.write_bytes(b"fake")

        downloads: list[tuple[str, str]] = []
        states: list[bool] = []

        def fake_download(repo_id, filename, *, cache_dir):
            downloads.append((repo_id, filename))
            if filename == FIRERED_MODEL_FILE:
                return str(fr_onnx)
            if filename == FIRERED_TOKENIZER_FILE:
                return str(fr_tok)
            return str(ct_onnx)

        class FakeFireRedSession:
            def __init__(self, path, opts, providers=None):
                pass

            def get_outputs(self):
                return [SimpleNamespace(name="logits")]

            def run(self, outputs, feeds):
                seq_len = feeds["input_ids"].shape[1]
                # seq_minus_1 输出（跳过 [CLS]），在最后一个词元（index seq_len - 3）输出句号 class 2
                logits = np.zeros((1, seq_len - 1, 5), dtype=np.float32)
                logits[0, :, 0] = 1.0
                if seq_len >= 4:
                    logits[0, seq_len - 3, 2] = 5.0
                return [logits]

        class FakeTokenizer:
            @classmethod
            def from_file(cls, path):
                return cls()

            def token_to_id(self, token):
                return {"[CLS]": 101, "[SEP]": 102, "[UNK]": 100}.get(token, 100)

            def encode(self, text, add_special_tokens=False):
                return SimpleNamespace(ids=[10, 11] if text == "review" else [12])

        class FakeCtSession:
            def __init__(self, path, opts, providers=None):
                pass

            def get_inputs(self):
                return [SimpleNamespace(name="inputs"), SimpleNamespace(name="text_lengths")]

            def get_outputs(self):
                return [SimpleNamespace(name="logits")]

            def get_modelmeta(self):
                return SimpleNamespace(
                    custom_metadata_map={
                        "punctuations": "<unk>|_|，|。|？|、",
                        "tokens": "<unk>|好|的|patch",
                        "unk_symbol": "<unk>",
                    }
                )

            def run(self, outputs, feeds):
                n = int(feeds["text_lengths"][0])
                logits = np.zeros((1, n, 6), dtype=np.float32)
                logits[0, :, 1] = 1.0
                logits[0, n - 1, 3] = 5.0
                return [logits]

        fake_ort = ModuleType("onnxruntime")
        fake_ort.SessionOptions = lambda: SimpleNamespace()  # type: ignore[attr-defined]
        fake_ort.InferenceSession = FakeFireRedSession  # type: ignore[attr-defined]
        fake_tok_mod = ModuleType("tokenizers")
        fake_tok_mod.Tokenizer = FakeTokenizer  # type: ignore[attr-defined]

        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **kw: None)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

        with patch.dict(sys.modules, {"onnxruntime": fake_ort, "tokenizers": fake_tok_mod}):
            fr = FireRedPuncOnnx(model_cache_dir=hub_root, on_model_download=states.append)
            tags = fr.predict_tags(["开", "始", "review"])
            assert tags == ["", "", "。"]
            assert (FIRERED_REPO, FIRERED_MODEL_FILE) in downloads
            assert (FIRERED_REPO, FIRERED_TOKENIZER_FILE) in downloads
            assert True in states and states[-1] is False

        fake_ort.InferenceSession = FakeCtSession  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            ct = CtTransformerOnnx(model_cache_dir=hub_root)
            ct_tags = ct.predict_tags(["好", "的", "patch"])
            assert ct_tags == ["", "", "。"]
            assert (CT_TRANSFORMER_REPO, CT_TRANSFORMER_MODEL_FILE) in downloads


def test_diarize_punctuation_onnx_failure_falls_back_to_rule(tmp_path):
    from meetmin.pipeline import Pipeline

    messages: list[str] = []
    pipe = Pipeline(
        {
            "output_dir": tmp_path,
            "diarization": {"provider": "single"},
            "punctuation": {
                "provider": "firered_onnx",
                "model_path": str(tmp_path / "missing.onnx"),
            },
        },
        progress=messages.append,
    )
    session = pipe.new_session("punc-fallback")
    session.save_json(
        "asr_raw.json",
        {
            "session_id": session.session_id,
            "asr": {"provider": "faster_whisper"},
            "language": "zh",
            "utterances": [
                {"start": 0.0, "end": 1.0, "text": "网络出了点问题"},
                {"start": 2.0, "end": 3.0, "text": "今天刚刚搞定"},
            ],
        },
    )
    tr = pipe.diarize(session)
    assert [u["text"] for u in tr["utterances"]] == ["网络出了点问题。", "今天刚刚搞定。"]
    assert any("降级为规则标点" in m for m in messages)


def test_to_simplified_chinese_converts_traditional_and_respects_opencc_phrases(tmp_path):
    from meetmin.asr.zh_convert import to_simplified
    from meetmin.pipeline import Pipeline

    assert (
        to_simplified("在不開 DLC state 的時候，確認統計模組與預期行為怎麼工作？")
        == "在不开 DLC state 的时候，确认统计模组与预期行为怎么工作？"
    )
    # 词组规则（一目瞭然 -> 一目了然）与防误转保护词条（乾坤 -> 乾坤）
    assert to_simplified("這個方案一目瞭然，乾坤未定。") == "这个方案一目了然，乾坤未定。"

    pipe = Pipeline(
        {
            "output_dir": tmp_path,
            "diarization": {"provider": "single"},
            "punctuation": {"provider": "rule", "to_simplified": True},
        }
    )
    session = pipe.new_session("t2s")
    session.save_json(
        "asr_raw.json",
        {
            "session_id": session.session_id,
            "asr": {"provider": "faster_whisper"},
            "language": "zh",
            "utterances": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "這裡有這個變化，確認說沒有問題",
                    "words": [{"w": "這裡有這個變化", "s": 0.0, "e": 2.0}],
                }
            ],
        },
    )
    tr = pipe.diarize(session)
    assert tr["utterances"][0]["text"] == "这里有这个变化，确认说没有问题。"
    assert "这里有这个变化，确认说没有问题。" in session.transcript_md.read_text(encoding="utf-8")

    # 允许显式关闭繁转简
    pipe_keep = Pipeline(
        {
            "output_dir": tmp_path,
            "diarization": {"provider": "single"},
            "punctuation": {"provider": "rule", "to_simplified": False},
        }
    )
    tr_keep = pipe_keep.diarize(session)
    assert tr_keep["utterances"][0]["text"] == "這裡有這個變化，確認說沒有問題。"
