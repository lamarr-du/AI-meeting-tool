import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from meetmin.asr.faster_whisper_asr import FasterWhisperASR
from meetmin.asr.profile import select_profile
from meetmin.pipeline import Pipeline


@pytest.mark.parametrize(
    "memory, model",
    [
        (None, "small"),
        ((8192, 7000), "small"),
        ((16280, 4000), "small"),
        ((16280, 6144), "large-v3-turbo"),
    ],
)
def test_cpu_auto_checks_total_and_available_memory(memory, model):
    choice = select_profile({}, memory=memory, gpu=None)
    assert (choice.model, choice.device, choice.compute_type) == (model, "cpu", "int8")


@pytest.mark.parametrize(
    "free, peak, missing, expected",
    [
        (4096, None, [], "cpu"),
        (4096, 3100, [], "cpu"),
        (2755, 2500, [], "cpu"),
        (2756, 2500, ["cuDNN"], "cpu"),
        (2756, 2500, [], "cuda"),
    ],
)
def test_automatic_cuda_requires_measured_peak_and_headroom(free, peak, missing, expected):
    choice = select_profile({"cuda_peak_mib": peak}, memory=None, gpu=("GPU", free, 4096, missing))
    assert choice.device == expected
    assert choice.cpu_fallback_model == "small"
    if expected == "cuda":
        assert (choice.model, choice.compute_type) == ("large-v3-turbo", "int8_float16")


def test_pinned_settings_and_cpu_profiles():
    gpu = ("GPU", 4096, 4096, [])
    light = select_profile(
        {"profile": "cpu_light", "cuda_peak_mib": 2500}, memory=(32000, 25000), gpu=gpu
    )
    assert (light.model, light.device) == ("small", "cpu")
    manual = select_profile(
        {"device": "cuda", "model_size": "local/model", "compute_type": "float16"},
        memory=None,
        gpu=None,
    )
    assert (manual.model, manual.cpu_fallback_model, manual.compute_type) == (
        "local/model",
        "local/model",
        "float16",
    )


def test_cuda_load_oom_records_effective_cpu_model(monkeypatch):
    model = Mock()
    model.transcribe.return_value = ([], SimpleNamespace(language="zh", duration=1))
    module = ModuleType("faster_whisper")
    loader = Mock(side_effect=[RuntimeError("CUDA out of memory"), model])
    module.WhisperModel = loader
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr("meetmin.doctor.gpu_status", lambda: ("GPU", 4096, 4096, []))
    monkeypatch.setattr("meetmin.asr.faster_whisper_asr.memory_status", lambda: None)
    provider = FasterWhisperASR({"cuda_peak_mib": 2500})
    assert [(c.args[0], c.kwargs["device"]) for c in loader.call_args_list] == [
        ("large-v3-turbo", "cuda"),
        ("small", "cpu"),
    ]
    result = provider.transcribe(np.zeros(16000, dtype=np.float32))
    assert result.extra["asr_model"] == "small"
    assert result.extra["effective_device"] == "cpu"
    assert result.extra["cuda_oom_fallback"]
    provider.close()
    provider.close()
    assert not hasattr(provider, "model")


def test_pipeline_closes_provider_when_transcription_fails(tmp_path, monkeypatch):
    provider = SimpleNamespace(
        name="faster_whisper", transcribe=Mock(side_effect=RuntimeError("decode")), close=Mock()
    )
    monkeypatch.setattr("meetmin.asr.get_asr", lambda _: provider)
    monkeypatch.setattr("meetmin.pipeline.to_mono_16k", lambda _: np.zeros(16000, dtype=np.float32))
    pipeline = Pipeline({"output_dir": str(tmp_path)})
    session = pipeline.new_session("release")
    session.audio_path.touch()
    with pytest.raises(RuntimeError, match="decode"):
        pipeline.transcribe(session)
    provider.close.assert_called_once()
    assert not (session.root / "asr_raw.json").exists()
