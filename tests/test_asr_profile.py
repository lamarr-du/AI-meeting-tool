import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from meetmin.asr.cuda import CudaStatus, gpu_status
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
    "memory, peak, reason, expected",
    [
        (None, None, None, ("cuda", "large-v3-turbo")),
        (None, 2500, None, ("cuda", "large-v3-turbo")),
        (("GPU", 4096, 4096), 3100, None, ("cuda", "large-v3-turbo")),
        (("GPU", 2755, 4096), 2500, None, ("cuda", "small")),
        (None, 2500, "cuDNN 缺失", ("cpu", "small")),
        (("GPU", 2756, 4096), 2500, None, ("cuda", "large-v3-turbo")),
    ],
)
def test_automatic_cuda_uses_optional_peak_and_headroom(memory, peak, reason, expected):
    choice = select_profile(
        {"cuda_peak_mib": peak}, memory=None, gpu=CudaStatus(reason=reason, memory=memory)
    )
    assert (choice.device, choice.model) == expected
    assert choice.cpu_fallback_model == "small"
    assert choice.compute_type == ("int8_float16" if choice.device == "cuda" else "int8")


def test_pinned_settings_and_cpu_profiles():
    gpu = CudaStatus(memory=("GPU", 4096, 4096))
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
    choice = select_profile({"compute_type": "float32"}, memory=None, gpu=None)
    assert (choice.device, choice.compute_type) == ("cpu", "float32")


def test_cuda_preflight_identifies_missing_dll_in_bundle(monkeypatch, tmp_path):
    bundle = tmp_path / "cuda"
    bundle.mkdir()
    (bundle / "cublasLt64_12.dll").touch()
    module = ModuleType("ctranslate2")
    module.get_cuda_device_count = lambda: 1
    module.get_supported_compute_types = lambda _: {"int8_float16"}
    monkeypatch.setitem(sys.modules, "ctranslate2", module)
    monkeypatch.setattr("meetmin.asr.cuda.sys.platform", "win32")
    monkeypatch.setattr("meetmin.asr.cuda.sys._MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr("meetmin.asr.cuda._register_dll_directory", Mock())
    monkeypatch.setattr("meetmin.asr.cuda.ctypes.CDLL", Mock())
    assert gpu_status().reason == "CUDA 运行库缺失：cublas64_12.dll"


def test_cuda_load_oom_tries_small_cuda_before_cpu(monkeypatch):
    model = Mock()
    model.transcribe.return_value = ([], SimpleNamespace(language="zh", duration=1))
    module = ModuleType("faster_whisper")
    loader = Mock(side_effect=[RuntimeError("CUDA out of memory")] * 2 + [model])
    module.WhisperModel = loader
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr("meetmin.doctor.gpu_status", lambda _: CudaStatus())
    monkeypatch.setattr("meetmin.asr.faster_whisper_asr.memory_status", lambda: None)
    provider = FasterWhisperASR({"cuda_peak_mib": 2500})
    assert [(c.args[0], c.kwargs["device"]) for c in loader.call_args_list] == [
        ("large-v3-turbo", "cuda"),
        ("small", "cuda"),
        ("small", "cpu"),
    ]
    result = provider.transcribe(np.zeros(16000, dtype=np.float32))
    assert result.extra["asr_model"] == "small"
    assert result.extra["effective_device"] == "cpu"
    assert result.extra["cuda_oom_fallback"]
    assert [step["to"] for step in result.extra["fallbacks"]] == [
        "small / cuda / int8_float16",
        "small / cpu / int8",
    ]
    provider.close()
    provider.close()
    assert not hasattr(provider, "model")


@pytest.mark.parametrize(
    "message, expected_device",
    [
        ("Library cublas64_12.dll is not found or cannot be loaded", "cpu"),
        ("CUDA out of memory", "cuda"),
    ],
)
def test_second_track_fallback_preserves_first_track_metadata(
    monkeypatch, message, expected_device
):
    def failed():
        raise RuntimeError(message)
        yield

    info = SimpleNamespace(language="zh", duration=1)
    turbo = Mock()
    turbo.transcribe.side_effect = [([], info), (failed(), info)]
    fallback = Mock()
    fallback.transcribe.return_value = ([], info)
    module = ModuleType("faster_whisper")
    module.WhisperModel = Mock(side_effect=[turbo, fallback])
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr("meetmin.doctor.gpu_status", lambda _: CudaStatus())
    monkeypatch.setattr("meetmin.asr.faster_whisper_asr.memory_status", lambda: None)

    provider = FasterWhisperASR({})
    audio = np.zeros(16000, dtype=np.float32)
    first = provider.transcribe(audio)
    second = provider.transcribe(audio)
    assert first.extra["asr_model"] == "large-v3-turbo"
    assert first.extra["effective_device"] == "cuda"
    assert first.extra["fallbacks"] == []
    assert second.extra["asr_model"] == "small"
    assert second.extra["effective_device"] == expected_device
    assert second.extra["fallbacks"][0]["reason"] == message
    fallback.transcribe.assert_called_once()


def test_live_fallback_keeps_the_repair_repetition_guard(monkeypatch):
    first, second = Mock(), Mock()
    first.transcribe.side_effect = RuntimeError("CUDA out of memory")
    second.transcribe.return_value = ([], None)
    module = ModuleType("faster_whisper")
    module.WhisperModel = Mock(side_effect=[first, second])
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr("meetmin.doctor.gpu_status", lambda _: CudaStatus())
    monkeypatch.setattr("meetmin.asr.faster_whisper_asr.memory_status", lambda: None)

    provider = FasterWhisperASR({})
    options = {**provider.build_stream_options(), "no_repeat_ngram_size": 4}
    provider.decode_window(np.zeros(16000, dtype=np.float32), 0, options=options)
    assert second.transcribe.call_args.kwargs["no_repeat_ngram_size"] == 4


def test_download_error_does_not_trigger_cuda_fallback(monkeypatch):
    module = ModuleType("faster_whisper")
    module.WhisperModel = Mock()
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr("meetmin.doctor.gpu_status", lambda _: CudaStatus())
    download = Mock(side_effect=RuntimeError("download failed"))
    monkeypatch.setattr("meetmin.asr.faster_whisper_asr.download_model", download)
    with pytest.raises(RuntimeError, match="download failed"):
        FasterWhisperASR({}, on_model_download=Mock())
    download.assert_called_once()
    module.WhisperModel.assert_not_called()


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
