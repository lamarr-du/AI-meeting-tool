import json
import wave

import numpy as np
import pytest

from meetmin.asr.base import Utterance
from meetmin.asr.live_whisper import LiveTrack
from meetmin.asr.model_download import cached_model
from meetmin.asr.quality import Decoded
from meetmin.audio.ring_buffer import SAMPLE_RATE, AudioRingBuffer, IdleTick, PcmChunk
from meetmin.live_pipeline import LiveSessionCoordinator
from meetmin.session import Session


def test_audio_ring_keeps_absolute_sample_ranges():
    ring = AudioRingBuffer(4 / SAMPLE_RATE)
    ring.append(PcmChunk(0, np.array([1, 2, 3], dtype="<i2").tobytes()))
    ring.append(PcmChunk(3, np.array([4, 5, 6], dtype="<i2").tobytes()))
    np.testing.assert_array_equal(ring.read(2, 6) * 32768, [3, 4, 5, 6])
    with pytest.raises(ValueError, match="超出缓冲区"):
        ring.read(1, 5)
    with pytest.raises(ValueError, match="音频缺失"):
        ring.append(PcmChunk(7, b"\0\0"))


def test_idle_closes_a_draft_without_synthesizing_pcm_or_duplicate_text():
    track = LiveTrack(
        ring_sec=60,
        endpoint_ms=600,
        max_window_sec=15,
        vad=lambda audio: [{"start": 0, "end": len(audio)}],
    )
    pcm = np.ones(SAMPLE_RATE, dtype="<i2").tobytes()
    track.feed(PcmChunk(0, pcm))
    track.feed(PcmChunk(SAMPLE_RATE, pcm))
    calls = []

    def decode(audio, offset):
        calls.append((len(audio), offset))
        return [Decoded(Utterance(0.1, 1.9, "测试"), 1.0, -0.1, 0.0)]

    window = track.next_window(SAMPLE_RATE)
    assert window is not None and not window.final
    assert track.decode(window, decode) == (None, "测试")
    track.feed(IdleTick(2 * SAMPLE_RATE + 9600))
    window = track.next_window(SAMPLE_RATE)
    assert window is not None and window.final
    result, draft = track.decode(window, decode)
    assert result is not None and result.text == "测试"
    assert draft == ""
    assert track.audio.end == 2 * SAMPLE_RATE
    assert track.next_window(SAMPLE_RATE) is None
    assert calls == [(2 * SAMPLE_RATE, 0.0)]


def test_offline_model_requires_the_tokenizer(monkeypatch, tmp_path):
    import huggingface_hub

    folder = tmp_path / "snapshot"
    folder.mkdir()
    for name in ("model.bin", "config.json", "vocabulary.json"):
        (folder / name).touch()
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *a, **k: str(folder / "model.bin")
    )
    assert cached_model("large-v3-turbo", cache_dir=tmp_path) is None
    (folder / "tokenizer.json").touch()
    assert cached_model("large-v3-turbo", cache_dir=tmp_path) == folder.resolve()


def test_live_inherits_asr_compute_type(tmp_path):
    coordinator = LiveSessionCoordinator(
        {"asr": {"compute_type": "int8_float32"}},
        Session(tmp_path, "live").ensure(),
        lambda event: None,
        dual_track=False,
    )
    assert coordinator.pipe.cfg["asr"]["compute_type"] == "int8_float32"


@pytest.mark.parametrize("bleed_enabled", [False, True])
def test_live_dual_track_archives_standard_artifacts(monkeypatch, tmp_path, bleed_enabled):
    from faster_whisper import vad

    text = "测试实时录音的外放去重功能"

    class Provider:
        def __init__(self, cfg, **_kwargs):
            assert cfg["model_size"] == "large-v3-turbo" and cfg["local_files_only"]

        def build_stream_options(self):
            return {}

        def stream_metadata(self):
            return {"effective_device": "cpu"}

        def decode_window(self, audio, offset, *, options):
            end = offset + len(audio) / SAMPLE_RATE
            return [
                Decoded(Utterance(offset, end, text, words=[{"w": text, "s": offset, "e": end}]))
            ]

        def close(self):
            pass

    monkeypatch.setattr("meetmin.live_pipeline.FasterWhisperASR", Provider)
    monkeypatch.setattr(
        vad, "get_speech_timestamps", lambda audio, **_: [{"start": 0, "end": len(audio)}]
    )
    session = Session(tmp_path, "live").ensure()
    take = session.root / ("audio_parts-" + "0" * 32)
    take.mkdir()
    pcm = np.ones(SAMPLE_RATE // 10, dtype="<i2").tobytes()
    for filename in ("audio.wav", "microphone.wav"):
        with wave.open(str(take / filename), "wb") as wav:
            wav.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            wav.writeframes(pcm)
    session.publish_dual_recording(take)

    events = []
    coordinator = LiveSessionCoordinator(
        {
            "diarization": {"provider": "single"},
            "punctuation": {"provider": "rule"},
            "live": {"bleed_suppression": {"enabled": bleed_enabled}},
        },
        session,
        events.append,
        dual_track=True,
    )
    coordinator.start()
    for track in ("loopback", "microphone"):
        coordinator.feed(track, PcmChunk(0, pcm))
    transcript = coordinator.finish()
    assert transcript is not None, coordinator.error
    assert [u["speaker"] for u in transcript["utterances"]] == (
        ["说话人A"] if bleed_enabled else ["说话人A", "我"]
    )
    raw = json.loads((session.root / "asr_raw.json").read_text("utf-8"))
    assert "speaker" not in raw["utterances"][0]
    assert raw["microphone"]["utterances"][0]["words"][0]["w"] == text
    assert session.recording_paths()[1].is_file()
    logged = [
        json.loads(line) for line in (session.root / "events.jsonl").read_text("utf-8").splitlines()
    ]
    assert [event for event in events if event["type"].startswith("asr.")] == logged
    mic = next(event for event in logged if event.get("track") == "microphone")
    assert mic["suppressed_bleed"] is bleed_enabled
    assert session.transcript_md.is_file()
