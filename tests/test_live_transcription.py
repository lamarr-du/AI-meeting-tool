import json
import wave
from threading import get_ident

import numpy as np
import pytest

from meetmin.asr.base import Utterance
from meetmin.asr.live_whisper import LiveTrack, ReviewSpan, review_redecode
from meetmin.asr.model_download import cached_model
from meetmin.asr.quality import LIVE_REVIEW_TEXT, Decoded, QualityOptions
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


def test_mute_ends_the_previous_turn_and_excludes_silence_from_inference():
    vad_calls = []

    def vad(audio):
        vad_calls.append(audio.copy())
        return [{"start": 0, "end": len(audio)}]

    track = LiveTrack(ring_sec=60, endpoint_ms=600, max_window_sec=15, vad=vad)
    samples = SAMPLE_RATE // 10
    speech = np.ones(samples, dtype="<i2").tobytes()
    silence = bytes(samples * 2)
    track.feed(PcmChunk(0, speech))
    track.feed(PcmChunk(samples, silence, muted=True))
    track.feed(PcmChunk(2 * samples, silence, muted=True))
    assert len(vad_calls) == 1
    track.feed(PcmChunk(3 * samples, speech))
    track.close()

    for start in (0, 3 * samples):
        window = track.next_window(SAMPLE_RATE)
        assert window is not None and window.final
        assert (window.start, window.end) == (start, start + samples)
        track.decode(window, lambda audio, offset: [])
    assert track.audio.end == 4 * samples
    assert track.next_window(SAMPLE_RATE) is None
    assert len(vad_calls) == 2 and all(np.all(audio) for audio in vad_calls)


def test_repeated_draft_is_deferred_when_idle_closes_the_window():
    track = LiveTrack(
        ring_sec=60,
        endpoint_ms=600,
        max_window_sec=15,
        vad=lambda audio: [{"start": 0, "end": len(audio)}],
    )
    pcm = np.ones(SAMPLE_RATE, dtype="<i2").tobytes()
    track.feed(PcmChunk(0, pcm))
    calls = []

    def decode(audio, offset):
        calls.append(offset)
        return [Decoded(Utterance(0, 1, "你认为" * 50), compression_ratio=20)]

    window = track.next_window(SAMPLE_RATE)
    assert window is not None and not window.final
    assert track.decode(window, decode) == (None, "")
    track.feed(IdleTick(2 * SAMPLE_RATE))
    window = track.next_window(SAMPLE_RATE)
    assert window is not None and window.final
    assert track.decode(window, decode) == (
        ReviewSpan(0, 1, ("high_compression", "text_loop")),
        "",
    )
    assert not track.has_speech
    assert calls == [0]


def test_repair_discards_words_from_the_added_context():
    words = [
        {"w": "上一句", "s": 0.5, "e": 1.2},
        {"w": "继续", "s": 1.2, "e": 1.4},
        {"w": "讨论", "s": 1.4, "e": 1.85},
        {"w": "下一句", "s": 1.85, "e": 2.5},
    ]
    raw = Utterance(0.5, 2.5, "上一句继续讨论下一句", words=words)
    result, reason = review_redecode(
        ReviewSpan(1, 2, ()), [Decoded(raw, 1, -0.2)], QualityOptions()
    )
    assert result is not None, reason
    assert result.text == "继续讨论" and result.words == words[1:3]


@pytest.mark.parametrize(
    ("outcome", "repair_max_clips", "expected_calls"),
    [("accepted", 1, 2), ("low_confidence", 1, 2), ("retry_limit", 0, 1)],
)
def test_live_repairs_recorded_clip_after_stop(
    monkeypatch, tmp_path, outcome, repair_max_clips, expected_calls
):
    from faster_whisper import vad

    events = []
    calls = []
    closed = []
    good_text = "对，对，继续看下一个问题"

    class Provider:
        def __init__(self, cfg, **_kwargs):
            assert cfg["compute_type"] == "int8_float32"
            self.worker = get_ident()

        def build_stream_options(self):
            return {"no_repeat_ngram_size": 0}

        def stream_metadata(self):
            return {"effective_device": "cuda"}

        def decode_window(self, audio, offset, *, options):
            assert get_ident() == self.worker
            calls.append((offset, len(audio), options["no_repeat_ngram_size"]))
            if len(calls) == 1:
                return [Decoded(Utterance(0, 1, "你认为" * 50), compression_ratio=20)]
            assert any(e.get("quality", {}).get("status") == "pending" for e in events)
            return [
                Decoded(
                    Utterance(0.05, 0.9, good_text, words=[{"w": good_text, "s": 0.05, "e": 0.9}]),
                    avg_logprob=(-0.2 if outcome == "accepted" else -1.2),
                )
            ]

        def close(self):
            assert get_ident() == self.worker
            closed.append(True)

    monkeypatch.setattr("meetmin.live_pipeline.FasterWhisperASR", Provider)
    monkeypatch.setattr(
        vad, "get_speech_timestamps", lambda audio, **_: [{"start": 0, "end": len(audio)}]
    )
    session = Session(tmp_path, "repair").ensure()
    pcm = np.ones(SAMPLE_RATE, dtype="<i2").tobytes()
    with wave.open(str(session.root / "audio.wav"), "wb") as wav:
        wav.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        wav.writeframes(pcm)

    coordinator = LiveSessionCoordinator(
        {
            "asr": {"compute_type": "int8_float32"},
            "diarization": {"provider": "single"},
            "punctuation": {"provider": "rule"},
            "live": {"repair_max_clips": repair_max_clips, "hop_sec": 0.5},
        },
        session,
        events.append,
        dual_track=False,
    )
    coordinator.start()
    coordinator.feed("loopback", PcmChunk(0, pcm))
    transcript = coordinator.finish()

    assert transcript is not None, coordinator.error
    assert len(closed) == 1
    assert calls == [(0, SAMPLE_RATE, 0)] + [(0, SAMPLE_RATE, 4)] * (expected_calls - 1)
    assert not any("你认为" in event.get("text", "") for event in events)
    pending = next(event for event in events if event["type"] == "asr.commit")
    assert pending["text"] == LIVE_REVIEW_TEXT
    (report,) = transcript["asr"]["live_quality"]["issues"]
    assert (report["start"], report["end"]) == (0, 1)
    raw = json.loads((session.root / "asr_raw.json").read_text("utf-8"))
    (utterance,) = raw["utterances"]
    if outcome == "accepted":
        assert report["status"] == "accepted" and utterance["text"] == good_text
    else:
        assert report["detail"] == outcome and utterance["text"] == LIVE_REVIEW_TEXT
        assert transcript["utterances"][0]["text"] == LIVE_REVIEW_TEXT


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
