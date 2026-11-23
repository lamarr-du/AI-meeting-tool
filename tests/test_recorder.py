"""录音生命周期、时间轴和故障测试，无需声卡或 PyAudio。"""

import json
import threading
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from meetmin.audio.loopback_recorder import LoopbackRecorder, MicrophoneRecorder, RecordingError
from meetmin.audio.ring_buffer import IdleTick, PcmChunk
from meetmin.audio.segments import SessionLock, merge_segments
from meetmin.audio.session_recorder import RECORDING_DIAGNOSTICS, SessionRecorder
from meetmin.cli import cmd_recover
from meetmin.pipeline import Pipeline
from meetmin.session import Session


@pytest.mark.parametrize(
    "recorder_class,info",
    [(LoopbackRecorder, {"isLoopbackDevice": True}), (MicrophoneRecorder, {})],
)
def test_configured_target_rms_changes_recorded_level(recorder_class, info):
    samples = np.full(1600, 0.01, dtype="<f4")
    outputs = []
    for target in (0.03, 0.09):
        recorder = recorder_class(target_rms=target)
        control = recorder._create_level_control(16000, info)
        output = np.frombuffer(control.to_pcm(samples.tobytes(), len(samples)), dtype="<i2")
        outputs.append(float(np.mean(output)) / 32768)
    assert outputs == pytest.approx([0.03, 0.09], abs=0.0001)


def test_pipeline_passes_target_to_both_tracks():
    recorder = Pipeline({"recorder": {"target_rms": 0.08, "microphone": True}}).recorder(
        for_session=True
    )
    assert [track.target_rms for track in recorder._recorders] == [0.08, 0.08]
    with pytest.raises(ValueError, match="target_rms"):
        LoopbackRecorder(target_rms=0.5)


class Clock:
    value = 10.0

    def __call__(self):
        return self.value


class Stream:
    def __init__(self, clock, **kwargs):
        assert kwargs["start"] is False
        self.callback = kwargs["stream_callback"]
        self.clock = clock
        self.rate = kwargs["rate"]
        self.channels = kwargs["channels"]
        self.float_capture = kwargs["format"] == 1
        self.epoch = 1000 if kwargs["input_device_index"] == 0 else 2000
        self.active = False
        self.closed = False
        self.start_error = None
        self.close_error = None
        self.stop_error = None

    def get_time(self):
        return self.clock() + self.epoch  # Each stream has a different clock epoch.

    def start_stream(self):
        if self.start_error:
            raise self.start_error
        self.active = True

    def stop_stream(self):
        if self.stop_error:
            raise self.stop_error
        self.active = False

    def close(self):
        if self.close_error:
            raise self.close_error
        self.active = False
        self.closed = True

    def is_active(self):
        return self.active

    def emit(self, position, *, value=1000, frames=100, timing=True, received=None, flags=0):
        self.clock.value = received if received is not None else 10 + position + frames / self.rate
        samples = np.broadcast_to(value, (frames, self.channels))
        data = (
            (samples / 32768).astype("<f4") if self.float_capture else samples.astype("<i2")
        ).tobytes()
        times = (
            {
                "input_buffer_adc_time": self.epoch + 10 + position,
                "current_time": self.epoch + 10 + position + frames / self.rate,
            }
            if timing
            else {}
        )
        if isinstance(timing, dict):
            times = timing
        result = self.callback(data, frames, times, flags)
        if result[1] == 2:
            self.active = False
        return result


@pytest.fixture
def recording(monkeypatch, tmp_path):
    clock = Clock()
    streams = []
    instances = []

    class PA:
        def __init__(self):
            self.terminated = False
            instances.append(self)

        def get_device_info_by_index(self, index):
            return {
                "defaultSampleRate": 1000 if index == 0 else 2000,
                "maxInputChannels": 1,
                "index": index,
                "name": "fake",
                "hostApi": 2,
                "isLoopbackDevice": index == 0,
            }

        def get_host_api_info_by_type(self, _):
            return {"index": 2, "defaultInputDevice": 1}

        def open(self, **kwargs):
            stream = Stream(clock, **kwargs)
            streams.append(stream)
            return stream

        def terminate(self):
            self.terminated = True

    monkeypatch.setattr(
        "meetmin.audio.loopback_recorder._import_pyaudio",
        lambda: SimpleNamespace(
            PyAudio=PA,
            paInt16=8,
            paFloat32=1,
            paContinue=0,
            paAbort=2,
            paWASAPI=13,
        ),
    )
    rec = LoopbackRecorder(device=0, clock=clock)
    yield SimpleNamespace(
        rec=rec, clock=clock, streams=streams, instances=instances, path=tmp_path / "audio.wav"
    )
    # Tests explicitly release blocked threads before fixture teardown.
    if rec._thread is not None or rec._pa is not None:
        try:
            rec.stop()
        except RecordingError:
            pass


def read_audio(path, rate=1000):
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == rate
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")


def test_live_tap_idle_and_resampler_tail_match_the_saved_wav(recording):
    r = recording
    events = []
    idle = threading.Event()

    def tap(event):
        events.append(event)
        if isinstance(event, IdleTick):
            idle.set()

    r.rec.output_sample_rate = 16000
    r.rec.output_channels = 1
    r.rec.sample_rate = 48000
    r.rec.audio_tap = tap
    r.rec.start(r.path)
    for index in range(5):
        r.streams[-1].emit(index / 10, value=1000, frames=4800)
    r.clock.value = 11.5
    assert idle.wait(1)
    assert any(isinstance(event, PcmChunk) for event in events)
    r.rec.stop()

    sample_count = 0
    pcm = []
    for event in events:
        if isinstance(event, PcmChunk):
            assert event.start == sample_count
            pcm.append(event.pcm)
            sample_count += len(event.pcm) // 2
    assert b"".join(pcm) == read_audio(r.path, rate=16000).tobytes()


def recover_failed(path):
    directories = list(path.parent.glob("audio_parts-*"))
    assert len(directories) == 1
    merge_segments(directories[0], path)
    return read_audio(path)


def test_leading_middle_trailing_silence_and_timestamps(recording):
    r = recording
    r.rec.start(r.path)
    s = r.streams[-1]
    s.emit(0.5)
    s.emit(0.6)
    s.emit(1.5, value=2000)
    s.emit(1.6, value=2000)
    r.clock.value = 12
    info = r.rec.stop()
    data = read_audio(r.path)
    expected = np.zeros(2000, dtype="<i2")
    expected[500:700] = 1000
    expected[1500:1700] = 2000
    np.testing.assert_array_equal(data, expected)
    assert info["duration_sec"] == 2
    assert info["padded_sec"] == 1.6
    assert info["no_data_sec"] == pytest.approx(0.3)
    assert info["timing_mode"] == "device"
    assert info["finalized"] and not info["cleanup_pending"]


def test_all_silence_stop_is_idempotent_and_instance_reusable(recording):
    r = recording
    for i in range(2):
        path = r.path.with_name(f"{i}.wav")
        r.rec.start(path)
        assert r.rec.status()["duration_sec"] == 0
        r.clock.value += 1.25
        info = r.rec.stop()
        assert info == r.rec.stop()
        assert info["duration_sec"] == 1.25
        assert info["padded_sec"] == 1.25
        assert info["no_data_sec"] == 1.25  # Synthetic silence is not newly received audio.
        assert not np.any(read_audio(path))
        assert not r.rec.is_recording
    assert all(p.terminated for p in r.instances)


def test_one_hour_timeline_crosses_segments(recording):
    r = recording
    r.rec.start(r.path)
    r.streams[-1].emit(0, value=1234)
    r.streams[-1].emit(0.1, value=1234)
    r.clock.value = 3610
    info = r.rec.stop()
    assert info["duration_sec"] == 3600
    data = read_audio(r.path)
    assert len(data) == 3600000
    assert np.all(data[:200] == 1234)
    assert not np.any(data[200:])
    assert len(list(next(r.path.parent.glob("audio_parts-*")).glob("*.wav"))) == 12


def test_stop_trims_pending_block(recording):
    r = recording
    r.rec.start(r.path)
    r.streams[-1].emit(0, received=10.05)
    r.clock.value = 10.05
    info = r.rec.stop()
    assert info["duration_sec"] == 0.05
    np.testing.assert_array_equal(read_audio(r.path), np.full(50, 1000))


def test_quiet_loopback_is_boosted_before_pcm_and_recovery(recording):
    r = recording
    r.rec.start(r.path)
    stream = r.streams[-1]
    assert stream.float_capture
    stream.emit(0, value=0)
    stream.emit(0.1, value=0.025)  # Below one 16-bit PCM step at capture.
    stream.emit(0.2, value=2000, received=10.25)
    info = r.rec.stop()

    expected = np.repeat([0, 250, 2000], [100, 100, 50])
    np.testing.assert_array_equal(read_audio(r.path), expected)
    assert info["active_fraction"] == pytest.approx(0.6)
    recovered = r.path.with_name("recovered.wav")
    directory = next(r.path.parent.glob("audio_parts-*"))
    merge_segments(directory, recovered)
    np.testing.assert_array_equal(read_audio(recovered), expected)

    # A second recording must start without the previous session's level estimate.
    r.clock.value = 10
    r.rec.start(r.path.with_name("second.wav"))
    r.streams[-1].emit(0, value=0.025)
    info = r.rec.stop()
    np.testing.assert_array_equal(read_audio(info["path"]), np.full(100, 250))


def test_microphone_boosts_quiet_input_without_boosting_very_low_noise(recording):
    r = recording
    microphone = MicrophoneRecorder(clock=r.clock)
    microphone.start(r.path)
    stream = r.streams[-1]
    assert stream.float_capture
    stream.emit(0, value=0, frames=200)
    stream.emit(0.1, value=32, frames=200)
    low_noise = np.tile([2, -2], 100)
    stream.emit(0.2, value=low_noise[:, None], frames=200)
    stream.emit(0.3, value=2400, frames=200)
    info = microphone.stop()

    expected = np.concatenate((np.zeros(200), np.full(200, 512), low_noise, np.full(200, 2400)))
    np.testing.assert_array_equal(read_audio(r.path, rate=2000), expected)
    assert info["active_fraction"] == pytest.approx(0.5)


def test_loopback_boost_preserves_stereo_balance_and_peak_headroom(recording):
    r = recording
    r.rec.sample_rate = 48000
    r.rec.channels = 2
    r.rec.start(r.path)
    stream = r.streams[-1]
    stream.emit(0, value=[16, -8], frames=4800)
    samples = np.tile([16, -8], (4800, 1))
    samples[0, 1] = -29000
    stream.emit(0.1, value=samples, frames=4800)
    r.rec.stop()

    stereo = read_audio(r.path, rate=48000).reshape(-1, 2)
    assert stereo[0, 0] > 100
    assert stereo[0, 0] == pytest.approx(-2 * stereo[0, 1], abs=1)
    assert stereo[4800, 1] == round(-0.95 * 32768)


def test_session_recording_saves_and_recovers_16k_mono_tracks(recording, tmp_path):
    r = recording
    recorder = Pipeline(
        {"recorder": {"device": 0, "sample_rate": 48000, "channels": 2, "microphone": True}},
        progress=lambda _: None,
    ).recorder(for_session=True)
    for track in recorder._recorders:
        track._clock = r.clock
    session = Session(tmp_path, "format")
    recorder.start(session)
    loopback, microphone = r.streams[-2:]
    assert (loopback.rate, loopback.channels) == (48000, 2)
    assert (microphone.rate, microphone.channels) == (2000, 1)
    loopback.emit(0, value=[4000, 2000], frames=4800)
    microphone.emit(0, value=3000, frames=200)
    r.clock.value = 10.2
    info = recorder.stop()
    assert (info["sample_rate"], info["channels"]) == (16000, 1)
    assert (info["capture_sample_rate"], info["capture_channels"]) == (48000, 2)
    assert (info["microphone"]["sample_rate"], info["microphone"]["channels"]) == (16000, 1)
    paths = session.recording_paths()
    for path, track in zip(paths, ("loopback", "microphone")):
        assert path is not None
        with wave.open(str(path), "rb") as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
            assert wav.getnframes() == 3200
            data = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        assert np.mean(np.abs(data[200:1400])) > 1000
        assert np.max(np.abs(data[2000:])) < 10
        recovered = tmp_path / f"{track}.wav"
        merge_segments(path.parent / track, recovered)
        np.testing.assert_array_equal(read_audio(recovered, rate=16000), data)


@pytest.mark.parametrize("capture_silence", [True, False])
def test_loopback_reestimates_gain_after_silence(recording, capture_silence):
    r = recording
    r.rec.start(r.path)
    stream = r.streams[-1]
    stream.emit(0, value=4000)
    if capture_silence:
        stream.emit(0.1, value=0)
        stream.emit(0.2, value=0)
    stream.emit(0.3, value=8)
    r.rec.stop()

    audio = read_audio(r.path)
    np.testing.assert_array_equal(audio[:100], np.full(100, 4000))
    np.testing.assert_array_equal(audio[100:300], np.zeros(200))
    assert np.min(audio[300:]) > 600


def test_loopback_boosts_quieter_speech_after_loud_speech_without_a_gap(recording):
    r = recording
    r.rec.start(r.path)
    stream = r.streams[-1]
    stream.emit(0, value=4000)
    for block in range(1, 8):
        stream.emit(block / 10, value=8)
    r.rec.stop()

    audio = read_audio(r.path)
    assert np.min(audio[600:]) > 600


def test_backlog_does_not_insert_silence(recording, monkeypatch):
    r = recording
    release = threading.Event()
    original = r.rec._run
    monkeypatch.setattr(r.rec, "_run", lambda: (release.wait(3), original()))
    r.rec.start(r.path)
    try:
        for i in range(5):
            r.streams[-1].emit(i / 10)
        # Writer has not consumed anything, although the callbacks have all arrived.
        r.clock.value = 11
        assert r.rec.status()["no_data_sec"] == 0.5
    finally:
        release.set()
    info = r.rec.stop()
    data = read_audio(r.path)
    np.testing.assert_array_equal(data[:500], np.full(500, 1000))
    assert not np.any(data[500:])
    assert info["padded_sec"] == 0.5


def test_queue_full_reports_error_and_drains_accepted_audio(recording, monkeypatch):
    r = recording
    r.rec._queue_size = 2
    release = threading.Event()
    original = r.rec._run
    monkeypatch.setattr(r.rec, "_run", lambda: (release.wait(3), original()))
    r.rec.start(r.path)
    try:
        s = r.streams[-1]
        s.emit(0)
        s.emit(0.1)
        assert s.emit(0.2)[1] == 2
        assert r.rec.status()["failed"]
    finally:
        release.set()
    with pytest.raises(RecordingError, match="队列已满") as error:
        r.rec.stop()
    assert error.value.info["finalized"]
    assert len(recover_failed(r.path)) == 200


def test_writer_timeout_never_closes_its_file(recording, monkeypatch):
    r = recording
    release = threading.Event()
    original = r.rec._run
    monkeypatch.setattr(r.rec, "_run", lambda: (release.wait(3), original()))
    r.rec._stop_timeout = 0.01
    r.rec.start(r.path)
    audio_file = r.rec._audio_file
    try:
        with pytest.raises(RecordingError, match="超时") as error:
            r.rec.stop()
        assert not audio_file.closed
        assert error.value.info["cleanup_pending"]
        with pytest.raises(RuntimeError, match="清理尚未完成"):
            r.rec.start(r.path.with_name("next.wav"))
    finally:
        release.set()
        r.rec._stop_timeout = 3
    with pytest.raises(RecordingError, match="超时"):
        r.rec.stop()
    assert audio_file.closed
    assert not r.rec.status()["cleanup_pending"]


def test_write_failure_preserves_original_error_and_closes_file(recording, monkeypatch):
    r = recording
    r.rec.start(r.path)
    closed = r.rec._audio_file

    def fail(data):
        raise OSError("disk full")

    monkeypatch.setattr(r.rec._wf, "writeframesraw", fail)
    r.streams[-1].emit(0)
    with pytest.raises(RecordingError, match="disk full"):
        r.rec.stop()
    assert closed.closed
    assert r.streams[-1].closed
    assert r.instances[-1].terminated


def test_merge_failure_keeps_previous_recording_and_new_segments(recording, monkeypatch):
    r = recording
    r.path.write_bytes(b"old recording")
    r.rec.start(r.path, overwrite=True)
    r.streams[-1].emit(0)
    r.clock.value = 10.2

    def fail(*args, **kwargs):
        raise OSError("disk full while merging")

    monkeypatch.setattr("meetmin.audio.loopback_recorder.merge_segments", fail)
    with pytest.raises(RecordingError, match="disk full"):
        r.rec.stop()
    assert r.path.read_bytes() == b"old recording"
    directory = next(r.path.parent.glob("audio_parts-*"))
    restored = r.path.with_name("recovered.wav")
    merge_segments(directory, restored)
    assert len(read_audio(restored)) == 200


def test_stream_stopped_unexpectedly(recording):
    r = recording
    r.rec.start(r.path)
    r.streams[-1].active = False
    assert r.rec.status()["failed"]
    assert not r.rec.is_recording
    with pytest.raises(RecordingError, match="意外停止"):
        r.rec.stop()


def test_session_restarts_inactive_loopback_once_and_keeps_the_gap(recording, tmp_path):
    r = recording
    session = Session(tmp_path, "retry-active")
    recorder = SessionRecorder(r.rec)
    recorder.start(session)
    stream = r.streams[-1]
    stream.emit(0.0, value=1000)
    stream.active = False
    r.clock.value = 10.3
    info = recorder.status()
    assert not info["failed"] and info["stream_restarts"] == 1
    assert len(r.streams) == 1 and stream.active
    stream.emit(0.4, value=2000)
    r.clock.value = 10.6

    recorder.stop()
    np.testing.assert_array_equal(
        read_audio(session.audio_path), np.repeat([1000, 0, 2000, 0], [100, 300, 100, 100])
    )
    logs = [
        json.loads(line)
        for line in (session.root / RECORDING_DIAGNOSTICS).read_text("utf-8").splitlines()
    ]
    assert [entry["event"] for entry in logs] == [
        "recording.started",
        "recording.status_changed",
        "recording.stopped",
    ]
    assert logs[1]["tracks"]["loopback"]["stream_restarts"] == 1


def test_repeated_stream_failure_preserves_segments_and_logs_error(recording, tmp_path):
    r = recording
    session = Session(tmp_path, "retry-failed")
    recorder = SessionRecorder(r.rec)
    recorder.start(session)
    stream = r.streams[-1]
    stream.emit(0.0, value=1000)
    stream.active = False
    assert not recorder.status()["failed"]
    stream.active = False
    assert recorder.status()["failed"]
    with pytest.raises(RecordingError, match="意外停止"):
        recorder.stop()

    assert not session.audio_path.exists()
    entry = json.loads((session.root / RECORDING_DIAGNOSTICS).read_text("utf-8").splitlines()[-1])
    assert entry["event"] == "recording.failed"
    assert "意外停止" in entry["tracks"]["loopback"]["error"]
    assert entry["tracks"]["loopback"]["stream_restart_attempts"] == 1


def test_stop_detects_stream_failure_without_status_polling(recording):
    r = recording
    r.rec.start(r.path)
    r.streams[-1].active = False
    with pytest.raises(RecordingError, match="意外停止"):
        r.rec.stop()


def test_finalization_error_is_reported(recording, monkeypatch):
    r = recording
    r.rec.start(r.path)
    audio_file = r.rec._audio_file
    wav = r.rec._wf
    original_close = wav.close

    def fail_close():
        original_close()
        raise OSError("header flush failed")

    monkeypatch.setattr(wav, "close", fail_close)
    r.clock.value += 0.2
    with pytest.raises(RecordingError, match="header flush failed") as error:
        r.rec.stop()
    assert not error.value.info["finalized"]
    assert audio_file.closed
    monkeypatch.setattr(wav, "close", original_close)


@pytest.mark.parametrize("flags", [1, 2])
def test_callback_flags_abort_and_report(recording, flags):
    r = recording
    r.rec.start(r.path)
    assert r.streams[-1].emit(0, flags=flags)[1] == 2
    with pytest.raises(RecordingError, match="PortAudio flags"):
        r.rec.stop()


def test_invalid_callback_data_aborts(recording):
    r = recording
    r.rec.start(r.path)
    assert r.streams[-1].callback(b"x", 100, {}, 0)[1] == 2
    with pytest.raises(RecordingError, match="长度"):
        r.rec.stop()


def test_missing_timestamps_fallback_is_visible(recording):
    r = recording
    r.rec.start(r.path)
    r.streams[-1].emit(0.5, timing=False)
    r.clock.value = 11
    info = r.rec.stop()
    assert info["timing_mode"] == "arrival"
    assert info["padded_sec"] == 0.9
    np.testing.assert_array_equal(read_audio(r.path)[500:600], np.full(100, 1000))


def test_short_timestamp_glitch_keeps_pcm_and_the_next_real_gap(recording):
    r = recording
    r.rec.start(r.path)
    stream = r.streams[-1]
    stream.emit(0, value=1000)
    stream.emit(0, received=10.2, value=1200)
    stream.emit(0.2, timing=False, value=1400)
    stream.emit(0.3, value=1600)
    stream.emit(0.7, value=1800)
    r.clock.value = 11

    info = r.rec.stop()
    assert info["timing_mode"] == "device"
    assert info["repaired_timestamp_blocks"] == 2
    assert info["received_sec"] == 0.5
    np.testing.assert_array_equal(
        read_audio(r.path),
        np.repeat([1000, 1200, 1400, 1600, 0, 1800, 0], [100, 100, 100, 100, 300, 100, 200]),
    )


def test_adc_ahead_of_callback_does_not_abort_contiguous_audio(recording):
    r = recording
    r.rec.start(r.path)
    stream = r.streams[-1]
    origin = 10 + stream.epoch
    stream.emit(0)
    stream.emit(
        0.1,
        timing={"input_buffer_adc_time": origin + 0.22, "current_time": origin + 0.2},
    )
    stream.emit(0.2)
    r.clock.value = 10.3

    assert r.rec.stop()["repaired_timestamp_blocks"] == 1
    np.testing.assert_array_equal(read_audio(r.path), np.full(300, 1000))


@pytest.mark.parametrize("position", [-1, 0.02, 10000])
def test_broken_timestamps_do_not_silently_drop_or_pad_audio(recording, position):
    r = recording
    r.rec.start(r.path)
    r.streams[-1].emit(0)
    r.streams[-1].emit(position, received=10.2)
    with pytest.raises(RecordingError, match="时间戳|时钟|重叠"):
        r.rec.stop()
    assert len(recover_failed(r.path)) <= 200


def test_persistent_missing_timestamps_cannot_run_past_the_repair_limit(recording):
    r = recording
    r.rec.start(r.path)
    stream = r.streams[-1]
    stream.emit(0)
    for index in range(1, 4):
        stream.emit(index / 10, timing=False)
    with pytest.raises(RecordingError, match="时间戳") as error:
        r.rec.stop()
    assert error.value.info["repaired_timestamp_blocks"] == 2
    np.testing.assert_array_equal(recover_failed(r.path), np.full(300, 1000))


def test_bad_timestamp_after_a_gap_cannot_collapse_the_gap(recording):
    r = recording
    r.rec.start(r.path)
    stream = r.streams[-1]
    stream.emit(0)
    stream.emit(1.0, timing=False)
    with pytest.raises(RecordingError, match="时间戳"):
        r.rec.stop()
    np.testing.assert_array_equal(recover_failed(r.path), np.full(100, 1000))


def test_jitter_does_not_add_gaps(recording):
    r = recording
    r.rec.start(r.path)
    for i, offset in enumerate([0, 0.002, -0.001, 0.001, -0.002]):
        r.streams[-1].emit(i / 10 + offset)
    r.clock.value = 10.5
    info = r.rec.stop()
    assert info["padded_sec"] == 0
    np.testing.assert_array_equal(read_audio(r.path), np.full(500, 1000))


def test_start_failure_cleans_writer_and_can_retry(recording, monkeypatch):
    r = recording
    original = Stream.start_stream

    def fail(self):
        raise OSError("start failed")

    monkeypatch.setattr(Stream, "start_stream", fail)
    with pytest.raises(OSError, match="start failed"):
        r.rec.start(r.path)
    assert r.rec._thread is None
    assert r.streams[-1].closed and r.instances[-1].terminated
    monkeypatch.setattr(Stream, "start_stream", original)
    r.rec.start(r.path.with_name("retry.wav"))
    assert not r.rec.stop()["failed"]


def test_failed_stream_close_does_not_release_writer(recording):
    r = recording
    r.rec.start(r.path)
    s = r.streams[-1]
    s.close_error = OSError("close failed")
    audio_file = r.rec._audio_file
    try:
        with pytest.raises(RecordingError, match="close failed"):
            r.rec.stop()
        assert not r.rec._producer_done.is_set()
        assert not audio_file.closed
        assert not r.instances[-1].terminated
    finally:
        s.close_error = None
    with pytest.raises(RecordingError, match="close failed"):
        r.rec.stop()
    assert audio_file.closed


def test_cli_failure_does_not_continue_pipeline(recording, monkeypatch, capsys):
    import meetmin.cli as cli

    r = recording
    session = Session(str(r.path.parent), "test").ensure()
    pipe = SimpleNamespace(
        recorder=lambda **_: SessionRecorder(r.rec),
        new_session=lambda _: session,
        transcribe=lambda *a, **k: pytest.fail("must not transcribe"),
    )
    monkeypatch.setattr(cli, "Pipeline", lambda cfg: pipe)
    monkeypatch.setattr("meetmin.doctor.run_doctor", lambda *a, **k: True)

    def fail_stream(_):
        r.streams[-1].active = False

    monkeypatch.setattr(cli.time, "sleep", fail_stream)
    args = SimpleNamespace(
        audio=None, overwrite=False, session=None, file_url=None, max_seconds=None
    )
    with pytest.raises(SystemExit) as error:
        cli.cmd_run(args, {})
    assert error.value.code == 2
    output = capsys.readouterr().out
    assert "录音失败" in output and "已写时长" in output


def test_dual_recording_shares_boundaries_across_device_clocks(recording, monkeypatch, tmp_path):
    r = recording
    start_stream = Stream.start_stream

    def stagger_start(stream):
        start_stream(stream)
        if stream.rate == 1000:
            # The microphone starts later but uses the same timeline.
            r.clock.value += 0.15

    monkeypatch.setattr(Stream, "start_stream", stagger_start)
    session = Session(tmp_path, "dual")
    rec = SessionRecorder(r.rec, MicrophoneRecorder(channels=1, clock=r.clock))
    rec.start(session)
    with pytest.raises(RuntimeError, match="正在录音或恢复"):
        SessionLock(session.root)
    r.streams[0].emit(0.20, value=3000)
    r.streams[1].emit(0.20, value=32, frames=200)
    r.clock.value = 10.60
    info = rec.stop()

    loopback, microphone = session.recording_paths()
    assert microphone is not None and microphone.parent == loopback.parent
    assert info["duration_sec"] == info["microphone"]["duration_sec"] == 0.6
    assert info["timing_mode"] == info["microphone"]["timing_mode"] == "device"
    np.testing.assert_array_equal(read_audio(loopback), np.repeat([0, 3000, 0], [200, 100, 300]))
    np.testing.assert_array_equal(
        read_audio(microphone, 2000), np.repeat([0, 512, 0], [400, 200, 600])
    )
    assert all(stream.closed for stream in r.streams)

    rec = SessionRecorder(r.rec)
    rec.start(session, overwrite=True)
    r.clock.value = 11
    rec.stop()
    assert session.recording_paths() == (session.audio_path, None)
    assert microphone.is_file()


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_microphone_mute_preserves_wav_and_live_timeline(recording, tmp_path, rate):
    r = recording
    r.rec.sample_rate = r.rec.output_sample_rate = 16000
    mic = MicrophoneRecorder(clock=r.clock, sample_rate=rate, output_sample_rate=16000)
    recorder = SessionRecorder(r.rec, mic)
    events = {"loopback": [], "microphone": []}
    recorder.set_audio_tap(lambda name, event: events[name].append(event))
    session = Session(tmp_path, "muted")
    recorder.start(session)
    loopback, microphone = r.streams
    for index in range(6):
        start = index / 10
        loopback.emit(start, value=3000, frames=1600)
        if index == 2:
            recorder.set_microphone_muted(True)
        elif index == 4:
            # Unmuting mid-buffer must still discard that buffer's captured audio.
            recorder.set_microphone_muted(False)
        microphone.emit(start, value=6000, frames=rate // 10)
    recorder.stop()

    loopback_wav, microphone_wav = session.recording_paths()
    assert microphone_wav is not None
    for track, path in (("loopback", loopback_wav), ("microphone", microphone_wav)):
        audio = read_audio(path, 16000)
        assert len(audio) == 9600
        cursor = 0
        muted = []
        for event in events[track]:
            if isinstance(event, PcmChunk):
                assert event.start == cursor
                samples = np.frombuffer(event.pcm, dtype="<i2")
                np.testing.assert_array_equal(samples, audio[cursor : cursor + len(samples)])
                muted.extend([event.muted] * len(samples))
                cursor += len(samples)
        expected = np.zeros(len(audio), dtype=bool)
        if track == "microphone":
            expected[3200:8000] = True
        np.testing.assert_array_equal(muted, expected)
        assert not audio[expected].any()
        assert np.all(audio[~expected] > 1500)


def test_failed_dual_overwrite_preserves_active_take_and_can_export(recording, tmp_path):
    r = recording
    session = Session(tmp_path, "dual")
    rec = SessionRecorder(r.rec, MicrophoneRecorder(device=1, channels=1, clock=r.clock))
    rec.start(session)
    r.clock.value = 10.4
    rec.stop()
    original = session.recording_paths()

    r.clock.value = 10
    rec.start(session, overwrite=True)
    r.streams[-2].emit(0.05, value=1111)
    r.streams[-1].emit(0.05, value=2222, frames=200)
    r.streams[-1].active = False
    assert rec.status()["failed"]
    with pytest.raises(RecordingError, match="microphone"):
        rec.stop()
    assert session.recording_paths() == original
    assert not session.audio_path.exists()

    failed_take = next(p for p in session.root.glob("audio_parts-*") if p != original[0].parent)
    output = tmp_path / "recovered"
    args = SimpleNamespace(session="dual", take=failed_take.name, output=str(output))
    cmd_recover(args, {"output_dir": str(tmp_path)})
    assert 1111 in read_audio(output / "audio.wav")
    assert 2222 in read_audio(output / "microphone.wav", 2000)
    assert session.recording_paths() == original


def test_microphone_start_failure_never_publishes_single_track(recording, monkeypatch, tmp_path):
    r = recording
    start_stream = Stream.start_stream

    def fail_microphone(stream):
        if stream.rate == 2000:
            raise OSError("麦克风无法启动")
        start_stream(stream)

    monkeypatch.setattr(Stream, "start_stream", fail_microphone)
    session = Session(tmp_path, "dual")
    rec = SessionRecorder(r.rec, MicrophoneRecorder(device=1, channels=1, clock=r.clock))
    with pytest.raises(RecordingError, match="麦克风无法启动"):
        rec.start(session)
    assert not session.recording_manifest.exists()
    assert not session.audio_path.exists()
    assert all(stream.closed for stream in r.streams)
    with SessionLock(session.root):
        pass


def test_recover_interrupted_dual_recording_publishes_both_tracks(recording, tmp_path):
    r = recording
    session = Session(tmp_path, "interrupted")
    rec = SessionRecorder(r.rec, MicrophoneRecorder(channels=1, clock=r.clock))
    rec.start(session)
    r.streams[0].emit(0.1, value=111)
    r.streams[1].emit(0.2, value=222, frames=200)
    r.streams[0].active = r.streams[1].active = False
    with pytest.raises(RecordingError):
        rec.stop()
    assert session.recording_paths() == (session.audio_path, None)

    cmd_recover(
        SimpleNamespace(session=session.session_id, take=None, output=None),
        {"output_dir": str(tmp_path)},
    )
    loopback, microphone = session.recording_paths()
    assert microphone is not None
    assert np.flatnonzero(read_audio(loopback)).tolist() == list(range(100, 200))
    assert np.flatnonzero(read_audio(microphone, 2000)).tolist() == list(range(400, 600))
