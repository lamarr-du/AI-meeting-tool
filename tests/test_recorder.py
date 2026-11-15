"""录音生命周期、时间轴和故障测试，无需声卡或 PyAudio。"""

import threading
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from meetmin.audio.loopback_recorder import LoopbackRecorder, RecordingError


class Clock:
    value = 10.0

    def __call__(self):
        return self.value


class Stream:
    def __init__(self, clock, **kwargs):
        assert kwargs["start"] is False
        self.callback = kwargs["stream_callback"]
        self.clock = clock
        self.active = False
        self.closed = False
        self.start_error = None
        self.close_error = None
        self.stop_error = None

    def get_time(self):
        return self.clock() + 1000  # Deliberately different from the Python clock epoch.

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
        self.clock.value = received if received is not None else 10 + position + frames / 1000
        data = np.full(frames, value, dtype="<i2").tobytes()
        times = (
            {
                "input_buffer_adc_time": 1010 + position,
                "current_time": 1010 + position + frames / 1000,
            }
            if timing
            else {}
        )
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
                "defaultSampleRate": 1000,
                "maxInputChannels": 1,
                "index": index,
                "name": "fake",
            }

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
            paContinue=0,
            paAbort=2,
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


def read_audio(path):
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == 1000
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")


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


def test_stop_trims_pending_block(recording):
    r = recording
    r.rec.start(r.path)
    r.streams[-1].emit(0, received=10.05)
    r.clock.value = 10.05
    info = r.rec.stop()
    assert info["duration_sec"] == 0.05
    np.testing.assert_array_equal(read_audio(r.path), np.full(50, 1000))


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
    assert len(read_audio(r.path)) == 200


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


def test_stream_stopped_unexpectedly(recording):
    r = recording
    r.rec.start(r.path)
    r.streams[-1].active = False
    assert r.rec.status()["failed"]
    assert not r.rec.is_recording
    with pytest.raises(RecordingError, match="意外停止"):
        r.rec.stop()


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


@pytest.mark.parametrize("position", [-1, 0, 0.02, 10000])
def test_broken_timestamps_do_not_silently_drop_or_pad_audio(recording, position):
    r = recording
    r.rec.start(r.path)
    r.streams[-1].emit(0)
    r.streams[-1].emit(position, received=10.2)
    with pytest.raises(RecordingError, match="时间戳|时钟|重叠"):
        r.rec.stop()
    assert len(read_audio(r.path)) <= 200


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
    from meetmin.session import Session

    r = recording
    session = Session(str(r.path.parent), "test").ensure()
    pipe = SimpleNamespace(
        recorder=lambda: r.rec,
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
