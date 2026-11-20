from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest

from meetmin.config import load_config
from meetmin.gui import workflow
from meetmin.pipeline import Pipeline


def test_imported_audio_remains_available_after_original_moves(tmp_path):
    session = Pipeline({"output_dir": str(tmp_path / "data")}).new_session("计划会议")
    session.save_json("meeting.json", {"title": "计划会议"})
    original = tmp_path / "会议录音.WAV"
    original.write_bytes(b"audio")

    imported = workflow.import_source(session, original)
    original.unlink()

    assert workflow.available_audio(session) == (True, imported)
    assert imported.read_bytes() == b"audio"
    restored = workflow.history(str(tmp_path / "data"))[0]
    assert restored.session_id == session.session_id
    assert workflow.metadata(restored)["title"] == "计划会议"


@pytest.mark.parametrize("device_failed", [False, True])
def test_recording_stops_and_finalizes_on_button_or_device_failure(device_failed):
    stopped = Event()
    recorder = Mock()
    recorder.status.return_value = {"failed": device_failed}
    session = Mock()
    stages = []

    def meter(status):
        if not status["failed"]:
            stopped.set()

    result = workflow.record(recorder, session, stopped, stages.append, meter)

    recorder.start.assert_called_once_with(session)
    recorder.stop.assert_called_once()
    assert stages == ["正在录音", "正在保存录音…"]
    assert result == recorder.stop.return_value


def test_align_only_retry_does_not_need_original_audio_or_repeat_asr(tmp_path):
    cfg = load_config(tmp_path / "defaults.yaml")
    pipe = Mock(spec=Pipeline)
    pipe.cfg = cfg
    stages: list[str] = []
    session = Mock()

    assert (
        workflow.transcribe(pipe, session, stages.append, align_only=True)
        == pipe.diarize.return_value
    )
    pipe.transcribe.assert_not_called()
    pipe.diarize.assert_called_once_with(session)
    assert stages == ["正在整理转写与说话人…"]


def test_desktop_can_reuse_legacy_cli_external_audio(tmp_path):
    session = Pipeline({"output_dir": str(tmp_path)}).new_session("CLI")
    audio = Path(tmp_path / "original.wav")
    audio.touch()
    session.save_json("asr_raw.json", {"audio_path": str(audio)})
    assert workflow.available_audio(session) == (True, audio)
