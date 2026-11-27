from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest

from meetmin.config import load_config
from meetmin.gui import workflow
from meetmin.pipeline import Pipeline


def test_target_rms_settings_controls_and_save(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QPushButton

    from meetmin.config import DEFAULT_CONFIG, deep_merge
    from meetmin.gui import app
    from meetmin.gui.settings import SettingsDialog

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QApplication.instance() or QApplication([])
    cfg = deep_merge(DEFAULT_CONFIG, {"output_dir": str(tmp_path)})
    dialog = SettingsDialog(cfg, "")
    dialog.target_slider.setValue(87)
    assert dialog.target_rms.value() == pytest.approx(0.087)
    dialog.target_rms.setValue(0.123)
    assert dialog.target_slider.value() == 123
    next(
        button for button in dialog.findChildren(QPushButton) if button.text() == "恢复默认值"
    ).click()
    assert dialog.target_rms.value() == pytest.approx(0.03)
    dialog.target_rms.setValue(0.08)
    monkeypatch.setattr(app.MeetingWindow, "refresh_devices", lambda self: None)
    monkeypatch.setattr(app, "SettingsDialog", lambda *args: dialog)
    monkeypatch.setattr(dialog, "exec", lambda: 1)
    settings_path = str(tmp_path / "settings.ini")
    settings = QSettings(settings_path, QSettings.Format.IniFormat)
    window = app.MeetingWindow(cfg, settings)
    try:
        window.show_settings()
        settings.sync()
        assert cfg["recorder"]["target_rms"] == pytest.approx(0.08)
        restored = QSettings(settings_path, QSettings.Format.IniFormat)
        assert restored.value("target_rms", type=float) == pytest.approx(0.08)
        assert Pipeline(cfg).recorder().target_rms == pytest.approx(0.08)
    finally:
        window.close()
        dialog.close()
        application.processEvents()


def test_history_rename_and_delete(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QMessageBox

    from meetmin.config import DEFAULT_CONFIG, deep_merge
    from meetmin.gui import app

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr(app.MeetingWindow, "refresh_devices", lambda self: None)
    cfg = deep_merge(DEFAULT_CONFIG, {"output_dir": str(tmp_path)})
    first = Pipeline(cfg).new_session("first")
    first.save_json("meeting.json", {"title": "原名称", "extra": 42})
    first.audio_path.write_bytes(b"audio")
    second = Pipeline(cfg).new_session("second")
    window = app.MeetingWindow(
        cfg, QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    )
    window.refresh_history(second)
    try:
        window.show()
        application.processEvents()
        item = window.sessions.item(0)
        QTest.mouseMove(window.sessions.viewport(), window.sessions.visualItemRect(item).center())
        application.processEvents()
        assert window.sessions.button.isVisible()
        assert window.sessions.session_id == item.data(Qt.ItemDataRole.UserRole)
        monkeypatch.setattr(app.QInputDialog, "getText", lambda *a, **k: (" 新名称 ", True))
        window.history_action(first.session_id, "rename")
        assert workflow.metadata(first) == {"title": "新名称", "extra": 42}
        assert first.audio_path.read_bytes() == b"audio"
        assert window.session.session_id == second.session_id

        monkeypatch.setattr(app.QInputDialog, "getText", lambda *a, **k: ("   ", True))
        monkeypatch.setattr(app.QMessageBox, "warning", lambda *a, **k: None)
        window.history_action(first.session_id, "rename")
        assert workflow.metadata(first)["title"] == "新名称"

        monkeypatch.setattr(app.QMessageBox, "question", lambda *a, **k: QMessageBox.No)
        window.history_action(first.session_id, "delete")
        assert first.root.exists()
        monkeypatch.setattr(app.QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
        window.history_action(first.session_id, "delete")
        assert not first.root.exists()
        assert window.session.session_id == second.session_id
        window.history_action(second.session_id, "delete")
        assert window.session is None
        assert window.sessions.count() == 0
    finally:
        window.close()
        application.processEvents()


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


def test_live_speaker_name_refreshes_existing_rows_without_pinning(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from meetmin.gui.live_transcript import LiveTranscriptView

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QApplication.instance() or QApplication([])
    view = LiveTranscriptView()
    try:
        for segment, start in (("loopback-1", 0), ("loopback-2", 2000)):
            view.apply_event(
                {
                    "type": "asr.commit",
                    "segment_id": segment,
                    "track": "loopback",
                    "start_ms": start,
                    "text": "发言",
                    "speaker_pending": True,
                }
            )
            view.apply_event(
                {
                    "type": "diarization.assign",
                    "segment_id": segment,
                    "parts": [
                        {
                            "start_ms": start,
                            "speaker_id": 1,
                            "display_name": "说话人B",
                            "provisional": True,
                            "text": "发言",
                        }
                    ],
                }
            )
        view.apply_event(
            {
                "type": "speaker.rename",
                "speaker_id": 1,
                "display_name": "登记姓名",
                "pin_anchor": False,
            }
        )
        assert [(row.name, row.provisional) for row in view.model._rows] == [
            ("登记姓名", True),
            ("登记姓名", True),
        ]
    finally:
        view.close()
        application.processEvents()


def test_finished_transcript_speaker_names_persist_and_refresh(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from meetmin.config import DEFAULT_CONFIG, deep_merge
    from meetmin.gui import app

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr(app.MeetingWindow, "refresh_devices", lambda self: None)
    cfg = deep_merge(DEFAULT_CONFIG, {"output_dir": str(tmp_path)})
    pipe = Pipeline(cfg)
    session = pipe.new_session("imported")
    transcript = {
        "session_id": session.session_id,
        "duration": 2,
        "speakers": ["说话人A", "说话人B"],
        "utterances": [
            {"start": 0, "end": 1, "speaker": "说话人A", "text": "你好"},
            {"start": 1, "end": 2, "speaker": "说话人B", "text": "好的"},
        ],
    }
    pipe._save_transcript(session, transcript)
    pipe._save_corrected_transcript(session, transcript)
    session.save_text("minutes.md", "原纪要")
    window = app.MeetingWindow(
        cfg, QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    )
    window.refresh_history(session)
    monkeypatch.setattr(app.QInputDialog, "getItem", lambda *a, **k: ("说话人A", True))
    monkeypatch.setattr(app.QInputDialog, "getText", lambda *a, **k: ("张三", True))
    try:
        assert window.rename_speaker.isEnabled()
        window.rename_transcript_speaker()
        assert session.load_transcript()["utterances"][0]["speaker"] == "张三"
        assert "张三" in window.transcript.reader.toPlainText()
        assert "张三" in (session.root / "transcript_corrected.md").read_text(encoding="utf-8")
        assert "重新生成纪要" in window.stage.text()
        assert session.minutes_md.read_text(encoding="utf-8") == "原纪要"
        window.render_session()
        assert "张三" in window.transcript.reader.toPlainText()

        monkeypatch.setattr(app.QInputDialog, "getItem", lambda *a, **k: ("张三", True))
        monkeypatch.setattr(app.QInputDialog, "getText", lambda *a, **k: ("说话人B", True))
        warnings = []
        monkeypatch.setattr(app.QMessageBox, "warning", lambda *a: warnings.append(a[2]))
        window.rename_transcript_speaker()
        assert warnings and "重名" in warnings[0]
        assert session.load_transcript()["utterances"][0]["speaker"] == "张三"
    finally:
        window.close()
        application.processEvents()


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


@pytest.mark.parametrize("first_live, second_live", [(False, False), (True, False), (True, True)])
def test_recordings_can_overlap_background_transcription(
    tmp_path, monkeypatch, first_live, second_live
):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from meetmin.gui import app

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr(app.MeetingWindow, "refresh_devices", lambda self: None)
    cfg = {**load_config(tmp_path / "defaults.yaml"), "output_dir": str(tmp_path)}
    window = app.MeetingWindow(
        cfg, QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    )
    window.loopback_available = True
    window.update_controls()
    allow_merge, allow_first_asr, allow_second_asr = Event(), Event(), Event()
    started_recording, started_transcribing, live_sessions = [], [], []

    def title(session):
        return workflow.metadata(session)["title"]

    def wait_until(condition):
        for _ in range(300):
            if condition():
                return
            QTest.qWait(10)
        assert condition()

    def wait_for_gate(gate):
        assert gate.wait(10), "test did not release the background job"

    class Recorder:
        dual_track = False

        def start(self, session):
            self.session = session
            started_recording.append(title(session))

        def status(self):
            return {"failed": False}

        def set_audio_tap(self, _callback):
            pass

        def stop(self, *, on_device_released):
            on_device_released()
            if title(self.session) == "A":
                wait_for_gate(allow_merge)
            self.session.audio_path.write_bytes(title(self.session).encode())
            return {}

    def finish_transcript(session):
        session.save_text("transcript.md", f"会议 {title(session)} 的转写")
        return {"utterances": [{"text": title(session)}]}

    class LiveCoordinator:
        error = None

        def __init__(self, _cfg, session, emit, **_kwargs):
            self.session, self.emit = session, emit

        def start(self):
            name = title(self.session)
            live_sessions.append(name)
            self.emit({"type": "live.status", "message": f"实时转写会议 {name}"})

        def archive_started(self):
            pass

        def feed(self, _track, _event):
            pass

        def finish(self):
            if title(self.session) == "B":
                self.error = "实时音频积压超出缓冲区"
                return None
            started_transcribing.append("A")
            self.emit({"type": "live.status", "message": "仅属于 A 场的实时事件"})
            return finish_transcript(self.session)

        def close(self):
            if title(self.session) == "A":
                wait_for_gate(allow_first_asr)

    def transcribe(pipe, session, stage, **_kwargs):
        name = title(session)
        stage(f"正在转写会议 {name}")
        pipe.progress(f"会议 {name} 的日志")
        started_transcribing.append(name)
        if name == "A":
            wait_for_gate(allow_first_asr)
        elif name == "B":
            wait_for_gate(allow_second_asr)
        return finish_transcript(session)

    monkeypatch.setattr(Pipeline, "recorder", lambda *a, **k: Recorder())
    monkeypatch.setattr(workflow, "LiveSessionCoordinator", LiveCoordinator)
    monkeypatch.setattr(workflow, "transcribe", transcribe)

    def start(name, *, live=True):
        assert window.new.isEnabled()
        window.new.click()
        assert window.start.isEnabled()
        window.meeting_name.setText(name)
        window.live_enabled.setChecked(live)
        window.start.click()
        wait_until(lambda: name in started_recording)
        assert window.session is not None and title(window.session) == name
        return window.session

    try:
        first = start("A", live=first_live)
        window.stop.click()
        wait_until(window.new.isEnabled)
        assert not first.audio_path.exists()

        second = start("B", live=second_live)
        if second_live:
            view = window.transcript_stack.currentWidget()
            wait_until(lambda: "实时转写会议 B" in view.status.text())
        window.stop.click()
        wait_until(lambda: second.audio_path.exists() and "已排队" in window.stage.text())
        if second_live:
            assert "积压超出缓冲区" in window.log.toPlainText()
        assert not started_transcribing

        third = start("C", live=False)
        allow_merge.set()
        wait_until(lambda: started_transcribing == ["A"])
        assert title(window.session) == "C" and window.stop.isEnabled()
        allow_first_asr.set()
        wait_until(lambda: started_transcribing == ["A", "B"])
        assert title(window.session) == "C" and window.stop.isEnabled()
        assert "会议 A" not in window.log.toPlainText()
        assert "会议 B" not in window.log.toPlainText()
        assert live_sessions == (["A"] if first_live else []) + (["B"] if second_live else [])

        window.stop.click()
        wait_until(lambda: third.audio_path.exists() and "已排队" in window.stage.text())
        window.refresh_history(second)
        assert "正在转写会议 B" in window.stage.text()
        assert "会议 B 的日志" in window.log.toPlainText()
        allow_second_asr.set()
        wait_until(lambda: not window.jobs)
        assert started_transcribing == ["A", "B", "C"]
        for session, name in ((first, "A"), (second, "B"), (third, "C")):
            assert session.audio_path.read_bytes() == name.encode()
            assert f"会议 {name} 的转写" in session.transcript_md.read_text(encoding="utf-8")
    finally:
        allow_merge.set()
        allow_first_asr.set()
        allow_second_asr.set()
        window.request_stop()
        wait_until(lambda: not window.jobs)
        window.close()
        application.processEvents()


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
    pipe.diarize.assert_called_once()
    assert pipe.diarize.call_args.args == (session,)
    callback = pipe.diarize.call_args.kwargs["on_model_download"]
    callback(True)
    callback(False)
    assert stages == [
        "正在整理转写与说话人…",
        "正在下载说话人识别模型，首次下载可能需要较长时间…",
        "正在整理转写与说话人…",
    ]


def test_desktop_can_reuse_legacy_cli_external_audio(tmp_path):
    session = Pipeline({"output_dir": str(tmp_path)}).new_session("CLI")
    audio = Path(tmp_path / "original.wav")
    audio.touch()
    session.save_json("asr_raw.json", {"audio_path": str(audio)})
    assert workflow.available_audio(session) == (True, audio)


def test_default_desktop_config_falls_back_to_repo_config_and_respects_user_override(tmp_path):
    pytest.importorskip("PySide6")
    from meetmin.gui.app import _load_default_desktop_config

    repo_root = Path(__file__).resolve().parents[1]
    empty_appdata = tmp_path / "appdata"
    empty_appdata.mkdir()
    fallback_cfg = _load_default_desktop_config(empty_appdata)
    assert fallback_cfg["recorder"]["microphone"] is True
    assert fallback_cfg["asr"]["glossary"]["path"] == str(repo_root / "glossary.yaml")
    assert fallback_cfg["asr"]["glossary"]["correction_path"] == str(repo_root / "glossary.yaml")

    custom_config = empty_appdata / "config.yaml"
    custom_config.write_text(
        "asr:\n  beam_size: 3\nrecorder:\n  microphone: false\n", encoding="utf-8"
    )
    override_cfg = _load_default_desktop_config(empty_appdata)
    assert override_cfg["asr"]["beam_size"] == 3
    assert override_cfg["recorder"]["microphone"] is False
    assert override_cfg["asr"]["glossary"]["path"] is None
