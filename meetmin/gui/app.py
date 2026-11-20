from __future__ import annotations

import datetime as dt
import hashlib
import math
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any

import yaml
from PySide6.QtCore import QSettings, QStandardPaths, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QTextDocument
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..asr.model_download import ModelDownloadProgress
from ..audio.loopback_recorder import RecordingError, list_loopback_devices, list_microphone_devices
from ..audio.session_recorder import SessionRecorder
from ..config import deep_merge, load_config
from ..pipeline import Pipeline
from ..session import Session
from ..summarizer.llm_client import MockLLM, get_llm
from ..textutil import ts
from . import workflow
from .settings import SettingsDialog
from .style import STYLESHEET


def label(text: str = "", style: str = "") -> QLabel:
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setObjectName(style)
    return widget


class MarkdownPreview(QTextBrowser):
    def loadResource(self, resource_type: int, name: QUrl | str) -> Any:
        # Imported or generated Markdown can contain references to local or remote files.
        return None

    def show_file(self, path: Path | None, empty_text: str, heading: str = "") -> None:
        self.document().setDocumentMargin(16)
        try:
            if path is not None and path.is_file():
                text = path.read_text(encoding="utf-8")
                if heading and text.startswith(heading + "\n"):
                    text = text[len(heading) :].lstrip("\n")
                features = QTextDocument.MarkdownFeature
                self.document().setMarkdown(
                    text,
                    features.MarkdownDialectGitHub | features.MarkdownNoHTML,
                )
                self.verticalScrollBar().setValue(0)
                return
        except (OSError, ValueError) as error:
            empty_text = f"无法读取结果：{error}"
        self.setPlainText(empty_text)


class Job(QThread):
    stage = Signal(str)
    log = Signal(str)
    meter = Signal(dict)
    model_download = Signal(object)

    def __init__(self, action: Callable[[Job], Any], parent: QWidget):
        super().__init__(parent)
        self.action = action
        self.result: Any = None
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            self.result = self.action(self)
        except Exception as error:
            self.error = error


class MeetingWindow(QMainWindow):
    def __init__(self, cfg: dict, settings: QSettings):
        super().__init__()
        self.cfg = cfg
        self.settings = settings
        self.api_key = ""
        self.session: Session | None = None
        self.job: Job | None = None
        self.job_done: Callable[[Any], None] | None = None
        self.recorder: SessionRecorder | None = None
        self.stop_requested: Event | None = None
        self.recorded_at = 0.0
        self.close_after_recording = False
        self.has_audio = False
        self.recovery: list[Path] = []
        self.transcript_path: Path | None = None
        self.minutes_path: Path | None = None
        self.setWindowTitle("MeetMin · 会议纪要")
        self.resize(1100, 780)
        self.setMinimumSize(890, 650)
        self.build_interface()
        self.refresh_history()
        self.refresh_devices()

    def build_interface(self) -> None:
        workspace = QWidget()
        workspace.setObjectName("workspace")
        self.setCentralWidget(workspace)
        layout = QHBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(228)
        navigation = QVBoxLayout(sidebar)
        navigation.setContentsMargins(18, 24, 18, 20)
        navigation.setSpacing(10)
        navigation.addWidget(label("MeetMin", "brand"))
        self.new = QPushButton("＋ 新建会议")
        self.new.clicked.connect(self.new_meeting)
        navigation.addWidget(self.new)
        navigation.addSpacing(10)
        navigation.addWidget(label("会议历史", "section"))
        self.sessions = QListWidget()
        self.sessions.setAccessibleName("会议历史")
        self.sessions.currentItemChanged.connect(self.select_session)
        navigation.addWidget(self.sessions, 1)
        self.settings_button = QPushButton("设置")
        self.settings_button.clicked.connect(self.show_settings)
        navigation.addWidget(self.settings_button)
        layout.addWidget(sidebar)

        detail = QWidget()
        content = QVBoxLayout(detail)
        content.setContentsMargins(24, 22, 24, 20)
        content.setSpacing(12)
        self.title = label("新建会议", "title")
        self.title.setWordWrap(True)
        content.addWidget(self.title)
        content.addWidget(self.build_capture_card())

        self.stage = label("", "status")
        self.stage.setWordWrap(True)
        content.addWidget(self.stage)
        self.indicator = QProgressBar()
        self.indicator.setRange(0, 0)
        self.indicator.setTextVisible(False)
        content.addWidget(self.indicator)

        actions = QHBoxLayout()
        self.retry = QPushButton("转写音频")
        self.retry.clicked.connect(self.retry_transcription)
        self.generate = QPushButton("生成纪要")
        self.generate.setObjectName("primary")
        self.generate.clicked.connect(self.generate_minutes)
        self.more = QToolButton()
        self.more.setText("更多操作")
        self.more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.more)
        self.export_transcript = menu.addAction("导出转写…")
        self.export_transcript.triggered.connect(lambda: self.export_file(self.transcript_path))
        self.export_minutes = menu.addAction("导出纪要…")
        self.export_minutes.triggered.connect(lambda: self.export_file(self.minutes_path))
        menu.addSeparator()
        self.web_prompt = menu.addAction("导出 AI 网页提示词…")
        self.web_prompt.triggered.connect(self.export_prompt)
        self.web_minutes = menu.addAction("导入 AI 网页纪要…")
        self.web_minutes.triggered.connect(self.import_minutes)
        menu.addSeparator()
        self.open_folder = menu.addAction("打开会话文件夹")
        self.open_folder.triggered.connect(self.open_session_folder)
        self.more.setMenu(menu)
        actions.addWidget(self.generate)
        actions.addWidget(self.retry)
        actions.addWidget(self.more)
        actions.addStretch()
        content.addLayout(actions)

        self.tabs = QTabWidget()
        self.transcript = MarkdownPreview(openLinks=False)
        self.minutes = MarkdownPreview(openLinks=False)
        self.log = QPlainTextEdit(readOnly=True, maximumBlockCount=300)
        self.tabs.addTab(self.transcript, "转写")
        self.tabs.addTab(self.minutes, "纪要")
        self.tabs.addTab(self.log, "处理日志")
        content.addWidget(self.tabs, 1)
        layout.addWidget(detail, 1)

    def build_capture_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        grid = QGridLayout(card)
        grid.setContentsMargins(15, 13, 15, 13)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        heading = label("录音或导入", "section")
        self.meeting_name = QLineEdit(placeholderText="会议名称（可选）", maxLength=100)
        self.meeting_name.setAccessibleName("会议名称")
        self.start = QPushButton("开始录音")
        self.start.setObjectName("primary")
        self.start.clicked.connect(self.start_recording)
        self.stop = QPushButton("停止录音")
        self.stop.setObjectName("stop")
        self.stop.clicked.connect(self.request_stop)
        self.import_button = QPushButton("导入音频…")
        self.import_button.clicked.connect(self.import_audio)
        grid.addWidget(heading, 0, 0, 1, 2)
        grid.addWidget(self.meeting_name, 0, 2)
        grid.addWidget(self.start, 0, 3)
        grid.addWidget(self.stop, 0, 3)
        grid.addWidget(self.import_button, 0, 4)

        self.loopback_devices = QComboBox()
        self.loopback_devices.setAccessibleName("电脑声音设备")
        self.microphone_devices = QComboBox()
        self.microphone_devices.setAccessibleName("麦克风设备")
        self.microphone = QCheckBox("同时录制麦克风")
        self.microphone.setChecked(bool(self.cfg["recorder"].get("microphone")))
        self.microphone.toggled.connect(self.update_controls)
        self.refresh_audio = QPushButton("刷新设备")
        self.refresh_audio.clicked.connect(self.refresh_devices)
        grid.addWidget(label("电脑声音", "muted"), 1, 0)
        grid.addWidget(self.loopback_devices, 1, 1, 1, 2)
        grid.addWidget(self.refresh_audio, 1, 3, 1, 2)
        grid.addWidget(self.microphone, 2, 0, 1, 2)
        grid.addWidget(self.microphone_devices, 2, 2, 1, 3)

        self.device_hint = label("", "muted")
        self.device_hint.setWordWrap(True)
        grid.addWidget(self.device_hint, 3, 0, 1, 5)
        self.meter_area = QWidget()
        meter_layout = QGridLayout(self.meter_area)
        meter_layout.setContentsMargins(0, 0, 0, 0)
        self.track_labels = [label(), label()]
        self.track_levels = [QProgressBar(), QProgressBar()]
        for index, (caption, level) in enumerate(zip(self.track_labels, self.track_levels)):
            level.setAccessibleName(("电脑声音电平", "麦克风电平")[index])
            level.setTextVisible(False)
            meter_layout.addWidget(caption, index, 0)
            meter_layout.addWidget(level, index, 1)
        grid.addWidget(self.meter_area, 4, 0, 1, 5)
        grid.setColumnStretch(2, 1)
        self.capture_card = card
        self.loopback_available = self.mic_available = False
        return card

    def refresh_devices(self) -> None:
        if sys.platform != "win32":
            self.device_hint.setText("电脑声音录制需要 Windows。你可以导入已有音频。")
            self.update_controls()
            return
        errors = []
        for combo, list_devices, key, default in (
            (self.loopback_devices, list_loopback_devices, "device", "默认输出设备"),
            (self.microphone_devices, list_microphone_devices, "microphone_device", "默认麦克风"),
        ):
            selected = combo.currentData() if combo.count() else self.cfg["recorder"].get(key)
            combo.clear()
            try:
                devices = list_devices()
            except (OSError, RuntimeError) as error:
                devices = []
                errors.append(str(error))
            combo.addItem(default if devices else "未找到设备", None)
            for device in devices:
                combo.addItem(f"{device['name']}  [{device['index']}]", device["index"])
                if isinstance(selected, str) and selected.casefold() in device["name"].casefold():
                    selected = device["index"]
            combo.setCurrentIndex(max(0, combo.findData(selected)))
        self.loopback_available = self.loopback_devices.count() > 1
        self.mic_available = self.microphone_devices.count() > 1
        hint = (
            "双轨录音建议佩戴耳机，避免麦克风重复收录电脑声音。"
            if self.loopback_available
            else "未检测到 Windows 内录设备。请连接输出设备后刷新，或导入已有音频。"
        )
        if errors:
            self.log.appendPlainText("\n".join(errors))
        self.device_hint.setText(hint)
        self.update_controls()

    def refresh_history(self, selected: Session | None = None) -> None:
        self.sessions.blockSignals(True)
        self.sessions.clear()
        chosen = None
        try:
            for session in workflow.history(self.cfg["output_dir"]):
                title = workflow.metadata(session).get("title") or session.session_id
                item = QListWidgetItem(str(title))
                item.setData(Qt.ItemDataRole.UserRole, session.session_id)
                item.setToolTip(session.session_id)
                self.sessions.addItem(item)
                if selected and session.session_id == selected.session_id:
                    chosen = item
            if chosen is not None:
                self.sessions.setCurrentItem(chosen)
        except OSError as error:
            self.log.appendPlainText(f"无法读取会议历史：{error}")
        finally:
            self.sessions.blockSignals(False)
        self.select_session(chosen)

    def select_session(self, item: QListWidgetItem | None, previous=None) -> None:
        session = (
            Session(self.cfg["output_dir"], item.data(Qt.ItemDataRole.UserRole)) if item else None
        )
        if self.session is None or session is None or self.session.root != session.root:
            self.log.clear()
        self.session = session
        self.render_session()

    def render_session(self) -> None:
        session = self.session
        session_id = session.session_id if session else ""
        self.transcript_path = None
        self.minutes_path = None
        self.recovery = []
        self.has_audio = False
        message = "从录音或导入开始。"
        if session is not None:
            self.title.setText(str(workflow.metadata(session).get("title") or session.session_id))
            if session.minutes_md.is_file():
                self.minutes_path = session.minutes_md
            corrected = session.root / "transcript_corrected.md"
            if session.transcript_md.is_file():
                self.transcript_path = corrected if corrected.is_file() else session.transcript_md
            try:
                self.has_audio = workflow.available_audio(session)[0]
                self.recovery = workflow.recovery_sources(session)
                message = self.session_status(session)
            except (OSError, ValueError) as error:
                message = f"无法读取会话音频：{error}"
        else:
            self.title.setText("新建会议")
        self.stage.setText(message)
        self.transcript.show_file(
            self.transcript_path, "转写完成后在这里查看带时间戳的发言。", f"# 会议转写 {session_id}"
        )
        self.minutes.show_file(
            self.minutes_path,
            "完成转写后，可生成纪要或导入 AI 网页纪要。",
            f"# {session_id} 会议纪要",
        )
        self.update_controls()

    def session_status(self, session: Session) -> str:
        if self.minutes_path is not None:
            return "已生成纪要"
        if self.transcript_path is not None:
            return "转写已保存，可以生成纪要。"
        if (session.root / "asr_raw.json").is_file():
            return "原始转写已保存，可以继续整理说话人。"
        if self.has_audio:
            return "音频已保存，可以开始转写。"
        if self.recovery:
            return "检测到未完成的录音，可以从已保存的分段恢复。"
        return "该会话暂无可用音频。"

    def update_controls(self) -> None:
        running = self.job is not None
        stop = self.stop_requested
        recording = stop is not None
        cleanup = self.recorder is not None and not recording
        ready = not running and not cleanup
        self.capture_card.setVisible(self.session is None or recording)
        self.new.setEnabled(ready)
        self.sessions.setEnabled(ready)
        self.settings_button.setEnabled(ready)
        self.start.setVisible(not recording)
        self.start.setEnabled(ready and self.loopback_available)
        self.stop.setVisible(recording)
        self.stop.setEnabled(stop is not None and not stop.is_set())
        self.import_button.setEnabled(ready)
        self.meeting_name.setEnabled(ready)
        self.loopback_devices.setEnabled(ready and self.loopback_available)
        self.microphone.setEnabled(ready and self.mic_available)
        self.microphone_devices.setEnabled(
            ready and self.mic_available and self.microphone.isChecked()
        )
        self.refresh_audio.setEnabled(ready and sys.platform == "win32")
        self.meter_area.setVisible(recording)
        self.indicator.setVisible(running and (stop is None or stop.is_set()))
        raw = self.session is not None and (self.session.root / "asr_raw.json").is_file()
        self.retry.setVisible(cleanup or bool(self.recovery) or raw or self.has_audio)
        self.retry.setEnabled(not running)
        self.retry.setText(
            "重试保存录音"
            if cleanup
            else "恢复录音…"
            if self.recovery
            else "继续整理转写"
            if raw and self.transcript_path is None
            else "重新转写"
            if self.transcript_path
            else "转写音频"
        )
        self.generate.setVisible(self.session is not None)
        self.generate.setEnabled(ready and self.transcript_path is not None)
        self.generate.setText("重新生成纪要" if self.minutes_path is not None else "生成纪要")
        self.more.setVisible(self.session is not None)
        self.more.setEnabled(ready)
        self.export_transcript.setEnabled(self.transcript_path is not None)
        self.export_minutes.setEnabled(self.minutes_path is not None)
        can_use_web = self.transcript_path is not None
        self.web_prompt.setEnabled(can_use_web)
        self.web_minutes.setEnabled(can_use_web)

    def launch(
        self,
        starting: str,
        action: Callable[[Job], Any],
        done: Callable[[Any], None] | None = None,
    ) -> None:
        assert self.job is None
        self.show_stage(starting)
        self.job = Job(action, self)
        self.job_done = done
        self.job.stage.connect(self.show_stage)
        self.job.log.connect(self.log.appendPlainText)
        self.job.meter.connect(self.show_meter)
        self.job.model_download.connect(self.show_model_download)
        self.job.finished.connect(self.job_finished)
        self.job.start()
        self.update_controls()

    def show_stage(self, message: str) -> None:
        self.indicator.setRange(0, 0)
        self.stage.setText(message)

    def show_model_download(self, progress: ModelDownloadProgress) -> None:
        downloaded, total = progress.downloaded, progress.total
        if not total:
            amount = f"{downloaded / 2**20:.1f} MiB"
            self.show_stage(f"正在下载 Whisper 模型 {progress.model} · 已下载 {amount}")
            return
        fraction = downloaded / total
        self.indicator.setRange(0, 1000)
        self.indicator.setValue(int(fraction * 1000))
        size = f"{downloaded / 2**20:.1f} / {total / 2**20:.1f} MiB"
        if progress.complete:
            self.stage.setText(f"Whisper 模型 {progress.model} 已下载 · {size}")
            return
        if downloaded == total:
            timing = "正在校验文件"
        elif progress.bytes_per_second and progress.bytes_per_second > 0:
            seconds = math.ceil((total - downloaded) / progress.bytes_per_second)
            timing = f"预计剩余 {ts(seconds)}"
        else:
            timing = "正在估算剩余时间"
        self.stage.setText(
            f"正在下载 Whisper 模型 {progress.model} · {fraction:.1%} · {size} · {timing}"
        )

    def job_finished(self) -> None:
        job = self.job
        assert job is not None
        done = self.job_done
        self.job = self.job_done = None
        job.deleteLater()
        was_recording = self.recorder is not None
        if was_recording:
            self.stop_requested = None
            if not isinstance(job.error, RecordingError) or not job.error.info["cleanup_pending"]:
                self.recorder = None
        self.render_session()
        if job.error is not None:
            self.close_after_recording = False
            error = str(job.error)
            if self.recorder is not None:
                error += "\n录音文件仍在收尾，请点击「重试保存录音」。"
            elif was_recording and self.recovery:
                error += "\n已写入的分段仍在，可以点击「恢复录音」。"
            self.log.appendPlainText(error)
            self.stage.setText("处理失败 · " + error)
        elif done is not None:
            done(job.result)
        if self.close_after_recording and not self.job and self.recorder is None:
            self.close_after_recording = False
            self.close()

    def new_meeting(self) -> None:
        self.sessions.setCurrentRow(-1)
        self.tabs.setCurrentWidget(self.transcript)
        self.meeting_name.setFocus()

    def make_session(self, default_title: str) -> Session | None:
        title = self.meeting_name.text().strip() or default_title
        try:
            session = Pipeline(self.cfg).new_session(title)
            session.save_json("meeting.json", {"title": title})
        except (OSError, RuntimeError) as error:
            self.stage.setText(f"无法新建会话：{error}")
            return None
        self.meeting_name.clear()
        self.refresh_history(session)
        self.tabs.setCurrentWidget(self.transcript)
        return session

    def start_recording(self) -> None:
        session = self.make_session(dt.datetime.now().strftime("%m月%d日 %H:%M 会议"))
        if session is None:
            return
        cfg = deep_merge(
            self.cfg,
            {
                "recorder": {
                    "device": self.loopback_devices.currentData(),
                    "microphone": self.mic_available and self.microphone.isChecked(),
                    "microphone_device": self.microphone_devices.currentData(),
                }
            },
        )
        recorder = self.recorder = Pipeline(cfg).recorder(for_session=True)
        stop_requested = self.stop_requested = Event()
        self.recorded_at = time.monotonic()
        self.show_meter({"microphone": {} if recorder.dual_track else None})
        self.launch(
            "正在启动录音…",
            lambda job: workflow.record(
                recorder, session, stop_requested, job.stage.emit, job.meter.emit
            ),
            lambda result: self.recording_saved(session, result),
        )

    def request_stop(self) -> None:
        if self.stop_requested is not None:
            self.stop_requested.set()
            self.stage.setText("正在保存录音…")
            self.update_controls()

    def show_meter(self, status: dict) -> None:
        if self.stop_requested is None:
            return
        for index, (name, track) in enumerate(
            (("电脑声音", status), ("麦克风", status.get("microphone")))
        ):
            caption, meter = self.track_labels[index], self.track_levels[index]
            caption.setVisible(track is not None)
            meter.setVisible(track is not None)
            if track is None:
                continue
            no_data = track.get("no_data_sec", 0)
            duration = ts(track.get("duration_sec", 0))
            warning = " · 10 秒未收到数据，请检查设备" if no_data >= 10 else ""
            if not warning and track.get("silent_sec", 0) >= 30:
                warning = " · 持续低电平"
            caption.setText(f"{name}  {duration}{warning}")
            level = track.get("level", 0) if no_data < 2 else 0
            meter.setValue(min(100, int(max(0, level) ** 0.5 * 200)))
        if not self.stop_requested.is_set():
            self.stage.setText(f"正在录音 · {ts(time.monotonic() - self.recorded_at)}")

    def recording_saved(self, session: Session, result: dict) -> None:
        for caption, track in (("电脑声音", result), ("麦克风", result.get("microphone"))):
            if track and track.get("padded_sec", 0):
                self.log.appendPlainText(
                    f"{caption}有 {track['padded_sec']:.1f} 秒无数据，已补静音。"
                )
        if not self.close_after_recording:
            self.transcribe_session(session)

    def import_audio(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "导入会议音频",
            "",
            "音频 (*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus *.wma);;全部文件 (*)",
        )
        if not filename:
            return
        source = Path(filename)
        session = self.make_session(source.stem)
        if session is not None:
            self.launch(
                "正在复制音频到会话文件夹…",
                lambda _job: workflow.import_source(session, source),
                lambda _path: self.transcribe_session(session),
            )

    def transcribe_session(self, session: Session, *, align_only: bool = False) -> None:
        self.launch(
            "正在检查会话音频…",
            lambda job: workflow.transcribe(
                Pipeline(self.cfg, progress=job.log.emit),
                session,
                job.stage.emit,
                align_only=align_only,
                on_model_download=job.model_download.emit,
            ),
            self.transcription_saved,
        )

    def transcription_saved(self, result: dict) -> None:
        self.tabs.setCurrentWidget(self.transcript)
        if not result["utterances"]:
            self.stage.setText("转写未检测到发言。请检查录音设备或换一段音频。")
            self.generate.setEnabled(False)
        elif result.get("diarization_status", {}).get("status") == "fallback":
            self.stage.setText("转写已保存，内录说话人分离失败，已按单人显示。")

    def retry_transcription(self) -> None:
        session = self.session
        assert session is not None
        recorder = self.recorder
        if recorder is not None:
            self.launch(
                "正在收尾…",
                lambda _job: recorder.stop(),
                lambda result: self.recording_saved(session, result),
            )
        elif self.recovery:
            take = self.recovery[0]
            if len(self.recovery) > 1:
                name, ok = QInputDialog.getItem(
                    self, "恢复录音", "录音分段", [p.name for p in self.recovery], 0, False
                )
                if not ok:
                    return
                take = session.root / name
            self.launch(
                "正在恢复已保存的录音分段…",
                lambda _job: workflow.restore_recording(session, take),
                lambda _result: self.transcribe_session(session),
            )
        else:
            align_only = self.transcript_path is None and (session.root / "asr_raw.json").is_file()
            self.transcribe_session(session, align_only=align_only)

    def generate_minutes(self) -> None:
        session = self.session
        assert session is not None
        try:
            llm = get_llm(self.cfg, api_key=self.api_key or None)
        except (RuntimeError, ValueError) as error:
            self.stage.setText(f"请检查纪要设置：{error}")
            return
        if not isinstance(llm, MockLLM):
            prompt = (
                f"将把会议转写文本发送到下列接口生成纪要。\n\n"
                f"{self.cfg['llm']['base_url']}\n模型 {self.cfg['llm']['model']}"
            )
            if not workflow.api_key_available(self.cfg, self.api_key):
                prompt += "\n\n当前未配置 API Key。如果接口需要密钥，请先打开设置填写。"
            if self.minutes_path is not None:
                prompt += "\n\n生成成功后将替换当前纪要。"
            if (
                QMessageBox.question(
                    self, "确认生成纪要", prompt, defaultButton=QMessageBox.StandardButton.No
                )
                != QMessageBox.StandardButton.Yes
            ):
                return
        self.launch(
            "正在生成纪要…",
            lambda job: Pipeline(self.cfg, progress=job.log.emit).summarize(session, llm=llm),
            lambda _result: self.tabs.setCurrentWidget(self.minutes),
        )

    def export_file(self, path: Path | None) -> None:
        if path is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "导出文件", path.name, "文本文件 (*.md *.txt)"
        )
        if filename:
            try:
                target = Path(filename)
                if target.resolve() != path.resolve():
                    shutil.copyfile(path, target)
                self.stage.setText(f"已导出至 {target}")
            except OSError as error:
                self.stage.setText(f"导出失败：{error}")

    def export_prompt(self) -> None:
        session = self.session
        assert session is not None
        self.launch(
            "正在整理 AI 网页提示词…",
            lambda job: Pipeline(self.cfg, progress=job.log.emit).export_web_prompt(
                session, overwrite=True
            ),
            self.export_file,
        )

    def import_minutes(self) -> None:
        session = self.session
        assert session is not None
        filename, _ = QFileDialog.getOpenFileName(
            self, "导入 AI 网页纪要", "", "文本文件 (*.md *.txt);;全部文件 (*)"
        )
        if not filename:
            return
        overwrite = self.minutes_path is not None
        if (
            overwrite
            and QMessageBox.question(
                self,
                "替换纪要",
                "要用导入的文件替换当前纪要吗？",
                defaultButton=QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.launch(
            "正在导入纪要…",
            lambda job: Pipeline(self.cfg, progress=job.log.emit).import_web_minutes(
                session, filename, overwrite=overwrite
            ),
            lambda _result: self.tabs.setCurrentWidget(self.minutes),
        )

    def open_session_folder(self) -> None:
        if self.session is not None and not QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self.session.root))
        ):
            self.stage.setText(f"无法打开会话文件夹：{self.session.root}")

    def show_settings(self) -> None:
        dialog = SettingsDialog(self.cfg, self.api_key, self)
        if not dialog.exec():
            return
        output = dialog.output.text()
        selected = self.session if output == self.cfg["output_dir"] else None
        self.cfg["output_dir"] = output
        self.cfg["llm"].update(
            base_url=dialog.url.text().strip(), model=dialog.model.text().strip()
        )
        self.api_key = dialog.key.text().strip()
        self.settings.setValue("output_dir", output)
        self.settings.setValue("llm_url", self.cfg["llm"]["base_url"])
        self.settings.setValue("llm_model", self.cfg["llm"]["model"])
        self.refresh_history(selected)

    def closeEvent(self, event) -> None:
        if self.stop_requested is not None:
            self.close_after_recording = True
            self.request_stop()
            event.ignore()
        elif self.job is not None or self.recorder is not None:
            event.ignore()
            QMessageBox.information(
                self, "正在处理会话", "请等待当前任务完成。录音收尾失败时请先重试保存。"
            )
        else:
            event.accept()


def run(config: Path | None) -> int:
    app = QApplication([])
    app.setOrganizationName("MeetMin")
    app.setApplicationName("MeetMin")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    settings = QSettings()
    try:
        if config is not None:
            config = config.expanduser().resolve(strict=True)
            settings.beginGroup(
                "configurations/" + hashlib.sha256(str(config).encode()).hexdigest()
            )
            cfg = load_config(config)
            output = config.parent / Path(cfg["output_dir"]).expanduser()
            model = cfg["asr"].get("model_size")
            if isinstance(model, str) and (config.parent / Path(model).expanduser()).is_dir():
                cfg["asr"]["model_size"] = str(config.parent / Path(model).expanduser())
        else:
            data = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.AppLocalDataLocation
            )
            output = Path(data) / "data"
            cfg = load_config(Path(data) / "config.yaml")
            if not (Path(data) / "config.yaml").is_file():
                cfg["recorder"]["microphone"] = True
        cfg["output_dir"] = settings.value("output_dir", str(output), type=str)
        cfg["llm"]["base_url"] = settings.value("llm_url", cfg["llm"]["base_url"], type=str)
        cfg["llm"]["model"] = settings.value("llm_model", cfg["llm"]["model"], type=str)
        if getattr(sys, "frozen", False):
            cfg["diarization"]["provider"] = "single"
        window = MeetingWindow(cfg, settings)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError) as error:
        QMessageBox.critical(None, "无法启动 MeetMin", str(error))
        return 1
    window.show()
    return app.exec()
