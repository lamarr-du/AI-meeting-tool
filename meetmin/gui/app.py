from __future__ import annotations

import datetime as dt
import hashlib
import math
import shutil
import sys
import time
from collections.abc import Callable
from html import escape
from pathlib import Path
from queue import SimpleQueue
from threading import Event
from typing import Any, cast

import yaml
from PySide6.QtCore import QProcess, QSettings, QSize, QStandardPaths, Qt, QThread, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFontDatabase,
    QTextBlockFormat,
    QTextCursor,
    QTextDocument,
    QTextFrameFormat,
    QTextLength,
    QTextTable,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
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
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..asr.model_download import ModelDownloadProgress
from ..audio.loopback_recorder import RecordingError, list_loopback_devices, list_microphone_devices
from ..audio.session_recorder import RECORDING_DIAGNOSTICS, SessionRecorder
from ..config import MAX_TARGET_RMS, MIN_TARGET_RMS, deep_merge, load_config
from ..diarization.voiceprints import Enrollment
from ..live_pipeline import SpeakerRenames
from ..pipeline import Pipeline
from ..session import Session
from ..speakers import is_generated_speaker, is_person_name
from ..summarizer.llm_client import MockLLM, get_llm
from ..textutil import ts
from . import workflow
from .live_transcript import LiveTranscriptView
from .settings import SettingsDialog
from .style import (
    HISTORY_DATE,
    AudioLevel,
    DeviceComboBox,
    HistoryDelegate,
    apply_light_theme,
    line_icon,
    style_property,
)


def label(text: str = "", style: str = "") -> QLabel:
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setObjectName(style)
    return widget


def session_date(session: Session | None) -> dt.date | None:
    if session is None:
        return None
    parts = session.session_id.rsplit("-", 3)[-3:]
    if len(parts) == 3 and len(parts[0]) == 8:
        try:
            return dt.datetime.strptime(parts[0], "%Y%m%d").date()
        except ValueError:
            pass
    return None


class MarkdownBrowser(QTextBrowser):
    def loadResource(self, resource_type: int, name: QUrl | str) -> Any:
        # Imported or generated Markdown can contain references to local or remote files.
        return None


class MarkdownPreview(QStackedWidget):
    def __init__(self, name: str):
        super().__init__()
        self.reader = MarkdownBrowser(openLinks=False)
        self.reader.setAccessibleName(name)
        self.addWidget(self.reader)
        empty = QWidget()
        layout = QVBoxLayout(empty)
        layout.setContentsMargins(20, 14, 20, 18)
        layout.setSpacing(12)
        mark = label(style="emptyMark")
        mark.setFixedSize(42, 42)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setPixmap(line_icon("note").pixmap(QSize(20, 20), self.devicePixelRatioF()))
        self.message = label(style="emptyMessage")
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
        layout.addWidget(mark, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.message)
        layout.addStretch(1)
        self.addWidget(empty)

    def show_file(self, path: Path | None, empty_text: str, heading: str = "") -> None:
        document = self.reader.document()
        document.setDocumentMargin(26)
        try:
            if path is not None and path.is_file():
                text = path.read_text(encoding="utf-8")
                if heading and text.startswith(heading + "\n"):
                    text = text[len(heading) :].lstrip("\n")
                features = QTextDocument.MarkdownFeature
                document.setMarkdown(text, features.MarkdownDialectGitHub | features.MarkdownNoHTML)
                if not document.isEmpty():
                    document.setLayoutEnabled(False)
                    try:
                        self.format_document(document)
                    finally:
                        document.setLayoutEnabled(True)
                    self.reader.verticalScrollBar().setValue(0)
                    self.setCurrentWidget(self.reader)
                    return
        except (OSError, ValueError) as error:
            empty_text = f"无法读取结果：{error}"
        self.reader.clear()
        self.message.setText(empty_text)
        self.setCurrentIndex(1)

    @staticmethod
    def format_document(document: QTextDocument) -> None:
        for frame in document.rootFrame().childFrames():
            if not isinstance(frame, QTextTable):
                continue
            table_format = frame.format()
            table_format.setBorder(1)
            table_format.setBorderStyle(QTextFrameFormat.BorderStyle.BorderStyle_Solid)
            table_format.setBorderBrush(QColor("#e5def0"))
            table_format.setCellPadding(8)
            table_format.setCellSpacing(0)
            table_format.setTopMargin(7)
            table_format.setWidth(QTextLength(QTextLength.Type.PercentageLength, 100))
            table_format.setHeaderRowCount(1)
            frame.setFormat(table_format)
            for column in range(frame.columns()):
                cell = frame.cellAt(0, column)
                cell_format = cell.format()
                cell_format.setBackground(QColor("#f7f3fc"))
                cell.setFormat(cell_format)

        block = document.begin()
        while block.isValid():
            cursor = QTextCursor(block)
            if cursor.currentTable() is None:
                fmt = block.blockFormat()
                if fmt.headingLevel():
                    fmt.setTopMargin(16 if block.position() else 0)
                    fmt.setBottomMargin(6)
                else:
                    fmt.setBottomMargin(max(7, fmt.bottomMargin()))
                    fmt.setLineHeight(
                        142, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value
                    )
                cursor.setBlockFormat(fmt)
            block = block.next()


class Job(QThread):
    stage = Signal(str)
    log = Signal(str)
    meter = Signal(dict)
    model_download = Signal(object)
    live_event = Signal(dict)

    def __init__(self, action: Callable[[Job], Any], parent: QWidget):
        super().__init__(parent)
        self.action = action
        self.result: Any = None
        self.error: Exception | None = None
        self.speaker_renames: SpeakerRenames = SimpleQueue()

    def run(self) -> None:
        try:
            self.result = self.action(self)
        except Exception as error:
            self.error = error


class HistoryList(QListWidget):
    action_requested = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.button = QToolButton(self.viewport())
        self.button.setObjectName("historyActions")
        self.button.setText("⋯")
        self.button.setAccessibleName("会议操作")
        self.button.setFixedSize(28, 28)
        self.button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.session_id = ""
        menu = QMenu(self.button)
        menu.addAction("改名…", lambda: self.action_requested.emit(self.session_id, "rename"))
        menu.addAction("删除…", lambda: self.action_requested.emit(self.session_id, "delete"))
        self.button.setMenu(menu)
        self.button.hide()
        self.verticalScrollBar().valueChanged.connect(self.button.hide)

    def mouseMoveEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        if item is None:
            self.button.hide()
        else:
            self.session_id = item.data(Qt.ItemDataRole.UserRole)
            rect = self.visualItemRect(item)
            self.button.move(rect.right() - self.button.width() - 7, rect.center().y() - 14)
            self.button.show()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self.button.hide()
        super().leaveEvent(event)


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
        self.exported_prompt_path: Path | None = None
        self.setWindowTitle("MeetMin · 会议纪要")
        self.resize(1180, 780)
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
        sidebar.setFixedWidth(226)
        navigation = QVBoxLayout(sidebar)
        navigation.setContentsMargins(16, 24, 16, 16)
        navigation.setSpacing(12)
        branding = QHBoxLayout()
        branding.setSpacing(10)
        mark = label(style="brandMark")
        mark.setFixedSize(38, 38)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setPixmap(line_icon("audio").pixmap(QSize(20, 20), self.devicePixelRatioF()))
        branding.addWidget(mark)
        branding.addWidget(label("MeetMin", "brand"), 1)
        navigation.addLayout(branding)
        navigation.addSpacing(4)
        self.new = QPushButton("＋ 新建会议")
        self.new.setObjectName("newMeeting")
        self.new.clicked.connect(self.new_meeting)
        navigation.addWidget(self.new)
        navigation.addSpacing(8)
        navigation.addWidget(label("会议历史", "navCaption"))
        self.sessions = HistoryList()
        self.sessions.setObjectName("historyList")
        self.sessions.setUniformItemSizes(True)
        self.sessions.setMouseTracking(True)
        self.sessions.setItemDelegate(HistoryDelegate(self.sessions))
        self.sessions.setAccessibleName("会议历史")
        self.sessions.currentItemChanged.connect(self.select_session)
        self.sessions.action_requested.connect(self.history_action)
        navigation.addWidget(self.sessions, 1)
        divider = QFrame()
        divider.setObjectName("navDivider")
        divider.setFixedHeight(1)
        navigation.addWidget(divider)
        self.settings_button = QPushButton("设置")
        self.settings_button.setObjectName("settingsButton")
        self.settings_button.setIcon(line_icon("settings"))
        self.settings_button.clicked.connect(self.show_settings)
        navigation.addWidget(self.settings_button)
        layout.addWidget(sidebar)

        detail = QWidget()
        content = QVBoxLayout(detail)
        content.setContentsMargins(28, 24, 28, 24)
        content.setSpacing(16)
        heading = QVBoxLayout()
        heading.setSpacing(3)
        self.title = label("新建会议", "title")
        self.title.setWordWrap(True)
        self.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        heading.addWidget(self.title)
        self.date = label(style="muted")
        heading.addWidget(self.date)
        content.addLayout(heading)
        content.addWidget(self.build_capture_card())

        self.status_strip = QFrame()
        self.status_strip.setObjectName("statusStrip")
        status = QVBoxLayout(self.status_strip)
        status.setContentsMargins(16, 10, 12, 10)
        status.setSpacing(8)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.status_dot = label(style="statusDot")
        self.status_dot.setFixedSize(8, 8)
        actions.addWidget(self.status_dot, alignment=Qt.AlignmentFlag.AlignVCenter)
        self.stage = label("", "status")
        self.stage.setWordWrap(True)
        self.stage.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        actions.addWidget(self.stage, 1)
        status.addLayout(actions)
        self.prompt_file_actions = QWidget()
        file_actions = QHBoxLayout(self.prompt_file_actions)
        file_actions.setContentsMargins(0, 0, 0, 0)
        open_prompt = QPushButton("打开文件")
        open_prompt.clicked.connect(self.open_exported_prompt)
        reveal_prompt = QPushButton("在文件夹中显示")
        reveal_prompt.clicked.connect(lambda: self.open_exported_prompt(reveal=True))
        file_actions.addWidget(open_prompt)
        file_actions.addWidget(reveal_prompt)
        file_actions.addStretch()
        status.addWidget(self.prompt_file_actions)
        self.indicator = QProgressBar()
        self.indicator.setAccessibleName("处理进度")
        self.indicator.setRange(0, 0)
        self.indicator.setTextVisible(False)
        self.indicator.setFixedHeight(7)
        status.addWidget(self.indicator)
        content.addWidget(self.status_strip)
        self.retry = QPushButton("转写音频")
        self.retry.clicked.connect(self.retry_transcription)
        self.generate = QPushButton("生成纪要")
        self.generate.clicked.connect(self.generate_minutes)
        self.more = QToolButton()
        self.more.setObjectName("moreActions")
        self.more.setText("更多操作")
        self.more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.more)
        self.export_transcript = menu.addAction("导出转写…")
        self.export_transcript.triggered.connect(lambda: self.export_file(self.transcript_path))
        self.export_minutes = menu.addAction("导出纪要…")
        self.export_minutes.triggered.connect(lambda: self.export_file(self.minutes_path))
        menu.addSeparator()
        self.edit_speaker = menu.addAction("编辑说话人姓名…")
        self.edit_speaker.triggered.connect(self.edit_speaker_name)
        self.enroll_speakers = menu.addAction("登记到公共声纹库…")
        self.enroll_speakers.triggered.connect(self.enroll_speaker_voiceprints)
        menu.addSeparator()
        self.web_prompt = menu.addAction("导出 AI 网页提示词…")
        self.web_prompt.triggered.connect(self.export_prompt)
        self.web_minutes = menu.addAction("导入 AI 网页纪要…")
        self.web_minutes.triggered.connect(self.import_minutes)
        menu.addSeparator()
        self.open_folder = menu.addAction("打开会话文件夹")
        self.open_folder.triggered.connect(self.open_session_folder)
        self.more.setMenu(menu)
        actions.addWidget(self.retry)
        actions.addWidget(self.generate)
        actions.addWidget(self.more)

        result = QFrame()
        result.setObjectName("resultCard")
        result_layout = QVBoxLayout(result)
        result_layout.setContentsMargins(1, 1, 1, 1)
        self.tabs = QTabWidget()
        self.transcript = MarkdownPreview("会议转写")
        self.live_transcript = LiveTranscriptView()
        self.live_transcript.rename_requested.connect(self.rename_live_speaker)
        self.transcript_stack = QStackedWidget()
        self.transcript_stack.addWidget(self.transcript)
        self.transcript_stack.addWidget(self.live_transcript)
        self.minutes = MarkdownPreview("会议纪要")
        self.log = QPlainTextEdit(readOnly=True, maximumBlockCount=300)
        self.log.setAccessibleName("处理日志")
        self.tabs.addTab(self.transcript_stack, "转写")
        self.tabs.addTab(self.minutes, "纪要")
        self.tabs.addTab(self.log, "处理日志")
        result_layout.addWidget(self.tabs)
        content.addWidget(result, 1)
        layout.addWidget(detail, 1)

    def build_capture_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("captureCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 20, 22, 20)
        self.capture_setup = self.build_capture_setup()
        self.capture_recording = self.build_recording_panel()
        layout.addWidget(self.capture_setup)
        layout.addWidget(self.capture_recording)
        self.capture_card = card
        self.loopback_available = self.mic_available = False
        return card

    def build_capture_setup(self) -> QWidget:
        panel = QWidget()
        columns = QHBoxLayout(panel)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(18)
        form = QVBoxLayout()
        form.setSpacing(9)
        form.addWidget(label("录音或导入", "captureTitle"))
        form.addSpacing(9)
        form.addWidget(label("会议名称", "fieldLabel"))
        self.meeting_name = QLineEdit(placeholderText="输入会议名称（可选）", maxLength=100)
        self.meeting_name.setAccessibleName("会议名称")
        form.addWidget(self.meeting_name)
        form.addSpacing(9)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.start = QPushButton("开始录音")
        self.start.setMinimumWidth(106)
        self.start.setProperty("primaryAction", True)
        self.start.setIcon(line_icon("audio", "#ffffff"))
        self.start.clicked.connect(self.start_recording)
        self.import_button = QPushButton("导入音频…")
        self.import_button.setMinimumWidth(104)
        self.import_button.clicked.connect(self.import_audio)
        buttons.addWidget(self.start)
        buttons.addWidget(self.import_button)
        buttons.addStretch()
        form.addLayout(buttons)
        self.live_enabled = QCheckBox("边录边转写")
        self.live_enabled.setChecked(
            cast(
                bool,
                self.settings.value(
                    "live_enabled", bool((self.cfg.get("live") or {}).get("enabled")), type=bool
                ),
            )
        )
        self.live_enabled.toggled.connect(
            lambda checked: self.settings.setValue("live_enabled", checked)
        )
        form.addWidget(self.live_enabled)
        form.addStretch()
        columns.addLayout(form, 5)

        device_panel = QFrame()
        device_panel.setObjectName("devicePanel")
        devices = QVBoxLayout(device_panel)
        devices.setContentsMargins(16, 10, 16, 14)
        devices.setSpacing(6)
        device_heading = QHBoxLayout()
        device_heading.addWidget(label("录音设备", "section"), 1)
        self.refresh_audio = QPushButton("刷新设备")
        self.refresh_audio.setObjectName("refreshAudio")
        self.refresh_audio.clicked.connect(self.refresh_devices)
        device_heading.addWidget(self.refresh_audio)
        devices.addLayout(device_heading)
        self.loopback_devices = DeviceComboBox("电脑声音设备")
        self.microphone_devices = DeviceComboBox("麦克风设备")
        devices.addWidget(label("电脑声音", "fieldLabel"))
        devices.addWidget(self.loopback_devices)
        devices.addSpacing(4)
        self.microphone = QCheckBox("同时录制麦克风")
        self.microphone.setChecked(bool(self.cfg["recorder"].get("microphone")))
        self.microphone.toggled.connect(self.update_controls)
        devices.addWidget(self.microphone)
        devices.addWidget(self.microphone_devices)
        columns.addWidget(device_panel, 6)

        self.device_hint = label("", "muted")
        self.device_hint.setWordWrap(True)
        devices.addWidget(self.device_hint)
        return panel

    def build_recording_panel(self) -> QWidget:
        panel = QWidget()
        columns = QHBoxLayout(panel)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(18)
        timer = QVBoxLayout()
        timer.setSpacing(8)
        self.record_badge = label("正在启动录音…", "recordBadge")
        timer.addWidget(self.record_badge, alignment=Qt.AlignmentFlag.AlignLeft)
        self.record_clock = label("00:00:00", "recordClock")
        self.record_clock.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.record_clock.setAccessibleName("录音时长")
        timer.addWidget(self.record_clock)
        controls = QHBoxLayout()
        self.stop = QPushButton("停止录音")
        self.stop.setObjectName("stop")
        self.stop.setIcon(line_icon("stop", "#89253e"))
        self.stop.clicked.connect(self.request_stop)
        controls.addWidget(self.stop)
        self.mute_microphone = QPushButton("禁言麦克风")
        self.mute_microphone.setObjectName("muteMicrophone")
        self.mute_microphone.setCheckable(True)
        self.mute_microphone.setToolTip("仅控制 MeetMin 录制的麦克风音轨")
        self.mute_microphone.toggled.connect(self.toggle_microphone_mute)
        controls.addWidget(self.mute_microphone)
        self.record_hint = label("", "muted")
        self.record_hint.setWordWrap(True)
        controls.addWidget(self.record_hint, 1)
        timer.addLayout(controls)
        timer.addStretch()
        columns.addLayout(timer, 5)

        meter_panel = QFrame()
        meter_panel.setObjectName("meterPanel")
        meters = QVBoxLayout(meter_panel)
        meters.setContentsMargins(15, 13, 15, 13)
        meters.setSpacing(10)
        self.tracks: list[tuple[QWidget, QLabel, AudioLevel]] = []
        for name, color in (("电脑声音", "#7655d6"), ("麦克风", "#9270c2")):
            track = QWidget()
            readings = QVBoxLayout(track)
            readings.setContentsMargins(0, 0, 0, 0)
            readings.setSpacing(5)
            caption = label(name, "fieldLabel")
            caption.setWordWrap(True)
            level = AudioLevel(f"{name}电平", color)
            readings.addWidget(caption)
            readings.addWidget(level)
            self.tracks.append((track, caption, level))
            meters.addWidget(track)
        columns.addWidget(meter_panel, 6)
        return panel

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
                date = session_date(session)
                if date is not None:
                    caption = f"{date.month} 月 {date.day} 日"
                    item.setData(HISTORY_DATE, caption)
                    item.setData(Qt.ItemDataRole.AccessibleTextRole, f"{title}，{caption}")
                item.setToolTip(f"<qt>{escape(str(title))}<br>{escape(session.session_id)}</qt>")
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

    def history_action(self, session_id: str, action: str) -> None:
        session = Session(self.cfg["output_dir"], session_id)
        title = str(workflow.metadata(session).get("title") or session_id)
        selected = self.session
        if action == "rename":
            name, accepted = QInputDialog.getText(self, "改名", "会议名称", text=title)
            if not accepted:
                return
            name = name.strip()
            if not name:
                QMessageBox.warning(self, "无法改名", "会议名称不能为空。")
                return
            try:
                if not session.root.is_dir() or session.root.is_symlink():
                    raise FileNotFoundError("会话文件夹不存在或不可用")
                metadata = workflow.metadata(session)
                metadata["title"] = name
                session.save_json("meeting.json", metadata)
            except OSError as error:
                QMessageBox.warning(self, "改名失败", str(error))
                return
        elif action == "delete":
            if (
                QMessageBox.question(
                    self,
                    "删除会议",
                    f"确定删除会议「{title}」吗？\n\n会话文件夹中的音频、转写和纪要等文件都将被永久删除。",
                    defaultButton=QMessageBox.StandardButton.No,
                )
                != QMessageBox.StandardButton.Yes
            ):
                return
            try:
                shutil.rmtree(session.root)
            except OSError as error:
                QMessageBox.warning(self, "删除失败", str(error))
                self.refresh_history(selected)
                return
            if selected is not None and selected.session_id == session_id:
                selected = None
        else:
            return
        self.sessions.button.hide()
        self.refresh_history(selected)

    def render_session(self) -> None:
        session = self.session
        session_id = session.session_id if session else ""
        self.transcript_path = None
        self.minutes_path = None
        self.recovery = []
        self.has_audio = False
        message, tone = "", "neutral"
        date = session_date(session)
        self.date.setVisible(date is not None)
        self.date.setText(f"{date.year} 年 {date.month} 月 {date.day} 日" if date else "")
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
                if self.minutes_path is not None or self.transcript_path is not None:
                    tone = "success"
                elif self.recovery:
                    tone = "warning"
            except (OSError, ValueError) as error:
                message = f"无法读取会话音频：{error}"
                tone = "error"
        else:
            self.title.setText("新建会议")
        self.set_status(message, tone)
        if self.stop_requested is None:
            self.transcript_stack.setCurrentWidget(self.transcript)
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

    def recording_active(self) -> bool:
        return self.stop_requested is not None and not self.stop_requested.is_set()

    def set_status(
        self, message: str, tone: str = "neutral", *, prompt: Path | None = None
    ) -> None:
        style_property(self.status_dot, "tone", tone)
        self.stage.setText(message)
        self.exported_prompt_path = prompt
        self.prompt_file_actions.setVisible(prompt is not None)
        self.status_strip.setVisible(bool(message) and not self.recording_active())

    def update_controls(self) -> None:
        running = self.job is not None
        stop = self.stop_requested
        capturing = self.recording_active()
        cleanup = self.recorder is not None and stop is None
        ready = not running and not cleanup
        setup = self.session is None and not running and stop is None
        self.capture_card.setVisible(setup or capturing)
        self.capture_setup.setVisible(setup)
        self.capture_recording.setVisible(capturing)
        style_property(self.new, "primaryAction", self.session is not None and ready)
        self.new.setEnabled(ready)
        self.sessions.setEnabled(ready)
        self.settings_button.setEnabled(ready)
        self.start.setEnabled(ready and self.loopback_available)
        self.stop.setEnabled(capturing)
        self.mute_microphone.setVisible(
            capturing and self.recorder is not None and self.recorder.dual_track
        )
        self.import_button.setEnabled(ready)
        self.meeting_name.setEnabled(ready)
        self.live_enabled.setEnabled(ready)
        self.loopback_devices.setEnabled(ready and self.loopback_available)
        self.microphone.setEnabled(ready and self.mic_available)
        self.microphone_devices.setEnabled(
            ready and self.mic_available and self.microphone.isChecked()
        )
        self.refresh_audio.setEnabled(ready and sys.platform == "win32")
        self.indicator.setVisible(running and not capturing)
        raw = self.session is not None and (self.session.root / "asr_raw.json").is_file()
        self.retry.setVisible(
            not running and (cleanup or bool(self.recovery) or raw or self.has_audio)
        )
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
        show_actions = self.session is not None and not running
        self.generate.setVisible(show_actions)
        self.generate.setEnabled(ready and self.transcript_path is not None)
        self.generate.setText("重新生成纪要" if self.minutes_path is not None else "生成纪要")
        style_property(self.generate, "primaryAction", self.minutes_path is None)
        self.more.setVisible(show_actions)
        self.more.setEnabled(ready)
        self.export_transcript.setEnabled(self.transcript_path is not None)
        self.export_minutes.setEnabled(self.minutes_path is not None)
        self.edit_speaker.setEnabled(ready and self.transcript_path is not None)
        self.enroll_speakers.setEnabled(ready and self.transcript_path is not None)
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
        self.job.live_event.connect(self.live_transcript.apply_event)
        self.job.finished.connect(self.job_finished)
        self.job.start()
        self.update_controls()

    def show_stage(self, message: str) -> None:
        self.indicator.setRange(0, 0)
        if self.recording_active():
            self.record_badge.setText(message)
        self.set_status(message)

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
            self.set_status(f"Whisper 模型 {progress.model} 已下载 · {size}")
            return
        if downloaded == total:
            timing = "正在校验文件"
        elif progress.bytes_per_second and progress.bytes_per_second > 0:
            seconds = math.ceil((total - downloaded) / progress.bytes_per_second)
            timing = f"预计剩余 {ts(seconds)}"
        else:
            timing = "正在估算剩余时间"
        self.set_status(
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
            if (
                was_recording
                and self.session is not None
                and (self.session.root / RECORDING_DIAGNOSTICS).is_file()
            ):
                error += "\n录音诊断日志已保存到会话文件夹。"
            self.log.appendPlainText(error)
            self.set_status("处理失败 · " + error, "error")
        elif done is not None:
            done(job.result)
        if self.close_after_recording and not self.job and self.recorder is None:
            self.close_after_recording = False
            self.close()

    def new_meeting(self) -> None:
        self.sessions.setCurrentRow(-1)
        self.tabs.setCurrentWidget(self.transcript_stack)
        self.meeting_name.setFocus()

    def make_session(self, default_title: str) -> Session | None:
        title = self.meeting_name.text().strip() or default_title
        try:
            session = Pipeline(self.cfg).new_session(title)
            session.save_json("meeting.json", {"title": title})
        except (OSError, RuntimeError) as error:
            self.set_status(f"无法新建会话：{error}", "error")
            return None
        self.meeting_name.clear()
        self.refresh_history(session)
        self.tabs.setCurrentWidget(self.transcript_stack)
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
        self.mute_microphone.setChecked(False)
        self.recorded_at = time.monotonic()
        self.show_meter({"microphone": {} if recorder.dual_track else None})
        live = self.live_enabled.isChecked()
        self.record_hint.setText("停止后归档转写" if live else "停止后自动转写")
        if live:
            self.live_transcript.reset(dual_track=recorder.dual_track)
            self.transcript_stack.setCurrentWidget(self.live_transcript)
            self.tabs.setCurrentWidget(self.transcript_stack)

        def record_job(job: Job) -> dict:
            args = (recorder, session, stop_requested, job.stage.emit, job.meter.emit)
            if live:
                return workflow.record_live(
                    *args, cfg=cfg, emit=job.live_event.emit, renames=job.speaker_renames
                )
            return workflow.record(*args)

        self.launch(
            "正在启动录音…",
            record_job,
            lambda result: self.recording_saved(session, result),
        )

    def request_stop(self) -> None:
        if self.stop_requested is not None:
            self.live_transcript.renaming_enabled = False
            self.stop_requested.set()
            self.set_status("正在保存录音…")
            self.update_controls()

    def toggle_microphone_mute(self, muted: bool) -> None:
        if self.recorder is not None and self.recording_active():
            self.recorder.set_microphone_muted(muted)
        self.mute_microphone.setText("解除麦克风禁言" if muted else "禁言麦克风")
        _, caption, meter = self.tracks[1]
        caption.setText("麦克风已禁言" if muted else "麦克风")
        meter.setValue(0)

    def rename_live_speaker(self, speaker: int, name: str) -> None:
        if self.job is not None and self.recording_active():
            self.job.speaker_renames.put((speaker, name))

    def show_meter(self, status: dict) -> None:
        if self.stop_requested is None:
            return
        for (widget, caption, meter), (name, track) in zip(
            self.tracks, (("电脑声音", status), ("麦克风", status.get("microphone")))
        ):
            widget.setVisible(track is not None)
            if track is None:
                continue
            if name == "麦克风" and self.mute_microphone.isChecked():
                continue
            no_data = track.get("no_data_sec", 0)
            duration = ts(track.get("duration_sec", 0))
            warning = " · 10 秒未收到数据，请检查设备" if no_data >= 10 else ""
            if not warning and track.get("silent_sec", 0) >= 30:
                warning = " · 持续低电平"
            caption.setText(f"{name}  {duration}{warning}")
            level = track.get("level", 0) if no_data < 2 else 0
            meter.setValue(min(100, int(max(0, level) ** 0.5 * 200)))
        if self.recording_active():
            elapsed = max(0, int(time.monotonic() - self.recorded_at))
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.record_clock.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def recording_saved(self, session: Session, result: dict) -> None:
        for caption, track in (("电脑声音", result), ("麦克风", result.get("microphone"))):
            if track and track.get("padded_sec", 0):
                self.log.appendPlainText(
                    f"{caption}有 {track['padded_sec']:.1f} 秒无数据，已补静音。"
                )
            if track and track.get("repaired_timestamp_blocks", 0):
                self.log.appendPlainText(
                    f"{caption}有 {track['repaired_timestamp_blocks']} 个音频块的采集时间戳异常，"
                    "已按连续样本保留录音。"
                )
            if track and track.get("stream_restarts", 0):
                self.log.appendPlainText(
                    f"{caption}的录音流曾中断并已自动重启。中断前后的音频均已保留。"
                )
        if transcript := result.get("live_transcript"):
            self.set_status("实时转写已归档。", "success")
            self.transcription_saved(transcript)
        elif error := result.get("live_error"):
            self.log.appendPlainText(f"实时转写未完成：{error}")
            self.set_status("录音已保存，可继续转写。", "warning")
        elif not self.close_after_recording:
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
        self.tabs.setCurrentWidget(self.transcript_stack)
        if not result["utterances"]:
            self.set_status("转写未检测到发言。请检查录音设备或换一段音频。", "warning")
            self.generate.setEnabled(False)
        elif result.get("diarization_status", {}).get("status") == "fallback":
            self.set_status("转写已保存，内录说话人分离失败，已按单人显示。", "warning")

    def edit_speaker_name(self) -> None:
        session = self.session
        if session is None:
            return
        try:
            transcript = session.load_transcript()
        except (OSError, ValueError) as exc:
            self.set_status(f"无法读取转写 · {exc}", "error")
            return
        examples: dict[str, dict] = {}
        for row in transcript["utterances"]:
            name = row.get("speaker")
            if isinstance(name, str) and (is_generated_speaker(name) or is_person_name(name)):
                examples.setdefault(name, row)
        if not examples:
            self.set_status("本场没有可编辑姓名的说话人。", "warning")
            return
        current, accepted = QInputDialog.getItem(
            self, "编辑说话人姓名", "说话人", list(examples), editable=False
        )
        if not accepted:
            return
        example = examples[current]
        name, accepted = QInputDialog.getText(
            self,
            "编辑说话人姓名",
            f"{current} · {ts(example['start'])}\n{example['text'][:100]}\n\n姓名",
            text=current if is_person_name(current) else "",
        )
        name = name.strip()
        if not accepted or name == current:
            return
        if not is_person_name(name):
            self.set_status("请输入姓名。", "warning")
            return
        self.launch(
            "正在保存说话人姓名…",
            lambda job: Pipeline(self.cfg, progress=job.log.emit).rename_speakers(
                session, {current: name}
            ),
            lambda _: self.set_status(f"已保存本场说话人姓名 · {name}", "success"),
        )

    def enroll_speaker_voiceprints(self) -> None:
        session = self.session
        if session is None:
            return
        library = Pipeline(self.cfg).voiceprint_library.path.resolve()
        if (
            QMessageBox.question(
                self,
                "登记到公共声纹库",
                "将本场已命名且有足够样本的说话人姓名和声纹保存到下列位置，供后续会议自动识别。"
                f"\n\n{library}",
                defaultButton=QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.launch(
            "正在登记公共声纹…",
            lambda job: Pipeline(self.cfg, progress=job.log.emit).enroll_voiceprints(session),
            self.voiceprints_enrolled,
        )

    def voiceprints_enrolled(self, result: Enrollment) -> None:
        messages = []
        if result.added:
            messages.append("已登记 " + "、".join(result.added))
        if result.existing:
            messages.append("已在库中 " + "、".join(result.existing))
        self.set_status(" · ".join(messages), "success")

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
            self.set_status(f"请检查纪要设置：{error}", "error")
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

    def export_file(self, path: Path | None, *, offer_prompt_actions: bool = False) -> None:
        if path is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "导出文件", path.name, "文本文件 (*.md *.txt)"
        )
        if filename:
            try:
                target = Path(filename).absolute()
                if target.resolve() != path.resolve():
                    shutil.copyfile(path, target)
                self.set_status(
                    f"已导出至 {target}", "success", prompt=target if offer_prompt_actions else None
                )
            except OSError as error:
                self.set_status(f"导出失败：{error}", "error")

    def export_prompt(self) -> None:
        session = self.session
        assert session is not None
        self.launch(
            "正在整理 AI 网页提示词…",
            lambda job: Pipeline(self.cfg, progress=job.log.emit).export_web_prompt(
                session, overwrite=True
            ),
            lambda path: self.export_file(path, offer_prompt_actions=True),
        )

    def open_exported_prompt(self, *, reveal: bool = False) -> None:
        path = self.exported_prompt_path
        if path is None:
            return
        if not path.is_file():
            self.set_status(f"找不到导出的提示词：{path}", "error")
            return
        if reveal and sys.platform == "win32":
            opened, _ = QProcess.startDetached("explorer.exe", ["/select,", str(path)])
        elif reveal and sys.platform == "darwin":
            opened, _ = QProcess.startDetached("open", ["-R", str(path)])
        else:
            opened = QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(path.parent if reveal else path))
            )
        if not opened:
            self.set_status(
                f"无法打开{'文件夹' if reveal else '文件'}：{path}", "error", prompt=path
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
            self.set_status(f"无法打开会话文件夹：{self.session.root}", "error")

    def show_settings(self) -> None:
        dialog = SettingsDialog(self.cfg, self.api_key, self)
        if not dialog.exec():
            return
        output = dialog.output.text()
        selected = self.session if output == self.cfg["output_dir"] else None
        self.cfg["output_dir"] = output
        self.cfg["recorder"]["target_rms"] = dialog.target_rms.value()
        bleed = dialog.bleed_suppression.isChecked()
        self.cfg.setdefault("live", {}).setdefault("bleed_suppression", {})["enabled"] = bleed
        self.cfg["llm"].update(
            base_url=dialog.url.text().strip(), model=dialog.model.text().strip()
        )
        self.api_key = dialog.key.text().strip()
        self.settings.setValue("output_dir", output)
        self.settings.setValue("target_rms", dialog.target_rms.value())
        self.settings.setValue("bleed_suppression", bleed)
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
    apply_light_theme(app)
    settings = QSettings()
    try:
        if config is not None:
            config = config.expanduser().resolve(strict=True)
            settings.beginGroup(
                "configurations/" + hashlib.sha256(str(config).encode()).hexdigest()
            )
            cfg = load_config(config)
            output = config.parent / Path(cfg["output_dir"]).expanduser()
        else:
            data = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.AppLocalDataLocation
            )
            output = Path(data) / "data"
            cfg = load_config(Path(data) / "config.yaml")
            if not (Path(data) / "config.yaml").is_file():
                cfg["recorder"]["microphone"] = True
        cfg["output_dir"] = settings.value("output_dir", str(output), type=str)
        target_rms = cast(
            float, settings.value("target_rms", cfg["recorder"]["target_rms"], type=float)
        )
        if MIN_TARGET_RMS <= target_rms <= MAX_TARGET_RMS:
            cfg["recorder"]["target_rms"] = target_rms
        live = cfg.get("live") or {}
        bleed = live.get("bleed_suppression") or {}
        bleed["enabled"] = settings.value(
            "bleed_suppression", bleed.get("enabled", True), type=bool
        )
        live["bleed_suppression"] = bleed
        cfg["live"] = live
        cfg["llm"]["base_url"] = settings.value("llm_url", cfg["llm"]["base_url"], type=str)
        cfg["llm"]["model"] = settings.value("llm_model", cfg["llm"]["model"], type=str)
        if getattr(sys, "frozen", False):
            provider = str(cfg["diarization"].get("provider", "campplus_onnx")).lower()
            if provider in ("campplus_onnx", "campplus", "onnx", "3dspeaker"):
                try:
                    import onnxruntime

                    if "CPUExecutionProvider" not in onnxruntime.get_available_providers():
                        cfg["diarization"]["provider"] = "single"
                except (ImportError, OSError):
                    cfg["diarization"]["provider"] = "single"
            elif provider != "single":
                cfg["diarization"]["provider"] = "single"
        window = MeetingWindow(cfg, settings)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError) as error:
        QMessageBox.critical(None, "无法启动 MeetMin", str(error))
        return 1
    window.show()
    return app.exec()
