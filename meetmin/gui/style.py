from __future__ import annotations

import sys

from PySide6.QtCore import QLineF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QProgressBar,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionComboBox,
    QStylePainter,
    QWidget,
)

TEXT = "#242038"
MUTED = "#635C76"
ACCENT = "#6546C4"
HISTORY_DATE = Qt.ItemDataRole.UserRole + 1

STYLESHEET = """
QMainWindow, QWidget#workspace, QDialog { background: #ffffff; }
QWidget#sidebar { background: #f7f5fc; border-right: 1px solid #e8e3f1; }
QFrame#captureCard {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #f2ecff, stop: 1 #fbf9ff);
    border: 1px solid #e3d9f7;
    border-radius: 14px;
}
QFrame#devicePanel, QFrame#meterPanel {
    background: #ffffff; border: 1px solid #e8e0f5; border-radius: 11px;
}
QFrame#statusStrip { background: #f9f7fd; border: 1px solid #e8e3f1; border-radius: 11px; }
QFrame#resultCard { background: #ffffff; border: 1px solid #e8e3f1; border-radius: 13px; }
QFrame#navDivider { border: 0; border-top: 1px solid #e7e1ef; }
QLabel#brandMark, QLabel#emptyMark { background: #ece4ff; border-radius: 10px; }
QLabel#brand { font-size: 24px; font-weight: 700; }
QLabel#title { font-size: 29px; font-weight: 700; }
QLabel#captureTitle { font-size: 19px; font-weight: 700; }
QLabel#section, QLabel#fieldLabel { font-weight: 600; }
QLabel#navCaption, QLabel#muted { color: #635c76; font-size: 13px; }
QLabel#navCaption { font-weight: 600; }
QLabel#emptyMessage { color: #635c76; font-size: 15px; font-weight: 500; }
QLabel#status { font-weight: 600; }
QLabel#statusDot { background: #7a5cd6; border-radius: 4px; }
QLabel#statusDot[tone="success"] { background: #238365; }
QLabel#statusDot[tone="warning"] { background: #a46115; }
QLabel#statusDot[tone="error"] { background: #bd3b57; }
QLabel#recordBadge {
    background: #fbe5ed; color: #992942; border-radius: 13px;
    padding: 4px 11px; min-height: 18px; font-size: 13px; font-weight: 600;
}
QLabel#recordClock { font-size: 43px; font-weight: 600; }
QPushButton, QToolButton, QLineEdit, QComboBox {
    background: #ffffff; border: 1px solid #dcd4e9; border-radius: 7px;
    padding: 7px 12px; min-height: 22px;
}
QPushButton, QToolButton { font-weight: 600; }
QPushButton:hover, QToolButton:hover, QComboBox:hover { background: #f4effc; }
QPushButton:pressed, QToolButton:pressed { background: #eae1fa; }
QPushButton:focus, QToolButton:focus, QLineEdit:focus, QComboBox:focus {
    border: 1px solid #7655d6;
}
QPushButton[primaryAction="true"] {
    background: #6546c4; border-color: #6546c4; color: white;
}
QPushButton[primaryAction="true"]:hover { background: #5436ae; }
QPushButton[primaryAction="true"]:pressed { background: #482d99; }
QPushButton[primaryAction="true"]:focus { border-color: #2c1769; }
QPushButton#stop {
    background: #f8d9e4; border-color: #eab5c7; color: #89253e;
}
QPushButton#stop:hover { background: #f2c5d5; }
QPushButton#stop:focus { border-color: #a33151; }
QPushButton#muteMicrophone:checked {
    background: #f8d9e4; border-color: #a33151; color: #89253e;
}
QPushButton#newMeeting { text-align: left; }
QPushButton#settingsButton { background: transparent; text-align: left; border-color: transparent; }
QPushButton#settingsButton:hover { background: #ece5fa; }
QPushButton#settingsButton:focus { border-color: #7655d6; }
QPushButton#refreshAudio { background: transparent; border-color: transparent; color: #6546c4; }
QPushButton#refreshAudio:hover { background: #f4effc; }
QPushButton#refreshAudio:focus { border-color: #7655d6; }
QPushButton:disabled, QPushButton[primaryAction="true"]:disabled, QPushButton#stop:disabled,
QPushButton#refreshAudio:disabled, QToolButton:disabled, QComboBox:disabled, QLineEdit:disabled {
    color: #716981; background: #f2eff6; border-color: #ebe6f1;
}
QComboBox { padding-right: 24px; }
QComboBox::drop-down { border: 0; width: 22px; }
QComboBox::down-arrow { image: none; }
QComboBox QAbstractItemView {
    selection-background-color: #eee7fc;
    selection-color: #432c89;
}
QToolButton#moreActions { padding-right: 24px; }
QToolButton::menu-indicator { subcontrol-origin: padding; subcontrol-position: center right; }
QListWidget#historyList { background: transparent; border: 0; outline: 0; }
QToolButton#historyActions { padding: 0; border: 0; border-radius: 5px; font-size: 18px; }
QToolButton#historyActions::menu-indicator { image: none; width: 0; }
QTabWidget::pane { border: 0; border-top: 1px solid #ebe6f2; }
QTabWidget::tab-bar { left: 16px; }
QTabBar::tab {
    background: transparent; border-bottom: 2px solid transparent;
    padding: 12px 15px; color: #635c76;
}
QTabBar::tab:hover { color: #5436ae; background: #f9f6fe; }
QTabBar::tab:selected { color: #5436ae; border-bottom-color: #7655d6; font-weight: 600; }
QTabBar::tab:focus { background: #f1ebfb; }
QTextBrowser, QPlainTextEdit { background: transparent; border: 0; }
QTextBrowser { font-size: 15px; }
QPlainTextEdit { padding: 12px; }
QProgressBar { background: #eae3f7; border: 0; border-radius: 4px; }
QProgressBar::chunk { background: #7655d6; border-radius: 4px; }
QMenu { background: white; border: 1px solid #dcd4e9; padding: 4px; }
QMenu::item { padding: 8px 18px; border-radius: 4px; }
QMenu::item:selected { background: #eee7fc; color: #432c89; }
QMenu::separator { height: 1px; background: #ebe6f2; margin: 4px 8px; }
"""


def apply_light_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = app.style().standardPalette()
    for role, color in (
        (QPalette.ColorRole.Window, "#ffffff"),
        (QPalette.ColorRole.Base, "#ffffff"),
        (QPalette.ColorRole.AlternateBase, "#f8f5fc"),
        (QPalette.ColorRole.Button, "#ffffff"),
        (QPalette.ColorRole.WindowText, TEXT),
        (QPalette.ColorRole.Text, TEXT),
        (QPalette.ColorRole.ButtonText, TEXT),
        (QPalette.ColorRole.PlaceholderText, MUTED),
        (QPalette.ColorRole.Highlight, ACCENT),
        (QPalette.ColorRole.HighlightedText, "#ffffff"),
        (QPalette.ColorRole.Link, ACCENT),
    ):
        palette.setColor(role, QColor(color))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#716981"))
    app.setPalette(palette)
    font = app.font()
    if sys.platform == "win32":
        font.setFamily("Microsoft YaHei UI")
    font.setPixelSize(14)
    app.setFont(font)
    app.setStyleSheet(STYLESHEET)


def style_property(widget: QWidget, name: str, value: str | bool) -> None:
    if widget.property(name) != value:
        widget.setProperty(name, value)
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()


def line_icon(kind: str, color: str = ACCENT) -> QIcon:
    pixmap = QPixmap(40, 40)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.GlobalColor.transparent)
    with QPainter(pixmap) as painter:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(color), 1.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        if kind == "audio":
            for x, height in zip((4, 8, 12, 16), (6, 13, 16, 9)):
                painter.drawLine(QLineF(x, 10 - height / 2, x, 10 + height / 2))
        elif kind == "note":
            painter.drawRoundedRect(QRectF(4, 2.5, 12, 15), 2, 2)
            painter.drawLine(7, 8, 13, 8)
            painter.drawLine(7, 12, 12, 12)
        elif kind == "settings":
            for y, x in ((5, 7), (10, 13), (15, 8)):
                painter.drawLine(3, y, 17, y)
                painter.drawLine(x, y - 2, x, y + 2)
        elif kind == "stop":
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(QRectF(5, 5, 10, 10), 2, 2)
    return QIcon(pixmap)


class DeviceComboBox(QComboBox):
    def __init__(self, name: str):
        super().__init__()
        self.setAccessibleName(name)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(12)
        self.currentTextChanged.connect(self.setToolTip)

    def paintEvent(self, event) -> None:
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        field = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox, option, QStyle.SubControl.SC_ComboBoxEditField, self
        )
        option.currentText = self.fontMetrics().elidedText(
            option.currentText, Qt.TextElideMode.ElideRight, field.width()
        )
        painter = QStylePainter(self)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)
        arrow = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox, option, QStyle.SubControl.SC_ComboBoxArrow, self
        )
        x, y = arrow.center().x(), arrow.center().y()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(MUTED), 1.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(x - 4, y - 2, x, y + 2)
        painter.drawLine(x, y + 2, x + 4, y - 2)
        painter.end()


class HistoryDelegate(QStyledItemDelegate):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.note = line_icon("note", MUTED)
        self.active_note = line_icon("note")

    def sizeHint(self, option, index) -> QSize:
        return QSize(0, 56)

    def paint(self, painter, option, index) -> None:
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        focused = bool(option.state & QStyle.StateFlag.State_HasFocus)
        rect = QRectF(option.rect).adjusted(1, 2, -2, -2)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if selected or hovered or focused:
            painter.setBrush(QColor("#ede5fc" if selected else "#f0eaf9"))
            painter.setPen(QPen(QColor(ACCENT if focused else "#ded2f5"), 1))
            painter.drawRoundedRect(rect, 8, 8)
        if selected:
            painter.setBrush(QColor(ACCENT))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(rect.x(), rect.y() + 13, 3, 26), 1.5, 1.5)

        icon = self.active_note if selected else self.note
        icon.paint(painter, int(rect.x()) + 10, int(rect.y()) + 16, 19, 19)
        title = str(index.data() or "")
        date = str(index.data(HISTORY_DATE) or "")
        title_rect = rect.adjusted(38, 3 if date else 0, -38, -24 if date else 0)
        font = QFont(option.font)
        font.setWeight(QFont.Weight.DemiBold if selected else QFont.Weight.Normal)
        painter.setFont(font)
        painter.setPen(QColor(TEXT if selected else MUTED))
        painter.drawText(
            title_rect,
            Qt.AlignmentFlag.AlignVCenter,
            painter.fontMetrics().elidedText(
                title, Qt.TextElideMode.ElideRight, int(title_rect.width())
            ),
        )
        if date:
            font.setPixelSize(12)
            painter.setFont(font)
            painter.setPen(QColor(MUTED))
            painter.drawText(rect.adjusted(38, 25, -9, -4), Qt.AlignmentFlag.AlignVCenter, date)
        painter.restore()


class AudioLevel(QProgressBar):
    def __init__(self, name: str, color: str):
        super().__init__()
        self.setRange(0, 100)
        self.setValue(0)
        self.setAccessibleName(name)
        self.setTextVisible(False)
        self.setFixedHeight(13)
        self.color = QColor(color)

    def paintEvent(self, event) -> None:
        count = min(30, max(1, self.width() // 12))
        gap = 4
        segment = self.width() / count
        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            for index in range(count):
                painter.setBrush(
                    self.color if index * 100 < self.value() * count else QColor("#e9e2f3")
                )
                width = segment if index == count - 1 else max(1, segment - gap)
                painter.drawRoundedRect(QRectF(index * segment, 0, width, self.height()), 3, 3)
