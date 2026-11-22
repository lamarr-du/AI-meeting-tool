"""实时草稿与已提交正文使用独立的 Qt 视图。"""

from __future__ import annotations

from bisect import bisect_right

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QPersistentModelIndex,
    QPoint,
    QRect,
    QSize,
    Qt,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QListView,
    QPushButton,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from ..textutil import ts

_SPEAKERS = {"loopback": "电脑声音", "microphone": "我"}


class LiveTranscriptModel(QAbstractListModel):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._rows: list[tuple[int, str]] = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            return self._rows[index.row()][1]
        return None

    def clear(self) -> None:
        self.beginResetModel()
        self._rows.clear()
        self.endResetModel()

    def append(self, event: dict) -> None:
        start = event["start_ms"]
        row = bisect_right(self._rows, start, key=lambda item: item[0])
        text = f"{ts(start / 1000)}  ·  {_SPEAKERS[event['track']]}\n{event['text']}"
        self.beginInsertRows(QModelIndex(), row, row)
        self._rows.insert(row, (start, text))
        self.endInsertRows()


class _UtteranceDelegate(QStyledItemDelegate):
    def __init__(self, view: QListView):
        super().__init__(view)
        self.view = view

    def sizeHint(self, option, index):
        width = max(40, self.view.viewport().width() - 16)
        size = option.fontMetrics.boundingRect(
            QRect(0, 0, width, 10000),
            Qt.TextFlag.TextWordWrap.value,
            index.data(Qt.ItemDataRole.DisplayRole),
        )
        return QSize(width, size.height() + 14)


class LiveTranscriptView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self.status = QLabel(textFormat=Qt.TextFormat.PlainText, wordWrap=True)
        self.status.setObjectName("muted")
        layout.addWidget(self.status)
        self.model = LiveTranscriptModel(self)
        self.list = QListView()
        self.list.setAccessibleName("实时转写正文")
        self.list.setModel(self.model)
        self.list.setWordWrap(True)
        self.list.setResizeMode(QListView.ResizeMode.Adjust)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setItemDelegate(_UtteranceDelegate(self.list))
        layout.addWidget(self.list, 1)
        self.jump = QPushButton("回到底部")
        self.jump.clicked.connect(self.list.scrollToBottom)
        self.list.verticalScrollBar().valueChanged.connect(self._scroll_changed)
        layout.addWidget(self.jump, alignment=Qt.AlignmentFlag.AlignRight)
        # Preserve the list's height when the user scrolls away from the bottom.
        policy = self.jump.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.jump.setSizePolicy(policy)
        self.drafts = {}
        for track in _SPEAKERS:
            text = self.drafts[track] = QLabel(textFormat=Qt.TextFormat.PlainText, wordWrap=True)
            text.setAccessibleName(f"{_SPEAKERS[track]}实时草稿")
            text.setObjectName("muted")
            text.setFixedHeight(text.fontMetrics().lineSpacing() * 2 + 6)
            text.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            font = text.font()
            font.setItalic(True)
            text.setFont(font)
            layout.addWidget(text)
        self.reset(dual_track=False)

    def reset(self, *, dual_track: bool) -> None:
        self.model.clear()
        self.status.setText("正在启动实时转写…")
        for text in self.drafts.values():
            text.clear()
        self.drafts["microphone"].setVisible(dual_track)
        self.jump.hide()

    def _scroll_changed(self, _value: int) -> None:
        bar = self.list.verticalScrollBar()
        self.jump.setVisible(bar.maximum() - bar.value() > 30)

    def apply_event(self, event: dict) -> None:
        kind = event["type"]
        if kind == "live.status":
            self.status.setText(event["message"])
        elif kind == "asr.interim":
            label = self.drafts[event["track"]]
            label.setText(f"{_SPEAKERS[event['track']]} · {event['text']}" if event["text"] else "")
        elif kind == "asr.commit":
            bar = self.list.verticalScrollBar()
            follow = bar.maximum() - bar.value() <= 30
            top = self.list.indexAt(QPoint(2, 0))
            anchor = QPersistentModelIndex(top)
            offset = self.list.visualRect(top).top() if top.isValid() else 0
            self.model.append(event)
            if follow:
                self.list.scrollToBottom()
            elif anchor.isValid():
                self.list.scrollTo(
                    self.model.index(anchor.row(), 0), QAbstractItemView.ScrollHint.PositionAtTop
                )
                bar.setValue(bar.value() - offset)
            self._scroll_changed(bar.value())
