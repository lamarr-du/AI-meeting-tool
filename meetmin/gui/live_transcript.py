"""实时草稿与已提交正文使用独立的 Qt 视图。"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace

from PySide6.QtCore import (
    QAbstractListModel,
    QEvent,
    QModelIndex,
    QPersistentModelIndex,
    QPoint,
    QRect,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QMouseEvent, QPalette, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QInputDialog,
    QLabel,
    QListView,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from ..textutil import ts

_SPEAKERS = {"loopback": "电脑声音", "microphone": "我"}


@dataclass
class _Row:
    group: str
    start: int
    text: str
    name: str
    speaker: int | None = None
    provisional: bool = False


class LiveTranscriptModel(QAbstractListModel):
    SpeakerRole = Qt.ItemDataRole.UserRole.value + 1

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._rows: list[_Row] = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and index.row() < len(self._rows):
            row = self._rows[index.row()]
            if role == Qt.ItemDataRole.DisplayRole:
                return f"{ts(row.start / 1000)} · {row.name}\n{row.text}"
            if role == self.SpeakerRole:
                return row
        return None

    def clear(self) -> None:
        self.beginResetModel()
        self._rows.clear()
        self.endResetModel()

    def append(self, event: dict) -> None:
        if event.get("suppressed_bleed"):
            return
        self._insert(
            _Row(
                event["segment_id"],
                event["start_ms"],
                event["text"],
                "说话人待定" if event.get("speaker_pending") else _SPEAKERS[event["track"]],
                provisional=bool(event.get("speaker_pending")),
            )
        )

    def _insert(self, item: _Row) -> None:
        row = bisect_right(self._rows, item.start, key=lambda value: value.start)
        self.beginInsertRows(QModelIndex(), row, row)
        self._rows.insert(row, item)
        self.endInsertRows()

    def _positions(self, group: str) -> list[int]:
        return [index for index, item in enumerate(self._rows) if item.group == group]

    def _remove(self, positions: list[int]) -> None:
        for row in reversed(positions):
            self.beginRemoveRows(QModelIndex(), row, row)
            self._rows.pop(row)
            self.endRemoveRows()

    def remove(self, group: str) -> None:
        self._remove(self._positions(group))

    def _change(self, index: int, item: _Row) -> None:
        self._rows[index] = item
        model_index = self.index(index, 0)
        self.dataChanged.emit(
            model_index, model_index, [self.SpeakerRole, Qt.ItemDataRole.DisplayRole.value]
        )

    def update_speaker(self, update: dict) -> None:
        group = update["segment_id"]
        positions = self._positions(group)
        if not positions:
            return
        parts = [
            _Row(
                group,
                part["start_ms"],
                part["text"],
                part["display_name"],
                part["speaker_id"],
                part["provisional"],
            )
            for part in update["parts"]
        ]
        if [self._rows[index].start for index in positions] == [part.start for part in parts]:
            for index, part in zip(positions, parts, strict=True):
                self._change(index, part)
            return
        self._remove(positions)
        for part in parts:
            self._insert(part)

    def rename(self, speaker: int, name: str, *, pin_anchor: bool = True) -> None:
        for index, item in enumerate(self._rows):
            if item.speaker == speaker:
                self._change(
                    index, replace(item, name=name, provisional=item.provisional and not pin_anchor)
                )


class _UtteranceDelegate(QStyledItemDelegate):
    def __init__(self, view: QListView):
        super().__init__(view)
        self.view = view

    @staticmethod
    def _badge(option, row: _Row) -> QRect:
        return QRect(
            option.rect.left() + 8,
            option.rect.top() + 5,
            min(180, option.fontMetrics.horizontalAdvance(row.name) + 16),
            option.fontMetrics.height() + 4,
        )

    def sizeHint(self, option, index):
        width = max(40, self.view.viewport().width() - 16)
        row = index.data(LiveTranscriptModel.SpeakerRole)
        size = option.fontMetrics.boundingRect(
            QRect(0, 0, width - 16, 10000),
            Qt.TextFlag.TextWordWrap.value,
            row.text if row is not None else "",
        )
        return QSize(width, size.height() + option.fontMetrics.height() + 20)

    def paint(self, painter, option, index) -> None:
        row = index.data(LiveTranscriptModel.SpeakerRole)
        if row is None:
            return
        style_option = QStyleOptionViewItem(option)
        self.initStyleOption(style_option, index)
        style_option.text = ""
        self.view.style().drawControl(
            QStyle.ControlElement.CE_ItemViewItem, style_option, painter, self.view
        )
        badge = self._badge(option, row)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        foreground = option.palette.color(
            QPalette.ColorRole.HighlightedText if selected else QPalette.ColorRole.Text
        )
        fill = option.palette.color(QPalette.ColorRole.Highlight)
        fill.setAlpha(30 if row.provisional else 65)
        painter.save()
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        pen = QPen(foreground)
        if row.provisional:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(fill)
        painter.drawRoundedRect(badge, 5, 5)
        painter.setPen(foreground)
        painter.drawText(
            badge.adjusted(7, 0, -7, 0),
            Qt.AlignmentFlag.AlignCenter.value,
            option.fontMetrics.elidedText(
                row.name, Qt.TextElideMode.ElideRight, badge.width() - 14
            ),
        )
        time_rect = QRect(badge.right() + 9, badge.top(), 86, badge.height())
        painter.drawText(time_rect, Qt.AlignmentFlag.AlignVCenter.value, ts(row.start / 1000))
        body = option.rect.adjusted(8, badge.height() + 9, -8, -3)
        painter.drawText(body, Qt.TextFlag.TextWordWrap.value, row.text)
        painter.restore()

    def editorEvent(self, event, model, option, index):
        row = index.data(LiveTranscriptModel.SpeakerRole)
        if (
            isinstance(event, QMouseEvent)
            and event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
            and row is not None
            and row.speaker is not None
            and self._badge(option, row).contains(event.position().toPoint())
        ):
            parent = self.view.parentWidget()
            if isinstance(parent, LiveTranscriptView):
                parent.prompt_rename(row.speaker, row.name)
                return True
        return super().editorEvent(event, model, option, index)


class LiveTranscriptView(QWidget):
    rename_requested = Signal(int, str)

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
        self.renaming_enabled = True

    def prompt_rename(self, speaker: int, current: str) -> None:
        if self.renaming_enabled:
            name, accepted = QInputDialog.getText(self, "说话人改名", "姓名", text=current)
            if accepted and name.strip():
                self.rename_requested.emit(speaker, name)

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
        elif kind == "speaker.rename":
            self.model.rename(
                event["speaker_id"], event["display_name"], pin_anchor=event.get("pin_anchor", True)
            )
        elif kind in (
            "asr.commit",
            "diarization.assign",
            "diarization.reconcile",
            "asr.suppress_bleed",
        ):
            bar = self.list.verticalScrollBar()
            follow = bar.maximum() - bar.value() <= 30
            top = self.list.indexAt(QPoint(2, 0))
            anchor = QPersistentModelIndex(top)
            original_row = top.row()
            offset = self.list.visualRect(top).top() if top.isValid() else 0
            if kind == "asr.commit":
                self.model.append(event)
            elif kind == "asr.suppress_bleed":
                self.model.remove(event["segment_id"])
            else:
                for update in event["updates"] if kind == "diarization.reconcile" else [event]:
                    self.model.update_speaker(update)
            if follow:
                self.list.scrollToBottom()
            elif original_row >= 0 and self.model.rowCount():
                row = (
                    anchor.row()
                    if anchor.isValid()
                    else min(original_row, self.model.rowCount() - 1)
                )
                self.list.scrollTo(
                    self.model.index(row, 0), QAbstractItemView.ScrollHint.PositionAtTop
                )
                bar.setValue(bar.value() - offset)
            self._scroll_changed(bar.value())
