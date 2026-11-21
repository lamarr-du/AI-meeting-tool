from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
)

from ..config import DEFAULT_TARGET_RMS, MAX_TARGET_RMS, MIN_TARGET_RMS


class SettingsDialog(QDialog):
    def __init__(self, cfg: dict, api_key: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("桌面版设置")
        self.setMinimumWidth(520)
        self.output = QLineEdit(cfg["output_dir"])
        self.url = QLineEdit(cfg["llm"].get("base_url", ""))
        self.model = QLineEdit(cfg["llm"].get("model", ""))
        self.key = QLineEdit(api_key)
        self.key.setEchoMode(QLineEdit.EchoMode.Password)

        self.target_rms = QDoubleSpinBox()
        self.target_rms.setAccessibleName("录音目标电平")
        self.target_rms.setDecimals(3)
        self.target_rms.setRange(MIN_TARGET_RMS, MAX_TARGET_RMS)
        self.target_rms.setSingleStep(0.001)
        self.target_rms.setValue(cfg["recorder"].get("target_rms", DEFAULT_TARGET_RMS))
        self.target_slider = QSlider(Qt.Orientation.Horizontal)
        self.target_slider.setAccessibleName("录音目标电平滑块")
        self.target_slider.setRange(1, 300)
        self.target_slider.setValue(round(self.target_rms.value() * 1000))
        self.target_slider.valueChanged.connect(
            lambda value: self.target_rms.setValue(value / 1000)
        )
        self.target_rms.valueChanged.connect(
            lambda value: self.target_slider.setValue(round(value * 1000))
        )
        target = QHBoxLayout()
        target.addWidget(self.target_slider, 1)
        target.addWidget(self.target_rms)
        reset_target = QPushButton("恢复默认值")
        reset_target.clicked.connect(lambda: self.target_rms.setValue(DEFAULT_TARGET_RMS))
        target_help = QLabel("影响新录音的电脑声音和麦克风；值越高，背景噪声也可能越明显。")
        target_help.setWordWrap(True)
        target_help.setObjectName("muted")

        browse = QPushButton("选择…")
        browse.clicked.connect(self.choose_folder)
        folder = QHBoxLayout()
        folder.addWidget(self.output)
        folder.addWidget(browse)

        env = cfg["llm"].get("api_key_env", "LLM_API_KEY")
        self.key.setPlaceholderText("已从环境变量读取" if os.environ.get(env) else "按需填写")
        help_text = QLabel("在此输入的 API Key 仅用于本次运行。留空时使用环境变量。")
        help_text.setWordWrap(True)
        help_text.setObjectName("muted")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        save = buttons.button(QDialogButtonBox.StandardButton.Save)
        save.setText("保存")
        save.setProperty("primaryAction", True)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QFormLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(12)
        layout.addRow("会话存储位置", folder)
        layout.addRow("纪要 API 地址", self.url)
        layout.addRow("纪要模型", self.model)
        layout.addRow("API Key", self.key)
        layout.addRow(help_text)
        layout.addRow("录音目标电平（RMS）", target)
        layout.addRow("", reset_target)
        layout.addRow(target_help)
        layout.addRow(buttons)

    def choose_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择会话存储位置", self.output.text())
        if selected:
            self.output.setText(selected)

    def accept(self) -> None:
        try:
            if not self.output.text().strip():
                raise ValueError("请选择会话存储位置。")
            url = urlsplit(self.url.text().strip())
            if self.url.text().strip() and (
                url.scheme not in ("http", "https") or not url.hostname or url.username is not None
            ):
                raise ValueError("请输入有效的 HTTP 或 HTTPS API 地址，账号信息请填在 API Key 中。")
            folder = Path(self.output.text().strip()).expanduser().resolve()
            folder.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "设置未保存", str(error))
            return
        self.output.setText(str(folder))
        super().accept()
