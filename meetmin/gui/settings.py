from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
)


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
