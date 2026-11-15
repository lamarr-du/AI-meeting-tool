STYLESHEET = """
QWidget { color: #192e33; font-size: 14px; }
QMainWindow, QWidget#workspace, QDialog { background: #f4f7f6; }
QWidget#sidebar { background: #e9f0ed; border-right: 1px solid #d8e4df; }
QWidget#card { background: white; border: 1px solid #dbe6e1; border-radius: 12px; }
QLabel#brand { color: #075e56; font-size: 26px; font-weight: 700; }
QLabel#title { font-size: 27px; font-weight: 700; }
QLabel#section { font-size: 15px; font-weight: 700; }
QLabel#muted { color: #506a69; }
QLabel#status { color: #075e56; font-weight: 600; }
QPushButton, QToolButton, QLineEdit, QComboBox {
    background: #ffffff; border: 1px solid #cbdad3; border-radius: 7px; padding: 6px 11px;
    min-height: 21px;
}
QPushButton:hover, QToolButton:hover, QComboBox:hover { background: #eaf5f0; }
QPushButton:focus, QToolButton:focus, QLineEdit:focus, QComboBox:focus, QListWidget:focus {
    border: 1px solid #0b766b;
}
QPushButton#primary { background: #0a665b; color: white; border-color: #0a665b; font-weight: 600; }
QPushButton#primary:hover { background: #08554d; }
QPushButton#stop { background: #a33136; color: white; border-color: #a33136; font-weight: 600; }
QPushButton#stop:hover { background: #87282d; }
QPushButton:disabled, QPushButton#primary:disabled, QPushButton#stop:disabled,
QToolButton:disabled, QComboBox:disabled, QLineEdit:disabled {
    color: #7c8d88; background: #eef2f0; border-color: #dde6e1;
}
QListWidget { background: transparent; border: 1px solid transparent; outline: 0; padding: 2px; }
QListWidget::item { padding: 9px; border-radius: 6px; margin: 1px 0; }
QListWidget::item:selected { background: #cee6dd; color: #164239; }
QListWidget::item:hover:!selected { background: #dfece6; }
QTabWidget::pane, QTextBrowser, QPlainTextEdit {
    background: white; border: 1px solid #dbe6e1; border-radius: 8px;
}
QTabBar::tab {
    background: transparent; border: 0; border-bottom: 2px solid transparent;
    padding: 9px 16px; color: #506a69;
}
QTabBar::tab:selected { color: #075e56; border-bottom: 2px solid #0b766b; }
QProgressBar { background: #e4eeea; border: 0; border-radius: 4px; min-height: 6px; }
QProgressBar::chunk { background: #209180; border-radius: 4px; }
QMenu { background: white; border: 1px solid #cbdad3; }
QMenu::item { padding: 7px 18px; }
QMenu::item:selected { background: #dceee7; }
"""
