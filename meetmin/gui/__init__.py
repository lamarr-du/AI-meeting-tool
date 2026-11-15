"""可选桌面入口，核心 CLI 无需安装 Qt。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meetmin-gui", description="MeetMin 会议纪要桌面版")
    parser.add_argument("--config", type=Path, help="加载现有 MeetMin YAML 配置")
    args = parser.parse_args(argv)
    try:
        from .app import run
    except ModuleNotFoundError as error:
        if error.name != "PySide6":
            raise
        print('桌面版需要 PySide6。请先执行 python -m pip install ".[gui]"。', file=sys.stderr)
        return 1
    return run(args.config)
