"""安装后的命令入口必须能在项目目录之外运行。"""

import os
import subprocess
import sys
import sysconfig
from importlib.metadata import metadata
from pathlib import Path

import pytest


@pytest.mark.parametrize("entry", ["console", "module", "legacy"])
def test_cli_help_outside_checkout(tmp_path, entry):
    if entry == "console":
        executable = Path(sysconfig.get_path("scripts")) / (
            "meetmin.exe" if os.name == "nt" else "meetmin"
        )
        assert executable.is_file(), "请先安装项目及开发依赖：python -m pip install -e '.[dev]'"
        command = [str(executable)]
    elif entry == "module":
        command = [sys.executable, "-m", "meetmin"]
    else:
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / "main.py")]
    result = subprocess.run(
        [*command, "--help"], cwd=tmp_path, capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    for name in ("doctor", "devices", "record", "transcribe", "diarize", "summarize", "run"):
        assert name in result.stdout


def test_distribution_extras():
    package = metadata("meetmin")
    extras = package.get_all("Provides-Extra")
    assert "diar" in extras and "dev" in extras
    dependencies = package.get_all("Requires-Dist")
    assert any("speechbrain" in dep and 'extra == "diar"' in dep for dep in dependencies)
    assert any("pyaudiowpatch" in dep and 'sys_platform == "win32"' in dep for dep in dependencies)
