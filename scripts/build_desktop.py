"""在 Windows 上用 PyInstaller 生成包含 Python 运行时的桌面版目录。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("请在 Windows 上构建 Windows 桌面版。")
    root = Path(__file__).resolve().parents[1]
    work = (root / "build" / "desktop").resolve()
    dist = (root / "dist" / "desktop").resolve()
    if not all(path.is_relative_to(root) for path in (work, dist, (dist / "MeetMin").resolve())):
        raise SystemExit("构建和分发目录必须位于项目内。")
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Qt uses the Windows ICU ABI. Search System32 ahead of other tools that ship ICU DLLs.
    system_dlls = Path(os.environ["SYSTEMROOT"]) / "System32"
    conda_bin = Path(sys.prefix) / "Library" / "bin"
    search = [str(system_dlls)]
    if conda_bin.is_dir():
        search.append(str(conda_bin))
    env["PATH"] = os.pathsep.join([*search, env.get("PATH", "")])
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--onedir",
            "--windowed",
            "--name",
            "MeetMin",
            "--distpath",
            str(dist),
            "--workpath",
            str(work / "pyinstaller"),
            "--specpath",
            str(work),
            "--paths",
            str(root),
            "--collect-data",
            "faster_whisper",
            "--collect-binaries",
            "ctranslate2",
            "--exclude-module",
            "speechbrain",
            "--exclude-module",
            "torch",
            "--exclude-module",
            "torchaudio",
            "--exclude-module",
            "sklearn",
            str(root / "meetmin" / "gui" / "__main__.py"),
        ],
        cwd=root,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
