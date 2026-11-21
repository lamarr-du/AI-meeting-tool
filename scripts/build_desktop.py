"""在 Windows 上用 PyInstaller 生成包含 Python 运行时的桌面版目录。"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

CUDA_DLL_PATTERNS = (
    "cublas*64_12.dll",
    "cudart64_12.dll",
    "cudnn*64_9.dll",
    "nvJitLink_120_0.dll",
    "nvrtc64_120_0.dll",
    "nvrtc-builtins64_12*.dll",
)


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("请在 Windows 上构建 Windows 桌面版。")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from meetmin.asr.cuda import WINDOWS_CUDA_LIBRARIES, windows_cuda_directory

    parser = argparse.ArgumentParser(description=__doc__)
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument("--cuda-dir", type=Path, help="包含 CUDA 12 和 cuDNN 9 DLL 的目录")
    runtime.add_argument("--cpu-only", action="store_true", help="构建不附带 CUDA 运行库的便携包")
    args = parser.parse_args()
    libraries: list[Path] = []
    if not args.cpu_only:
        source = args.cuda_dir or windows_cuda_directory()
        required = (*WINDOWS_CUDA_LIBRARIES, "cudnn_ops64_9.dll", "cudnn_graph64_9.dll")
        if source is None or any(not (source / name).is_file() for name in required):
            raise SystemExit(
                "CUDA 运行库不完整。用 --cuda-dir 指定目录，或用 --cpu-only 构建 CPU 包。"
            )
        libraries = sorted({path for pattern in CUDA_DLL_PATTERNS for path in source.glob(pattern)})

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
    if libraries:
        internal = dist / "MeetMin" / "_internal"
        cuda = internal / "cuda"
        cuda.mkdir()
        for library in libraries:
            shutil.copy2(library, cuda / library.name)
        # The wheel ships a cuDNN loader that may be a different release from
        # the cuDNN components above. Keep the packaged runtime as a matched set.
        (internal / "ctranslate2" / "cudnn64_9.dll").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
