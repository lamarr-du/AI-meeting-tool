"""CUDA detection shared by Whisper and the read-only preflight."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from importlib.util import find_spec
from pathlib import Path

WINDOWS_CUDA_LIBRARIES = ("cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll")


@dataclass(frozen=True)
class CudaStatus:
    reason: str | None = None
    memory: tuple[str, int, int] | None = None


def windows_cuda_directory() -> Path | None:
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle is not None:
        path = Path(bundle) / "cuda"
        if path.is_dir():
            return path
    configured = os.environ.get("MEETMIN_CUDA_DLL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    torch = find_spec("torch")
    if torch and torch.origin:
        path = Path(torch.origin).parent / "lib"
        if all((path / library).is_file() for library in WINDOWS_CUDA_LIBRARIES):
            return path
    return None


@cache
def _register_dll_directory(path: Path) -> object:
    # Keep the directory registered for libraries loaded lazily by cuDNN.
    return os.add_dll_directory(str(path))


def prepare_cuda_runtime() -> Path | None:
    if sys.platform == "win32" and (path := windows_cuda_directory()) is not None:
        _register_dll_directory(path)
        return path
    return None


def _gpu_memory() -> tuple[str, int, int] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        rows = result.stdout.splitlines()
        # CTranslate2 uses CUDA device 0; with more than one physical card,
        # nvidia-smi's first row need not be that device.
        if len(rows) == 1:
            name, free, total = rows[0].rsplit(",", 2)
            return name.strip(), int(free), int(total)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def gpu_status(compute_type: str = "int8_float16") -> CudaStatus:
    try:
        directory = prepare_cuda_runtime()
        from ctranslate2 import get_cuda_device_count, get_supported_compute_types

        if get_cuda_device_count() == 0:
            return CudaStatus("CTranslate2 未检测到 CUDA 设备")
        libraries = (
            WINDOWS_CUDA_LIBRARIES
            if sys.platform == "win32"
            else ("libcublasLt.so.12", "libcublas.so.12", "libcudnn.so.9")
        )
        for library in libraries:
            if directory is not None and not (directory / library).is_file():
                return CudaStatus(f"CUDA 运行库缺失：{library}")
            try:
                ctypes.CDLL(str(directory / library) if directory else library)
            except OSError:
                return CudaStatus(f"CUDA 运行库无法加载：{library}")
        if compute_type not in ("default", "auto") and (
            compute_type not in get_supported_compute_types("cuda")
        ):
            return CudaStatus(f"CUDA 设备不支持计算类型 {compute_type}")
    except (ImportError, OSError, RuntimeError) as error:
        return CudaStatus(f"CUDA 初始化失败：{error}")
    return CudaStatus(memory=_gpu_memory())
