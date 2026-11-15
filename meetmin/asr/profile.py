"""Resolve Whisper model, device and precision from a shared hardware policy."""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path

MiB = 1024**2
_CUDA_HEADROOM_MIB = 256
_CUDA_BUDGET_MIB = 3072
_TURBO = "large-v3-turbo"


def memory_status() -> tuple[int, int] | None:
    """Return (total, available) RAM in MiB; unknown hosts use the light profile."""
    try:
        if sys.platform == "win32":

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_uint32),
                    ("load", ctypes.c_uint32),
                    ("total", ctypes.c_uint64),
                    ("available", ctypes.c_uint64),
                    ("total_page", ctypes.c_uint64),
                    ("available_page", ctypes.c_uint64),
                    ("total_virtual", ctypes.c_uint64),
                    ("available_virtual", ctypes.c_uint64),
                    ("extended", ctypes.c_uint64),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.total // MiB, status.available // MiB
            return None
        fields = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, _, value = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                fields[key] = int(value.split()[0]) // 1024
        return fields["MemTotal"], fields["MemAvailable"]
    except (OSError, ValueError, KeyError):
        return None


@dataclass(frozen=True)
class AsrProfile:
    requested: str
    selected: str
    model: str
    device: str
    compute_type: str
    cpu_fallback_model: str
    reason: str


def select_profile(
    config: dict,
    *,
    memory: tuple[int, int] | None,
    gpu: tuple[str, int, int, list[str]] | None,
) -> AsrProfile:
    requested = config.get("profile", "auto")
    if requested not in ("auto", "cpu_light", "cpu_standard", "gpu_standard"):
        raise ValueError("asr.profile 必须是 auto、cpu_light、cpu_standard 或 gpu_standard")
    device = config.get("device", "auto")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("asr.device 必须是 auto、cpu 或 cuda")
    peak = config.get("cuda_peak_mib")
    if peak is not None and (type(peak) is not int or peak <= 0):
        raise ValueError("asr.cuda_peak_mib 必须是正整数或 null")

    model = config.get("model_size")
    compute = config.get("compute_type")
    enough_ram = memory is not None and memory[0] >= 15 * 1024 and memory[1] >= 6 * 1024
    standard_cpu = requested == "cpu_standard" or (requested != "cpu_light" and enough_ram)
    cpu_profile = "cpu_standard" if standard_cpu else "cpu_light"
    cpu_model = model or (_TURBO if standard_cpu else "small")

    reason = "按可用内存选择 CPU 档位"
    if device == "cuda":
        reason = "显式指定 CUDA"
    elif device == "cpu" or requested in ("cpu_light", "cpu_standard"):
        reason = "配置指定 CPU"
    elif not gpu or gpu[3]:
        reason = "未找到 CUDA 设备或运行库"
    elif peak is None:
        reason = "尚未配置目标模型的实测 CUDA 峰值"
    elif peak > _CUDA_BUDGET_MIB or min(gpu[1], gpu[2]) < peak + _CUDA_HEADROOM_MIB:
        reason = "实测峰值或当前空闲显存超出选档条件"
    else:
        device = "cuda"
        reason = "实测峰值与空闲显存符合选档条件"
    if device == "cuda":
        return AsrProfile(
            requested,
            "gpu_standard",
            model or _TURBO,
            "cuda",
            compute or "int8_float16",
            cpu_model,
            reason,
        )
    return AsrProfile(
        requested, cpu_profile, cpu_model, "cpu", compute or "int8", cpu_model, reason
    )
