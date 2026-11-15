"""启动前的只读检查；不下载模型、不发送网络请求。"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

from .asr.profile import memory_status, select_profile


def gpu_status():
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
        )
        name, free_text, total_text = [
            x.strip() for x in result.stdout.splitlines()[0].split(",")[:3]
        ]
        free, total = int(free_text), int(total_text)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None
    # DLL/SO 加载可检出缺失及间接依赖，但无法保证推理一定成功。
    libraries = [
        ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll")
        if os.name == "nt"
        else ("libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9")
    ][0]
    missing = []
    for lib in libraries:
        try:
            ctypes.CDLL(lib)
        except OSError:
            missing.append(lib)
    return name, free, total, missing


def _model_cached(model):
    path = Path(str(model)).expanduser()
    if path.is_dir():
        return (path / "model.bin").is_file()
    repo = {
        "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    }.get(str(model), str(model))
    repo = repo if "/" in repo else "Systran/faster-whisper-" + repo
    root = Path(
        os.environ.get("HF_HUB_CACHE")
        or Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"
    )
    snapshots = root / ("models--" + repo.replace("/", "--")) / "snapshots"
    return any(p.is_file() for p in snapshots.glob("*/model.bin"))


def run_doctor(cfg, *, recording=True, device=None, file_url=None, check_asr=True):
    print("预检（不联网、不下载模型）：")
    ok = True
    if recording:
        try:
            from .audio.loopback_recorder import (
                LoopbackRecorder,
                MicrophoneRecorder,
                _import_pyaudio,
            )

            pa = _import_pyaudio().PyAudio()
            try:
                recorder = cfg.get("recorder", {})
                checks = [("内录", LoopbackRecorder(device=device))]
                if recorder.get("microphone"):
                    checks.append(
                        ("麦克风", MicrophoneRecorder(device=recorder.get("microphone_device")))
                    )
                for label, capture in checks:
                    try:
                        d = capture._resolve_device(pa)
                        print(f"  {label}设备：[{d['index']}] {d['name']}")
                    except Exception as e:
                        print(f"  错误：无法选择{label}设备：{e}")
                        ok = False
            finally:
                pa.terminate()
        except Exception as e:
            print(f"  错误：无法选择内录设备：{e}")
            ok = False
    asr = cfg.get("asr", {}) or {}
    asr_provider = asr.get("provider", "faster_whisper") if check_asr else None
    if asr_provider == "faster_whisper":
        for mod in ("faster_whisper", "av"):
            if importlib.util.find_spec(mod) is None:
                print(f"  错误：缺少 {mod}；请安装 requirements.txt")
                ok = False
        gpu = gpu_status()
        memory = memory_status()
        if memory:
            print(f"  内存：可用 {memory[1] / 1024:.1f} GiB / 总计 {memory[0] / 1024:.1f} GiB")
        try:
            profile = select_profile(asr, memory=memory, gpu=gpu)
        except ValueError as error:
            print(f"  错误：{error}")
            ok = False
        else:
            model = profile.model
            print(
                f"  转写档位：{profile.selected} · {model} / {profile.device} / {profile.compute_type}（{profile.reason}）"
            )
            print(
                f"  Whisper {model}：{'缓存中发现模型文件' if _model_cached(model) else '未检测到缓存，请提前下载或配置本地模型目录'}"
            )
            if profile.device == "cuda" and profile.cpu_fallback_model != model:
                fallback = profile.cpu_fallback_model
                if not _model_cached(fallback):
                    print(f"  CPU 回退模型 {fallback} 未检测到缓存，请提前准备。")
        if gpu:
            name, free, total, missing = gpu
            print(f"  GPU：{name}，空闲 {free} MiB / 总计 {total} MiB")
            if missing:
                print(
                    "  CUDA 运行库无法加载："
                    + ", ".join(missing)
                    + "；Windows 需要 CUDA 12 和 cuDNN 9，建议使用 CPU int8"
                )
            else:
                print(
                    "  CUDA 库已加载；实际推理仍需验证。建议显存充足时使用 int8_float16，不启用批量推理。"
                )
        else:
            print("  未检测到 NVIDIA GPU；CPU int8 可用（速度取决于机器和模型）。")
    elif asr_provider == "dashscope":
        dc = cfg.get("dashscope_asr", {}) or {}
        if not (file_url or dc.get("file_url")):
            print("  错误：DashScope 转写需要 --file-url 或配置 file_url")
            ok = False
        key = dc.get("api_key_env", "DASHSCOPE_API_KEY")
        if not os.environ.get(key):
            print(f"  错误：未设置 {key}")
            ok = False
    d = cfg.get("diarization", {}) or {}
    if d.get("enabled", True) and d.get("provider", "ecapa_speechbrain") not in ("single",):
        missing = [
            m for m in ("torch", "speechbrain", "sklearn") if importlib.util.find_spec(m) is None
        ]
        if missing:
            print("  可选说话人分离依赖缺失：" + ", ".join(missing) + "；将降级为单说话人")
        cache = Path.home() / ".cache" / "meetmin" / "speechbrain"
        print(
            f"  声纹模型：{'缓存目录存在，请确认文件完整' if cache.is_dir() and any(cache.iterdir()) else '未检测到缓存，请提前下载'} ({cache})"
        )
    llm = cfg.get("llm", {}) or {}
    if llm.get("provider", "openai_compatible") not in ("mock", "none", "offline", ""):
        key = llm.get("api_key_env", "LLM_API_KEY")
        print(
            f"  LLM 密钥环境变量 {key}：{'已设置' if os.environ.get(key) else '未设置（服务可能要求鉴权）'}；模型名未联网验证"
        )
    p = Path(cfg.get("output_dir", "data")).resolve()
    while not p.exists():
        p = p.parent
    print(f"  磁盘剩余：{shutil.disk_usage(p).free / 1024**3:.1f} GiB（{p}）")
    return ok
