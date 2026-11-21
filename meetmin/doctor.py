"""启动前的只读检查；不下载模型、不发送网络请求。"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

from .asr.cuda import gpu_status as gpu_status
from .asr.profile import memory_status, select_profile
from .model_cache import hub_cache_dir


def _model_cached(model, cache_dir):
    path = Path(str(model)).expanduser()
    if path.is_dir():
        return (path / "model.bin").is_file()
    repo = {
        "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    }.get(str(model), str(model))
    repo = repo if "/" in repo else "Systran/faster-whisper-" + repo
    snapshots = cache_dir / ("models--" + repo.replace("/", "--")) / "snapshots"
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
    configured_cache = cfg.get("model_cache_dir")
    asr_provider = asr.get("provider", "faster_whisper") if check_asr else None
    if asr_provider == "faster_whisper":
        for mod in ("faster_whisper", "av"):
            if importlib.util.find_spec(mod) is None:
                print(f"  错误：缺少 {mod}；请安装 requirements.txt")
                ok = False
        gpu = gpu_status(asr.get("compute_type") or "int8_float16")
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
            cache = hub_cache_dir(configured_cache)
            print(
                f"  转写档位：{profile.selected} · {model} / {profile.device} / {profile.compute_type}（{profile.reason}）"
            )
            print(
                f"  Whisper {model}：{'缓存中发现模型文件' if _model_cached(model, cache) else '未检测到缓存，请提前下载或配置本地模型目录'}"
            )
            print(f"  Whisper 下载目录：{cache}")
            if profile.device == "cuda" and profile.cpu_fallback_model != model:
                fallback = profile.cpu_fallback_model
                if not _model_cached(fallback, cache):
                    print(f"  CPU 回退模型 {fallback} 未检测到缓存，请提前准备。")
        if gpu is not None and gpu.memory is not None:
            name, free, total = gpu.memory
            print(f"  GPU：{name}，空闲 {free} MiB / 总计 {total} MiB")
        if gpu is not None:
            print(f"  CUDA：{gpu.reason or '设备和运行库检查通过，实际推理仍需验证。'}")
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
    diar_provider = str(d.get("provider", "campplus_onnx")).lower()
    if d.get("enabled", True) and diar_provider not in ("single",):
        if diar_provider in ("campplus_onnx", "campplus", "onnx", "3dspeaker"):
            if importlib.util.find_spec("onnxruntime") is None:
                print("  说话人分离依赖缺失：onnxruntime；将降级为单说话人")
            if d.get("model_path"):
                model_file = Path(str(d["model_path"])).expanduser()
                print(
                    f"  声纹模型：{'本地文件存在' if model_file.is_file() else '未找到本地模型文件'} ({model_file})"
                )
            else:
                cache = (
                    hub_cache_dir(configured_cache) / "models--csukuangfj--speaker-embedding-models"
                )
                cached = any(
                    cache.glob(
                        "snapshots/*/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
                    )
                )
                print(
                    f"  声纹模型(CAM++ ONNX)：{'已缓存' if cached else '未检测到缓存，首次运行将自动下载（约 27 MB）'} ({cache})"
                )
        else:
            missing = [m for m in ("torch", "speechbrain") if importlib.util.find_spec(m) is None]
            if missing:
                print("  可选说话人分离依赖缺失：" + ", ".join(missing) + "；将降级为单说话人")
            cache = hub_cache_dir(configured_cache) / "models--speechbrain--spkrec-ecapa-voxceleb"
            cached = any(cache.glob("snapshots/*/embedding_model.ckpt"))
            print(
                f"  声纹模型(ECAPA)：{'缓存目录存在，请确认文件完整' if cached else '未检测到缓存，请提前下载'} ({cache})"
            )
    pcfg = cfg.get("punctuation", {}) or {}
    punc_provider = str(pcfg.get("provider", "firered_onnx")).lower()
    if asr_provider == "faster_whisper" and punc_provider not in ("rule", "none", "off", ""):
        if punc_provider in ("firered_onnx", "firered", "fireredpunc"):
            missing = [
                m for m in ("onnxruntime", "tokenizers") if importlib.util.find_spec(m) is None
            ]
            if missing:
                print("  标点模型依赖缺失：" + ", ".join(missing) + "；将降级为规则标点")
            if pcfg.get("model_path"):
                mp = Path(str(pcfg["model_path"])).expanduser()
                exists = (mp / "punc.q8w.onnx").is_file() if mp.is_dir() else mp.is_file()
                print(
                    f"  标点模型(FireRedPunc)：{'本地文件存在' if exists else '未找到本地模型文件'} ({mp})"
                )
            else:
                cache = hub_cache_dir(configured_cache) / "models--jiangzhuo9357--fireredpunc-onnx"
                cached = any(cache.glob("snapshots/*/punc.q8w.onnx")) and any(
                    cache.glob("snapshots/*/tokenizer.json")
                )
                print(
                    f"  标点模型(FireRedPunc ONNX)：{'已缓存' if cached else '未检测到缓存，首次运行将自动下载（约 155 MB）'} ({cache})"
                )
        elif punc_provider in (
            "ct_transformer_onnx",
            "ct_transformer",
            "ct-transformer",
            "ct_punc",
            "ct",
        ):
            if importlib.util.find_spec("onnxruntime") is None:
                print("  标点模型依赖缺失：onnxruntime；将降级为规则标点")
            if pcfg.get("model_path"):
                mp = Path(str(pcfg["model_path"])).expanduser()
                exists = (mp / "model.int8.onnx").is_file() if mp.is_dir() else mp.is_file()
                print(
                    f"  标点模型(CT-Transformer)：{'本地文件存在' if exists else '未找到本地模型文件'} ({mp})"
                )
            else:
                cache = (
                    hub_cache_dir(configured_cache)
                    / "models--ranger810--sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8"
                )
                cached = any(cache.glob("snapshots/*/model.int8.onnx"))
                print(
                    f"  标点模型(CT-Transformer ONNX)：{'已缓存' if cached else '未检测到缓存，首次运行将自动下载（约 72 MB）'} ({cache})"
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
