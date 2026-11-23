"""配置加载：config.yaml 覆盖内置默认值。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TARGET_RMS = 0.03
MIN_TARGET_RMS = 0.001
MAX_TARGET_RMS = 0.3

DEFAULT_CONFIG: dict[str, Any] = {
    "output_dir": "data",  # 会话产物目录：<output_dir>/sessions/<session_id>/
    "model_cache_dir": None,  # 默认工程或独立便携版目录下的 .cache
    "voiceprint_library_dir": None,  # 默认 <output_dir>/voiceprints
    "recorder": {
        "device": None,  # 内录设备索引或名称片段；null=默认输出设备的 loopback
        "channels": None,  # null=跟随设备（最多取 2）
        "sample_rate": None,  # null=跟随设备默认采样率
        "microphone": False,
        "microphone_device": None,  # null=默认 WASAPI 输入
        "target_rms": DEFAULT_TARGET_RMS,  # 录音自适应增益目标电平
        "chunk_ms": 100,  # 每次读取的块长（毫秒）
        "segment_seconds": 300,  # 单个录音分段时长
        "sync_seconds": 10,  # 活动分段的磁盘同步间隔
    },
    "asr": {
        "provider": "faster_whisper",  # faster_whisper（本地，零 token）| dashscope（云端 Paraformer）
        "language": "zh",  # null=自动检测
        "profile": "auto",
        "model_size": None,  # null 由档位选择；也可指定本地模型目录
        "cuda_peak_mib": None,  # 当前模型和计算类型在目标 GPU 上的实测峰值
        "device": "auto",  # auto/cpu/cuda
        "compute_type": None,  # 跟随设备，CPU int8、CUDA int8_float16
        "beam_size": 1,  # 1=贪心解码，速度优先
        "condition_on_previous_text": False,  # 关闭前文条件，减少重复和幻觉
        "no_repeat_ngram_size": 0,
        "repetition_penalty": 1.0,
        "hallucination_silence_threshold": None,
        "quality": {},  # 默认值由本地 Whisper 的 QualityOptions 提供
        "punctuation_prompt": None,
        "hotwords": None,
        "glossary": {"path": None, "terms": None, "max_terms": 20, "correction_path": None},
        "word_timestamps": True,  # 词级时间戳（供说话人分离对齐，建议保留）
        "hf_endpoint": None,  # 国内可设 https://hf-mirror.com 下载模型
        "hf_disable_xet": True,  # 避免部分网络环境访问 Xet 重建服务时出现 401
    },
    "live": {
        "enabled": True,
        "model_size": "large-v3-turbo",
        "device": "auto",
        "beam_size": 1,
        "hop_sec": 1.8,
        "backpressure_hop_sec": 4.0,
        "max_window_sec": 15,
        "ring_buffer_sec": 60,
        "endpoint_silence_ms": 600,
        "flush_timeout_sec": 12,
        "diarization": {
            "enabled": True,
            "min_embed_sec": 1.5,
            "online_assign_sim": 0.58,
            "online_new_cluster_sim": 0.40,
            "online_ema_alpha": 0.15,
            "warmup_recluster_sec": [20, 45, 90],
            "steady_recluster_sec": 45,
            "bounded_max_embeddings": 1000,
            "bounded_window_sec": 900,
            "prototypes_per_speaker": 15,
            "freeze_after_sec": 120,
            "relabel_hysteresis_margin": 0.08,
        },
        "bleed_suppression": {
            "enabled": True,
            "window_ms": 600,
            "min_bigram_similarity": 0.75,
        },
    },
    "diarization": {
        "enabled": True,
        "provider": "campplus_onnx",  # campplus_onnx（默认，纯 ONNX 零 Torch）| ecapa_speechbrain | single
        "num_speakers": None,  # 已知人数时可填，聚类更准
        "model_path": None,  # 可选：指定本地 ONNX 声纹模型路径（campplus_onnx 模式生效）
        "window_mode": "speech_spans",  # speech_spans（按语音段切分，推荐）| sliding（固定滑窗）
        "window_sec": 2.5,  # speech_spans 模式下为长段切分上限，sliding 模式下为窗长
        "hop_sec": 1.25,  # sliding 模式下为步长
        "merge_gap_sec": 0.3,  # speech_spans 模式下合并短停顿相邻片段的间隔阈值
        "min_cluster_dur_sec": 1.2,  # 短于该时长的片段不参与主聚类，仅在聚类后重分配
        "cluster_method": "auto",  # auto（自动人数 average / 指定人数 ward）| average | ward | spectral
        "cluster_threshold": None,  # null 随模型自动选默认值（campplus_onnx: 0.54，ecapa_speechbrain: 0.64）
        "min_window_rms": 0.003,  # 过窗能量阈值，过滤近静音窗口
        "min_active_ratio": 0.45,  # 20ms 帧级语音活跃占比门槛，低于此值不参与主聚类
        "min_cluster_size": 2,  # 主簇最少核心段数，小簇将重分配到最近主簇
        "min_cluster_sec": 4.0,  # 主簇最少核心累计时长（秒）
        "min_group_sec": 0.4,  # 短于该时长的说话人片段并入相邻片段
    },
    "punctuation": {
        "provider": "firered_onnx",  # firered_onnx（默认，中英高精度 155 MB）| ct_transformer_onnx（轻量 72 MB）| rule（零模型）
        "model_path": None,  # 可选：指定本地 ONNX 标点模型文件或目录
        "to_simplified": True,  # 统一将中文转写结果中的繁体字转为简体中文（解决 Whisper 混出繁体问题）
        "max_sentence_tokens": 85,  # 超长独白软切分上限（词元数）
        "max_sentence_sec": 18.0,  # 超长独白软切分上限（秒）
        "left_context_tokens": 15,  # 跨块只读左上下文词元数，稳定句界预测
        "enable_polish": True,  # 同步生成轻量零幻觉口语顺滑视图 polished_text
        "paragraph_max_gap_sec": 2.2,  # transcript.md 同说话人停顿换段阈值（秒）
        "paragraph_max_chars": 220,  # transcript.md 单段目标字数上限
        "paragraph_max_duration_sec": 45.0,  # transcript.md 单段持续时长上限（秒）
    },
    "dashscope_asr": {  # 云端 ASR（步骤2的可选接口，需要音频有可访问 URL）
        "base_url": "https://dashscope.aliyuncs.com",
        "api_key_env": "DASHSCOPE_API_KEY",
        "model": "paraformer-v2",
        "speaker_count": None,
        "language_hints": ["zh", "en"],
        "poll_interval_sec": 3,
        "timeout_sec": 1800,
    },
    "llm": {  # 大模型接口（步骤3，OpenAI 兼容 chat/completions）
        "provider": "openai_compatible",  # openai_compatible | mock（离线联调）
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",  # API Key 从该环境变量读取，不写进配置文件
        "model": "qwen-3.8-flash",  # ← 按你实际开通的模型名修改（如 qwen-flash/qwen-plus）
        "temperature": 0.3,
        "max_output_tokens": 2000,
        "timeout_sec": 120,
        "retries": 3,  # 最大请求次数（含首次请求）
        "extra_request": {},  # 透传给接口的额外字段，如 DashScope 关闭思考：{"enable_thinking": false}
    },
    "summarizer": {
        "strategy": "auto",  # auto | direct（整篇一次调用，最省）| map_reduce（长会议两级压缩）
        "merge_same_speaker": True,  # 相邻同发言人合并，显著省 token
        "with_timestamps": False,  # 时间戳对纪要价值低且耗 token，默认不进 prompt
        "use_polished_text": False,  # true 时优先将去口癖的 polished_text 喂给纪要模型
        "max_chunk_chars": 4000,  # map_reduce 每段字符预算
        "direct_threshold_chars": 9000,  # 转写低于该长度直接一次调用
        "chars_per_token": 1.5,  # 估算用：每 token 约多少字符
        "price_per_1k_input": 0.0,  # 元/千 token，用于成本估算（0=不估算）
        "price_per_1k_output": 0.0,
        "guess_speaker_roles": False,  # 让模型根据内容推断"说话人A≈主持人"等（多花极少 token）
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None) -> dict:
    """加载 config.yaml（不存在则用默认值），与内置默认深合并。"""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    p = Path(path) if path else Path("config.yaml")
    if p.exists():
        user = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(user, dict):
            raise ValueError(f"配置文件格式错误（应为 YAML 映射）: {p}")
        cfg = deep_merge(cfg, user)
    config_dir = p.resolve().parent
    if (cache_dir := cfg.get("model_cache_dir")) is not None:
        if not isinstance(cache_dir, str) or not cache_dir.strip():
            raise ValueError("model_cache_dir 必须是目录路径或 null")
        cfg["model_cache_dir"] = str(config_dir / Path(cache_dir).expanduser())
    if (voiceprint_dir := cfg.get("voiceprint_library_dir")) is not None:
        if not isinstance(voiceprint_dir, str) or not voiceprint_dir.strip():
            raise ValueError("voiceprint_library_dir 必须是目录路径或 null")
        cfg["voiceprint_library_dir"] = str(config_dir / Path(voiceprint_dir).expanduser())
    asr = cfg.get("asr")
    if isinstance(asr, dict) and isinstance(model := asr.get("model_size"), str):
        local_model = config_dir / Path(model).expanduser()
        if local_model.is_dir():
            asr["model_size"] = str(local_model)
    live = cfg.get("live")
    if isinstance(live, dict) and isinstance(model := live.get("model_size"), str):
        local_model = config_dir / Path(model).expanduser()
        if local_model.is_dir():
            live["model_size"] = str(local_model)
    glossary = asr.get("glossary") if isinstance(asr, dict) else None
    if isinstance(glossary, dict):
        for key in ("path", "correction_path"):
            value = glossary.get(key)
            if isinstance(value, str) and value.strip():
                glossary[key] = str(config_dir / Path(value).expanduser())
    for section in ("diarization", "punctuation"):
        sec = cfg.get(section)
        if isinstance(sec, dict) and isinstance(mp := sec.get("model_path"), str) and mp.strip():
            sec["model_path"] = str(config_dir / Path(mp).expanduser())
    return cfg
