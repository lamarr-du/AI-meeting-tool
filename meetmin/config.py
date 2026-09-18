"""配置加载：config.yaml 覆盖内置默认值。"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "output_dir": "data",  # 会话产物目录：<output_dir>/sessions/<session_id>/

    "recorder": {
        "device": None,        # 内录设备索引或名称片段；null=默认输出设备的 loopback
        "channels": None,      # null=跟随设备（最多取 2）
        "sample_rate": None,   # null=跟随设备默认采样率
        "chunk_ms": 100,       # 每次读取的块长（毫秒）
    },

    "asr": {
        "provider": "faster_whisper",  # faster_whisper（本地，零 token）| dashscope（云端 Paraformer）
        "language": "zh",              # null=自动检测
        "model_size": "small",         # tiny/base/small/medium/large-v3/large-v3-turbo
        "device": "auto",              # auto/cpu/cuda
        "compute_type": "int8",        # int8 最省资源；有 GPU 可用 float16
        "beam_size": 1,                # 1=贪心解码，速度优先
        "word_timestamps": True,       # 词级时间戳（供说话人分离对齐，建议保留）
        "hf_endpoint": None,           # 国内可设 https://hf-mirror.com 下载模型
        "hf_disable_xet": True,        # 避免部分网络环境访问 Xet 重建服务时出现 401
    },

    "diarization": {
        "enabled": True,
        "provider": "ecapa_speechbrain",  # ecapa_speechbrain | single
        "num_speakers": None,             # 已知人数时可填，聚类更准
        "window_sec": 1.5,
        "hop_sec": 0.75,
        "cluster_threshold": 0.55,        # 余弦距离聚类阈值（num_speakers 为 null 时生效）
        "min_window_rms": 0.003,          # 过窗能量阈值，过滤近静音窗口
        "min_group_sec": 0.4,             # 短于该时长的说话人片段并入相邻片段
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
        "model": "qwen-3.8-flash",           # ← 按你实际开通的模型名修改（如 qwen-flash/qwen-plus）
        "temperature": 0.3,
        "max_output_tokens": 2000,
        "timeout_sec": 120,
        "retries": 3,
        "extra_request": {},  # 透传给接口的额外字段，如 DashScope 关闭思考：{"enable_thinking": false}
    },

    "summarizer": {
        "strategy": "auto",            # auto | direct（整篇一次调用，最省）| map_reduce（长会议两级压缩）
        "merge_same_speaker": True,    # 相邻同发言人合并，显著省 token
        "with_timestamps": False,      # 时间戳对纪要价值低且耗 token，默认不进 prompt
        "max_chunk_chars": 4000,       # map_reduce 每段字符预算
        "direct_threshold_chars": 9000,  # 转写低于该长度直接一次调用
        "chars_per_token": 1.5,        # 估算用：每 token 约多少字符
        "price_per_1k_input": 0.0,     # 元/千 token，用于成本估算（0=不估算）
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
    return cfg
