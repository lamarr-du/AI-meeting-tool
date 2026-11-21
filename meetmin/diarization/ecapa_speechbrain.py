"""说话人分离：SpeechBrain ECAPA-TDNN 声纹嵌入 + 纯 NumPy 稳健聚类（兼容选项）。

流程：基于 ASR 语音区间切分（无 speech_spans 时退化为滑窗）→ 帧级语音活跃度与时长筛选 →
ECAPA 声纹嵌入与缓存 → 凝聚层次/谱聚类 + 小簇过滤 + L2 归一化质心二次重分配 →
相邻同簇合并且异簇边界取中点消重叠。首次运行自动下载 ECAPA 模型（约 80MB）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..model_cache import configure_xet_cache, hub_cache_dir
from .base import Diarizer
from .clustering import (
    SR,
    build_cache_signature,
    build_windows,
    cluster_embeddings,
    extract_wave_slice,
    load_embedding_cache,
    save_embedding_cache,
)
from .clustering import (
    filter_knn_outliers as _filter_knn_outliers,  # noqa: F401
)
from .clustering import (
    merge_non_overlapping as _merge_non_overlapping,
)
from .clustering import (
    reassign_by_centroids as _reassign_by_centroids,  # noqa: F401
)

ECAPA_REPO = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_FILES = [
    "hyperparams.yaml",
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "classifier.ckpt",
    "label_encoder.txt",
]
DEFAULT_CLUSTER_THRESHOLD = 0.64


class EcapaSpeechbrainDiarizer(Diarizer):
    name = "ecapa_speechbrain"
    supports_cache = True

    def __init__(
        self,
        dcfg: dict,
        *,
        model_cache_dir: str | Path | None = None,
        on_model_download: Callable[[bool], None] | None = None,
    ):
        configure_xet_cache(model_cache_dir)
        self._hub_cache = hub_cache_dir(model_cache_dir)
        try:
            import torch  # noqa: F401  (speechbrain 依赖)
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
        except ImportError as e:
            raise ImportError(
                "需要 speechbrain（会引入 torch）：pip install -r requirements-optional.txt"
            ) from e
        self.dcfg = dcfg
        self._on_model_download = on_model_download
        self._torch = torch
        self._classifier = EncoderClassifier
        self._strategy = LocalStrategy.NO_LINK
        self.encoder: Any = None

    def _load_encoder(self):
        if self.encoder is None:
            from huggingface_hub import snapshot_download, try_to_load_from_cache

            cached = try_to_load_from_cache(
                ECAPA_REPO, "hyperparams.yaml", cache_dir=self._hub_cache
            )
            source = Path(cached).parent if isinstance(cached, str) else None
            if source is None or not all((source / name).is_file() for name in ECAPA_FILES):
                if self._on_model_download is not None:
                    self._on_model_download(True)
                try:
                    source = Path(
                        snapshot_download(
                            ECAPA_REPO, cache_dir=self._hub_cache, allow_patterns=ECAPA_FILES
                        )
                    )
                finally:
                    if self._on_model_download is not None:
                        self._on_model_download(False)
            self.encoder = self._classifier.from_hparams(
                source=str(source),
                overrides={"pretrained_path": str(source)},
                local_strategy=self._strategy,
                run_opts={"device": "cpu"},
            )

    def _build_windows(
        self,
        audio16k: np.ndarray,
        speech_spans: list[tuple[float, float]] | None,
    ) -> tuple[list[int], list[int], list[bool], str, float, float, float]:
        return build_windows(audio16k, speech_spans, self.dcfg)

    def _compute_embeddings(
        self, audio16k: np.ndarray, starts: list[int], ends: list[int]
    ) -> np.ndarray:
        if not starts:
            return np.empty((0, 0), dtype=np.float32)
        wavs = [extract_wave_slice(audio16k, a, b) for a, b in zip(starts, ends)]
        lengths = {len(w) for w in wavs}
        if len(lengths) == 1:
            return np.concatenate(
                [
                    self._embed(np.stack(wavs[i : i + 32]).astype(np.float32))
                    for i in range(0, len(wavs), 32)
                ]
            )
        batches: list[np.ndarray] = []
        batch_size = 16
        for i in range(0, len(wavs), batch_size):
            chunk = wavs[i : i + batch_size]
            max_len = max(len(w) for w in chunk)
            buf = np.zeros((len(chunk), max_len), dtype=np.float32)
            rel_lens = np.zeros(len(chunk), dtype=np.float32)
            for j, w in enumerate(chunk):
                buf[j, : len(w)] = w
                rel_lens[j] = len(w) / max_len
            batches.append(self._embed(buf, wav_lens=rel_lens))
        return np.concatenate(batches, axis=0)

    # ------------------------------------------------------------------
    def diarize(self, audio16k: np.ndarray, *, speech_spans=None, cache_path=None) -> list[dict]:
        if audio16k is None or len(audio16k) < SR:
            return []
        min_rms = float(self.dcfg.get("min_window_rms", 0.003))
        starts, ends, core_flags, mode, win_sec, hop_sec, merge_gap_sec = self._build_windows(
            audio16k, speech_spans
        )
        audio_sha256 = hashlib.sha256(memoryview(np.ascontiguousarray(audio16k))).hexdigest()
        signature = build_cache_signature(
            model=ECAPA_REPO,
            mode=mode,
            win_sec=win_sec,
            hop_sec=hop_sec,
            merge_gap_sec=merge_gap_sec,
            min_rms=min_rms,
            speech_spans=speech_spans,
            audio_sha256=audio_sha256,
        )
        emb = load_embedding_cache(cache_path, signature, starts, ends)
        if emb is None:
            emb = self._compute_embeddings(audio16k, starts, ends)
            save_embedding_cache(cache_path, emb, starts, ends, signature)
        if not starts:
            return []

        core_mask = np.asarray(core_flags, dtype=bool)
        durations = np.asarray([(e - s) / SR for s, e in zip(starts, ends)], dtype=np.float64)
        if "_cluster" in self.__dict__:
            # 兼容外部测试直接以单参函数/Mock 替换实例方法 diarizer._cluster
            labels = self._cluster(emb)
        else:
            labels = self._cluster(emb, core_mask=core_mask, durations=durations)

        seg_merge_gap = (
            max(hop_sec, merge_gap_sec) if mode == "sliding" else max(merge_gap_sec, 0.5)
        )
        return _merge_non_overlapping(starts, ends, labels, merge_gap_sec=seg_merge_gap)

    # ------------------------------------------------------------------
    def _embed(self, windows: np.ndarray, wav_lens: np.ndarray | None = None) -> np.ndarray:
        self._load_encoder()
        torch = self._torch
        with torch.no_grad():
            w = torch.from_numpy(windows)  # [B, T]，单声道批次
            if wav_lens is not None:
                emb = self.encoder.encode_batch(w, wav_lens=torch.from_numpy(wav_lens))
            else:
                emb = self.encoder.encode_batch(w)  # [B, 1, D]
        E = emb.squeeze(1).cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        return E / np.clip(norms, 1e-9, None)

    def _cluster(
        self,
        emb: np.ndarray,
        *,
        core_mask: np.ndarray | None = None,
        durations: np.ndarray | None = None,
    ) -> np.ndarray:
        return cluster_embeddings(
            emb,
            self.dcfg,
            default_threshold=DEFAULT_CLUSTER_THRESHOLD,
            core_mask=core_mask,
            durations=durations,
        )
