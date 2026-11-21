"""说话人分离：SpeechBrain ECAPA-TDNN 声纹嵌入 + 稳健聚类（全本地、零 token）。

流程：基于 ASR 语音区间切分（无 speech_spans 时退化为滑窗）→ 帧级语音活跃度与时长筛选 →
ECAPA 声纹嵌入与缓存 → 凝聚层次/谱聚类 + 小簇过滤 + L2 归一化质心二次重分配 →
相邻同簇合并且异簇边界取中点消重叠。首次运行自动下载 ECAPA 模型（约 80MB）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import secrets
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from ..model_cache import configure_xet_cache, hub_cache_dir
from .base import Diarizer

SR = 16000
_FRAME_SAMPLES = 320  # 20ms @ 16kHz
_MIN_EMBED_SAMPLES = int(0.4 * SR)
_MIN_VALID_SAMPLES = 400
ECAPA_REPO = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_FILES = [
    "hyperparams.yaml",
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "classifier.ckpt",
    "label_encoder.txt",
]


def _rms(wav: np.ndarray) -> float:
    if len(wav) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))


def _frame_active_ratio(wav: np.ndarray, frame_rms_threshold: float) -> float:
    n_frames = len(wav) // _FRAME_SAMPLES
    if n_frames <= 0:
        return 1.0 if _rms(wav) >= frame_rms_threshold else 0.0
    frames = wav[: n_frames * _FRAME_SAMPLES].reshape(n_frames, _FRAME_SAMPLES)
    frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    return float(np.mean(frame_rms >= frame_rms_threshold))


def _estimate_frame_rms_threshold(
    audio16k: np.ndarray,
    speech_spans: list[tuple[float, float]] | None,
    min_window_rms: float,
) -> float:
    """根据语音区间帧能量分布估算自适应帧门限，兼顾低音量录音与噪声过滤。"""
    chunks: list[np.ndarray] = []
    if speech_spans:
        for s, e in speech_spans:
            a = max(0, int(round(s * SR)))
            b = min(len(audio16k), int(round(e * SR)))
            seg = audio16k[a:b]
            n_frames = len(seg) // _FRAME_SAMPLES
            if n_frames > 0:
                frames = seg[: n_frames * _FRAME_SAMPLES].reshape(n_frames, _FRAME_SAMPLES)
                chunks.append(np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1)))
    else:
        n_frames = len(audio16k) // _FRAME_SAMPLES
        if n_frames > 0:
            frames = audio16k[: n_frames * _FRAME_SAMPLES].reshape(n_frames, _FRAME_SAMPLES)
            chunks.append(np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1)))
    if not chunks:
        return max(min_window_rms, 0.015)
    all_rms = np.concatenate(chunks)
    p75 = float(np.percentile(all_rms, 75))
    upper = max(min_window_rms, 0.02)
    return float(np.clip(p75 * 0.25, min_window_rms, upper))


def _extract_wave_slice(audio16k: np.ndarray, start: int, end: int) -> np.ndarray:
    """提取用于声纹编码的波形；极短片段向两侧补齐上下文到 0.4s，不改变原时间标注。"""
    if end - start >= _MIN_EMBED_SAMPLES or len(audio16k) <= _MIN_EMBED_SAMPLES:
        return audio16k[start:end]
    center = (start + end) // 2
    half = _MIN_EMBED_SAMPLES // 2
    ext_start = max(0, min(center - half, len(audio16k) - _MIN_EMBED_SAMPLES))
    ext_end = min(len(audio16k), ext_start + _MIN_EMBED_SAMPLES)
    return audio16k[ext_start:ext_end]


def _spectral_cluster(
    emb: np.ndarray,
    *,
    num_speakers: int | None,
    p_percentile: float = 0.75,
    max_speakers: int = 10,
) -> np.ndarray:
    """余弦亲和矩阵 p-pruning + 归一化拉普拉斯特征间隙估计人数 + KMeans。"""
    from sklearn.cluster import KMeans

    n = emb.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int)
    sim = np.clip(emb @ emb.T, 0.0, 1.0).astype(np.float64)
    np.fill_diagonal(sim, 0.0)
    if p_percentile > 0.0 and n > 2:
        pruned = np.zeros_like(sim)
        k_keep = max(2, int(round((1.0 - p_percentile) * (n - 1))))
        k_keep = min(k_keep, n - 1)
        for i in range(n):
            top = np.argpartition(sim[i], -k_keep)[-k_keep:]
            pruned[i, top] = sim[i, top]
        affinity = 0.5 * (pruned + pruned.T)
    else:
        affinity = 0.5 * (sim + sim.T)

    degree = np.sum(affinity, axis=1)
    inv_sqrt = np.where(degree > 1e-12, 1.0 / np.sqrt(np.maximum(degree, 1e-12)), 0.0)
    laplacian = np.eye(n) - (inv_sqrt[:, None] * affinity * inv_sqrt[None, :])
    laplacian = 0.5 * (laplacian + laplacian.T)
    eigvals, eigvecs = np.linalg.eigh(laplacian)

    if num_speakers is not None:
        k = min(int(num_speakers), n)
    else:
        k_max = min(max_speakers, n - 1)
        if k_max < 2:
            return np.zeros(n, dtype=int)
        gaps = np.diff(eigvals[: k_max + 1])
        k = int(np.argmax(gaps)) + 1

    if k <= 1:
        return np.zeros(n, dtype=int)
    spectral_emb = eigvecs[:, :k]
    norms = np.linalg.norm(spectral_emb, axis=1, keepdims=True)
    spectral_emb = spectral_emb / np.clip(norms, 1e-9, None)
    return KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(spectral_emb)


def _filter_knn_outliers(
    core_emb: np.ndarray,
    *,
    min_keep: int,
    min_knn_sim: float = 0.20,
) -> np.ndarray:
    """剔除与最近邻几乎无相似度的孤立离群点，防止固定人数聚类时异常窗占坑。"""
    n = core_emb.shape[0]
    if n < max(min_keep + 4, 8):
        return np.arange(n, dtype=int)
    sim = core_emb @ core_emb.T
    np.fill_diagonal(sim, -1.0)
    k_nn = min(3, n - 1)
    top_sim = np.partition(sim, n - k_nn, axis=1)[:, -k_nn:]
    mean_sim = np.mean(top_sim, axis=1)
    keep = np.where(mean_sim >= min_knn_sim)[0]
    if len(keep) < min_keep:
        return np.arange(n, dtype=int)
    return keep


def _reassign_by_centroids(
    all_emb: np.ndarray,
    core_idx: np.ndarray,
    core_labels: np.ndarray,
    durations: np.ndarray,
    *,
    min_cluster_size: int,
    min_cluster_sec: float,
    keep_top_k: int | None = None,
) -> np.ndarray:
    """保留达标主簇，计算 L2 归一化质心，将所有样本按最大余弦相似度重分配。"""
    unique_labels = sorted({int(x) for x in core_labels})
    stats: list[tuple[int, int, float, np.ndarray]] = []
    for lab in unique_labels:
        mask = core_labels == lab
        idx = core_idx[mask]
        cnt = int(np.sum(mask))
        sec = float(np.sum(durations[idx]))
        stats.append((lab, cnt, sec, idx))

    if keep_top_k is not None:
        stats.sort(key=lambda item: (item[2], item[1]), reverse=True)
        kept = stats[:keep_top_k]
    else:
        kept = [
            item for item in stats if item[1] >= min_cluster_size and item[2] >= min_cluster_sec
        ]
        if not kept:
            stats.sort(key=lambda item: (item[2], item[1]), reverse=True)
            kept = stats[:1]

    centroids: list[np.ndarray] = []
    kept_labels: list[int] = []
    for lab, _, _, idx in kept:
        centroid = np.mean(all_emb[idx], axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-9:
            centroid = centroid / norm
        centroids.append(centroid)
        kept_labels.append(lab)

    centroid_matrix = np.stack(centroids, axis=0)
    similarities = all_emb @ centroid_matrix.T
    best = np.argmax(similarities, axis=1)
    return np.asarray([kept_labels[int(b)] for b in best], dtype=int)


def _merge_non_overlapping(
    starts: list[int],
    ends: list[int],
    labels: np.ndarray,
    *,
    merge_gap_sec: float,
) -> list[dict]:
    """合并相邻同簇时间段；相邻异簇若有重叠则以两窗中心的中点作为精确边界。"""
    if not starts:
        return []
    segs: list[dict] = []
    cur_label = int(labels[0])
    cur_start = starts[0] / SR
    cur_end = ends[0] / SR
    prev_mid = (cur_start + cur_end) / 2.0

    for i in range(1, len(starts)):
        st = starts[i] / SR
        en = ends[i] / SR
        lab = int(labels[i])
        mid = (st + en) / 2.0
        if lab == cur_label and st <= cur_end + merge_gap_sec:
            cur_end = max(cur_end, en)
            prev_mid = mid
        else:
            if st < cur_end:
                boundary = (prev_mid + mid) / 2.0
                boundary = max(cur_start, min(en, boundary))
                cur_end = boundary
                new_start = boundary
            else:
                new_start = st
            if cur_end > cur_start:
                segs.append({"start": cur_start, "end": cur_end, "label": cur_label})
            cur_label, cur_start, cur_end = lab, new_start, en
            prev_mid = mid
    if cur_end > cur_start:
        segs.append({"start": cur_start, "end": cur_end, "label": cur_label})
    return segs


class EcapaSpeechbrainDiarizer(Diarizer):
    name = "ecapa_speechbrain"

    def __init__(self, dcfg: dict, *, model_cache_dir: str | Path | None = None):
        configure_xet_cache(model_cache_dir)
        self._hub_cache = hub_cache_dir(model_cache_dir)
        try:
            import torch  # noqa: F401  (speechbrain 依赖)
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
        except ImportError as e:
            raise ImportError(
                "需要 speechbrain + scikit-learn（会引入 torch）：pip install -r requirements-optional.txt"
            ) from e
        self.dcfg = dcfg
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
                source = Path(
                    snapshot_download(
                        ECAPA_REPO, cache_dir=self._hub_cache, allow_patterns=ECAPA_FILES
                    )
                )
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
        """构建待提取声纹的时间段，并标记哪些高质量段参与主聚类（core）。"""
        configured_mode = str(self.dcfg.get("window_mode", "speech_spans")).lower()
        mode = "speech_spans" if configured_mode == "speech_spans" and speech_spans else "sliding"
        win_sec = float(self.dcfg.get("window_sec", 2.5))
        hop_sec = float(self.dcfg.get("hop_sec", 1.25))
        merge_gap_sec = float(self.dcfg.get("merge_gap_sec", 0.3))
        min_rms = float(self.dcfg.get("min_window_rms", 0.003))
        min_active_ratio = float(self.dcfg.get("min_active_ratio", 0.45))
        min_cluster_dur = float(self.dcfg.get("min_cluster_dur_sec", 1.2))
        frame_thr = _estimate_frame_rms_threshold(audio16k, speech_spans, min_rms)

        starts: list[int] = []
        ends: list[int] = []
        core_flags: list[bool] = []
        total_sec = len(audio16k) / SR

        if mode == "speech_spans" and speech_spans:
            merged: list[list[float]] = []
            for s, e in sorted(speech_spans):
                s = max(0.0, float(s))
                e = min(total_sec, float(e))
                if e <= s:
                    continue
                if merged and s - merged[-1][1] <= merge_gap_sec:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])

            max_chunk = max(win_sec, 0.5)
            for s, e in merged:
                dur = e - s
                if dur <= max_chunk:
                    sub_ranges = [(s, e)]
                else:
                    n_chunks = max(1, int(math.ceil(dur / max_chunk)))
                    step = dur / n_chunks
                    sub_ranges = [(s + i * step, s + (i + 1) * step) for i in range(n_chunks)]
                for rs, re in sub_ranges:
                    a = max(0, int(round(rs * SR)))
                    b = min(len(audio16k), int(round(re * SR)))
                    if b <= a:
                        continue
                    wav = _extract_wave_slice(audio16k, a, b)
                    if len(wav) < _MIN_VALID_SAMPLES or _rms(wav) < min_rms:
                        continue
                    active_ratio = _frame_active_ratio(wav, frame_thr)
                    is_core = (re - rs) >= min_cluster_dur and active_ratio >= min_active_ratio
                    starts.append(a)
                    ends.append(b)
                    core_flags.append(is_core)
        else:
            win = max(int(win_sec * SR) // 2 * 2, _MIN_VALID_SAMPLES)
            hop = max(int(hop_sec * SR) // 2 * 2, 200)

            def in_speech(t: float) -> bool:
                if not speech_spans:
                    return True
                for s, e in speech_spans:
                    if s - 0.05 <= t <= e + 0.05:
                        return True
                return False

            for st in range(0, max(0, len(audio16k) - win) + 1, hop):
                mid = (st + win / 2) / SR
                if not in_speech(mid):
                    continue
                wav = audio16k[st : st + win]
                if _rms(wav) < min_rms:
                    continue
                active_ratio = _frame_active_ratio(wav, frame_thr)
                starts.append(st)
                ends.append(st + win)
                core_flags.append(active_ratio >= min_active_ratio)

        return starts, ends, core_flags, mode, win_sec, hop_sec, merge_gap_sec

    def _compute_embeddings(
        self, audio16k: np.ndarray, starts: list[int], ends: list[int]
    ) -> np.ndarray:
        if not starts:
            return np.empty((0, 0), dtype=np.float32)
        wavs = [_extract_wave_slice(audio16k, a, b) for a, b in zip(starts, ends)]
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
        metadata = {
            "schema_version": 2,
            "model": ECAPA_REPO,
            "sample_rate": SR,
            "window_mode": mode,
            "window_sec": win_sec,
            "hop_sec": hop_sec,
            "merge_gap_sec": merge_gap_sec,
            "min_window_rms": min_rms,
            "speech_spans": speech_spans,
            "audio_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(audio16k))).hexdigest(),
        }
        signature = json.dumps(metadata, sort_keys=True)
        emb = None
        if cache_path is not None and Path(cache_path).is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    candidate = cached["embeddings"]
                    if (
                        str(cached["metadata"].item()) == signature
                        and np.array_equal(cached["starts"], starts)
                        and np.array_equal(cached["ends"], ends)
                        and candidate.ndim == 2
                        and candidate.shape[0] == len(starts)
                        and np.isfinite(candidate).all()
                    ):
                        emb = candidate.copy()
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                pass  # 损坏或不匹配的缓存重新生成
        if emb is None:
            emb = self._compute_embeddings(audio16k, starts, ends)
            if cache_path is not None:
                path = Path(cache_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name("." + path.name + "." + secrets.token_hex(8) + ".tmp")
                try:
                    with temporary.open("xb") as f:
                        np.savez_compressed(
                            f,
                            embeddings=emb,
                            starts=np.asarray(starts, dtype=np.int64),
                            ends=np.asarray(ends, dtype=np.int64),
                            metadata=np.asarray(signature),
                        )
                        f.flush()
                        os.fsync(f.fileno())
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
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
        from sklearn.cluster import AgglomerativeClustering

        n_total = emb.shape[0]
        if n_total <= 1:
            return np.zeros(n_total, dtype=int)

        if core_mask is None or len(core_mask) != n_total or not bool(np.any(core_mask)):
            core_idx = np.arange(n_total, dtype=int)
        else:
            core_idx = np.where(core_mask)[0]

        if durations is None or len(durations) != n_total:
            default_dur = float(self.dcfg.get("window_sec", 2.5))
            durations = np.full(n_total, default_dur, dtype=np.float64)

        k = self.dcfg.get("num_speakers")
        thr = float(self.dcfg.get("cluster_threshold", 0.64))
        method = str(self.dcfg.get("cluster_method", "auto")).lower()
        min_cluster_size = max(1, int(self.dcfg.get("min_cluster_size", 2)))
        min_cluster_sec = max(0.0, float(self.dcfg.get("min_cluster_sec", 4.0)))

        if k is not None:
            target_k = int(k)
            if target_k <= 1 or len(core_idx) <= 1:
                return np.zeros(n_total, dtype=int)
            target_k = min(target_k, len(core_idx))
            # 先剔除极端孤立离群点，避免异常窗占掉固定人数名额
            inlier_rel = _filter_knn_outliers(emb[core_idx], min_keep=target_k)
            fit_idx = core_idx[inlier_rel]
            fit_emb = emb[fit_idx]

            if method in ("auto", "ward"):
                model = AgglomerativeClustering(
                    n_clusters=target_k, metric="euclidean", linkage="ward"
                )
                fit_labels = model.fit_predict(fit_emb)
            elif method == "average":
                model = AgglomerativeClustering(
                    n_clusters=target_k, metric="cosine", linkage="average"
                )
                fit_labels = model.fit_predict(fit_emb)
            elif method == "spectral":
                fit_labels = _spectral_cluster(fit_emb, num_speakers=target_k)
            else:
                raise ValueError(
                    f"未知 cluster_method: {method}（可选 auto | average | ward | spectral）"
                )
            return _reassign_by_centroids(
                emb,
                fit_idx,
                fit_labels,
                durations,
                min_cluster_size=1,
                min_cluster_sec=0.0,
                keep_top_k=target_k,
            )

        # 自动人数模式
        if len(core_idx) <= 1:
            return np.zeros(n_total, dtype=int)
        core_emb = emb[core_idx]
        if method in ("auto", "average"):
            model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=thr, metric="cosine", linkage="average"
            )
            core_labels = model.fit_predict(core_emb)
        elif method == "ward":
            euclid_thr = math.sqrt(max(0.0, 2.0 * thr))
            model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=euclid_thr, metric="euclidean", linkage="ward"
            )
            core_labels = model.fit_predict(core_emb)
        elif method == "spectral":
            core_labels = _spectral_cluster(core_emb, num_speakers=None)
        else:
            raise ValueError(
                f"未知 cluster_method: {method}（可选 auto | average | ward | spectral）"
            )

        labels = _reassign_by_centroids(
            emb,
            core_idx,
            core_labels,
            durations,
            min_cluster_size=min_cluster_size,
            min_cluster_sec=min_cluster_sec,
        )
        n = len({int(x) for x in labels})
        if n > 12:
            logging.getLogger("meetmin").warning(
                "分离出 %d 个说话人簇，疑似阈值过低；可调大 diarization.cluster_threshold 或填 num_speakers",
                n,
            )
        return labels
