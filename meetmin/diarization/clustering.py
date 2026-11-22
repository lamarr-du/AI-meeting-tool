"""声纹切窗、缓存与纯 NumPy 稳健聚类（零 sklearn/scipy/torch 依赖）。

供 campplus_onnx 与 ecapa_speechbrain 共用：
- 基于 ASR speech_spans 的语音段切分（无 speech_spans 时退化为滑窗）
- 基于语音区能量的自适应 20ms 帧级语音活跃度筛选
- 纯 NumPy 凝聚层次聚类（average / ward）与归一化拉普拉斯谱聚类
- kNN 孤立离群预筛 + 小簇过滤 + L2 归一化质心二次重分配
- 相邻同簇合并与异簇中点去重叠边界
"""

from __future__ import annotations

import json
import logging
import math
import os
import secrets
import zipfile
from collections import deque
from pathlib import Path

import numpy as np

SR = 16000
_FRAME_SAMPLES = 320  # 20ms @ 16kHz
_MIN_EMBED_SAMPLES = int(0.4 * SR)
_MIN_VALID_SAMPLES = 400


def rms(wav: np.ndarray) -> float:
    if len(wav) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))


def frame_active_ratio(wav: np.ndarray, frame_rms_threshold: float) -> float:
    n_frames = len(wav) // _FRAME_SAMPLES
    if n_frames <= 0:
        return 1.0 if rms(wav) >= frame_rms_threshold else 0.0
    frames = wav[: n_frames * _FRAME_SAMPLES].reshape(n_frames, _FRAME_SAMPLES)
    frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    return float(np.mean(frame_rms >= frame_rms_threshold))


class StreamingRmsTracker:
    """Retain up to 60 seconds of voiced 20 ms frame RMS values."""

    def __init__(self, minimum: float):
        self.minimum = minimum
        self.frames: deque[float] = deque(maxlen=60 * SR // _FRAME_SAMPLES)

    def update(self, audio: np.ndarray) -> float:
        usable = len(audio) // _FRAME_SAMPLES * _FRAME_SAMPLES
        frames = audio[:usable].reshape(-1, _FRAME_SAMPLES)
        levels = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
        self.frames.extend(levels[levels >= self.minimum])
        if len(self.frames) < 50:
            return max(self.minimum, 0.015)
        p75 = float(np.percentile(self.frames, 75))
        return float(np.clip(p75 * 0.25, self.minimum, max(self.minimum, 0.02)))


def linear_sum_assignment_max(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-weight rectangular assignment, using shortest augmenting paths."""
    costs = -np.asarray(weights, dtype=np.float64)
    transposed = costs.shape[0] > costs.shape[1]
    if transposed:
        costs = costs.T
    rows, cols = costs.shape
    u, v = np.zeros(rows + 1), np.zeros(cols + 1)
    matched, previous = np.zeros(cols + 1, dtype=int), np.zeros(cols + 1, dtype=int)
    for row in range(1, rows + 1):
        matched[0] = row
        col = 0
        slack = np.full(cols + 1, np.inf)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[col] = True
            active = matched[col]
            remaining = np.flatnonzero(~used)
            reduced = costs[active - 1, remaining - 1] - u[active] - v[remaining]
            improved = remaining[reduced < slack[remaining]]
            slack[improved] = reduced[reduced < slack[remaining]]
            previous[improved] = col
            next_col = int(remaining[np.argmin(slack[remaining])])
            delta = slack[next_col]
            u[matched[used]] += delta
            v[used] -= delta
            slack[~used] -= delta
            col = next_col
            if matched[col] == 0:
                break
        while col:
            prior = previous[col]
            matched[col] = matched[prior]
            col = prior
    col_indices = np.flatnonzero(matched[1:])
    row_indices = matched[col_indices + 1] - 1
    if transposed:
        row_indices, col_indices = col_indices, row_indices
    order = np.argsort(row_indices)
    return row_indices[order], col_indices[order]


def estimate_frame_rms_threshold(
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


def extract_wave_slice(audio16k: np.ndarray, start: int, end: int) -> np.ndarray:
    """提取用于声纹编码的波形；极短片段向两侧补齐上下文到 0.4s，不改变原时间标注。"""
    if end - start >= _MIN_EMBED_SAMPLES or len(audio16k) <= _MIN_EMBED_SAMPLES:
        return audio16k[start:end]
    center = (start + end) // 2
    half = _MIN_EMBED_SAMPLES // 2
    ext_start = max(0, min(center - half, len(audio16k) - _MIN_EMBED_SAMPLES))
    ext_end = min(len(audio16k), ext_start + _MIN_EMBED_SAMPLES)
    return audio16k[ext_start:ext_end]


def build_windows(
    audio16k: np.ndarray,
    speech_spans: list[tuple[float, float]] | None,
    dcfg: dict,
) -> tuple[list[int], list[int], list[bool], str, float, float, float]:
    """构建待提取声纹的时间段，并标记哪些高质量段参与主聚类（core）。"""
    configured_mode = str(dcfg.get("window_mode", "speech_spans")).lower()
    mode = "speech_spans" if configured_mode == "speech_spans" and speech_spans else "sliding"
    win_sec = float(dcfg.get("window_sec", 2.5))
    hop_sec = float(dcfg.get("hop_sec", 1.25))
    merge_gap_sec = float(dcfg.get("merge_gap_sec", 0.3))
    min_rms = float(dcfg.get("min_window_rms", 0.003))
    min_active_ratio = float(dcfg.get("min_active_ratio", 0.45))
    min_cluster_dur = float(dcfg.get("min_cluster_dur_sec", 1.2))
    frame_thr = estimate_frame_rms_threshold(audio16k, speech_spans, min_rms)

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
                wav = extract_wave_slice(audio16k, a, b)
                if len(wav) < _MIN_VALID_SAMPLES or rms(wav) < min_rms:
                    continue
                active_ratio = frame_active_ratio(wav, frame_thr)
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
            if rms(wav) < min_rms:
                continue
            active_ratio = frame_active_ratio(wav, frame_thr)
            starts.append(st)
            ends.append(st + win)
            core_flags.append(active_ratio >= min_active_ratio)

    return starts, ends, core_flags, mode, win_sec, hop_sec, merge_gap_sec


def load_embedding_cache(
    cache_path: str | Path | None,
    signature: str,
    starts: list[int],
    ends: list[int],
) -> np.ndarray | None:
    if cache_path is None or not Path(cache_path).is_file():
        return None
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
                return candidate.copy()
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        pass
    return None


def save_embedding_cache(
    cache_path: str | Path | None,
    emb: np.ndarray,
    starts: list[int],
    ends: list[int],
    signature: str,
) -> None:
    if cache_path is None:
        return
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


def reuse_live_embeddings(
    cache_path: str | Path | None, signature: str, starts: list[int], ends: list[int]
) -> dict[int, np.ndarray]:
    """Reuse only exact speech windows from a live cache for this model and PCM."""
    if cache_path is None or not Path(cache_path).is_file():
        return {}
    expected = json.loads(signature)
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata"].item()))
            if not isinstance(metadata, dict) or metadata.get("window_mode") != "live_commit":
                return {}
            keys = ("schema_version", "model", "sample_rate", "audio_sha256")
            if any(metadata.get(key) != expected[key] for key in keys):
                return {}
            vectors, old_starts, old_ends = cached["embeddings"], cached["starts"], cached["ends"]
            if (
                vectors.ndim != 2
                or vectors.shape[1] != 192
                or len(vectors) != len(old_starts)
                or len(vectors) != len(old_ends)
                or old_starts.ndim != 1
                or old_ends.ndim != 1
                or not np.isfinite(vectors).all()
            ):
                return {}
            # Short live windows may be padded using different surrounding PCM.
            known = {
                (int(a), int(b)): vector
                for a, b, vector in zip(old_starts, old_ends, vectors, strict=True)
                if b - a >= _MIN_EMBED_SAMPLES
            }
            return {
                index: known[a, b]
                for index, (a, b) in enumerate(zip(starts, ends, strict=True))
                if (a, b) in known
            }
    except (OSError, ValueError, KeyError, EOFError, TypeError, zipfile.BadZipFile):
        return {}


def build_cache_signature(
    *,
    model: str,
    mode: str,
    win_sec: float,
    hop_sec: float,
    merge_gap_sec: float,
    min_rms: float,
    speech_spans: list[tuple[float, float]] | None,
    audio_sha256: str,
) -> str:
    metadata = {
        "schema_version": 2,
        "model": model,
        "sample_rate": SR,
        "window_mode": mode,
        "window_sec": win_sec,
        "hop_sec": hop_sec,
        "merge_gap_sec": merge_gap_sec,
        "min_window_rms": min_rms,
        "speech_spans": speech_spans,
        "audio_sha256": audio_sha256,
    }
    return json.dumps(metadata, sort_keys=True)


def _agglomerative_average_cosine(
    emb: np.ndarray,
    *,
    threshold: float | None = None,
    n_clusters: int | None = None,
) -> np.ndarray:
    """纯 NumPy 凝聚层次聚类（余弦距离 + average linkage），O(N^2) 加速。"""
    n = emb.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int)
    if n_clusters is not None:
        if n_clusters <= 1:
            return np.zeros(n, dtype=int)
        if n_clusters >= n:
            return np.arange(n, dtype=int)

    dist = np.clip(1.0 - (emb @ emb.T).astype(np.float64), 0.0, 2.0)
    np.fill_diagonal(dist, np.inf)
    alive = np.ones(n, dtype=bool)
    sizes = np.ones(n, dtype=np.float64)
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    row_min_idx = np.argmin(dist, axis=1)
    row_min_val = dist[np.arange(n), row_min_idx]
    n_alive = n

    while n_alive > 1:
        if n_clusters is not None and n_alive <= n_clusters:
            break
        i = int(np.argmin(row_min_val))
        min_d = float(row_min_val[i])
        if threshold is not None and min_d >= threshold:
            break
        j = int(row_min_idx[i])
        if i > j:
            i, j = j, i
        si, sj = sizes[i], sizes[j]
        new_d = (si * dist[i] + sj * dist[j]) / (si + sj)
        dist[i, :] = new_d
        dist[:, i] = new_d
        dist[i, i] = np.inf
        dist[j, :] = np.inf
        dist[:, j] = np.inf
        alive[j] = False
        row_min_val[j] = np.inf
        sizes[i] = si + sj
        members[i].extend(members.pop(j))
        n_alive -= 1

        # 仅重算行 i 及原本最近邻指向 i/j 的活跃行
        dirty = alive & ((row_min_idx == i) | (row_min_idx == j))
        dirty[i] = True
        for r in np.where(dirty)[0]:
            idx = int(np.argmin(dist[r]))
            row_min_idx[r] = idx
            row_min_val[r] = dist[r, idx]
        # 其余活跃行若到新簇 i 的距离更近则更新
        other = alive & (~dirty)
        other_idx = np.where(other)[0]
        if len(other_idx) > 0:
            better = dist[other_idx, i] < row_min_val[other_idx]
            upd = other_idx[better]
            row_min_idx[upd] = i
            row_min_val[upd] = dist[upd, i]

    labels = np.zeros(n, dtype=int)
    for new_lab, root in enumerate(sorted(members)):
        for idx in members[root]:
            labels[idx] = new_lab
    return labels


def _agglomerative_ward(
    emb: np.ndarray,
    *,
    threshold: float | None = None,
    n_clusters: int | None = None,
) -> np.ndarray:
    """纯 NumPy Ward 层次聚类（L2 归一化向量，与 sklearn Ward 等价），O(N^2) 加速。"""
    n = emb.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int)
    if n_clusters is not None:
        if n_clusters <= 1:
            return np.zeros(n, dtype=int)
        if n_clusters >= n:
            return np.arange(n, dtype=int)

    X = emb.astype(np.float64)
    sq_dist = np.clip(2.0 - 2.0 * (X @ X.T), 0.0, 4.0)
    # sklearn Ward 合并距离为 sqrt(2 * delta_SSE) = sqrt((2 * n_i * n_j / (n_i + n_j)) * ||mu_i - mu_j||^2)
    # 单点合并时 2 * n_i * n_j / (n_i + n_j) = 1，因此平方合并距离等于欧氏距离平方
    cost = sq_dist.copy()
    np.fill_diagonal(cost, np.inf)
    thr_sq = None if threshold is None else float(threshold) ** 2

    alive = np.ones(n, dtype=bool)
    sizes = np.ones(n, dtype=np.float64)
    centroids = X.copy()
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    row_min_idx = np.argmin(cost, axis=1)
    row_min_val = cost[np.arange(n), row_min_idx]
    n_alive = n

    while n_alive > 1:
        if n_clusters is not None and n_alive <= n_clusters:
            break
        i = int(np.argmin(row_min_val))
        min_c = float(row_min_val[i])
        if thr_sq is not None and min_c >= thr_sq:
            break
        j = int(row_min_idx[i])
        if i > j:
            i, j = j, i
        si, sj = sizes[i], sizes[j]
        new_s = si + sj
        centroids[i] = (si * centroids[i] + sj * centroids[j]) / new_s
        sizes[i] = new_s
        alive[j] = False
        cost[j, :] = np.inf
        cost[:, j] = np.inf
        row_min_val[j] = np.inf
        members[i].extend(members.pop(j))
        n_alive -= 1

        alive_idx = np.where(alive)[0]
        diffs = centroids[alive_idx] - centroids[i]
        d2 = np.sum(diffs * diffs, axis=1)
        s_other = sizes[alive_idx]
        new_c = (2.0 * new_s * s_other / (new_s + s_other)) * d2
        cost[i, alive_idx] = new_c
        cost[alive_idx, i] = new_c
        cost[i, i] = np.inf

        dirty = alive & ((row_min_idx == i) | (row_min_idx == j))
        dirty[i] = True
        for r in np.where(dirty)[0]:
            idx = int(np.argmin(cost[r]))
            row_min_idx[r] = idx
            row_min_val[r] = cost[r, idx]
        other = alive & (~dirty)
        other_idx = np.where(other)[0]
        if len(other_idx) > 0:
            better = cost[other_idx, i] < row_min_val[other_idx]
            upd = other_idx[better]
            row_min_idx[upd] = i
            row_min_val[upd] = cost[upd, i]

    labels = np.zeros(n, dtype=int)
    for new_lab, root in enumerate(sorted(members)):
        for idx in members[root]:
            labels[idx] = new_lab
    return labels


def _kmeans_plusplus(
    X: np.ndarray, k: int, *, n_init: int = 10, max_iter: int = 100, seed: int = 0
) -> np.ndarray:
    """确定性纯 NumPy KMeans++（用于谱聚类特征空间，零 sklearn 依赖）。"""
    n = X.shape[0]
    if k <= 1 or n <= 1:
        return np.zeros(n, dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)

    rng = np.random.default_rng(seed)
    best_inertia = float("inf")
    best_labels = np.zeros(n, dtype=int)

    for _ in range(n_init):
        # KMeans++ 初始化
        centers = np.empty((k, X.shape[1]), dtype=np.float64)
        first = int(rng.integers(0, n))
        centers[0] = X[first]
        min_d2 = np.sum((X - centers[0]) ** 2, axis=1)
        for c_i in range(1, k):
            total = float(np.sum(min_d2))
            if total <= 1e-12:
                choice = int(rng.integers(0, n))
            else:
                probs = min_d2 / total
                choice = int(rng.choice(n, p=probs))
            centers[c_i] = X[choice]
            d2 = np.sum((X - centers[c_i]) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2)

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c_i in range(k):
                mask = labels == c_i
                if np.any(mask):
                    centers[c_i] = np.mean(X[mask], axis=0)
        inertia = float(np.sum((X - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels


def spectral_cluster(
    emb: np.ndarray,
    *,
    num_speakers: int | None,
    p_percentile: float = 0.75,
    max_speakers: int = 10,
) -> np.ndarray:
    """余弦亲和矩阵 p-pruning + 归一化拉普拉斯特征间隙估计人数 + 纯 NumPy KMeans++。"""
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
    return _kmeans_plusplus(spectral_emb, k, n_init=20, seed=0)


def filter_knn_outliers(
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


def reassign_by_centroids(
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


def cluster_embeddings(
    emb: np.ndarray,
    dcfg: dict,
    *,
    default_threshold: float,
    core_mask: np.ndarray | None = None,
    durations: np.ndarray | None = None,
) -> np.ndarray:
    """执行主聚类、离群防护、小簇过滤与 L2 质心二次重分配。"""
    n_total = emb.shape[0]
    if n_total <= 1:
        return np.zeros(n_total, dtype=int)

    if core_mask is None or len(core_mask) != n_total or not bool(np.any(core_mask)):
        core_idx = np.arange(n_total, dtype=int)
    else:
        core_idx = np.where(core_mask)[0]

    if durations is None or len(durations) != n_total:
        default_dur = float(dcfg.get("window_sec", 2.5))
        durations = np.full(n_total, default_dur, dtype=np.float64)

    k = dcfg.get("num_speakers")
    raw_thr = dcfg.get("cluster_threshold")
    thr = float(default_threshold if raw_thr is None else raw_thr)
    method = str(dcfg.get("cluster_method", "auto")).lower()
    min_cluster_size = max(1, int(dcfg.get("min_cluster_size", 2)))
    min_cluster_sec = max(0.0, float(dcfg.get("min_cluster_sec", 4.0)))

    if k is not None:
        target_k = int(k)
        if target_k <= 1 or len(core_idx) <= 1:
            return np.zeros(n_total, dtype=int)
        target_k = min(target_k, len(core_idx))
        inlier_rel = filter_knn_outliers(emb[core_idx], min_keep=target_k)
        fit_idx = core_idx[inlier_rel]
        fit_emb = emb[fit_idx]

        if method in ("auto", "ward"):
            fit_labels = _agglomerative_ward(fit_emb, n_clusters=target_k)
        elif method == "average":
            fit_labels = _agglomerative_average_cosine(fit_emb, n_clusters=target_k)
        elif method == "spectral":
            fit_labels = spectral_cluster(fit_emb, num_speakers=target_k)
        else:
            raise ValueError(
                f"未知 cluster_method: {method}（可选 auto | average | ward | spectral）"
            )
        return reassign_by_centroids(
            emb,
            fit_idx,
            fit_labels,
            durations,
            min_cluster_size=1,
            min_cluster_sec=0.0,
            keep_top_k=target_k,
        )

    if len(core_idx) <= 1:
        return np.zeros(n_total, dtype=int)
    core_emb = emb[core_idx]
    if method in ("auto", "average"):
        core_labels = _agglomerative_average_cosine(core_emb, threshold=thr)
    elif method == "ward":
        euclid_thr = math.sqrt(max(0.0, 2.0 * thr))
        core_labels = _agglomerative_ward(core_emb, threshold=euclid_thr)
    elif method == "spectral":
        core_labels = spectral_cluster(core_emb, num_speakers=None)
    else:
        raise ValueError(f"未知 cluster_method: {method}（可选 auto | average | ward | spectral）")

    labels = reassign_by_centroids(
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


def merge_non_overlapping(
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
