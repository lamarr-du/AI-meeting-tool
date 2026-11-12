"""说话人分离：SpeechBrain ECAPA-TDNN 声纹嵌入 + 凝聚层次聚类（全本地、零 token）。

流程：滑窗（只在 ASR 语音区间内取窗）→ 逐窗嵌入 → cosine 平均链接聚类 →
相邻同簇窗合并为说话人时间段。首次运行自动下载 ECAPA 模型（约 80MB）。
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import secrets
import zipfile

import numpy as np

from .base import Diarizer

SR = 16000


class EcapaSpeechbrainDiarizer(Diarizer):
    name = "ecapa_speechbrain"

    def __init__(self, dcfg: dict):
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
        self._strategy = LocalStrategy.COPY
        self.encoder = None

    def _load_encoder(self):
        if self.encoder is None:
            self.encoder = self._classifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(Path.home() / ".cache" / "meetmin" / "speechbrain"),
                local_strategy=self._strategy,
                run_opts={"device": "cpu"},
            )

    # ------------------------------------------------------------------
    def diarize(self, audio16k: np.ndarray, *, speech_spans=None, cache_path=None) -> list[dict]:
        if audio16k is None or len(audio16k) < SR:
            return []
        win_sec = float(self.dcfg.get("window_sec", 1.5))
        hop_sec = float(self.dcfg.get("hop_sec", 0.75))
        min_rms = float(self.dcfg.get("min_window_rms", 0.003))
        win = max(int(win_sec * SR) // 2 * 2, 400)
        hop = max(int(hop_sec * SR) // 2 * 2, 200)

        def in_speech(t: float) -> bool:
            if not speech_spans:
                return True
            for s, e in speech_spans:
                if s - 0.05 <= t <= e + 0.05:
                    return True
            return False

        starts, wavs = [], []
        for st in range(0, max(0, len(audio16k) - win) + 1, hop):
            mid = (st + win / 2) / SR
            if not in_speech(mid):
                continue
            w = audio16k[st : st + win]
            if float(np.sqrt(np.mean(np.square(w, dtype=np.float64)))) < min_rms:
                continue
            starts.append(st)
            wavs.append(w)
        metadata = {
            "schema_version": 1, "model": "speechbrain/spkrec-ecapa-voxceleb",
            "sample_rate": SR, "window_samples": win, "hop_samples": hop,
            "min_window_rms": min_rms, "speech_spans": speech_spans,
            "audio_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(audio16k))).hexdigest(),
        }
        signature = json.dumps(metadata, sort_keys=True)
        emb = None
        if cache_path is not None and Path(cache_path).is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    candidate = cached["embeddings"]
                    if (str(cached["metadata"].item()) == signature
                            and np.array_equal(cached["starts"], starts)
                            and candidate.ndim == 2 and candidate.shape[0] == len(starts)
                            and np.isfinite(candidate).all()):
                        emb = candidate.copy()
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                pass  # 损坏或不匹配的缓存重新生成
        if emb is None:
            emb = (np.concatenate([self._embed(np.stack(wavs[i:i + 32]).astype(np.float32))
                                   for i in range(0, len(wavs), 32)])
                   if starts else np.empty((0, 0), dtype=np.float32))
            if cache_path is not None:
                path = Path(cache_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name("." + path.name + "." + secrets.token_hex(8) + ".tmp")
                try:
                    with temporary.open("xb") as f:
                        np.savez_compressed(f, embeddings=emb, starts=np.asarray(starts, dtype=np.int64),
                                            ends=np.asarray(starts, dtype=np.int64) + win,
                                            metadata=np.asarray(signature))
                        f.flush()
                        os.fsync(f.fileno())
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
        if not starts:
            return []
        labels = self._cluster(emb)

        # 相邻同簇窗 → 时间段
        segs: list[dict] = []
        cur_label = int(labels[0])
        cur_start = starts[0]
        cur_end = starts[0] + win
        for i in range(1, len(starts)):
            if int(labels[i]) == cur_label and starts[i] <= cur_end + hop:
                cur_end = starts[i] + win
            else:
                segs.append({"start": cur_start / SR, "end": cur_end / SR, "label": cur_label})
                cur_label, cur_start, cur_end = int(labels[i]), starts[i], starts[i] + win
        segs.append({"start": cur_start / SR, "end": cur_end / SR, "label": cur_label})
        return segs

    # ------------------------------------------------------------------
    def _embed(self, windows: np.ndarray) -> np.ndarray:
        self._load_encoder()
        torch = self._torch
        with torch.no_grad():
            w = torch.from_numpy(windows).unsqueeze(1)  # [B, 1, T]
            emb = self.encoder.encode_batch(w)          # [B, 1, D]
        E = emb.squeeze(1).cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        return E / np.clip(norms, 1e-9, None)

    def _cluster(self, emb: np.ndarray) -> np.ndarray:
        from sklearn.cluster import AgglomerativeClustering

        k = self.dcfg.get("num_speakers")
        thr = float(self.dcfg.get("cluster_threshold", 0.55))
        if emb.shape[0] == 1:
            return np.zeros(1, dtype=int)
        if k:
            model = AgglomerativeClustering(n_clusters=int(k), metric="cosine", linkage="average")
        else:
            model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=thr, metric="cosine", linkage="average"
            )
        labels = model.fit_predict(emb)
        n = len(set(int(x) for x in labels))
        if n > 12:
            import logging

            logging.getLogger("meetmin").warning(
                "分离出 %d 个说话人簇，疑似阈值过低；可调大 diarization.cluster_threshold 或填 num_speakers", n
            )
        return labels
