"""说话人分离：SpeechBrain ECAPA-TDNN 声纹嵌入 + 凝聚层次聚类（全本地、零 token）。

流程：滑窗（只在 ASR 语音区间内取窗）→ 逐窗嵌入 → cosine 平均链接聚类 →
相邻同簇窗合并为说话人时间段。首次运行自动下载 ECAPA 模型（约 80MB）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Diarizer

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

SR = 16000


class EcapaSpeechbrainDiarizer(Diarizer):
    name = "ecapa_speechbrain"

    def __init__(self, dcfg: dict):
        try:
            import torch  # noqa: F401  (speechbrain 依赖)
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as e:
            raise ImportError(
                "需要 speechbrain + scikit-learn（会引入 torch）：pip install -r requirements-optional.txt"
            ) from e
        self.dcfg = dcfg
        self._torch = torch
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(_PROJECT_ROOT / ".cache" / "speechbrain"),
            run_opts={"device": "cpu"},
        )

    # ------------------------------------------------------------------
    def diarize(self, audio16k: np.ndarray, *, speech_spans=None) -> list[dict]:
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
        if not starts:
            return []

        emb = self._embed(np.stack(wavs).astype(np.float32))
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
