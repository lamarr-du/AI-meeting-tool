"""纯 ONNX Runtime 中文/中英标点恢复后端（零 PyTorch、零 sherpa-onnx 依赖）。

支持两款 Apache-2.0 开源 ONNX 标点模型（直接复用 faster-whisper 已安装的 onnxruntime / tokenizers）：
  - firered_onnx（默认）：FireRedPunc Q8W（155 MB，中英混读技术会议精度最优）
  - ct_transformer_onnx（轻量）：CT-Transformer INT8（72 MB 单文件，极速低内存）
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..model_cache import configure_xet_cache, hub_cache_dir
from .faster_whisper_asr import _configure_huggingface_download

FIRERED_REPO = "jiangzhuo9357/fireredpunc-onnx"
FIRERED_MODEL_FILE = "punc.q8w.onnx"
FIRERED_TOKENIZER_FILE = "tokenizer.json"
FIRERED_LABELS = ("", "，", "。", "？", "！")

CT_TRANSFORMER_REPO = "ranger810/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8"
CT_TRANSFORMER_MODEL_FILE = "model.int8.onnx"


class Punctuator(Protocol):
    name: str

    def predict_tags(self, tokens: list[str]) -> list[str]:
        """输入无标点词元列表，返回等长标点标签列表（'' / '，' / '。' / '？' / '！' / '、'）。"""
        ...


def _ort_session_options(threads: int | None = None) -> Any:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    n_threads = threads if threads is not None else min(2, max(1, os.cpu_count() or 2))
    opts.intra_op_num_threads = max(1, int(n_threads))
    opts.inter_op_num_threads = 1
    return opts


def _download_hf_file(
    repo_id: str,
    filename: str,
    *,
    cache_dir: Path,
    on_model_download: Callable[[bool], None] | None = None,
    local_files_only: bool = False,
) -> Path:
    import huggingface_hub

    cached = huggingface_hub.try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir)
    if isinstance(cached, str) and Path(cached).is_file():
        return Path(cached)
    if local_files_only:
        raise FileNotFoundError(f"本地标点模型未缓存：{repo_id}/{filename}")
    if on_model_download is not None:
        on_model_download(True)
    try:
        downloaded = huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )
        return Path(downloaded)
    finally:
        if on_model_download is not None:
            on_model_download(False)


class FireRedPuncOnnx:
    """FireRedPunc 8-bit 权重量化 ONNX 标点恢复器（中英双语高精度）。"""

    name = "firered_onnx"

    def __init__(
        self,
        pcfg: dict | None = None,
        *,
        acfg: dict | None = None,
        model_cache_dir: str | Path | None = None,
        on_model_download: Callable[[bool], None] | None = None,
    ):
        self.pcfg = pcfg or {}
        self.acfg = acfg or {}
        self.model_cache_dir = model_cache_dir
        self._on_model_download = on_model_download
        self._sess: Any = None
        self._tokenizer: Any = None
        self._out_name = ""
        self._cls_id = 101
        self._sep_id = 102
        self._unk_id = 100

    def _resolve_files(self) -> tuple[Path, Path]:
        explicit = self.pcfg.get("model_path")
        if explicit:
            p = Path(str(explicit)).expanduser()
            if p.is_dir():
                model_file = p / FIRERED_MODEL_FILE
                tok_file = p / FIRERED_TOKENIZER_FILE
            else:
                model_file = p
                tok_file = p.parent / FIRERED_TOKENIZER_FILE
            if not model_file.is_file():
                raise FileNotFoundError(f"指定的标点模型文件不存在：{model_file}")
            if not tok_file.is_file():
                raise FileNotFoundError(f"未在模型同目录找到分词器文件：{tok_file}")
            return model_file, tok_file

        _configure_huggingface_download(self.acfg)
        configure_xet_cache(self.model_cache_dir)
        cache_dir = hub_cache_dir(self.model_cache_dir)
        model_file = _download_hf_file(
            FIRERED_REPO,
            FIRERED_MODEL_FILE,
            cache_dir=cache_dir,
            on_model_download=self._on_model_download,
            local_files_only=bool(self.pcfg.get("local_files_only")),
        )
        tok_file = _download_hf_file(
            FIRERED_REPO,
            FIRERED_TOKENIZER_FILE,
            cache_dir=cache_dir,
            on_model_download=self._on_model_download,
            local_files_only=bool(self.pcfg.get("local_files_only")),
        )
        return model_file, tok_file

    def _ensure_loaded(self) -> None:
        if self._sess is not None:
            return
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("FireRedPunc ONNX 需要 onnxruntime 与 tokenizers 依赖") from exc

        model_file, tok_file = self._resolve_files()
        threads = self.pcfg.get("threads")
        opts = _ort_session_options(int(threads) if threads is not None else None)
        self._sess = ort.InferenceSession(str(model_file), opts, providers=["CPUExecutionProvider"])
        self._out_name = self._sess.get_outputs()[0].name
        self._tokenizer = Tokenizer.from_file(str(tok_file))
        self._cls_id = int(self._tokenizer.token_to_id("[CLS]") or 101)
        self._sep_id = int(self._tokenizer.token_to_id("[SEP]") or 102)
        self._unk_id = int(self._tokenizer.token_to_id("[UNK]") or 100)

    def predict_tags(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return []
        self._ensure_loaded()
        word_sub_ids: list[list[int]] = []
        for token in tokens:
            text = token.lower() if token.isascii() else token
            enc = self._tokenizer.encode(text, add_special_tokens=False).ids
            word_sub_ids.append(list(enc) if enc else [self._unk_id])

        tags = [""] * len(tokens)
        pos = 0
        max_subtokens = 460
        context_words = 24
        while pos < len(tokens):
            ctx_start = max(0, pos - context_words)
            sub_ids = [self._cls_id]
            for j in range(ctx_start, pos):
                if len(sub_ids) + len(word_sub_ids[j]) + 64 >= max_subtokens:
                    break
                sub_ids.extend(word_sub_ids[j])
            word_last_idx: list[int] = []
            end = pos
            while end < len(tokens):
                piece = word_sub_ids[end]
                if word_last_idx and len(sub_ids) + len(piece) + 1 > max_subtokens:
                    break
                if not word_last_idx and len(sub_ids) + len(piece) + 1 > max_subtokens:
                    piece = piece[: max(1, max_subtokens - len(sub_ids) - 1)]
                sub_ids.extend(piece)
                word_last_idx.append(len(sub_ids) - 1)
                end += 1
            sub_ids.append(self._sep_id)
            arr = np.array([sub_ids], dtype=np.int64)
            mask = np.ones_like(arr, dtype=np.int64)
            logits = self._sess.run(
                [self._out_name],
                {"input_ids": arr, "attention_mask": mask},
            )[0][0]
            preds = logits.argmax(axis=-1).tolist()
            # FireRedPunc ONNX 输出已跳过索引 0 的 [CLS]，输入位 k 对应输出位 k - 1。
            for offset, last_sub in enumerate(word_last_idx):
                out_idx = last_sub - 1
                if 0 <= out_idx < len(preds):
                    cls_id = int(preds[out_idx])
                    if 0 <= cls_id < len(FIRERED_LABELS):
                        tags[pos + offset] = FIRERED_LABELS[cls_id]
            pos = end
        return tags


class CtTransformerOnnx:
    """CT-Transformer INT8 单文件 ONNX 标点恢复器（72 MB，极速低内存）。"""

    name = "ct_transformer_onnx"

    def __init__(
        self,
        pcfg: dict | None = None,
        *,
        acfg: dict | None = None,
        model_cache_dir: str | Path | None = None,
        on_model_download: Callable[[bool], None] | None = None,
    ):
        self.pcfg = pcfg or {}
        self.acfg = acfg or {}
        self.model_cache_dir = model_cache_dir
        self._on_model_download = on_model_download
        self._sess: Any = None
        self._in0 = ""
        self._in1 = ""
        self._out0 = ""
        self._id2punct: list[str] = []
        self._token2id: dict[str, int] = {}
        self._unk_id = 0
        self._us_id = 1
        self._dot_ids: set[int] = set()
        self._comma_id = -1

    def _resolve_model(self) -> Path:
        explicit = self.pcfg.get("model_path")
        if explicit:
            p = Path(str(explicit)).expanduser()
            model_file = p / CT_TRANSFORMER_MODEL_FILE if p.is_dir() else p
            if not model_file.is_file():
                raise FileNotFoundError(f"指定的标点模型文件不存在：{model_file}")
            return model_file
        _configure_huggingface_download(self.acfg)
        configure_xet_cache(self.model_cache_dir)
        cache_dir = hub_cache_dir(self.model_cache_dir)
        return _download_hf_file(
            CT_TRANSFORMER_REPO,
            CT_TRANSFORMER_MODEL_FILE,
            cache_dir=cache_dir,
            on_model_download=self._on_model_download,
            local_files_only=bool(self.pcfg.get("local_files_only")),
        )

    def _ensure_loaded(self) -> None:
        if self._sess is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("CT-Transformer ONNX 需要 onnxruntime 依赖") from exc

        model_file = self._resolve_model()
        threads = self.pcfg.get("threads")
        opts = _ort_session_options(int(threads) if threads is not None else None)
        self._sess = ort.InferenceSession(str(model_file), opts, providers=["CPUExecutionProvider"])
        self._in0 = self._sess.get_inputs()[0].name
        self._in1 = self._sess.get_inputs()[1].name
        self._out0 = self._sess.get_outputs()[0].name
        meta = self._sess.get_modelmeta().custom_metadata_map
        self._id2punct = meta["punctuations"].split("|")
        punct2id = {p: i for i, p in enumerate(self._id2punct)}
        tokens = meta["tokens"].split("|")
        self._token2id = {t: i for i, t in enumerate(tokens)}
        self._unk_id = self._token2id[meta["unk_symbol"]]
        self._us_id = punct2id.get("_", 1)
        self._dot_ids = {punct2id[p] for p in ("。", "？") if p in punct2id}
        self._comma_id = punct2id.get("，", -1)

    def _run_window(self, ids: list[int]) -> list[int]:
        logits = self._sess.run(
            [self._out0],
            {
                self._in0: np.array(ids, dtype=np.int32).reshape(1, -1),
                self._in1: np.array([len(ids)], dtype=np.int32),
            },
        )[0][0]
        return [int(x) for x in logits.argmax(axis=-1).tolist()]

    def predict_tags(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return []
        self._ensure_loaded()
        ids = [
            self._token2id.get(
                t.lower() if t.isascii() else t,
                self._token2id.get(t, self._unk_id),
            )
            for t in tokens
        ]
        n = len(ids)
        win = 160
        if n <= win + 20:
            pids = self._run_window(ids)
        else:
            pids = []
            start = 0
            while start < n:
                end = min(n, start + win)
                sub_p = self._run_window(ids[start:end])
                if end == n:
                    pids.extend(sub_p)
                    break
                cut = -1
                lo = max(20, len(sub_p) // 2)
                for k in range(len(sub_p) - 1, lo, -1):
                    if sub_p[k] in self._dot_ids:
                        cut = k
                        break
                if cut == -1 and self._comma_id != -1:
                    for k in range(len(sub_p) - 1, lo, -1):
                        if sub_p[k] == self._comma_id:
                            cut = k
                            break
                if cut == -1:
                    cut = max(0, len(sub_p) - 16)
                pids.extend(sub_p[: cut + 1])
                start += cut + 1
        return [
            "" if (p == self._us_id or p < 0 or p >= len(self._id2punct)) else self._id2punct[p]
            for p in pids[:n]
        ]


def get_punctuator(
    cfg: dict,
    *,
    on_model_download: Callable[[bool], None] | None = None,
) -> Punctuator | None:
    """按配置创建标点恢复器；provider 为 rule/none/off 时返回 None。"""
    pcfg = cfg.get("punctuation") or {}
    provider = str(pcfg.get("provider", "firered_onnx")).strip().lower()
    if provider in ("rule", "none", "off", "disabled", ""):
        return None
    acfg = cfg.get("asr") or {}
    cache_dir = cfg.get("model_cache_dir")
    if provider in ("firered_onnx", "firered", "fireredpunc"):
        return FireRedPuncOnnx(
            pcfg,
            acfg=acfg,
            model_cache_dir=cache_dir,
            on_model_download=on_model_download,
        )
    if provider in ("ct_transformer_onnx", "ct_transformer", "ct-transformer", "ct_punc", "ct"):
        return CtTransformerOnnx(
            pcfg,
            acfg=acfg,
            model_cache_dir=cache_dir,
            on_model_download=on_model_download,
        )
    raise ValueError(
        f"未知 punctuation provider: {provider}（可选 firered_onnx | ct_transformer_onnx | rule）"
    )
