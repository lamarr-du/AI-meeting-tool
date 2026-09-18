"""离线单元测试（无网络、无大模型）。

运行： python tests/test_offline.py        # 或 pytest tests/test_offline.py
"""
from __future__ import annotations

import math
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


class _TmpDir:
    """仓库内 .tmp 下的临时目录（不依赖系统 temp，兼容受限沙箱）。"""

    def __init__(self):
        base = Path(__file__).resolve().parents[1] / ".tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"test-{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True)

    def __enter__(self) -> str:
        return str(self.path)

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)
        return False

from meetmin.asr.base import ASRResult, Utterance
from meetmin.asr.faster_whisper_asr import _configure_huggingface_download
from meetmin.audio import dsp
from meetmin.config import DEFAULT_CONFIG, deep_merge, load_config
from meetmin.diarization.align import assign_speakers, relabel
from meetmin.session import Session
from meetmin.summarizer.llm_client import MockLLM
from meetmin.summarizer.minutes import generate_minutes, render_transcript, split_chunks


def _utts(n, spk_cycle=("说话人A", "说话人B")):
    out = []
    for i in range(n):
        out.append(
            Utterance(
                start=i * 5.0,
                end=i * 5.0 + 4.0,
                text=f"这是第{i}句测试内容，讨论排期与方案。" * 2,
                speaker=spk_cycle[i % len(spk_cycle)],
            )
        )
    return out


# ---------------- dsp ----------------
def test_resample_length_and_energy():
    t = np.arange(48000, dtype=np.float32) / 48000.0
    x = np.sin(2 * math.pi * 440 * t)
    y = dsp.resample(x, 48000, 16000)
    assert abs(len(y) - 16000) <= 2, len(y)
    assert 0.3 < dsp.rms(y) < 0.9


def test_wav_roundtrip():
    with _TmpDir() as td:
        p = Path(td) / "a.wav"
        x = (np.sin(np.linspace(0, 200, 8000)) * 0.5).astype(np.float32)
        dsp.save_wav_mono(p, x, 16000)
        y, sr = dsp.load_wav_mono(p)
        assert sr == 16000 and len(y) == len(x)
        assert np.abs(x - y).max() < 1e-3


# ---------------- config / session ----------------
def test_config_merge():
    cfg = load_config(str(Path(__file__).parents[1] / "config.yaml"))
    assert cfg["asr"]["provider"] == "faster_whisper"
    assert cfg["llm"]["model"] == "qwen-3.8-flash"
    merged = deep_merge(DEFAULT_CONFIG, {"summarizer": {"direct_threshold_chars": 10}})
    assert merged["summarizer"]["direct_threshold_chars"] == 10
    assert merged["summarizer"]["max_chunk_chars"] == DEFAULT_CONFIG["summarizer"]["max_chunk_chars"]


def test_huggingface_download_settings_set_before_model_import():
    import os

    previous_endpoint = os.environ.pop("HF_ENDPOINT", None)
    previous_xet = os.environ.pop("HF_HUB_DISABLE_XET", None)
    try:
        _configure_huggingface_download(
            {"hf_endpoint": "https://hf-mirror.com", "hf_disable_xet": True}
        )
        assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
        assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    finally:
        if previous_endpoint is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = previous_endpoint
        if previous_xet is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = previous_xet


def test_session_io():
    with _TmpDir() as td:
        s = Session(td, "sid-1").ensure()
        s.save_json("transcript.json", {"utterances": [{"start": 0, "end": 1, "text": "甲"}]})
        loaded = s.load_transcript()
        assert loaded["utterances"][0]["text"] == "甲"


# ---------------- diarization align ----------------
def test_assign_splits_at_speaker_change():
    segments = [
        {"start": 0.0, "end": 5.0, "label": 0},
        {"start": 5.0, "end": 10.0, "label": 1},
    ]
    words = [
        {"w": "我", "s": 0.5, "e": 1.0}, {"w": "来说", "s": 1.0, "e": 2.5},
        {"w": "两句", "s": 2.5, "e": 4.0},
        {"w": "同", "s": 5.2, "e": 5.6}, {"w": "意", "s": 5.6, "e": 6.2},
    ]
    u = Utterance(0.5, 6.2, "我来说两句同意", words=words)
    out = assign_speakers([u], segments, min_group_sec=0.4)
    assert len(out) == 2, [ (o.speaker, o.text) for o in out]
    assert out[0].speaker == 0 and out[1].speaker == 1
    assert "同意" in out[1].text


def test_assign_smoothing_absorbs_blip():
    segments = [
        {"start": 0.0, "end": 8.0, "label": 0},
        {"start": 8.0, "end": 9.0, "label": 1},  # 极短的串扰窗
        {"start": 9.0, "end": 16.0, "label": 0},
    ]
    ws = [{"w": "开", "s": 1.0, "e": 1.6}, {"w": "场", "s": 1.6, "e": 2.2},
          {"w": "串", "s": 8.1, "e": 8.35},  # 落入 label1，但组时长 0.25s < 0.4 → 应被吸收
          {"w": "结", "s": 10.0, "e": 10.6}, {"w": "束", "s": 10.6, "e": 11.2}]
    u = Utterance(1.0, 11.2, "开场串结束", words=ws)
    out = assign_speakers([u], segments, min_group_sec=0.4)
    assert len(out) == 1 and out[0].speaker == 0, [(o.speaker, o.text) for o in out]
    assert "开场" in out[0].text and "结束" in out[0].text


def test_relabel_order_of_appearance():
    us = [Utterance(0, 1, "a", speaker=3), Utterance(1, 2, "b", speaker=0),
          Utterance(2, 3, "c", speaker=3)]
    speakers = relabel(us)
    assert us[0].speaker == "说话人A" and us[1].speaker == "说话人B"
    assert us[2].speaker == "说话人A" and speakers == ["说话人A", "说话人B"]


# ---------------- minutes（省 token 编排）----------------
def test_render_merge_saves_lines():
    us = [Utterance(0, 1, "第一句", speaker="说话人A"), Utterance(1, 2, "第二句", speaker="说话人A"),
          Utterance(2, 3, "第三句", speaker="说话人B")]
    merged = render_transcript(us, merge_same_speaker=True)
    assert merged.count("\n") == 1 and "第一句第二句" in merged.replace(" ", "")
    unmerged = render_transcript(us, merge_same_speaker=False)
    assert unmerged.count("\n") == 2


def test_split_chunks_bounds_and_coverage():
    text = "\n".join(f"第{i}行内容，长度大约二十个字符左右。" for i in range(200))
    chunks = split_chunks(text, 300)
    assert all(len(c) <= 300 for c in chunks)
    assert "\n".join(chunks).replace("\n", "") == text.replace("\n", "")
    long_line = "啊" * 1000
    chunks2 = split_chunks(long_line, 300)
    assert "".join(chunks2) == long_line


def test_generate_minutes_direct():
    tr = {"duration": 60, "speakers": ["说话人A"],
          "utterances": [u.to_dict() for u in _utts(4)]}
    cfg = deep_merge(DEFAULT_CONFIG, {"summarizer": {"strategy": "auto", "direct_threshold_chars": 100000}})
    md, usage = generate_minutes(tr, MockLLM(), config=cfg)
    assert "mock 纪要" in md and usage.calls == 1


def test_generate_minutes_map_reduce():
    tr = {"duration": 3600, "speakers": ["说话人A", "说话人B"],
          "utterances": [u.to_dict() for u in _utts(40)]}
    cfg = deep_merge(DEFAULT_CONFIG, {"summarizer": {
        "strategy": "map_reduce", "max_chunk_chars": 400, "direct_threshold_chars": 300}})
    md, usage = generate_minutes(tr, MockLLM(), config=cfg)
    assert "mock 纪要" in md
    assert usage.calls >= 2  # map 多次 + reduce 一次
    assert usage.prompt_tokens > 0


def test_generate_minutes_empty_raises():
    cfg = deep_merge(DEFAULT_CONFIG, {})
    try:
        generate_minutes({"utterances": []}, MockLLM(), config=cfg)
        assert False, "应当抛错"
    except RuntimeError:
        pass


# ---------------- 简易 runner ----------------
def main() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback

            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
