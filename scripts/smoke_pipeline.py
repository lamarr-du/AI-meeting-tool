"""端到端冒烟：FakeASR + MockLLM 跑通「(1)→(2)→(3)」编排，不联网、不调大模型。

运行： python scripts/smoke_pipeline.py
产物目录： data/sessions/smoke-*（transcript.json / transcript.md / minutes.md）
"""
from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import meetmin.asr as asr_pkg
from meetmin.asr.base import ASRProvider, ASRResult, Utterance
from meetmin.audio import dsp
from meetmin.config import load_config
from meetmin.pipeline import Pipeline
from meetmin.summarizer.llm_client import MockLLM


class FakeASR(ASRProvider):
    """假装是本地 whisper：返回固定 utterances 并回传音频。"""

    name = "faster_whisper"

    def transcribe(self, audio, *, file_path=None, progress_cb=None, **kw) -> ASRResult:
        utts = [
            Utterance(0.5, 4.0, "我们过一下这周的排期，先说后端。",
                      words=[{"w": "我们", "s": 0.5, "e": 1.4}, {"w": "过一下", "s": 1.4, "e": 2.6},
                             {"w": "这周", "s": 2.6, "e": 3.2}, {"w": "的排期", "s": 3.2, "e": 4.0}]),
            Utterance(5.0, 9.0, "后端周三能联调，测试需要两天。",
                      words=[{"w": "后端", "s": 5.0, "e": 5.9}, {"w": "周三", "s": 5.9, "e": 6.7},
                             {"w": "能联调", "s": 6.7, "e": 7.6}, {"w": "测试", "s": 7.6, "e": 8.2},
                             {"w": "两天", "s": 8.4, "e": 9.0}]),
            Utterance(10.5, 13.0, "好，就这么定。",
                      words=[{"w": "好", "s": 10.5, "e": 10.8}, {"w": "就这么定", "s": 10.9, "e": 13.0}]),
        ]
        if progress_cb:
            progress_cb(13.0, 16.0)
        return ASRResult(utterances=utts, language="zh", audio_16k=audio, extra={"asr_model": "fake"})


def main() -> None:
    asr_pkg.get_asr = lambda cfg: FakeASR()  # 注入假 ASR
    cfg = load_config(ROOT / "config.yaml")
    cfg["summarizer"]["strategy"] = "direct"
    cfg["summarizer"]["direct_threshold_chars"] = 100000
    cfg["diarization"]["provider"] = "single"  # 冒烟不拉 ECAPA 模型

    # 合成一段 16 秒音频作为"录音产物"
    sess_dir = ROOT / "data" / "sessions"
    sid = "smoke-%06d" % np.random.randint(0, 999999)
    session = Pipeline(cfg).new_session(sid)
    t = np.arange(16 * 16000, dtype=np.float32) / 16000
    tone = (0.2 * np.sin(2 * math.pi * 300 * t)).astype(np.float32)
    dsp.save_wav_mono(session.audio_path, tone, 16000)

    pipe = Pipeline(cfg)
    transcript = pipe.transcribe(session)
    assert transcript["speakers"], "说话人名单为空"
    assert len(transcript["utterances"]) == 3
    # 用 MockLLM 生成纪要
    path, usage = pipe.summarize(session, llm=MockLLM())
    assert path.exists() and usage.calls >= 1
    print("\n--- transcript.md（前 6 行）---")
    print("\n".join(session.transcript_md.read_text(encoding="utf-8").splitlines()[:6]))
    print("\n--- minutes.md ---")
    print(path.read_text(encoding="utf-8"))
    print(f"\nSMOKE OK  会话目录: {session.root}")


if __name__ == "__main__":
    main()
