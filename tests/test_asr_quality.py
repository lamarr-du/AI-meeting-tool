from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from meetmin.asr.base import Utterance
from meetmin.asr.faster_whisper_asr import FasterWhisperASR
from meetmin.asr.quality import Decoded, QualityOptions, is_silence_hallucination, review
from meetmin.audio.dsp import TARGET_SR, save_wav_mono
from meetmin.pipeline import Pipeline


def test_detects_repeats_compression_and_only_voiced_gaps():
    first = Utterance(0, 1, "现在开始测试。")
    repeated = Utterance(1.1, 2, "现在开始测试！")
    compressed = Utterance(8, 9, "下一段")
    rows = [Decoded(first), Decoded(repeated), Decoded(compressed, compression_ratio=3)]
    decode = Mock()
    result, report = review(
        rows, 20, QualityOptions(), lambda: [(0, 2), (4, 6), (8, 9), (11, 12)], decode
    )
    decode.assert_not_called()
    assert result == [first, repeated, compressed]
    assert [(i["start"], i["reasons"]) for i in report["issues"]] == [
        (1.1, ["adjacent_repeat"]),
        (4, ["speech_gap"]),
        (8, ["high_compression"]),
    ]
    assert all(issue["status"] == "detected" for issue in report["issues"])


def _segment(start, end, text):
    word = SimpleNamespace(word=text, start=start, end=end)
    return SimpleNamespace(
        start=start,
        end=end,
        text=text,
        words=[word],
        compression_ratio=1,
        avg_logprob=-0.3,
        no_speech_prob=0.1,
    )


def test_local_retry_uses_audio_slice_and_records_original_words(tmp_path, monkeypatch):
    asr = object.__new__(FasterWhisperASR)
    asr.acfg = {"quality": {"mode": "retry"}, "condition_on_previous_text": True}
    asr._device, asr._compute = "cpu", "int8"
    model = asr.model = Mock()
    model.transcribe.side_effect = [
        (
            [
                _segment(0, 1, "今天开始测试"),
                _segment(1, 2, "今天开始测试"),
                _segment(5, 6, "就这样"),
            ],
            SimpleNamespace(duration=6, language="zh"),
        ),
        ([_segment(0.1, 0.9, "保留模型参数")], None),
        ([_segment(0.1, 1.7, "然后核对结果")], None),
    ]
    monkeypatch.setattr("meetmin.asr.get_asr", lambda _: asr)
    monkeypatch.setattr(
        "meetmin.asr.faster_whisper_asr._speech_spans", lambda _: [(0, 2), (2.5, 4.5), (5, 6)]
    )
    pipeline = Pipeline({"output_dir": str(tmp_path)})
    session = pipeline.new_session("quality")
    waveform = np.repeat(np.arange(6, dtype=np.float32) / 8, TARGET_SR)
    save_wav_mono(session.audio_path, waveform, TARGET_SR)
    raw = pipeline.transcribe(session)
    calls = model.transcribe.call_args_list
    assert not hasattr(asr, "model")
    assert [len(call.args[0]) for call in calls] == [6 * TARGET_SR, TARGET_SR, 2 * TARGET_SR]
    assert calls[0].kwargs["condition_on_previous_text"] is True
    assert calls[1].kwargs["condition_on_previous_text"] is False
    assert calls[1].kwargs["language"] == "zh"
    assert calls[1].kwargs["beam_size"] == 3
    assert calls[1].kwargs["no_repeat_ngram_size"] == 4
    assert np.isclose(calls[1].args[0][0], waveform[TARGET_SR], atol=0.0001)
    assert [u["text"] for u in raw["utterances"]] == [
        "今天开始测试",
        "保留模型参数",
        "然后核对结果",
        "就这样",
    ]
    assert raw["utterances"][2]["words"] == [{"w": "然后核对结果", "s": 2.6, "e": 4.2}]
    quality = raw["asr"]["quality"]
    assert quality["attempted"] == quality["accepted"] == 2
    assert quality["issues"][0]["original_utterance"]["words"][0]["w"] == "今天开始测试"
    assert [issue["status"] for issue in quality["issues"]] == ["accepted", "accepted"]


@pytest.mark.parametrize(
    "candidate, rejection",
    [
        (Decoded(Utterance(1, 2, "今天开始测试")), "still_repeated"),
        (Decoded(Utterance(0.9, 2, "另外一段话")), "invalid_timing"),
        (
            Decoded(Utterance(1.1, 2, "另外一段话", words=[{"w": "另外", "s": 0.9, "e": 1.5}])),
            "invalid_timing",
        ),
        (Decoded(Utterance(1.1, 2, "另外一段话"), avg_logprob=-1), "low_confidence"),
        (Decoded(Utterance(1.1, 1.2, "这个")), "reduced_coverage"),
    ],
)
def test_retry_keeps_first_pass_when_candidate_is_worse(candidate, rejection):
    original = [Decoded(Utterance(0, 1, "今天开始测试")), Decoded(Utterance(1, 2, "今天开始测试"))]
    result, report = review(
        original, 2, QualityOptions(mode="retry"), lambda: [], lambda *_: [candidate]
    )
    assert result == [row.utterance for row in original]
    assert report["issues"][0]["status"] == "rejected"
    assert report["issues"][0]["detail"] == rejection


def test_long_gap_respects_clip_and_retry_limits_and_keeps_output_after_error():
    decoder = Mock(side_effect=RuntimeError("device failure"))
    settings = QualityOptions(mode="retry", max_retries=1)
    result, report = review([], 45, settings, lambda: [(0, 45)], decoder)
    assert result == []
    decoder.assert_called_once_with(0, 15)
    assert [issue["status"] for issue in report["issues"]] == ["failed", "skipped", "skipped"]
    assert all(issue["end"] - issue["start"] <= 20 for issue in report["issues"])
    decoder.reset_mock()
    vad = Mock()
    _, report = review([], 45, replace(settings, mode="off"), vad, decoder)
    vad.assert_not_called()
    decoder.assert_not_called()
    assert report == {"mode": "off"}


def test_global_hallucination_guard_requires_word_timestamps():
    asr = object.__new__(FasterWhisperASR)
    asr.acfg = {"word_timestamps": False, "hallucination_silence_threshold": 2}
    asr._device = "cpu"
    with pytest.raises(ValueError, match="需要 asr.word_timestamps"):
        asr.transcribe(np.zeros(TARGET_SR, dtype=np.float32))


def test_silence_hallucination_guard_filters_low_confidence_phrases_and_preserves_real_thanks():
    # 模拟 large-v3-turbo 在低频底噪上输出 no_speech_prob=0.0 但对数概率与词概率极低的幻觉
    hallu_thanks = Decoded(
        Utterance(20.5, 21.1, "谢谢大家"),
        compression_ratio=0.8,
        avg_logprob=-0.89,
        no_speech_prob=0.0,
        mean_word_prob=0.28,
    )
    hallu_sleepy = Decoded(
        Utterance(61.0, 61.9, "Sleepy state."),
        compression_ratio=0.9,
        avg_logprob=-1.91,
        no_speech_prob=0.0,
        mean_word_prob=0.08,
    )
    hallu_hum = Decoded(
        Utterance(10.0, 11.2, "再去怒了"),
        compression_ratio=0.9,
        avg_logprob=-0.85,
        no_speech_prob=0.0,
        mean_word_prob=0.32,
    )
    real_closing = Decoded(
        Utterance(5592.7, 5597.6, "我们今天就先到这里，谢谢大家。"),
        compression_ratio=1.1,
        avg_logprob=-0.21,
        no_speech_prob=0.0,
        mean_word_prob=0.89,
    )
    real_short_thanks = Decoded(
        Utterance(159.3, 160.2, "谢谢。"),
        compression_ratio=0.7,
        avg_logprob=-0.25,
        no_speech_prob=0.01,
        mean_word_prob=0.88,
    )

    assert is_silence_hallucination(hallu_thanks, band_ratio=0.04) is True
    assert is_silence_hallucination(hallu_sleepy, band_ratio=0.12) is True
    assert is_silence_hallucination(hallu_hum, band_ratio=0.05) is True
    assert (
        is_silence_hallucination(
            Decoded(Utterance(5.3, 5.9, "ご視聴ありがとうございました。"), avg_logprob=-0.4),
            language="zh",
        )
        is True
    )
    assert is_silence_hallucination(real_closing, band_ratio=0.52, language="zh") is False
    assert is_silence_hallucination(real_short_thanks, band_ratio=0.48, language="zh") is False
