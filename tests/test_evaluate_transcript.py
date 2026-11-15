import pytest

from scripts.evaluate_transcript import Turn, evaluate, read_reference


def test_reference_and_text_metrics(tmp_path):
    path = tmp_path / "label.txt"
    path.write_text(
        "\nAlice\n00:00:00\n看 ＤＲＣ Stat，@ 会议\n和 bypass。\nBob\n00:00:05\n看 bypass\n",
        encoding="utf-8-sig",
    )
    reference = read_reference(path)
    assert reference == [
        Turn("Alice", 0, "看 ＤＲＣ Stat，@ 会议 和 bypass。"),
        Turn("Bob", 5, "看 bypass"),
    ]
    hypothesis = {
        "duration": 10,
        "utterances": [{"text": "看 DRC stat, @会议和 bypass。看 bypassing"}],
    }
    result = evaluate(reference, hypothesis, ["DRC Stat", "@", "bypass", "missing"])
    assert result["character_edits"] == 3
    assert result["character_error_rate"] == 3 / result["reference_characters"]
    assert result["terms"]["recall"] == 3 / 4
    assert result["terms"]["terms"]["bypass"] == {"reference": 2, "hypothesis": 1, "credited": 1}
    assert result["speakers"] == {"status": "no_speaker_labels", "reference_count": 2}


def test_speaker_labels_are_permutation_invariant_and_score_unknown_as_error():
    pytest.importorskip("scipy")
    reference = [Turn("Alice", 0, "甲"), Turn("Bob", 2, "乙"), Turn("Alice", 4, "丙")]
    document = {
        "duration": 6,
        "utterances": [
            {"start": 0, "end": 3, "speaker": "说话人B", "text": "甲"},
            {"start": 3, "end": 4, "speaker": "说话人A", "text": "乙"},
            {"start": 4, "end": 5, "speaker": "说话人B", "text": "丙"},
            {"start": 5, "end": 6, "speaker": "未知", "text": ""},
        ],
    }
    result = evaluate(reference, document, [])["speakers"]
    assert result["count_matches"]
    assert result["covered_seconds"] == result["scored_seconds"] == 6
    assert result["attribution_error_rate"] == pytest.approx(1 / 3)
    document["diarization_status"] = {"status": "fallback"}
    assert evaluate(reference, document, [])["speakers"]["status"] == "diarization_fallback"
