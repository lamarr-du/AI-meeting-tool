import json

import pytest

from meetmin.config import load_config
from meetmin.glossary_correction import correct_transcript
from meetmin.pipeline import Pipeline


def test_replacements_are_longest_first_boundary_aware_and_reversible(tmp_path):
    glossary = tmp_path / "glossary.yaml"
    glossary.write_text(
        """terms:
  - term: DRC Stat
    aliases: [DRCST]
    mishearings: [DRC STD, DRCState]
  - term: DRC
    aliases: [drc]
  - term: BLC07
  - term: demosaic
    mishearings: [迪马赛克]
  - term: filter
    mishearings: [赛克处理阶段]
  - term: "@"
    mishearings: [爱他]
  - term: workaround
    needs_context: true
    mishearings: [walk around]
""",
        encoding="utf-8",
    )
    text = "检查drc std、drc、drcst、blc07和xblc07，迪马赛克和迪马赛克处理阶段，爱他 walk around"
    original = {"utterances": [{"start": 1, "end": 3, "speaker": "甲", "text": text}]}
    corrected = correct_transcript(original, glossary)
    output = corrected["utterances"][0]["text"]
    assert (
        output == "检查DRC Stat、DRC、DRCST、BLC07和xblc07，demosaic和迪马filter，爱他 walk around"
    )
    assert original["utterances"][0]["text"] == text
    audit = corrected["glossary_correction"]
    assert [record["before"] for record in audit["context_review"]] == ["爱他", "walk around"]
    for record in reversed(audit["changes"]):
        start, end = record["corrected_span"]
        assert record["utterance_index"] == 0 and record["start_sec"] == 1
        assert output[start:end] == record["after"]
        output = output[:start] + record["before"] + output[end:]
    assert output == text


def test_correction_rejects_conflicting_terms(tmp_path):
    glossary = tmp_path / "glossary.yaml"
    glossary.write_text(
        "terms: [{term: DRC, mishearings: [DIC]}, {term: ABC, mishearings: [dic]}]",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="同时指向"):
        correct_transcript({"utterances": []}, glossary)


def test_pipeline_keeps_source_and_regenerates_corrected_summary(tmp_path):
    glossary = tmp_path / "glossary.yaml"
    glossary.write_text("terms: [{term: demosaic, mishearings: [迪马赛克]}]", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("asr:\n  glossary:\n    correction_path: glossary.yaml\n", encoding="utf-8")
    cfg = load_config(config)
    cfg["output_dir"] = str(tmp_path)
    cfg["diarization"]["provider"] = "single"
    pipeline = Pipeline(cfg)
    session = pipeline.new_session("correction")
    raw = session.save_json(
        "asr_raw.json",
        {
            "session_id": session.session_id,
            "duration": 3,
            "language": "zh",
            "asr": {"provider": "faster_whisper"},
            "utterances": [{"start": 1, "end": 3, "text": "检查迪马赛克"}],
        },
    )
    raw_bytes = raw.read_bytes()
    original = pipeline.diarize(session)
    assert raw.read_bytes() == raw_bytes
    assert original["utterances"][0]["text"] == "检查迪马赛克。"
    assert "检查demosaic。" in (session.root / "transcript_corrected.md").read_text(
        encoding="utf-8"
    )

    original["utterances"][0]["text"] = "重跑迪马赛克。"
    path = session.save_transcript(original)
    original_bytes = path.read_bytes()
    prompt = pipeline.export_web_prompt(session, overwrite=True).read_text(encoding="utf-8")
    assert "重跑demosaic。" in prompt
    assert path.read_bytes() == original_bytes
    assert (
        json.loads((session.root / "transcript_corrected.json").read_text(encoding="utf-8"))[
            "utterances"
        ][0]["text"]
        == "重跑demosaic。"
    )

    cfg["asr"]["glossary"]["correction_path"] = None
    raw_prompt = pipeline.export_web_prompt(session, overwrite=True).read_text(encoding="utf-8")
    assert "重跑迪马赛克。" in raw_prompt
