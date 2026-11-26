import json
from pathlib import Path

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


def test_repository_isp_glossary_loads_without_conflict_and_corrects_explicit_short_terms(
    tmp_path,
):
    repo_glossary = Path(__file__).resolve().parents[1] / "glossary.yaml"
    # 复制仓库内公开版 glossary.yaml 到隔离目录，同时验证可选 glossary.local.yaml 合并能力
    isolated = tmp_path / "glossary.yaml"
    isolated.write_text(repo_glossary.read_text(encoding="utf-8"), encoding="utf-8")
    sample = {
        "utterances": [
            {
                "start": 0,
                "end": 5,
                "speaker": "甲",
                "text": "排查了感冒和反感冒、印的 8 上边码、鲁邦性、新毛子、REA LUT 以及迪马赛克、瑞玛赛克、突击问题。",
            }
        ]
    }
    result = correct_transcript(sample, isolated)
    corrected_text = result["utterances"][0]["text"]
    assert "gamma 和反 gamma" in corrected_text
    assert "int8" in corrected_text
    assert "熵编码" in corrected_text
    assert "鲁棒性" in corrected_text
    assert "C model" in corrected_text
    assert "REAE LUT" in corrected_text
    assert "demosaic" in corrected_text
    assert "remosaic" in corrected_text
    # "突击" 具有通用语境歧义，应保留原文并进入 context_review 待核对
    assert "突击" in corrected_text
    assert any(r["before"] == "突击" for r in result["glossary_correction"]["context_review"])

    # 验证本地未入库的 glossary.local.yaml 可自动合并纠错
    (tmp_path / "glossary.local.yaml").write_text(
        "terms:\n  - term: 测试员甲\n    needs_context: false\n    mishearings: [测士元甲]\n",
        encoding="utf-8",
    )
    local_res = correct_transcript(
        {"utterances": [{"start": 0, "end": 1, "speaker": "甲", "text": "找测士元甲确认"}]},
        isolated,
    )
    assert local_res["utterances"][0]["text"] == "找测试员甲确认"
