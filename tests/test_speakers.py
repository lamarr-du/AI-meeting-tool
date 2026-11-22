import json

import pytest

from meetmin.cli import main
from meetmin.pipeline import Pipeline
from meetmin.speakers import rename_speaker, rename_speakers, speaker_preview


def _transcript():
    return {
        "session_id": "meeting",
        "duration": 20,
        "speakers": ["说话人A", "说话人B", "未知", "我"],
        "utterances": [
            {"start": 1, "end": 2, "speaker": "说话人A", "text": "短句"},
            {"start": 3, "end": 5, "speaker": "说话人A", "text": "第一段较长的发言"},
            {"start": 5, "end": 7, "speaker": "说话人B", "text": "这是第二位说话人的检查事项"},
            {"start": 7, "end": 8, "speaker": "未知", "text": "未知的发言"},
            {"start": 9, "end": 10, "speaker": "我", "text": "本地麦克风发言"},
            {"start": 10, "end": 11, "speaker": "说话人A", "text": "简短句"},
            {"start": 12, "end": 14, "speaker": "说话人A", "text": "需要先显示的最长发言样本"},
        ],
    }


def test_preview_shows_three_longest_with_times_and_omits_unknown():
    lines = speaker_preview(_transcript())
    assert lines[:4] == [
        "[A] 说话人A",
        "  [00:12] 需要先显示的最长发言样本",
        "  [00:03] 第一段较长的发言",
        "  [00:10] 简短句",
    ]
    assert "[我] 我" in lines
    assert all("未知" not in line for line in lines)


@pytest.mark.parametrize(
    "assignments", [["A=张三", "B=张三"], ["A=张三", "说话人A=李四"], ["未知=张三"]]
)
def test_rename_rejects_invalid_batch(assignments):
    source = _transcript()
    with pytest.raises(ValueError):
        rename_speakers(source, assignments)
    assert source == _transcript()


def test_exact_display_name_can_be_renamed_even_if_it_contains_assignment_syntax():
    renamed = rename_speaker(_transcript(), "说话人A", "甲=负责人")
    renamed = rename_speaker(renamed, "甲=负责人", "张三")
    assert renamed["utterances"][0]["speaker"] == "张三"
    assert renamed["speaker_names"]["说话人A"] == "张三"


def test_cli_renames_current_transcript_and_feeds_summary(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    glossary = tmp_path / "glossary.yaml"
    glossary.write_text("terms: []\n", encoding="utf-8")
    config.write_text(
        json.dumps(
            {"output_dir": str(tmp_path), "asr": {"glossary": {"correction_path": str(glossary)}}}
        ),
        encoding="utf-8",
    )
    commands = ["--config", str(config)]
    session = Pipeline({"output_dir": str(tmp_path)}).new_session("meeting")
    source = _transcript()
    source["session_id"] = session.session_id
    session.save_transcript(source)
    raw = session.save_json("asr_raw.json", {"utterances": []})
    raw_bytes = raw.read_bytes()
    main([*commands, "speakers", "list", session.session_id])
    assert "[A] 说话人A" in capsys.readouterr().out

    main([*commands, "speakers", "rename", session.session_id, "A=张三", "B=李四"])
    output = capsys.readouterr().out
    assert "[00:12]" in output and "重新生成纪要" in output
    main([*commands, "speakers", "rename", session.session_id, "A=王五"])
    result = session.load_transcript()
    assert result["speakers"] == ["王五", "李四", "我"]
    assert result["speaker_names"]["说话人A"] == "王五"
    assert result["utterances"][3]["speaker"] == "未知"
    assert [(u["start"], u["end"], u["text"]) for u in result["utterances"]] == [
        (u["start"], u["end"], u["text"]) for u in source["utterances"]
    ]
    assert raw.read_bytes() == raw_bytes
    assert "**王五**" in session.transcript_md.read_text(encoding="utf-8")
    corrected = json.loads((session.root / "transcript_corrected.json").read_text(encoding="utf-8"))
    assert corrected["speakers"] == result["speakers"]
    main([*commands, "summarize", "-n", session.session_id, "--export-web"])
    prompt = (session.root / "minutes_web_prompt.txt").read_text(encoding="utf-8")
    assert "王五" in prompt and "李四" in prompt
