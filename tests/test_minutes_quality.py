"""纪要生成对空结果和各阶段截断的处理。"""

import pytest

from meetmin.config import DEFAULT_CONFIG, deep_merge
from meetmin.pipeline import Pipeline
from meetmin.session import Session
from meetmin.summarizer.llm_client import Usage


class Responses:
    name = "test"

    def __init__(self, *outputs):
        self.outputs = iter(outputs)

    def chat(self, messages, **kwargs):
        text, status = next(self.outputs)
        return text, Usage(calls=1, finish_status=status)


@pytest.fixture
def meeting(tmp_path):
    session = Session(tmp_path, "test").ensure()
    session.save_transcript(
        {"session_id": "test", "utterances": [{"text": "甲" * 350, "speaker": "甲"}]}
    )
    cfg = deep_merge(DEFAULT_CONFIG, {"summarizer": {"strategy": "direct"}})
    return session, cfg


@pytest.mark.parametrize(
    "status,expected",
    [
        ("complete", "完整性：接口未报告截断"),
        ("truncated", "完整性：不完整"),
        ("unknown", "完整性：无法确认"),
    ],
)
def test_direct_output_marks_completion(meeting, status, expected):
    session, cfg = meeting
    path, usage = Pipeline(cfg).summarize(session, llm=Responses(("纪要", status)))
    assert expected in path.read_text(encoding="utf-8")
    assert usage.finish_status == status


def test_empty_result_preserves_existing_minutes(meeting):
    session, cfg = meeting
    session.minutes_md.write_text("原纪要", encoding="utf-8")
    with pytest.raises(RuntimeError, match="空纪要"):
        Pipeline(cfg).summarize(session, llm=Responses(("  ", "complete")))
    assert session.minutes_md.read_text(encoding="utf-8") == "原纪要"


def test_intermediate_truncation_marks_final_minutes_incomplete(meeting):
    session, cfg = meeting
    cfg["summarizer"].update(strategy="map_reduce", max_chunk_chars=200)
    llm = Responses(("部分要点", "truncated"), ("完整要点", "complete"), ("纪要", "complete"))
    path, usage = Pipeline(cfg).summarize(session, llm=llm)
    assert "完整性：不完整" in path.read_text(encoding="utf-8")
    assert usage.calls == 3 and usage.finish_status == "truncated"


def test_intermediate_unknown_prevents_complete_label(meeting):
    session, cfg = meeting
    cfg["summarizer"].update(strategy="map_reduce", max_chunk_chars=200)
    llm = Responses(("要点", "unknown"), ("要点", "complete"), ("纪要", "complete"))
    path, usage = Pipeline(cfg).summarize(session, llm=llm)
    assert "完整性：无法确认" in path.read_text(encoding="utf-8")
    assert usage.finish_status == "unknown"


def test_empty_intermediate_segment_does_not_write_partial_minutes(meeting):
    session, cfg = meeting
    cfg["summarizer"].update(strategy="map_reduce", max_chunk_chars=200)
    session.minutes_md.write_text("原纪要", encoding="utf-8")
    llm = Responses(("要点", "complete"), ("  ", "complete"))
    with pytest.raises(RuntimeError, match="段摘要为空"):
        Pipeline(cfg).summarize(session, llm=llm)
    assert session.minutes_md.read_text(encoding="utf-8") == "原纪要"
