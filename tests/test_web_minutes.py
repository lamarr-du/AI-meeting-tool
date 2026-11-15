"""网页版纪要导出、导入及无密钥接口行为。"""

import pytest

from meetmin.cli import build_parser, cmd_run, cmd_summarize
from meetmin.config import DEFAULT_CONFIG, deep_merge
from meetmin.pipeline import Pipeline
from meetmin.session import Session
from meetmin.summarizer.llm_client import Usage


@pytest.fixture
def meeting(tmp_path):
    session = Session(tmp_path, "weekly").ensure()
    transcript = {
        "session_id": "weekly",
        "meeting_time": "2026-11-15 10:00",
        "duration": 90,
        "speakers": ["说话人A", "说话人B"],
        "utterances": [
            {"start": 0, "end": 1, "speaker": "说话人A", "text": "开头" + "重要内容" * 3000},
            {"start": 2, "end": 3, "speaker": "说话人B", "text": "最后一句"},
        ],
    }
    session.save_transcript(transcript)
    cfg = deep_merge(DEFAULT_CONFIG, {"output_dir": str(tmp_path)})
    return session, cfg


def args(*arguments):
    return build_parser().parse_args(["summarize", "-n", "weekly", *arguments])


def test_export_and_import_web_minutes(meeting, tmp_path, monkeypatch, capsys):
    session, cfg = meeting

    def no_network(*args, **kwargs):
        pytest.fail("网页版导出/导入不得发起网络请求")

    monkeypatch.setattr("requests.post", no_network)
    cmd_summarize(args("--export-web"), cfg)
    prompt = (session.root / "minutes_web_prompt.txt").read_text(encoding="utf-8")
    assert "开头" + "重要内容" * 3000 in prompt
    assert "说话人B: 最后一句" in prompt
    assert "2026-11-15 10:00" in prompt
    assert "不要编造" in prompt and "行动项" in prompt
    assert "文件包含会议内容" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="已存在"):
        cmd_summarize(args("--export-web"), cfg)
    cmd_summarize(args("--export-web", "--overwrite"), cfg)

    source = tmp_path / "web-answer.md"
    source.write_text("\ufeff### 结论\n已完成。\n", encoding="utf-8")
    cmd_summarize(args("--import-web", str(source)), cfg)
    assert "### 结论\n已完成。" in session.minutes_md.read_text(encoding="utf-8")
    assert "从 AI 网页导入" in session.minutes_md.read_text(encoding="utf-8")
    original = session.minutes_md.read_bytes()
    source.write_text("更新后的纪要", encoding="utf-8")
    with pytest.raises(SystemExit, match="已存在"):
        cmd_summarize(args("--import-web", str(source)), cfg)
    assert session.minutes_md.read_bytes() == original
    cmd_summarize(args("--import-web", str(source), "--overwrite"), cfg)
    assert "更新后的纪要" in session.minutes_md.read_text(encoding="utf-8")


def test_empty_or_invalid_import_preserves_minutes(meeting, tmp_path):
    session, cfg = meeting
    session.minutes_md.write_text("原始纪要", encoding="utf-8")
    source = tmp_path / "empty.txt"
    for value in (b"  \n", b"\xff\xfe"):
        source.write_bytes(value)
        with pytest.raises(SystemExit, match="纪要处理失败"):
            cmd_summarize(args("--import-web", str(source), "--overwrite"), cfg)
        assert session.minutes_md.read_text(encoding="utf-8") == "原始纪要"


def test_empty_transcript_rejected_without_output(meeting):
    session, cfg = meeting
    session.save_transcript({"utterances": [{"text": "  "}]})
    with pytest.raises(SystemExit, match="转写结果为空"):
        cmd_summarize(args("--export-web"), cfg)
    assert not (session.root / "minutes_web_prompt.txt").exists()


def test_missing_key_still_calls_local_compatible_model(meeting, monkeypatch):
    session, cfg = meeting
    cfg["llm"]["api_key_env"] = "MEETMIN_TEST_MISSING_KEY"
    cfg["summarizer"]["strategy"] = "direct"
    monkeypatch.delenv("MEETMIN_TEST_MISSING_KEY", raising=False)
    messages = []

    class LocalLLM:
        name = "local"

        def chat(self, messages):
            return "本地接口生成的纪要", Usage(calls=1)

    monkeypatch.setattr("meetmin.pipeline.get_llm", lambda cfg: LocalLLM())
    Pipeline(cfg, progress=messages.append).summarize(session)
    assert "本地接口生成的纪要" in session.minutes_md.read_text(encoding="utf-8")
    assert any("--export-web" in message for message in messages)


def test_failed_api_suggests_web_export(meeting, monkeypatch):
    session, cfg = meeting
    cfg["summarizer"]["strategy"] = "direct"
    messages = []

    class FailedLLM:
        name = "failed"

        def chat(self, messages):
            raise RuntimeError("接口不可用")

    monkeypatch.setattr("meetmin.pipeline.get_llm", lambda cfg: FailedLLM())
    with pytest.raises(RuntimeError, match="接口不可用"):
        Pipeline(cfg, progress=messages.append).summarize(session)
    assert any("--export-web" in message for message in messages)
    assert not session.minutes_md.exists()


def test_run_can_choose_web_export_before_recording(meeting, monkeypatch, capsys):
    session, cfg = meeting
    pipe = Pipeline(cfg)
    monkeypatch.setattr("meetmin.doctor.run_doctor", lambda *a, **kw: True)
    monkeypatch.setattr("meetmin.cli.Pipeline", lambda cfg: pipe)
    monkeypatch.setattr("meetmin.cli._record_session", lambda *a: session)
    monkeypatch.setattr("meetmin.cli._do_record", lambda *a: None)
    monkeypatch.setattr(pipe, "transcribe", lambda *a, **kw: None)
    monkeypatch.setattr(pipe, "diarize", lambda *a: None)
    monkeypatch.setattr(pipe, "summarize", lambda *a: pytest.fail("不应调用纪要 API"))
    run_args = build_parser().parse_args(["run", "-n", "weekly", "--export-web"])
    cmd_run(run_args, cfg)
    assert (session.root / "minutes_web_prompt.txt").exists()
    assert not session.minutes_md.exists()
    assert "--import-web" in capsys.readouterr().out
