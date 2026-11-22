import json
from pathlib import Path

from meetmin.asr.base import Utterance
from meetmin.diarization.live_recovery import restore_live_speaker_names
from meetmin.gui.workflow import transcribe
from meetmin.pipeline import Pipeline
from meetmin.session import Session


def _journal(path: Path) -> None:
    events = [
        {
            "type": "asr.commit",
            "track": "loopback",
            "segment_id": "a",
            "start_ms": 0,
            "end_ms": 3000,
        },
        {
            "type": "diarization.assign",
            "segment_id": "a",
            "parts": [{"start_ms": 0, "speaker_id": 3}],
        },
        {
            "type": "diarization.reconcile",
            "updates": [
                {
                    "segment_id": "a",
                    "parts": [
                        {"start_ms": 0, "speaker_id": 3},
                        {"start_ms": 2000, "speaker_id": 2},
                    ],
                }
            ],
            "merges": [{"from_id": 3, "to_id": 1}],
        },
        {"type": "speaker.rename", "speaker_id": 1, "display_name": "主持人"},
        {"type": "speaker.rename", "speaker_id": 2, "display_name": "嘉宾"},
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + '\n{"type":', "utf-8")


def _utterance() -> Utterance:
    words = [{"w": word, "s": i, "e": i + 1} for i, word in enumerate("甲乙丙丁")]
    return Utterance(0, 4, "甲乙丙丁", speaker="说话人A", words=words)


def test_replay_named_parts_from_torn_journal_without_dropping_the_tail(tmp_path):
    log = tmp_path / "events.jsonl"
    _journal(log)
    output = restore_live_speaker_names([_utterance()], log)
    assert [(part.speaker, part.text) for part in output] == [
        ("主持人", "甲乙"),
        ("嘉宾", "丙"),
        ("说话人A", "丁"),
    ]


def test_desktop_retranscription_recovers_recorded_names_from_existing_artifacts(tmp_path):
    session = Session(tmp_path, "recovered").ensure()
    session.save_json(
        "asr_raw.json",
        {
            "session_id": session.session_id,
            "utterances": [_utterance().to_dict()],
            "asr": {"provider": "faster_whisper"},
        },
    )
    _journal(session.root / "events.jsonl")
    pipe = Pipeline({"diarization": {"provider": "single"}, "punctuation": {"provider": "rule"}})
    saved = transcribe(pipe, session, lambda _: None, align_only=True)
    assert saved["speakers"] == ["主持人", "嘉宾", "说话人A"]
    assert saved == session.load_transcript()
