import wave

import numpy as np
import pytest

from meetmin.asr.base import Utterance
from meetmin.audio.dsp import to_mono_16k
from meetmin.diarization.clustering import build_cache_signature, save_embedding_cache
from meetmin.diarization.voiceprints import VoiceprintLibrary, audio_fingerprint, local_profiles
from meetmin.pipeline import Pipeline
from meetmin.session import Session

VECTORS = np.eye(192, dtype=np.float32)


def _cache(session, vectors, *, model="test-model", mode="speech_spans"):
    with wave.open(str(session.audio_path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes(len(vectors) * 64000))
    sha = audio_fingerprint(to_mono_16k(session.audio_path))
    starts = [i * 32000 for i in range(len(vectors))]
    save_embedding_cache(
        session.root / "embeddings.npz",
        np.stack(vectors),
        starts,
        [s + 32000 for s in starts],
        build_cache_signature(
            model=model,
            mode=mode,
            win_sec=2,
            hop_sec=0,
            merge_gap_sec=0,
            min_rms=0.003,
            speech_spans=None,
            audio_sha256=sha,
        ),
    )
    return sha


def _publish(pipe, session, vectors, *, model="test-model", mode="speech_spans", names=None):
    sha = _cache(session, vectors, model=model, mode=mode)
    a, b = "说话人A", (names or {}).get("说话人B", "说话人B")
    return pipe._publish_aligned_transcript(
        session,
        {
            "session_id": session.session_id,
            "audio_path": str(session.audio_path),
            "asr": {"provider": "faster_whisper"},
        },
        [Utterance(0, 4, "第一段", a), Utterance(4, 8, "第二段", b)],
        [a, b],
        speaker_names=names,
        voiceprint_audio_sha256=sha,
    )


def test_opt_in_enrollment_matches_later_archive_without_overwriting(tmp_path):
    pipe = Pipeline({"output_dir": tmp_path})
    first = Session(tmp_path, "first").ensure()
    result = _publish(pipe, first, VECTORS[[0, 0, 1, 1]])
    assert result["speakers"] == ["说话人A", "说话人B"]
    assert not pipe.voiceprint_library.path.parent.exists()
    with pytest.raises(ValueError, match="先编辑"):
        pipe.enroll_voiceprints(first)
    pipe.rename_speakers(first, {"说话人A": "甲"})
    assert not pipe.voiceprint_library.path.exists()
    assert pipe.enroll_voiceprints(first).added == ["甲"]
    registered = pipe.voiceprint_library.path.read_bytes()
    assert pipe.enroll_voiceprints(first).existing == ["甲"]
    assert pipe.voiceprint_library.path.read_bytes() == registered

    later = Session(tmp_path, "later").ensure()
    recovered = _publish(
        pipe, later, VECTORS[[0, 0, 2, 2]], mode="live_commit", names={"说话人B": "乙"}
    )
    assert recovered["speakers"] == ["甲", "乙"]
    assert [u["speaker"] for u in recovered["utterances"]] == ["甲", "乙"]
    assert pipe.voiceprint_library.path.read_bytes() == registered
    added = pipe.enroll_voiceprints(later)
    assert added.added == ["乙"] and added.existing == ["甲"]

    next_session = Session(tmp_path, "next").ensure()
    third = _publish(pipe, next_session, VECTORS[[2, 2, 0, 0]])
    assert third["speakers"] == ["乙", "甲"]
    assert pipe.voiceprint_library.match("other-model", {"A": VECTORS[0]}, reserved=set()) == {}


def test_ambiguous_matches_are_not_applied(tmp_path):
    library = VoiceprintLibrary(tmp_path)
    library.enroll("model", {"甲": VECTORS[0], "乙": VECTORS[1]})
    assert library.match("model", {"A": VECTORS[0] + VECTORS[1]}, reserved=set()) == {}
    assert library.match("model", {"A": VECTORS[0], "B": VECTORS[0]}, reserved=set()) == {}
    assert library.match("model", {"A": VECTORS[0]}, reserved={"甲"}) == {}
    assert library.match("model", {"A": VECTORS[2]}, reserved=set()) == {}


def test_legacy_enrollment_requires_matching_audio_and_clean_windows(tmp_path):
    pipe = Pipeline({"output_dir": tmp_path})
    session = Session(tmp_path, "legacy").ensure()
    transcript = _publish(pipe, session, VECTORS[[0, 0, 1, 1]])
    transcript.pop("speaker_voiceprints")
    session.save_transcript(transcript)
    pipe.rename_speakers(session, ["A=甲"])
    assert pipe.enroll_voiceprints(session).added == ["甲"]
    sha = audio_fingerprint(to_mono_16k(session.audio_path))
    assert (
        local_profiles(
            session.root / "embeddings.npz",
            [
                Utterance(0, 4, "甲", "甲"),
                Utterance(0, 4, "乙", "乙"),
            ],
            sha,
        )
        is None
    )
    with session.audio_path.open("r+b") as source:
        source.seek(44)
        source.write(b"\xff\x7f")
    with pytest.raises(ValueError, match="足够"):
        pipe.enroll_voiceprints(session)
