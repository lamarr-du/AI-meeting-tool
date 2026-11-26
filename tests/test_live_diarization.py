from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np

from meetmin.asr.base import Utterance
from meetmin.config import DEFAULT_CONFIG, deep_merge
from meetmin.diarization.clustering import (
    SR,
    build_cache_signature,
    linear_sum_assignment_max,
    reuse_live_embeddings,
    save_embedding_cache,
)
from meetmin.diarization.live_diarizer import ProgressiveDiarizer, VoiceTurn
from meetmin.diarization.voiceprints import VoiceprintLibrary
from meetmin.live_pipeline import LiveSessionCoordinator
from meetmin.session import Session

VECTORS = np.eye(2, 192, dtype=np.float32)


class Encoder:
    def __init__(self, *_args, **_kwargs):
        pass

    def embed_waveforms(self, wavs):
        return np.stack([VECTORS[0 if np.mean(wav) > 0 else 1] for wav in wavs])

    def _model_signature(self):
        return "test-campplus"


def _diarizer(monkeypatch, **overrides):
    monkeypatch.setattr("meetmin.diarization.live_diarizer.CampplusOnnxDiarizer", Encoder)
    cfg = deep_merge(DEFAULT_CONFIG, {"live": {"diarization": overrides}})
    return ProgressiveDiarizer(cfg)


def _turn(key, start, tones, *, width=2.0):
    end = start + len(tones) * width
    words = []
    chunks = []
    for tone in tones:
        chunks.append(np.full(round(width * SR), tone, dtype=np.float32))
    for index in range(int(end - start)):
        words.append({"w": f"字{index}", "s": start + index, "e": start + index + 1})
    text = "".join(word["w"] for word in words) or "嗯"
    return VoiceTurn(
        key, Utterance(start, end, text, words=words), text, np.concatenate(chunks), start * SR
    )


def test_maximum_assignment_handles_multiple_shapes_and_unmatched_weights():
    matrix = np.array([[10, 9, 0], [9, 0, 0]])
    row, col = linear_sum_assignment_max(matrix)
    assert list(zip(row, col, strict=True)) == [(0, 1), (1, 0)]
    row, col = linear_sum_assignment_max(matrix.T)
    assert list(zip(row, col, strict=True)) == [(0, 1), (1, 0)]
    row, col = linear_sum_assignment_max(np.empty((3, 0)))
    assert not len(row) and not len(col)


def test_live_voice_reconciles_a_mixed_turn_after_first_speaker_is_named(monkeypatch):
    voice = _diarizer(monkeypatch)
    assert voice.add(_turn("short", 0, (0.1,), width=1))[0]["parts"][0]["speaker_id"] is None
    for key, start, tone in (("a1", 2, 0.1), ("a2", 4, 0.1), ("b1", 6, -0.1), ("b2", 8, -0.1)):
        parts = voice.add(_turn(key, start, (tone,)))[0]["parts"]
        assert parts[0]["speaker_id"] == (0 if tone > 0 else 1)
    voice.rename(0, "主持人")
    voice.add(_turn("mixed", 10, (0.1, -0.1), width=2.5))
    patch = voice.reconcile()
    assert patch is not None
    parts = next(update["parts"] for update in patch["updates"] if update["segment_id"] == "mixed")
    assert [part["display_name"] for part in parts] == ["主持人", "说话人B"]
    mixed = [segment for segment in voice.archive() if segment.start >= 10]
    assert [word["w"] for segment in mixed for word in segment.words] == [
        f"字{i}" for i in range(5)
    ]


def test_live_voice_worker_matches_public_names_before_stop_and_keeps_manual_edit(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("meetmin.diarization.live_diarizer.CampplusOnnxDiarizer", Encoder)
    cfg = deep_merge(DEFAULT_CONFIG, {"output_dir": str(tmp_path)})
    session = Session(tmp_path, "live-voiceprints").ensure()
    updates = Queue()
    coordinator = LiveSessionCoordinator(cfg, session, updates.put, dual_track=True)
    library = coordinator.pipe.voiceprint_library
    library.enroll("test-campplus", {"登记姓名": VECTORS[1]})
    original = library.path.read_bytes()
    worker = Thread(target=coordinator._run_voice, daemon=True)
    worker.start()
    try:
        coordinator._voice_inbox.put(_turn("a1", 0, (0.1,)))
        coordinator._voice_inbox.put(_turn("a2", 2, (0.1,)))
        coordinator._voice_inbox.put(_turn("b1", 4, (-0.1,)))
        assert updates.get(timeout=2)["parts"][0]["display_name"] == "说话人A"
        assert updates.get(timeout=2)["parts"][0]["display_name"] == "说话人A"
        assert updates.get(timeout=2)["parts"][0]["display_name"] == "说话人B"
        assert updates.get(timeout=2) == {
            "type": "speaker.rename",
            "speaker_id": 1,
            "display_name": "登记姓名",
            "pin_anchor": False,
        }
        assert not coordinator._voice_done.is_set()
        assert not coordinator._voice.turns["b1"].locked

        coordinator._voice_inbox.put(_turn("b2", 6, (-0.1,)))
        assert updates.get(timeout=2)["parts"][0]["display_name"] == "登记姓名"

        coordinator._speaker_renames.put((1, "手动姓名"))
        manual = updates.get(timeout=2)
        assert manual["display_name"] == "手动姓名" and manual["pin_anchor"]
        coordinator._voice_inbox.put(_turn("b3", 8, (-0.1,)))
        assert updates.get(timeout=2)["parts"][0]["display_name"] == "手动姓名"
    finally:
        coordinator._voice_done.set()
        worker.join(timeout=3)
    assert not worker.is_alive()
    assert coordinator._voice.speaker_names == {1: "手动姓名"}
    assert [row.speaker for row in coordinator._voice.archive()] == [
        "说话人A",
        "说话人A",
        "手动姓名",
        "手动姓名",
        "手动姓名",
    ]
    assert library.path.read_bytes() == original


def test_live_voiceprint_name_is_withdrawn_if_two_clusters_compete(monkeypatch, tmp_path):
    voice = _diarizer(monkeypatch)
    library = VoiceprintLibrary(tmp_path)
    library.enroll("test-campplus", {"登记姓名": VECTORS[0]})
    voice.voiceprint_matcher = library.matcher(voice.model_signature)
    voice.add(_turn("a", 0, (0.1,)))
    assert voice.match_voiceprints()[0]["display_name"] == "登记姓名"
    assert voice.archive()[0].speaker == "登记姓名"

    voice.add(_turn("b", 2, (-0.1,)))
    voice.centroids[1] = VECTORS[0]  # A later re-clustering can produce competing candidates.
    assert voice.match_voiceprints()[0]["display_name"] == "说话人A"
    assert voice.speaker_names == {}
    assert [row.speaker for row in voice.archive()] == ["说话人A", "说话人B"]


def test_invalid_public_library_does_not_stop_live_diarization(monkeypatch, tmp_path):
    monkeypatch.setattr("meetmin.diarization.live_diarizer.CampplusOnnxDiarizer", Encoder)
    cfg = deep_merge(DEFAULT_CONFIG, {"output_dir": str(tmp_path)})
    session = Session(tmp_path, "invalid-library").ensure()
    events = []
    coordinator = LiveSessionCoordinator(cfg, session, events.append, dual_track=False)
    path = coordinator.pipe.voiceprint_library.path
    path.parent.mkdir()
    path.write_text("{", encoding="utf-8")

    coordinator._voice_inbox.put(_turn("a", 0, (0.1,)))
    coordinator._voice_done.set()
    coordinator._run_voice()
    assert "无法读取公共声纹库" in events[0]["message"]
    assert events[1]["type"] == "diarization.assign"
    assert events[1]["parts"][0]["display_name"] == "说话人A"
    assert coordinator._voice_warning is None


def test_two_named_speakers_keep_their_identity_when_clustering_collides(monkeypatch):
    voice = _diarizer(monkeypatch)
    voice.add(_turn("a", 0, (0.1,)))
    voice.add(_turn("b", 2, (-0.1,)))
    voice.rename(0, "甲")
    voice.rename(1, "乙")
    monkeypatch.setattr(
        "meetmin.diarization.live_diarizer.cluster_embeddings",
        lambda vectors, *args, **kwargs: np.zeros(len(vectors), dtype=int),
    )
    voice.reconcile()
    assert [item.speaker for item in voice.archive()] == ["甲", "乙"]
    assert voice.aliases == {}


def test_recluster_bounds_the_dense_matrix_and_keeps_old_prototypes(monkeypatch):
    voice = _diarizer(monkeypatch, bounded_max_embeddings=8, bounded_window_sec=30)
    voice.add(_turn("old", 0, (0.1, 0.1, 0.1)))
    for index in range(9):
        voice.add(_turn(f"near-{index}", 100 + index * 2, (-0.1,)))
    clustered = voice._cluster_input()
    assert len(clustered) == 8
    assert sum(record.turn_id == "old" for record in clustered) == 2


def test_live_cache_only_reuses_matching_pcm_and_exact_full_size_windows(tmp_path: Path):
    def signature(mode, digest):
        return build_cache_signature(
            model="test-campplus",
            mode=mode,
            win_sec=2.5,
            hop_sec=1.25,
            merge_gap_sec=0.3,
            min_rms=0.003,
            speech_spans=None,
            audio_sha256=digest,
        )

    path = tmp_path / "embeddings.npz"
    vectors = np.ones((2, 192), dtype=np.float32)
    save_embedding_cache(path, vectors, [0, SR], [SR, SR + 500], signature("live_commit", "abc"))
    known = reuse_live_embeddings(
        path, signature("speech_spans", "abc"), [0, SR, 1], [SR, SR + 500, SR]
    )
    assert list(known) == [0]
    np.testing.assert_array_equal(known[0], vectors[0])
    assert not reuse_live_embeddings(path, signature("speech_spans", "different"), [0], [SR])


def test_bleed_suppression_accepts_both_commit_orders_without_hiding_short_replies(tmp_path):
    session = Session(tmp_path, "echo").ensure()
    turn = Utterance(0, 2, "我们重新确认这一项验收标准")
    echo = Utterance(0.2, 2.2, turn.text)
    for earlier, later in (("loopback", "microphone"), ("microphone", "loopback")):
        coordinator = LiveSessionCoordinator(
            DEFAULT_CONFIG, session, lambda _: None, dual_track=True
        )
        tracks = {"loopback": turn, "microphone": echo}
        assert not coordinator._bleed_matches(earlier, 0, tracks[earlier])
        assert coordinator._bleed_matches(later, 0, tracks[later]) == [0]
        assert not coordinator._bleed_matches("microphone", 1, Utterance(0.3, 0.5, "嗯嗯"))
