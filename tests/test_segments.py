import subprocess
import sys
import wave
from types import SimpleNamespace

import pytest

from meetmin.audio.segments import SegmentedWriter, SessionLock, merge_segments
from meetmin.cli import cmd_recover


def samples(path):
    with wave.open(str(path), "rb") as wav:
        return wav.readframes(wav.getnframes())


def test_segment_boundaries_and_merge(tmp_path):
    output = tmp_path / "audio.wav"
    writer = SegmentedWriter(output, 10, 1, overwrite=False, segment_seconds=1, sync_seconds=10)
    audio = bytes(range(46))
    writer.writeframesraw(audio[:14])
    writer.writeframesraw(audio[14:])
    writer.close()
    try:
        assert [p.name for p in sorted(writer.directory.iterdir())] == [
            "000001.wav",
            "000002.wav",
            "000003.wav",
        ]
        result = merge_segments(writer.directory, output)
        assert result["duration_sec"] == 2.3
        assert samples(output) == audio
    finally:
        writer.release()


def test_forced_exit_recovers_active_segment(tmp_path):
    program = """
import sys, time
from pathlib import Path
from meetmin.audio.segments import SegmentedWriter
root = Path(sys.argv[1])
w = SegmentedWriter(root / "audio.wav", 10, 1, overwrite=False, segment_seconds=1, sync_seconds=10)
w.writeframesraw(bytes(range(30)))
w.checkpoint(force=True)
print("ready", flush=True)
time.sleep(60)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", program, str(tmp_path)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert proc.stdout.readline().strip() == "ready"
    finally:
        proc.kill()
        proc.wait(timeout=10)
        proc.stdout.close()
    directory = next(tmp_path.glob("audio_parts-*"))
    assert sorted(p.name for p in directory.iterdir()) == ["000001.wav", "000002.wav.part"]
    lock = SessionLock(tmp_path)
    try:
        result = merge_segments(directory, tmp_path / "audio.wav")
    finally:
        lock.close()
    assert result["partial"]
    assert samples(tmp_path / "audio.wav") == bytes(range(30))


def test_partial_tail_and_existing_output_are_safe(tmp_path):
    writer = SegmentedWriter(
        tmp_path / "audio.wav", 10, 2, overwrite=False, segment_seconds=1, sync_seconds=10
    )
    writer.writeframesraw(b"abcd" * 2)
    writer.checkpoint(force=True)
    writer.file.write(b"x")
    writer.file.flush()
    result = merge_segments(writer.directory, tmp_path / "recovered.wav")
    assert result["discarded_bytes"] == 1
    assert samples(tmp_path / "recovered.wav") == b"abcd" * 2
    with pytest.raises(FileExistsError):
        merge_segments(writer.directory, tmp_path / "recovered.wav")
    assert samples(tmp_path / "recovered.wav") == b"abcd" * 2
    writer.close()
    writer.release()


def test_missing_segment_rejected_without_deleting_sources(tmp_path):
    writer = SegmentedWriter(
        tmp_path / "audio.wav", 10, 1, overwrite=False, segment_seconds=1, sync_seconds=10
    )
    writer.writeframesraw(b"ab" * 11)
    writer.close()
    writer.release()
    (writer.directory / "000001.wav").rename(writer.directory / "000003.wav")
    with pytest.raises(ValueError, match="编号"):
        merge_segments(writer.directory, tmp_path / "audio.wav")
    assert not (tmp_path / "audio.wav").exists()
    assert len(list(writer.directory.iterdir())) == 2


def test_session_lock_rejects_concurrent_recovery(tmp_path):
    first = SessionLock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="正在录音或恢复"):
            SessionLock(tmp_path)
    finally:
        first.close()


def test_recovery_command_requires_choice_and_preserves_existing_audio(tmp_path, capsys):
    session = tmp_path / "sessions" / "meeting"
    session.mkdir(parents=True)
    path = session / "audio.wav"
    for data in (b"aa", b"bb"):
        writer = SegmentedWriter(path, 10, 1, overwrite=False, segment_seconds=1, sync_seconds=10)
        writer.writeframesraw(data)
        writer.close()
        writer.release()
    directories = sorted(p.name for p in session.glob("audio_parts-*"))
    args = SimpleNamespace(session="meeting", take=None, output=None)
    with pytest.raises(SystemExit, match="--take"):
        cmd_recover(args, {"output_dir": str(tmp_path)})
    args.take = directories[0]
    cmd_recover(args, {"output_dir": str(tmp_path)})
    original = samples(path)
    args.take = directories[1]
    with pytest.raises(SystemExit, match="恢复失败"):
        cmd_recover(args, {"output_dir": str(tmp_path)})
    assert samples(path) == original
    args.output = str(session / "recovered.wav")
    cmd_recover(args, {"output_dir": str(tmp_path)})
    assert {original, samples(session / "recovered.wav")} == {b"aa", b"bb"}
    assert "恢复完成" in capsys.readouterr().out


def test_format_mismatch_preserves_sources(tmp_path):
    directory = tmp_path / "parts"
    directory.mkdir()
    for index, rate in enumerate((10, 11), 1):
        with wave.open(str(directory / f"{index:06}.wav"), "wb") as wav:
            wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            wav.writeframes(b"ab" * 5)
    with pytest.raises(ValueError, match="格式不一致"):
        merge_segments(directory, tmp_path / "audio.wav")
    assert len(list(directory.iterdir())) == 2
    assert not (tmp_path / "audio.wav").exists()
