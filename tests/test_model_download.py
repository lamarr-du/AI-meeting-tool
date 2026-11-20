from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest

from meetmin.asr.model_download import ModelDownloadProgress, download_model


def test_complete_whisper_cache_skips_remote_requests(tmp_path, monkeypatch):
    weights = tmp_path / "model.bin"
    for path in (
        weights,
        tmp_path / "config.json",
        tmp_path / "tokenizer.json",
        tmp_path / "vocabulary.txt",
    ):
        path.touch()
    snapshot = Mock()
    report = Mock()
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *_: str(weights))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)

    assert download_model("tiny", report) == str(tmp_path)
    snapshot.assert_not_called()
    report.assert_not_called()


@pytest.mark.parametrize("size_known", [True, False])
def test_download_reports_aggregate_bytes_across_files_and_resumes(
    tmp_path, monkeypatch, size_known
):
    revision = "a" * 40
    files = [
        SimpleNamespace(
            filename="model.bin",
            file_size=1000 if size_known else None,
            will_download=True,
            commit_hash=revision,
        ),
        SimpleNamespace(
            filename="tokenizer.json", file_size=100, will_download=True, commit_hash=revision
        ),
        SimpleNamespace(
            filename="config.json", file_size=20, will_download=False, commit_hash=revision
        ),
    ]
    snapshot = Mock(return_value=files)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda *_, **kwargs: str(tmp_path / "model.bin") if kwargs else None,
    )
    order = []

    def fetch(repo, filename, *, revision, tqdm_class):
        assert repo == "Systran/faster-whisper-tiny" and revision == "a" * 40
        order.append(filename)
        size = 1000 if filename == "model.bin" else 100
        initial = 250 if filename == "model.bin" else 0
        with tqdm_class(total=size, initial=initial, unit="B") as bar:
            bar.update(size - initial)
            bar.refresh()
        path = tmp_path / filename
        path.write_bytes(bytes(size))
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)
    events = []
    assert download_model("tiny", events.append) == str(tmp_path)
    assert snapshot.call_args.kwargs["dry_run"]
    assert order == (
        ["tokenizer.json", "model.bin"] if size_known else ["model.bin", "tokenizer.json"]
    )
    assert events[0] == ModelDownloadProgress("tiny", 0, 1100 if size_known else None)
    assert any(event.downloaded == (350 if size_known else 250) for event in events)
    assert events[-1] == ModelDownloadProgress("tiny", 1100, 1100, complete=True)
