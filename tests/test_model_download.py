import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest

from meetmin.asr.model_download import ModelDownloadProgress, download_model
from meetmin.config import load_config
from meetmin.model_cache import hub_cache_dir, model_cache_root


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
    cached = Mock(return_value=str(weights))
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)

    assert download_model("tiny", report, cache_dir=tmp_path) == str(tmp_path)
    cached.assert_called_once_with("Systran/faster-whisper-tiny", "model.bin", cache_dir=tmp_path)
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
        lambda *_, revision=None, cache_dir=None: str(tmp_path / "model.bin") if revision else None,
    )
    order = []

    def fetch(repo, filename, *, revision, cache_dir, tqdm_class):
        assert repo == "Systran/faster-whisper-tiny" and revision == "a" * 40
        assert cache_dir == tmp_path
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
    assert download_model("tiny", events.append, cache_dir=tmp_path) == str(tmp_path)
    assert snapshot.call_args.kwargs["dry_run"]
    assert snapshot.call_args.kwargs["cache_dir"] == tmp_path
    assert order == (
        ["tokenizer.json", "model.bin"] if size_known else ["model.bin", "tokenizer.json"]
    )
    assert events[0] == ModelDownloadProgress("tiny", 0, 1100 if size_known else None)
    assert any(event.downloaded == (350 if size_known else 250) for event in events)
    assert events[-1] == ModelDownloadProgress("tiny", 1100, 1100, complete=True)


def test_project_and_portable_default_cache_locations(tmp_path, monkeypatch):
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(key, raising=False)
    assert hub_cache_dir() == Path(__file__).resolve().parents[1] / ".cache/huggingface/hub"

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "portable" / "MeetMin.exe"))
    assert model_cache_root() == tmp_path / "portable" / ".cache"

    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "meetmin").mkdir()
    (tmp_path / "meetmin" / "config.py").touch()
    assert model_cache_root() == tmp_path / ".cache"


def test_custom_model_cache_and_local_whisper_path_are_relative_to_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "environment-override"))
    local_model = tmp_path / "models" / "my-whisper"
    local_model.mkdir(parents=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model_cache_dir: weights\nasr:\n  model_size: models/my-whisper\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path.parent)
    cfg = load_config(config_path)
    assert Path(cfg["asr"]["model_size"]) == local_model
    assert hub_cache_dir(cfg["model_cache_dir"]) == tmp_path / "weights/huggingface/hub"
    assert hub_cache_dir() == tmp_path / "environment-override"
