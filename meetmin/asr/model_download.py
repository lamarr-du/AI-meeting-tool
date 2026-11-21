"""Download a Whisper model with byte counts from the Hub's actual file transfers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from io import StringIO
from pathlib import Path


@dataclass(frozen=True)
class ModelDownloadProgress:
    model: str
    downloaded: int
    total: int | None
    bytes_per_second: float | None = None
    complete: bool = False


def download_model(
    model: str, report: Callable[[ModelDownloadProgress], None], *, cache_dir: Path
) -> str:
    """Keep the normal Hub cache, including offline use and partial-file resume."""
    from faster_whisper.utils import _MODELS
    from huggingface_hub import hf_hub_download, snapshot_download, try_to_load_from_cache
    from tqdm.auto import tqdm

    # Match faster-whisper's aliases and snapshot contents so both entry points share a cache.
    repo = model if "/" in model else _MODELS.get(model)
    if repo is None:
        raise ValueError(f"未知 Whisper 模型：{model}")
    cached = try_to_load_from_cache(repo, "model.bin", cache_dir=cache_dir)
    if isinstance(cached, str):
        folder = Path(cached).parent
        if all((folder / name).is_file() for name in ("config.json", "tokenizer.json")) and any(
            path.is_file() for path in folder.glob("vocabulary.*")
        ):
            return str(folder)

    class QuietBar(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.update(file=StringIO(), disable=True)
            super().__init__(*args, **kwargs)

    files = snapshot_download(
        repo,
        cache_dir=cache_dir,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.*",
        ],
        tqdm_class=QuietBar,
        dry_run=True,
    )
    model_file = next((file for file in files if file.filename == "model.bin"), None)
    if model_file is None:
        raise FileNotFoundError(f"模型仓库缺少 model.bin：{repo}")

    pending = sorted(
        (file for file in files if file.will_download), key=lambda file: file.file_size or 0
    )
    total = (
        sum(file.file_size or 0 for file in pending)
        if all(file.file_size is not None for file in pending)
        else None
    )
    downloaded = 0

    class FileBar(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.update(file=StringIO(), disable=False, mininterval=0.25, miniters=1)
            super().__init__(*args, **kwargs)

        def display(self, msg=None, pos=None):
            current = downloaded + int(self.n)
            if total is not None:
                current = min(total, current)
            report(ModelDownloadProgress(model, current, total, self.format_dict.get("rate")))

    if pending:
        report(ModelDownloadProgress(model, 0, total))
    for file in pending:
        fetched = hf_hub_download(
            repo, file.filename, revision=file.commit_hash, cache_dir=cache_dir, tqdm_class=FileBar
        )
        downloaded += file.file_size if file.file_size is not None else Path(fetched).stat().st_size

    path = try_to_load_from_cache(
        repo, "model.bin", cache_dir=cache_dir, revision=model_file.commit_hash
    )
    if not isinstance(path, str):
        raise FileNotFoundError(f"找不到已下载的模型：{repo}")
    if pending:
        report(ModelDownloadProgress(model, downloaded, downloaded, complete=True))
    return str(Path(path).parent)
