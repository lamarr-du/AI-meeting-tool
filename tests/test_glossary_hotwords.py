from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from meetmin.asr.faster_whisper_asr import FasterWhisperASR
from meetmin.config import load_config
from meetmin.glossary import glossary_hotwords, load_terms, select_terms


def test_glossary_respects_priority_and_model_token_budget():
    prompt, terms = glossary_hotwords(
        ["too long" * 10, "DRC Stat", "@", "bypass"],
        existing="manual",
        language="zh",
        token_count=len,
        token_limit=35,
        max_terms=2,
    )
    assert terms == ["DRC Stat", "@"]
    assert prompt == "manual 本次会议涉及 DRC Stat、@。"


def test_glossary_keeps_existing_prompt_when_budget_is_exhausted():
    prompt, terms = glossary_hotwords(
        ["DRC Stat"],
        existing="existing",
        language=None,
        token_count=len,
        token_limit=8,
        max_terms=20,
    )
    assert prompt == "existing"
    assert terms == []


def test_project_glossary_merges_with_manual_and_punctuation_prompts(tmp_path, monkeypatch):
    folder = tmp_path / "project"
    folder.mkdir()
    path = folder / "terms.yaml"
    path.write_text(
        'terms:\n  - term: DRC Stat\n    aliases: [DRCST]\n    mishearings: [drcstd]\n  - term: "@"\n',
        encoding="utf-8",
    )
    config = folder / "config.yaml"
    config.write_text(
        'asr:\n  hotwords: patch\n  punctuation_prompt: "好的。"\n  glossary:\n'
        '    path: terms.yaml\n    terms: ["@", "drc stat", "@"]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    provider = object.__new__(FasterWhisperASR)
    provider.acfg = load_config(config)["asr"]
    provider._device, provider._compute = "cpu", "int8"
    provider.model = Mock(max_length=448)
    provider.model.hf_tokenizer.encode.side_effect = lambda text, **_: SimpleNamespace(
        ids=list(text)
    )
    provider.model.transcribe.return_value = ([], SimpleNamespace(language="zh", duration=1))

    result = provider.transcribe(np.zeros(16000, dtype=np.float32))
    options = provider.model.transcribe.call_args.kwargs
    assert options["hotwords"] == "好的。 patch 本次会议涉及 @、DRC Stat。"
    assert "initial_prompt" not in options
    assert result.extra["parameters"]["glossary"] == {
        "path": str(path.resolve()),
        "terms": ["@", "DRC Stat"],
    }
    assert result.extra["parameters"]["hotwords"] == options["hotwords"]
    provider.model.hf_tokenizer.encode.assert_called_with(
        " 好的。 patch 本次会议涉及 @、DRC Stat。", add_special_tokens=False
    )


def test_glossary_rejects_malformed_entries_and_unknown_priorities(tmp_path):
    file = tmp_path / "terms.yaml"
    file.write_text("terms: [{term: 42}]", encoding="utf-8")
    with pytest.raises(ValueError, match="非空 term"):
        load_terms(file)
    with pytest.raises(ValueError, match="找不到标准写法"):
        select_terms(["DRC Stat"], ["DRCST"])
