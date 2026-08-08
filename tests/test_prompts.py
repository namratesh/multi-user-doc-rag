"""Prompt registry: every pinned prompt must exist on disk and load."""

from __future__ import annotations

import pytest

from src.prompts import CURRENT_VERSIONS, load_prompt


@pytest.mark.parametrize("name", list(CURRENT_VERSIONS))
def test_pinned_prompt_loads_and_is_non_empty(name: str) -> None:
    prompt = load_prompt(name)
    assert prompt.strip()


@pytest.mark.parametrize("name,version", list(CURRENT_VERSIONS.items()))
def test_explicit_version_matches_pinned_default(name: str, version: str) -> None:
    assert load_prompt(name) == load_prompt(name, version)


def test_unknown_prompt_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        load_prompt("does-not-exist")


def test_unknown_prompt_version_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("classifier", "v999")
