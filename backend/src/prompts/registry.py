"""Versioned prompt loading.

Each prompt lives under `prompts/<name>/vN.txt`. `CURRENT_VERSIONS` pins the
version served by default for each prompt name -- to roll a prompt forward,
add a new `vN.txt` file alongside the existing ones and bump the pointer here
(old versions stay on disk for rollback/audit; git history documents the
change). Callers can still request an explicit version to A/B or roll back
without a code change elsewhere.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent

CURRENT_VERSIONS: dict[str, str] = {
    "classifier": "v1",
    "greet": "v1",
    "rephraser": "v1",
    "answer": "v3",
    "decompose": "v1",
    "guardrail": "v1",
}


@lru_cache(maxsize=None)
def load_prompt(name: str, version: str | None = None) -> str:
    resolved_version = version or CURRENT_VERSIONS[name]
    path = _PROMPTS_DIR / name / f"{resolved_version}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt {name!r} version {resolved_version!r} not found at {path}"
        )
    return path.read_text(encoding="utf-8").strip()
