"""Dummy per-user access-control map: email -> authorized company_id(s).

`company_id` values must match the ones produced by `ingest/chunker.py` exactly,
since retrieval filters vector-store records by this field.
"""

from __future__ import annotations

DUMMY_USERS: dict[str, list[str]] = {
    "alice@example.com": ["TCS", "Infosys"],
    "bob@example.com": ["Axis"],
    "carol@example.com": ["Hdfc"],
    "dave@example.com": ["TataTechnologies"],
    "eve@example.com": ["TCS", "Hdfc"],
}


def is_valid_user(email: str) -> bool:
    return email in DUMMY_USERS


def get_user_companies(email: str) -> list[str] | None:
    return DUMMY_USERS.get(email)
