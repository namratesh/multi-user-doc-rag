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

# Human-readable display name for each company_id, for use in UI copy
# (e.g. the assistant's greeting). Keys must match chunker.py's slugified
# company_id exactly.
COMPANY_DISPLAY_NAMES: dict[str, str] = {
    "TCS": "TCS",
    "Infosys": "Infosys",
    "Axis": "Axis Bank",
    "Hdfc": "HDFC",
    "TataTechnologies": "Tata Technologies",
}


def is_valid_user(email: str) -> bool:
    return email in DUMMY_USERS


def get_user_companies(email: str) -> list[str] | None:
    return DUMMY_USERS.get(email)


def get_company_display_names(company_ids: list[str]) -> list[str]:
    return [COMPANY_DISPLAY_NAMES.get(cid, cid) for cid in company_ids]


def get_company_catalog_text() -> str:
    """Renders the full company catalog as 'Display Name (id: company_id)'
    pairs, for prompts that need to recognize company mentions by name --
    e.g. query decomposition matching "axis" or a misspelling to `Axis`."""
    return ", ".join(f"{name} (id: {cid})" for cid, name in COMPANY_DISPLAY_NAMES.items())
