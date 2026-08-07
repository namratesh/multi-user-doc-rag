"""Split parsed earnings call transcripts into speaker-turn chunks with metadata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

QA_MARKER_PATTERN = re.compile(
    r"(?i)(?:"
    r"question[\s-]+and[\s-]+answer\s+session"
    r"|happy to take (?:your\s+)?questions"
    r"|open(?:ing)?\s+the\s+(?:floor|lines?)\s+for\s+questions"
    r")"
)

# Matches a speaker turn opener at the start of a line, e.g.
# "| Nehal Shah:", "## Moderator:", "- Sudheer Guntupalli:", "K Krithivasan:"
SPEAKER_PATTERN = re.compile(
    r"(?m)^[#|\s>-]*([A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z.'\-]*){0,3}):\s+"
)

# Markdown-header-style speaker turns with no colon, e.g. "## Yogesh Aggarwal", "## Moderator"
HEADER_SPEAKER_PATTERN = re.compile(
    r"(?m)^#+\s+((?:Moderator|Operator|[A-Z][a-z]+(?:\s+[A-Z][A-Za-z.'\-]*){1,3}))\s*$"
)

# A genuine speaker recurs; one-off matches (letter salutations, "Sub:", "Encl:") don't.
MIN_SPEAKER_OCCURRENCES = 2

MODERATOR_NAME_PATTERN = re.compile(r"(?i)^(moderator|operator)$")

# "MANAGEMENT:" / "CORPORATE PARTICIPANTS:" header that precedes a list of names,
# e.g. Axis/HDFC's "MANAGEMENT: MR. X - TITLE" block or Infosys' markdown headers.
MANAGEMENT_BLOCK_PATTERN = re.compile(r"(?im)^#*\s*(?:management|corporate participants)\s*:")
ANALYSTS_BLOCK_PATTERN = re.compile(r"(?im)^#*\s*analysts?\s*:?\s*$")
# A markdown heading (the call title) typically follows the management list directly.
NEXT_HEADING_PATTERN = re.compile(r"(?m)^#+\s")
TITLE_NAME_PATTERN = re.compile(
    r"(?:(?i:mr|ms|mrs|dr))\.?\s+([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,3})(?=\s*[-,\n])"
)
MARKDOWN_HEADER_NAME_PATTERN = re.compile(r"(?m)^#+\s+([A-Z][a-z]+(?:\s+[A-Z][A-Za-z.'\-]*){1,3})\s*$")

# Markdown table divider rows, e.g. "|----|----|"
TABLE_DIVIDER_PATTERN = re.compile(r"(?m)^\s*\|[-\s|]+\|\s*$")

QUARTER_PATTERN = re.compile(r"(?i)\bQ([1-4])\b")
YEAR_PATTERN = re.compile(r"(?i)\bFY\s?'?(\d{2,4})\b|\b(20\d{2})\b")

MAX_CHUNK_CHARS = 1500


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value)


def _clean_turn_text(text: str) -> str:
    text = TABLE_DIVIDER_PATTERN.sub(" ", text)
    text = text.replace("|", " ")
    return re.sub(r"\s+", " ", text).strip()


def derive_fiscal(metadata: dict, text: str) -> tuple[str | None, str | None]:
    period = metadata.get("period") or ""
    lookahead = text[:3000]

    q_match = QUARTER_PATTERN.search(period) or QUARTER_PATTERN.search(lookahead)
    quarter = f"Q{q_match.group(1)}" if q_match else None

    y_match = YEAR_PATTERN.search(period) or YEAR_PATTERN.search(lookahead)
    year = None
    if y_match:
        raw = y_match.group(1) or y_match.group(2)
        year = raw if len(raw) == 4 else f"20{raw}"

    return quarter, year


def split_sections(text: str) -> list[tuple[str, str]]:
    match = QA_MARKER_PATTERN.search(text)
    if not match:
        return [("prepared_remarks", text)]
    return [
        ("prepared_remarks", text[: match.start()]),
        ("qa", text[match.start() :]),
    ]


def _iter_speaker_matches(text: str) -> list[re.Match]:
    matches = list(SPEAKER_PATTERN.finditer(text)) + list(HEADER_SPEAKER_PATTERN.finditer(text))
    return sorted(matches, key=lambda m: m.start())


def find_valid_speakers(text: str, min_occurrences: int = MIN_SPEAKER_OCCURRENCES) -> set[str]:
    counts: dict[str, int] = {}
    for m in _iter_speaker_matches(text):
        name = m.group(1).strip()
        counts[name] = counts.get(name, 0) + 1
    return {name for name, count in counts.items() if count >= min_occurrences}


def extract_management_names(prepared_remarks: str) -> list[str]:
    """Pull management names from a "MANAGEMENT:"/"CORPORATE PARTICIPANTS:" header block, if present."""
    start = MANAGEMENT_BLOCK_PATTERN.search(prepared_remarks)
    if not start:
        return []
    # Infosys-style blocks list names as markdown headers, so a heading can't be used as
    # the end boundary there; Axis/HDFC-style blocks are plain text followed by the call
    # title heading, so it can.
    is_header_style = bool(re.match(r"\s*#", prepared_remarks[start.end() :]))
    end_matches = [ANALYSTS_BLOCK_PATTERN.search(prepared_remarks, start.end())]
    if not is_header_style:
        end_matches.append(NEXT_HEADING_PATTERN.search(prepared_remarks, start.end()))
    end_positions = [m.start() for m in end_matches if m]
    block_end = min(end_positions) if end_positions else start.end() + 800
    block = prepared_remarks[start.end() : block_end]

    names: list[str] = []
    seen: set[str] = set()
    for pattern in (MARKDOWN_HEADER_NAME_PATTERN, TITLE_NAME_PATTERN):
        for m in pattern.finditer(block):
            name = re.sub(r"\s+", " ", m.group(1)).strip().title()
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                names.append(name)
    return names


def classify_role(speaker: str, prepared_speakers: set[str], management_names: set[str]) -> str:
    """Classify a speaker as moderator, management, or analyst (default)."""
    if MODERATOR_NAME_PATTERN.match(speaker):
        return "moderator"
    normalized = speaker.lower()
    if normalized in prepared_speakers or normalized in management_names:
        return "management"
    return "analyst"


def split_speaker_turns(section_text: str, valid_speakers: set[str]) -> list[tuple[str | None, str]]:
    matches = [m for m in _iter_speaker_matches(section_text) if m.group(1).strip() in valid_speakers]
    if not matches:
        cleaned = _clean_turn_text(section_text)
        return [(None, cleaned)] if cleaned else []

    turns = []
    if matches[0].start() > 0:
        preamble = _clean_turn_text(section_text[: matches[0].start()])
        if preamble:
            turns.append((None, preamble))

    for i, m in enumerate(matches):
        speaker = m.group(1).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
        content = _clean_turn_text(section_text[m.end() : end])
        if content:
            turns.append((speaker, content))
    return turns


def _split_long_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    parts: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if current and length + len(word) + 1 > max_chars:
            parts.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += len(word) + 1
    if current:
        parts.append(" ".join(current))
    return parts


def chunk_document(parsed: dict) -> list[dict]:
    metadata = parsed["metadata"]
    text = parsed["text"]

    company_id = _slugify(metadata.get("company") or "Unknown")
    fiscal_quarter, fiscal_year = derive_fiscal(metadata, text)
    doc_id = "_".join(filter(None, [company_id, fiscal_quarter, fiscal_year])) or company_id

    valid_speakers = find_valid_speakers(text)
    sections = split_sections(text)

    prepared_text = next((t for s, t in sections if s == "prepared_remarks"), "")
    prepared_turns = split_speaker_turns(prepared_text, valid_speakers)
    prepared_speakers = {
        speaker.lower() for speaker, _ in prepared_turns if speaker and not MODERATOR_NAME_PATTERN.match(speaker)
    }

    management_names = extract_management_names(prepared_text) or [
        speaker for speaker, _ in prepared_turns if speaker and not MODERATOR_NAME_PATTERN.match(speaker)
    ]
    # dedupe while preserving order
    management_names = list(dict.fromkeys(management_names))
    management_names_lower = {name.lower() for name in management_names}

    chunks: list[dict] = []
    section_counters: dict[str, int] = {}
    qa_pair_id = None
    next_qa_pair_num = 0
    for section_type, section_text in sections:
        turns = prepared_turns if section_type == "prepared_remarks" else split_speaker_turns(section_text, valid_speakers)
        for speaker_name, turn_text in turns:
            speaker_role = (
                classify_role(speaker_name, prepared_speakers, management_names_lower) if speaker_name else None
            )

            if section_type == "qa":
                if speaker_role == "analyst":
                    next_qa_pair_num += 1
                    qa_pair_id = f"{doc_id}_qa_{next_qa_pair_num:03d}"
                elif speaker_role != "management":
                    qa_pair_id = None
                turn_qa_pair_id = qa_pair_id if speaker_role in ("analyst", "management") else None
            else:
                turn_qa_pair_id = None

            for piece in _split_long_text(turn_text):
                section_counters[section_type] = section_counters.get(section_type, 0) + 1
                idx = section_counters[section_type]
                chunks.append(
                    {
                        "company_id": company_id,
                        "doc_id": doc_id,
                        "section_type": section_type,
                        "speaker_name": speaker_name,
                        "speaker_role": speaker_role,
                        "qa_pair_id": turn_qa_pair_id,
                        "management_names": management_names,
                        "fiscal_quarter": fiscal_quarter,
                        "fiscal_year": fiscal_year,
                        "chunk_id": f"{doc_id}_{section_type}_{idx:03d}",
                        "text": piece,
                    }
                )
    return chunks


def chunk_parsed_file(parsed_path: Path) -> list[dict]:
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    return chunk_document(parsed)


def chunk_directory(parsed_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for parsed_path in sorted(parsed_dir.glob("*.json")):
        chunks = chunk_parsed_file(parsed_path)
        out_path = output_dir / parsed_path.name
        out_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
        print(f"{parsed_path.name}: {len(chunks)} chunks -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-dir", type=Path, default=Path("data/parsed"), help="Directory of parser.py JSON output")
    parser.add_argument("--output-dir", type=Path, default=Path("data/chunks"), help="Where to write chunked JSON files")
    args = parser.parse_args()

    chunk_directory(args.parsed_dir, args.output_dir)


if __name__ == "__main__":
    main()
