"""Load earnings call PDFs with Docling and extract text + metadata."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from docling.document_converter import DocumentConverter

from src.config.logger import get_logger

logger = get_logger(__name__)

# Matches quarter/fiscal-year tokens such as Q1FY24, Q1 FY2024, H1FY25, FY2024-25, FY24
PERIOD_PATTERN = re.compile(
    r"""(?ix)
    (?:
        [QH][1-4]\s?-?\s?FY\s?\d{2,4}          # Q1FY24, H2 FY2025
        | FY\s?\d{2,4}(?:-\d{2,4})?             # FY2024-25, FY24
        | (?:quarter|half\s+year)\s+ended\s+[A-Za-z]+\s+\d{1,2},?\s+\d{4}
        | (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s-]+\d{4}
    )
    """,
)

# Suffixes that mark the end of a company name in extracted text
COMPANY_SUFFIX_PATTERN = re.compile(
    r"(?im)^.*?\b[A-Z][\w&.,'\- ]{2,60}\b(?:Limited|Ltd\.?|Inc\.?|Corporation|Bank)\b"
)


@dataclass
class DocumentMetadata:
    filename: str
    source_path: str
    company: str | None
    period: str | None
    page_count: int | None
    char_count: int


@dataclass
class ParsedDocument:
    metadata: DocumentMetadata
    text: str


def _company_from_filename(stem: str) -> str:
    words = re.split(r"[_\-\s]+", stem.strip())
    cleaned = [w if w.isupper() else w.title() for w in words if w]
    return " ".join(cleaned)


def _period_from_filename(stem: str) -> str | None:
    match = PERIOD_PATTERN.search(stem)
    return match.group(0) if match else None


def _period_from_text(text: str) -> str | None:
    match = PERIOD_PATTERN.search(text[:5000])
    return match.group(0) if match else None


def _company_from_text(text: str) -> str | None:
    match = COMPANY_SUFFIX_PATTERN.search(text[:3000])
    if not match:
        return None
    candidate = match.group(0).splitlines()[-1].strip()
    return candidate or None


def extract_metadata(source_path: Path, text: str, page_count: int | None) -> DocumentMetadata:
    stem = source_path.stem

    period = _period_from_filename(stem) or _period_from_text(text)

    filename_company = _company_from_filename(re.sub(PERIOD_PATTERN, "", stem)).strip()
    company = filename_company or _company_from_text(text)

    return DocumentMetadata(
        filename=source_path.name,
        source_path=str(source_path),
        company=company,
        period=period,
        page_count=page_count,
        char_count=len(text),
    )


def _page_count(doc) -> int | None:
    try:
        return doc.num_pages()
    except (AttributeError, TypeError):
        pass
    pages = getattr(doc, "pages", None)
    return len(pages) if pages is not None else None


def parse_pdf(path: Path, converter: DocumentConverter | None = None) -> ParsedDocument:
    """Convert a single earnings call PDF into full text plus metadata."""
    logger.info("[parse_pdf] input path=%s", path)
    converter = converter or DocumentConverter()

    result = converter.convert(str(path))
    doc = result.document

    text = doc.export_to_markdown()
    metadata = extract_metadata(path, text, _page_count(doc))
    logger.info(
        "[parse_pdf] output company=%r period=%r pages=%s chars=%d",
        metadata.company,
        metadata.period,
        metadata.page_count,
        metadata.char_count,
    )

    return ParsedDocument(metadata=metadata, text=text)


def parse_directory(data_dir: Path, pattern: str = "*.pdf") -> list[ParsedDocument]:
    logger.info("[parse_directory] input data_dir=%s pattern=%r", data_dir, pattern)
    converter = DocumentConverter()
    parsed = []
    pdf_paths = sorted(data_dir.glob(pattern))
    for pdf_path in pdf_paths:
        logger.info("[parse_directory] step: parsing %s", pdf_path.name)
        parsed.append(parse_pdf(pdf_path, converter=converter))
    logger.info("[parse_directory] output documents=%d", len(parsed))
    return parsed


def _write_output(doc: ParsedDocument, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(doc.metadata.filename).stem
    payload = {"metadata": asdict(doc.metadata), "text": doc.text}
    (output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing PDFs")
    parser.add_argument("--output-dir", type=Path, default=None, help="If set, write parsed JSON files here")
    args = parser.parse_args()

    documents = parse_directory(args.data_dir)

    for doc in documents:
        m = doc.metadata
        logger.info(
            "[main] %s: company=%r period=%r pages=%s chars=%d",
            m.filename,
            m.company,
            m.period,
            m.page_count,
            m.char_count,
        )
        if args.output_dir:
            _write_output(doc, args.output_dir)


if __name__ == "__main__":
    main()
