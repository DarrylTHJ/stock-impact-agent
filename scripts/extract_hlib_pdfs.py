"""Extract HLIB PDFs into page-preserving JSON files without using an LLM."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfReader


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_DIR / "data" / "hlib_source"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "hlib_extracted"
DATE_PATTERN = re.compile(r"_(?P<date>\d{8})_HLIB(?: \(\d+\))?$", re.IGNORECASE)


def derive_metadata(path: Path) -> tuple[str, str, str]:
    """Derive category, subject, and YYYY-MM-DD date from a source filename."""
    match = DATE_PATTERN.search(path.stem)
    if not match:
        raise ValueError("filename must end with _YYYYMMDD_HLIB.pdf")
    raw_subject = path.stem[: match.start()].strip("_")
    if raw_subject == "Market_View":
        category = "Market View"
    elif raw_subject == "Economic_Update":
        category = "Economic Update"
    else:
        category = "Industry Insight"
    subject = raw_subject.replace("_", " ").replace("&", "&").strip()
    date = match.group("date")
    return category, subject, f"{date[:4]}-{date[4:6]}-{date[6:]}"


def extract_pdf(path: Path) -> dict:
    category, subject, source_date = derive_metadata(path)
    reader = PdfReader(path)
    pages = [
        {
            "page_number": page_number,
            "text": (page.extract_text() or "").strip(),
        }
        for page_number, page in enumerate(reader.pages, start=1)
    ]
    readable_pages = sum(bool(page["text"]) for page in pages)
    if readable_pages == 0:
        raise ValueError("PDF contains no selectable text and may require OCR")
    return {
        "source_file": path.name,
        "source_category": category,
        "source_subject": subject,
        "source_date": source_date,
        "page_count": len(pages),
        "readable_page_count": readable_pages,
        "extracted_at_utc": datetime.now(UTC).isoformat(),
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_dir.glob("*.pdf"))
    pending = [
        path for path in paths
        if args.overwrite or not (args.output_dir / f"{path.stem}.json").exists()
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    created = 0
    errors = 0
    log_path = args.output_dir / "extraction_log.jsonl"
    for position, path in enumerate(pending, start=1):
        try:
            payload = extract_pdf(path)
            output_path = args.output_dir / f"{path.stem}.json"
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            status = "created"
            detail = f"{payload['readable_page_count']}/{payload['page_count']} readable pages"
            created += 1
        except Exception as error:
            status = "error"
            detail = str(error)
            errors += 1
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({
                "source_file": path.name,
                "status": status,
                "detail": detail,
            }, ensure_ascii=False) + "\n")
        print(f"[{position}/{len(pending)}] {path.name}: {status} ({detail})", flush=True)

    print(f"Finished: {created} created, {errors} errors, {len(paths) - len(pending)} already present.")


if __name__ == "__main__":
    main()
