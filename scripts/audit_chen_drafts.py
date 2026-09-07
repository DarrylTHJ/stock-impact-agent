"""Audit saved Chen draft records before a human approves them for promotion.

This checks data-contract rules and whether each evidence quote appears in the
original timestamped transcript. It cannot judge whether the model's economic
interpretation is sensible; that remains a short human review task.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models import KnowledgeRecord


DRAFT_DIR = PROJECT_DIR / "data" / "chen_knowledge_records"
TRANSCRIPT_DIR = PROJECT_DIR / "data" / "chen_source_captions"
TIMESTAMP_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}–\d{2}:\d{2}:\d{2}$")


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def comparable(value: str) -> str:
    """Ignore caption punctuation differences, but retain all spoken words."""
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE)


def audit_draft(path: Path) -> list[str]:
    findings: list[str] = []
    draft = json.loads(path.read_text(encoding="utf-8"))
    video_id = draft.get("source_video_id")
    transcript_path = TRANSCRIPT_DIR / f"{video_id}.json"
    if not transcript_path.exists():
        return [f"{path.name}: source transcript is missing"]
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    source_text = " ".join(segment["text_original"] for segment in transcript["segments"])

    for index, raw_record in enumerate(draft.get("records", []), start=1):
        label = f"{path.name} record {index}"
        try:
            record = KnowledgeRecord.model_validate(raw_record)
        except Exception as error:
            findings.append(f"{label}: invalid schema: {error}")
            continue
        if record.source_name != "Chen" or record.source_link != transcript.get("video_url"):
            findings.append(f"{label}: source metadata does not match transcript")
        for evidence_index, evidence in enumerate(record.evidence, start=1):
            if not TIMESTAMP_PATTERN.fullmatch(evidence.location):
                findings.append(f"{label} evidence {evidence_index}: invalid timestamp format")
            if comparable(evidence.quote) not in comparable(source_text):
                findings.append(f"{label} evidence {evidence_index}: quote was not found verbatim in transcript")
        for company in record.impacted_companies:
            if any(item < 0 or item >= len(record.evidence) for item in company.evidence_indexes):
                findings.append(f"{label}: company evidence index is out of range")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", action="append", help="Audit only this video ID; repeatable.")
    args = parser.parse_args()
    paths = sorted(DRAFT_DIR.glob("*.json"))
    if args.video_id:
        wanted = set(args.video_id)
        paths = [path for path in paths if path.stem in wanted]

    findings = [finding for path in paths for finding in audit_draft(path)]
    print(f"Audited {len(paths)} draft file(s).")
    if findings:
        print(f"Found {len(findings)} issue(s):")
        print("\n".join(f"- {finding}" for finding in findings))
        raise SystemExit(1)
    print("PASS: all automated schema, timestamp, metadata, and verbatim-quote checks passed.")


if __name__ == "__main__":
    main()
