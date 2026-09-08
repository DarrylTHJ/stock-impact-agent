"""Transform page-preserving HLIB extracts into initial knowledge records."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field, model_validator

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models import BURSA_SECTORS, ImpactedCompany


DEFAULT_INPUT_DIR = PROJECT_DIR / "data" / "hlib_extracted"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "hlib_transformed_initial"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_DELAY_SECONDS = 45
MAX_REPORT_CHARACTERS = 140_000


class HlibEvidenceDraft(BaseModel):
    quote: str = Field(min_length=1)
    page_number: int = Field(ge=1)


class HlibRecordDraft(BaseModel):
    knowledge_type: Literal["sector_impact", "company_impact", "market_context"]
    trigger_event: str
    impacted_sector: str | None = None
    impacted_industry: str | None = None
    impact_direction: Literal["positive", "negative", "mixed"] | None = None
    reason: str
    impacted_companies: list[ImpactedCompany] = Field(default_factory=list)
    evidence: list[HlibEvidenceDraft] = Field(min_length=1)
    embedding_summary: str

    @model_validator(mode="after")
    def validate_scope(self) -> "HlibRecordDraft":
        if self.knowledge_type == "sector_impact":
            if self.impacted_sector not in BURSA_SECTORS or self.impact_direction is None:
                raise ValueError("sector_impact requires an official Bursa sector and direction")
        elif self.knowledge_type == "company_impact":
            if not self.impacted_companies or self.impact_direction is None:
                raise ValueError("company_impact requires a named company and direction")
        else:
            if self.impacted_sector is not None or self.impacted_industry is not None:
                raise ValueError("market_context cannot claim a sector or industry")
        return self


class HlibExtractionBatch(BaseModel):
    report_title: str
    report_type: str
    records: list[HlibRecordDraft] = Field(default_factory=list, max_length=15)


def is_disclaimer_page(text: str) -> bool:
    beginning = text[:700].lower()
    return "disclaimer" in beginning and "information contained in this report" in beginning


def format_report(source: dict) -> str:
    sections = []
    for page in source["pages"]:
        if is_disclaimer_page(page["text"]):
            continue
        sections.append(f"[Page {page['page_number']}]\n{page['text']}")
    text = "\n\n".join(sections)
    if len(text) > MAX_REPORT_CHARACTERS:
        raise ValueError(
            f"Report contains {len(text):,} characters, exceeding the "
            f"{MAX_REPORT_CHARACTERS:,}-character one-call limit"
        )
    return text


def build_prompt(source: dict, report_text: str) -> str:
    sectors = "; ".join(BURSA_SECTORS)
    return f"""
You are extracting auditable investment-research knowledge from one HLIB
Research report for a Bursa Malaysia event-impact system.

Website category: {source['source_category']}
Filename subject: {source['source_subject']}
Report date: {source['source_date']}

Read the complete report before extracting. Use only claims explicitly supported
by the report. Do not use outside knowledge or invent causal links. It is
acceptable to return no records.

Extract one record per distinct event-target relationship. Consolidate repeated
discussion of the same relationship. Preserve causal detail in `reason`, but do
not provide hidden chain-of-thought.

For sector_impact:
- impacted_sector must be exactly one of: {sectors}
- impact_direction must be positive, negative, or mixed
- the report must explicitly connect the event to the sector's demand, costs,
  earnings, outlook, sentiment, valuation, or index exposure
- impacted_industry may preserve a narrower report-supported label such as
  Automotive, Banking, Gloves, Semiconductors, or Renewable Energy
- do not automatically equate national-accounts categories with Bursa sectors;
  for example, stronger national manufacturing GDP alone does not prove a
  positive impact on Industrial Products & Services

For company_impact:
- use this when the explained impact is specific to named companies
- impacted_sector must be null unless the report independently makes a
  sector-wide claim
- do not create a company impact merely because a company appears in a rating,
  target-price, coverage, top-picks, or valuation table
- ticker must be null unless the report explicitly provides the Bursa stock code;
  do not guess it

For market_context:
- use this only for a useful macro or broad-market mechanism that cannot
  responsibly be assigned to one Bursa sector
- impacted_sector, impacted_industry, and impact_direction must be null
- do not extract isolated statistics without an explanatory implication

Evidence rules:
- each evidence item must copy a concise verbatim English excerpt from the
  stated page and provide its integer page_number
- use multiple evidence excerpts when the reasoning spans different pages
- ignore disclaimer pages and generic stock/sector rating definitions

Return the report's actual title, its internal report type (for example
Newsbreak, Sector View, or Data Pulse), and at most 15 knowledge records.
Each record requires: knowledge_type, trigger_event, impacted_sector,
impacted_industry, impact_direction, reason, impacted_companies, evidence, and
embedding_summary. Each company requires company_name, ticker, and zero-based
evidence_indexes.

Page-labelled report:
{report_text}
"""


def normalise_scope(batch: HlibExtractionBatch) -> HlibExtractionBatch:
    for record in batch.records:
        if record.knowledge_type == "company_impact":
            record.impacted_sector = None
        elif record.knowledge_type == "market_context":
            record.impacted_sector = None
            record.impacted_industry = None
            record.impact_direction = None
    return batch


def comparable_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("\u00ad", "").replace("�", "")
    return re.sub(r"\s+", "", value)


def evidence_matches_page(quote: str, page_text: str) -> bool:
    return comparable_text(quote) in comparable_text(page_text)


def is_quota_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(token in text for token in ("429", "resource_exhausted", "quota", "rate limit"))


def transform_one(client: genai.Client, model: str, source: dict) -> tuple[HlibExtractionBatch, list[dict]]:
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(source, format_report(source)),
        config={
            "response_mime_type": "application/json",
            "response_schema": HlibExtractionBatch,
            "temperature": 0,
        },
    )
    batch = normalise_scope(HlibExtractionBatch.model_validate_json(response.text))
    pages = {page["page_number"]: page["text"] for page in source["pages"]}
    warnings = []
    for record_index, record in enumerate(batch.records):
        for evidence_index, evidence in enumerate(record.evidence):
            page_text = pages.get(evidence.page_number)
            if page_text is None or not evidence_matches_page(evidence.quote, page_text):
                warnings.append({
                    "record_index": record_index,
                    "evidence_index": evidence_index,
                    "page_number": evidence.page_number,
                    "issue": "quote_not_found_on_stated_page",
                })
    return batch, warnings


def stable_source_id(path_name: str) -> str:
    stem = Path(path_name).stem.casefold()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def save_initial(output_path: Path, source: dict, batch: HlibExtractionBatch, model: str, warnings: list[dict]) -> None:
    records = []
    source_id = stable_source_id(source["source_file"])
    for index, draft in enumerate(batch.records, start=1):
        record = draft.model_dump()
        record["evidence"] = [
            {
                "quote": evidence["quote"],
                "translation": None,
                "location": f"Page {evidence['page_number']}",
            }
            for evidence in record["evidence"]
        ]
        record.update({
            "knowledge_id": f"hlib_{source_id}_{index:02d}",
            "source_name": "HLIB Research",
            "source_title": batch.report_title,
            "source_date": source["source_date"],
            "source_link": None,
            "source_file": source["source_file"],
            "source_category": source["source_category"],
            "source_report_type": batch.report_type,
            "extraction_model": model,
        })
        records.append(record)
    output_path.write_text(json.dumps({
        "source_file": source["source_file"],
        "source_category": source["source_category"],
        "source_report_type": batch.report_type,
        "source_title": batch.report_title,
        "transformed_at_utc": datetime.now(UTC).isoformat(),
        "evidence_warnings": warnings,
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--file", action="append", help="Transform only this extracted JSON filename or stem.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--delay-seconds", type=int, default=DEFAULT_DELAY_SECONDS)
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing from the local .env file.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input_dir.glob("*.json"))
    if args.file:
        wanted = {Path(value).stem for value in args.file}
        paths = [path for path in paths if path.stem in wanted]
        missing = wanted - {path.stem for path in paths}
        if missing:
            raise SystemExit(f"Extracted HLIB file not found: {', '.join(sorted(missing))}")
    pending = [
        path for path in paths
        if args.overwrite or not (args.output_dir / path.name).exists()
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"Pending HLIB reports: {len(pending)}", flush=True)
    client = genai.Client(api_key=api_key)
    log_path = args.output_dir / "transformation_log.jsonl"
    for position, path in enumerate(pending, start=1):
        source = json.loads(path.read_text(encoding="utf-8"))
        try:
            batch, warnings = transform_one(client, args.model, source)
            save_initial(args.output_dir / path.name, source, batch, args.model, warnings)
            status = "created"
            detail = f"{len(batch.records)} records; {len(warnings)} evidence warnings"
        except Exception as error:
            status = "quota_stop" if is_quota_error(error) else "error"
            detail = str(error)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({
                "source_file": source["source_file"],
                "status": status,
                "detail": detail,
            }, ensure_ascii=False) + "\n")
        print(f"[{position}/{len(pending)}] {source['source_file']}: {status} ({detail})", flush=True)
        if status == "quota_stop":
            print("Stopping safely at the quota limit; completed outputs will be skipped on resume.", flush=True)
            break
        if position < len(pending):
            time.sleep(args.delay_seconds)


if __name__ == "__main__":
    main()
