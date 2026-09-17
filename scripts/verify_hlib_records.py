"""Validate initial HLIB records against their fixed PDF evidence excerpts."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models import KnowledgeRecord


INITIAL_DIR = PROJECT_DIR / "data" / "hlib_transformed_initial"
EXTRACTED_DIR = PROJECT_DIR / "data" / "hlib_extracted"
VALIDATED_DIR = PROJECT_DIR / "data" / "hlib_transformed_validated"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_BATCH_SIZE = 12
DEFAULT_DELAY_SECONDS = 45
PAGE_PATTERN = re.compile(r"^Page (?P<number>\d+)$")


class VerificationFinding(BaseModel):
    candidate_id: str
    trigger_supported: bool
    causal_link_supported: bool
    target_supported: bool
    direction_supported: bool
    reason_supported: bool
    market_context_supported: bool
    supported_company_names: list[str] = Field(default_factory=list)
    reviewer_note: str = ""


class VerificationBatch(BaseModel):
    findings: list[VerificationFinding]


def source_audit_issues(record: dict, source: dict) -> list[str]:
    issues: list[str] = []
    try:
        KnowledgeRecord.model_validate(record)
    except Exception as error:
        return [f"invalid schema: {error}"]
    pages = {page["page_number"]: page["text"] for page in source["pages"]}
    for index, evidence in enumerate(record["evidence"], start=1):
        match = PAGE_PATTERN.fullmatch(evidence["location"])
        if not match:
            issues.append(f"evidence {index} has invalid page format")
            continue
        if int(match.group("number")) not in pages:
            issues.append(f"evidence {index} cites a page that does not exist")
    return issues


def candidate_payload(candidate_id: str, record: dict, source: dict) -> dict:
    pages = {page["page_number"]: page["text"] for page in source["pages"]}
    cited_numbers = sorted({
        int(PAGE_PATTERN.fullmatch(item["location"]).group("number"))
        for item in record["evidence"]
    })
    return {
        "candidate_id": candidate_id,
        "knowledge_type": record["knowledge_type"],
        "trigger_event": record["trigger_event"],
        "impacted_sector": record.get("impacted_sector"),
        "impacted_industry": record.get("impacted_industry"),
        "impact_direction": record.get("impact_direction"),
        "reason": record["reason"],
        "impacted_companies": [
            company["company_name"] for company in record.get("impacted_companies", [])
        ],
        "cited_pages": [
            {"page_number": number, "page_text": pages[number]}
            for number in cited_numbers
        ],
    }


def build_prompt(candidates: list[dict]) -> str:
    return f"""
You are a strict evidence verifier for an academic Bursa Malaysia event-impact
system. Use only the full text of the cited HLIB Research PDF pages supplied
with each candidate. Do not use outside knowledge and do not fill gaps with
plausible financial inference.

For every candidate decide these Boolean fields using only true or false:
- trigger_supported: the cited pages explicitly support the stated trigger event
- causal_link_supported: the cited pages explicitly connect that trigger to the
  claimed impact; separate mentions of an event and an outcome are insufficient
- target_supported: the cited pages explicitly support the named company,
  industry, or sector as the affected target; a narrower named industry can
  support its deterministically mapped Bursa main sector
- direction_supported: true when the pages support the candidate's stated
  positive, negative, or mixed direction for that exact target; return a
  Boolean, never repeat the direction label; `mixed` needs both positive and
  negative mechanisms
- reason_supported: all material causal details in the reason are supported
- market_context_supported: the pages support the stated broad market or
  macroeconomic context when the record type is market_context
- supported_company_names: retain only candidate company names explicitly
  named and connected to the impact on the cited pages; a phrase such as
  `larger producers` or `domestic developers` does not support inferred names

For an index-composition claim, direction may mean increased or decreased index
weight if the reason states that clearly. A rating or top-pick statement can
support a company only when the excerpt also states why it is affected.

Return JSON only with a `findings` array and exactly one finding for each
candidate_id. Every finding must use this structure:
{{"candidate_id": "...", "trigger_supported": true,
"causal_link_supported": true, "target_supported": true,
"direction_supported": true,
"reason_supported": true, "market_context_supported": false,
"supported_company_names": [], "reviewer_note": "..."}}

CANDIDATES:
{json.dumps(candidates, ensure_ascii=False)}
"""


def verify_batch(
    client: genai.Client, model: str, candidates: list[dict]
) -> VerificationBatch:
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(candidates),
        config={"response_mime_type": "application/json", "temperature": 0},
    )
    payload = json.loads(response.text)
    if isinstance(payload, list):
        payload = {"findings": payload}
    batch = VerificationBatch.model_validate(payload)
    expected = {item["candidate_id"] for item in candidates}
    actual = {item.candidate_id for item in batch.findings}
    if expected != actual:
        raise ValueError("Verifier did not return exactly one finding per candidate")
    return batch


def apply_finding(
    record: dict, finding: VerificationFinding
) -> tuple[dict | None, str]:
    cleaned = dict(record)
    supported = set(finding.supported_company_names)
    cleaned["impacted_companies"] = [
        company
        for company in record.get("impacted_companies", [])
        if company["company_name"] in supported
    ]
    if cleaned["knowledge_type"] == "market_context":
        approved = (
            finding.trigger_supported
            and finding.causal_link_supported
            and finding.market_context_supported
            and finding.reason_supported
        )
    else:
        approved = (
            finding.trigger_supported
            and finding.causal_link_supported
            and finding.target_supported
            and finding.direction_supported
            and finding.reason_supported
        )
        if cleaned["knowledge_type"] == "company_impact":
            approved = approved and bool(cleaned["impacted_companies"])
    if not approved:
        return None, "needs_more_evidence"
    KnowledgeRecord.model_validate(cleaned)
    return cleaned, "approved"


def chunks(items: list[dict], size: int) -> list[list[dict]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", action="append", help="Initial JSON filename or stem; repeatable.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--delay-seconds", type=int, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing from the local .env file.")
    VALIDATED_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(INITIAL_DIR.glob("*.json"))
    if args.file:
        wanted = {Path(name).stem for name in args.file}
        paths = [path for path in paths if path.stem in wanted]
        missing = wanted - {path.stem for path in paths}
        if missing:
            raise SystemExit(f"Initial HLIB file not found: {', '.join(sorted(missing))}")
    if not args.overwrite:
        paths = [path for path in paths if not (VALIDATED_DIR / path.name).exists()]

    source_files: dict[str, tuple[dict, list[dict]]] = {}
    candidates: list[dict] = []
    audit_decisions: dict[str, dict] = {}
    for path in paths:
        initial = json.loads(path.read_text(encoding="utf-8"))
        source_path = EXTRACTED_DIR / path.name
        if not source_path.exists():
            raise SystemExit(f"Extracted HLIB source missing: {source_path.name}")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source_files[path.name] = (initial, initial["records"])
        for record_index, record in enumerate(initial["records"]):
            candidate_id = f"{path.name}:{record_index}"
            issues = source_audit_issues(record, source)
            if issues:
                audit_decisions[candidate_id] = {
                    "knowledge_id": record["knowledge_id"],
                    "status": "needs_more_evidence",
                    "reviewer_note": "; ".join(issues),
                    "validation_stage": "local_source_audit",
                }
            else:
                candidates.append(candidate_payload(candidate_id, record, source))

    print(
        f"HLIB files: {len(paths)}; LLM candidates: {len(candidates)}; "
        f"local evidence failures: {len(audit_decisions)}",
        flush=True,
    )
    client = genai.Client(api_key=api_key)
    findings: dict[str, VerificationFinding] = {}
    batches = chunks(candidates, args.batch_size)
    for batch_index, batch_candidates in enumerate(batches, start=1):
        result = verify_batch(client, args.model, batch_candidates)
        findings.update({item.candidate_id: item for item in result.findings})
        print(f"HLIB verification batch {batch_index}: {len(batch_candidates)} records", flush=True)
        if batch_index < len(batches):
            time.sleep(args.delay_seconds)

    for filename, (initial, records) in source_files.items():
        approved_records: list[dict] = []
        decisions: list[dict] = []
        for record_index, record in enumerate(records):
            candidate_id = f"{filename}:{record_index}"
            if candidate_id in audit_decisions:
                decisions.append(audit_decisions[candidate_id])
                continue
            finding = findings[candidate_id]
            approved, status = apply_finding(record, finding)
            if approved is not None:
                approved_records.append(approved)
            decisions.append({
                "knowledge_id": record["knowledge_id"],
                "status": status,
                "validation_stage": "llm_evidence_validation",
                **finding.model_dump(),
            })
        (VALIDATED_DIR / filename).write_text(
            json.dumps({
                "source_file": initial["source_file"],
                "validated_at_utc": datetime.now(UTC).isoformat(),
                "validation_model": args.model,
                "records": approved_records,
                "decisions": decisions,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"Saved HLIB validation results for {len(source_files)} file(s).", flush=True)


if __name__ == "__main__":
    main()
