"""Verify candidate Chen claims against their fixed, source-derived evidence.

This is a separate gate after extraction. It does not invent replacements: a
candidate becomes promotable only when its quoted evidence supports its target,
direction, and reason. Otherwise it is demoted to grounded market context or
rejected.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from audit_chen_drafts import audit_draft
from models import KnowledgeRecord


DRAFT_DIR = PROJECT_DIR / "data" / "chen_transformed_initial"
VERIFIED_DIR = PROJECT_DIR / "data" / "chen_transformed_validated"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_BATCH_SIZE = 12
DEFAULT_DELAY_SECONDS = 45


class VerificationFinding(BaseModel):
    candidate_id: str
    target_supported: bool
    direction_supported: bool
    reason_supported: bool
    market_context_supported: bool
    supported_company_names: list[str] = Field(default_factory=list)
    reviewer_note: str = ""


class VerificationBatch(BaseModel):
    findings: list[VerificationFinding]


def candidate_payload(candidate_id: str, record: dict) -> dict:
    return {
        "candidate_id": candidate_id,
        "knowledge_type": record["knowledge_type"],
        "trigger_event": record["trigger_event"],
        "impacted_sector": record.get("impacted_sector"),
        "impacted_industry": record.get("impacted_industry"),
        "impact_direction": record.get("impact_direction"),
        "reason": record["reason"],
        "impacted_companies": [company["company_name"] for company in record.get("impacted_companies", [])],
        "evidence": [
            {"location": item["location"], "quote": item["quote"]}
            for item in record["evidence"]
        ],
    }


def build_prompt(candidates: list[dict]) -> str:
    return f"""
You are a strict evidence verifier for an academic Bursa Malaysia event-impact
system. Use only the provided original-language evidence quotes. Do not use
outside financial knowledge and do not make plausible inference.

For every candidate, decide four booleans:
- target_supported: the quote explicitly supports the named company, named
  industry, or official Bursa sector as the impact target. Broad investor or
  global-market commentary does NOT support a sector target.
- direction_supported: the quote explicitly supports the stated positive,
  negative, or mixed direction for that exact target.
- reason_supported: the quote explicitly supports the stated causal reason;
  reject details not mentioned in the quote (for example, inferred inflation).
- market_context_supported: the quote explicitly supports broad market,
  investor, macroeconomic, or policy context even if it does not support the
  candidate's claimed target.
- supported_company_names: include only company names from the candidate list
  that are explicitly named and connected to the claim in the evidence. Return
  an empty list when none are supported.

A named industry such as Automotive may support that industry, but must not be
treated as proof that every company in its official Bursa main sector is
affected. A named company must appear in the evidence to support company impact.
For a candidate whose knowledge_type is market_context, do not demand a sector
or company target. Set market_context_supported and reason_supported based on
the quote; its target/direction booleans are not used by the local policy.

Return JSON only with `findings`. Include exactly one finding for every
candidate_id. reviewer_note must be short and factual.

CANDIDATES:
{json.dumps(candidates, ensure_ascii=False)}
"""


def verify_batch(client: genai.Client, model: str, candidates: list[dict]) -> VerificationBatch:
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(candidates),
        config={"response_mime_type": "application/json", "temperature": 0},
    )
    payload = json.loads(response.text)
    if isinstance(payload, list):
        payload = {"findings": payload}
    batch = VerificationBatch.model_validate(payload)
    expected_ids = {candidate["candidate_id"] for candidate in candidates}
    actual_ids = {finding.candidate_id for finding in batch.findings}
    if actual_ids != expected_ids:
        raise ValueError("Verifier response did not contain exactly one result per candidate")
    return batch


def apply_finding(record: dict, finding: VerificationFinding, final: bool) -> tuple[dict | None, str]:
    """Apply the conservative local approval policy to a verifier finding."""
    supported_names = set(finding.supported_company_names)
    sanitised = dict(record)
    sanitised["impacted_companies"] = [
        company
        for company in record.get("impacted_companies", [])
        if company["company_name"] in supported_names
    ]
    # A record extracted as market context never claims a sector/company target.
    # It is evaluated only on whether its broad context and explanation are
    # supported; evidence recovery is reserved for missed sector/company links.
    if sanitised["knowledge_type"] == "market_context":
        if finding.market_context_supported and finding.reason_supported:
            sanitised["impacted_companies"] = []
            KnowledgeRecord.model_validate(sanitised)
            return sanitised, "approved_market_context"
        return None, "rejected_market_context"
    if finding.target_supported and finding.direction_supported and finding.reason_supported:
        if sanitised["knowledge_type"] == "company_impact" and not sanitised["impacted_companies"]:
            return None, "rejected"
        KnowledgeRecord.model_validate(sanitised)
        return sanitised, "approved"
    if not final:
        return None, "needs_more_evidence"
    if finding.market_context_supported and finding.reason_supported:
        demoted = dict(sanitised)
        demoted.update(
            {
                "knowledge_type": "market_context",
                "impacted_sector": None,
                "impacted_industry": None,
                "impact_direction": None,
                "impacted_companies": [],
            }
        )
        KnowledgeRecord.model_validate(demoted)
        return demoted, "demoted_to_market_context"
    return None, "rejected"


def chunks(items: list[dict], size: int) -> list[list[dict]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", action="append", help="Verify only this video ID; repeatable.")
    parser.add_argument("--limit", type=int, help="Verify at most this many eligible videos.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--delay-seconds", type=int, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--input-dir", type=Path, default=DRAFT_DIR,
        help="Candidate drafts to verify. Use recovered drafts for final verification.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=VERIFIED_DIR,
        help="Directory for verification decisions and promotable records.",
    )
    parser.add_argument(
        "--final", action="store_true",
        help="Allow the second-pass verifier to demote to market context or reject.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing from the local .env file.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input_dir.glob("*.json"))
    if args.video_id:
        wanted = set(args.video_id)
        paths = [path for path in paths if path.stem in wanted]
        missing = wanted - {path.stem for path in paths}
        if missing:
            raise SystemExit(f"Draft not found: {', '.join(sorted(missing))}")
    if not args.overwrite:
        paths = [path for path in paths if not (args.output_dir / path.name).exists()]
    eligible: list[Path] = []
    for path in paths:
        findings = audit_draft(path)
        if findings:
            print(f"Skipping {path.stem}: source audit failed ({len(findings)} issue(s))", flush=True)
        else:
            eligible.append(path)
    if args.limit is not None:
        eligible = eligible[: args.limit]

    candidates: list[dict] = []
    source_records: dict[str, tuple[dict, list[dict]]] = {}
    for path in eligible:
        draft = json.loads(path.read_text(encoding="utf-8"))
        source_records[path.name] = (draft, draft["records"])
        for record_index, record in enumerate(draft["records"]):
            candidate_id = f"{path.name}:{record_index}"
            candidates.append(candidate_payload(candidate_id, record))

    print(f"Eligible videos: {len(eligible)}; candidate records: {len(candidates)}", flush=True)
    if not candidates:
        return
    client = genai.Client(api_key=api_key)
    outcomes: dict[str, tuple[VerificationFinding, str]] = {}
    for batch_index, batch_candidates in enumerate(chunks(candidates, args.batch_size), start=1):
        verification = verify_batch(client, args.model, batch_candidates)
        for finding in verification.findings:
            record = next(item for item in batch_candidates if item["candidate_id"] == finding.candidate_id)
            original = source_records[finding.candidate_id.split(":", 1)[0]][1][int(finding.candidate_id.rsplit(":", 1)[1])]
            _, status = apply_finding(original, finding, args.final)
            outcomes[finding.candidate_id] = (finding, status)
        print(f"Verification batch {batch_index}: {len(batch_candidates)} candidate(s) checked", flush=True)
        if batch_index * args.batch_size < len(candidates):
            time.sleep(args.delay_seconds)

    for filename, (draft, records) in source_records.items():
        verified_records: list[dict] = []
        decisions: list[dict] = []
        for record_index, record in enumerate(records):
            candidate_id = f"{filename}:{record_index}"
            finding, status = outcomes[candidate_id]
            promoted, _ = apply_finding(record, finding, args.final)
            if promoted is not None:
                verified_records.append(promoted)
            decisions.append({"knowledge_id": record["knowledge_id"], "status": status, **finding.model_dump()})
        (args.output_dir / filename).write_text(
            json.dumps(
                {
                    "source_video_id": draft["source_video_id"],
                    "verified_at_utc": datetime.now(UTC).isoformat(),
                    "verification_model": args.model,
                    "records": verified_records,
                    "decisions": decisions,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"Saved verification results for {len(source_records)} video(s).", flush=True)


if __name__ == "__main__":
    main()
