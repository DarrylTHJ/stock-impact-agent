"""Search a full Chen transcript for stronger evidence for pending candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from transform_chen_transcripts import (
    TIMESTAMP_RANGE_PATTERN,
    format_transcript,
    materialise_evidence_quotes,
)


DRAFT_DIR = PROJECT_DIR / "data" / "chen_candidate_records"
INITIAL_VERIFICATION_DIR = PROJECT_DIR / "data" / "chen_initial_support_checks"
RECOVERED_DIR = PROJECT_DIR / "data" / "chen_recovered_evidence"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_DELAY_SECONDS = 45


class RecoveryFinding(BaseModel):
    candidate_id: str
    evidence_locations: list[str]
    note: str = ""


class RecoveryBatch(BaseModel):
    findings: list[RecoveryFinding]


def build_prompt(candidates: list[dict], transcript: str) -> str:
    return f"""
You are locating additional evidence in a complete timestamped Chen transcript.
You are not allowed to invent claims or rewrite them. For each candidate below,
find up to three timestamp ranges that explicitly support its exact target,
direction, and reason. A sector/company must be mentioned or unambiguously
described in the source. If no such evidence exists anywhere in the transcript,
return an empty evidence_locations array.

Use only timestamp ranges exactly as shown in the transcript: HH:MM:SS–HH:MM:SS.
Return JSON only as `{{"findings": [...]}}`, with one finding per candidate_id.

CANDIDATES:
{json.dumps(candidates, ensure_ascii=False)}

FULL TRANSCRIPT:
{transcript}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", action="append", help="Recover only this video ID; repeatable.")
    parser.add_argument(
        "--all-pending",
        action="store_true",
        help="Recover every video whose initial verification contains needs_more_evidence.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--delay-seconds", type=int, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.video_id and not args.all_pending:
        raise SystemExit("Provide --video-id or --all-pending.")

    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing from the local .env file.")
    RECOVERED_DIR.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=api_key)

    video_ids = list(args.video_id or [])
    if args.all_pending:
        for verification_path in sorted(INITIAL_VERIFICATION_DIR.glob("*.json")):
            verification_data = json.loads(verification_path.read_text(encoding="utf-8"))
            if any(item["status"] == "needs_more_evidence" for item in verification_data.get("decisions", [])):
                video_ids.append(verification_path.stem)
    video_ids = list(dict.fromkeys(video_ids))

    for position, video_id in enumerate(video_ids, start=1):
        output_path = RECOVERED_DIR / f"{video_id}.json"
        if output_path.exists() and not args.overwrite:
            print(f"{video_id}: already recovered; skipped", flush=True)
            continue
        draft_path = DRAFT_DIR / f"{video_id}.json"
        verification_path = INITIAL_VERIFICATION_DIR / f"{video_id}.json"
        transcript_path = PROJECT_DIR / "data" / "chen_source_captions" / f"{video_id}.json"
        if not all(path.exists() for path in (draft_path, verification_path, transcript_path)):
            raise SystemExit(f"{video_id}: raw draft, initial verification, or transcript is missing")

        draft = json.loads(draft_path.read_text(encoding="utf-8"))
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        pending_ids = {
            decision["knowledge_id"]
            for decision in verification["decisions"]
            if decision["status"] == "needs_more_evidence"
        }
        candidates = []
        for record in draft["records"]:
            if record["knowledge_id"] in pending_ids:
                candidates.append(
                    {
                        "candidate_id": record["knowledge_id"],
                        "target": record.get("impacted_companies") or record.get("impacted_industry") or record.get("impacted_sector"),
                        "direction": record.get("impact_direction"),
                        "reason": record["reason"],
                    }
                )
        findings: list[RecoveryFinding] = []
        if candidates:
            response = client.models.generate_content(
                model=args.model,
                contents=build_prompt(candidates, format_transcript(transcript["segments"])),
                config={"response_mime_type": "application/json", "temperature": 0},
            )
            response_payload = json.loads(response.text)
            # Gemini occasionally returns the inner list despite the requested
            # wrapper. Accept that equivalent shape, then validate it locally.
            if isinstance(response_payload, list):
                response_payload = {"findings": response_payload}
            findings = RecoveryBatch.model_validate(response_payload).findings
            if {item.candidate_id for item in findings} != {item["candidate_id"] for item in candidates}:
                raise ValueError(f"{video_id}: recovery response did not cover every pending candidate")

        by_id = {item.candidate_id: item for item in findings}
        recovered_records = []
        for record in draft["records"]:
            recovered = dict(record)
            finding = by_id.get(record["knowledge_id"])
            if finding:
                extra_evidence = [
                    {"quote": "pending source reconstruction", "translation": None, "location": location}
                    for location in finding.evidence_locations
                    if TIMESTAMP_RANGE_PATTERN.fullmatch(location)
                    and location not in {item["location"] for item in recovered["evidence"]}
                ]
                recovered["evidence"] = [*recovered["evidence"], *extra_evidence]
            recovered_records.append(recovered)
        recovered_payload = {"records": recovered_records}
        recovered_payload = materialise_evidence_quotes(recovered_payload, transcript["segments"])
        output_path.write_text(
            json.dumps(
                {
                    "source_video_id": video_id,
                    "recovered_at_utc": datetime.now(UTC).isoformat(),
                    "recovery_model": args.model,
                    "records": recovered_payload["records"],
                    "recovery_findings": [item.model_dump() for item in findings],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"{video_id}: recovered evidence for {len(findings)} candidate(s)", flush=True)
        if position < len(video_ids):
            time.sleep(args.delay_seconds)


if __name__ == "__main__":
    main()
