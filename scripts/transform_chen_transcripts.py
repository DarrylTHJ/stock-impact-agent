"""Create reviewable, evidence-grounded Chen knowledge-record drafts with Gemini.

Run a tiny pilot first. This script deliberately processes only one transcript
at a time and defaults to a conservative delay. It stops on quota/rate-limit
signals rather than cycling models rapidly or repeatedly retrying the API.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field, model_validator

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from models import BURSA_SECTORS, EvidenceItem, ImpactedCompany


DEFAULT_INPUT_DIR = PROJECT_DIR / "data" / "chen_transcripts"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "chen_extracted_drafts"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_DELAY_SECONDS = 45
MAX_TRANSCRIPT_CHARACTERS = 70_000
TIMESTAMP_RANGE_PATTERN = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2})–(?P<end>\d{2}:\d{2}:\d{2})$"
)


class DraftRecord(BaseModel):
    knowledge_type: Literal["sector_impact", "company_impact", "market_context"]
    trigger_event: str
    impacted_sector: str | None = None
    impacted_industry: str | None = None
    impact_direction: Literal["positive", "negative", "mixed"] | None = None
    reason: str
    impacted_companies: list[ImpactedCompany] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(min_length=1)
    embedding_summary: str

    @model_validator(mode="after")
    def enforce_impact_scope(self) -> "DraftRecord":
        if self.knowledge_type == "sector_impact":
            if self.impacted_sector not in BURSA_SECTORS or self.impact_direction is None:
                raise ValueError("sector_impact requires an official Bursa sector and direction")
        elif self.knowledge_type == "company_impact":
            if self.impacted_sector is not None:
                raise ValueError("company_impact must not imply a sector-wide impact")
            if not self.impacted_companies or self.impact_direction is None:
                raise ValueError("company_impact requires an impacted company and direction")
        elif self.impacted_sector is not None or self.impact_direction is not None:
            raise ValueError("market_context must not have a sector or direction")
        elif self.impacted_industry is not None:
            raise ValueError("market_context must not have an impacted industry")
        return self


class ExtractionBatch(BaseModel):
    records: list[DraftRecord] = Field(default_factory=list, max_length=15)


def format_seconds(value: float) -> str:
    minutes, seconds = divmod(int(value), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_timestamp(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def format_transcript(segments: list[dict]) -> str:
    text = "\n".join(
        f"[{format_seconds(segment['start_seconds'])}–{format_seconds(segment['end_seconds'])}] "
        f"{segment['text_original']}"
        for segment in segments
    )
    if len(text) > MAX_TRANSCRIPT_CHARACTERS:
        raise ValueError(
            f"Transcript is {len(text):,} characters; it exceeds the safe one-request limit "
            f"of {MAX_TRANSCRIPT_CHARACTERS:,} and needs a separate chunked review workflow."
        )
    return text


def build_prompt(transcript: str, title: str) -> str:
    sectors = "; ".join(BURSA_SECTORS)
    return f"""
You are extracting auditable investment-commentary knowledge from one Alfred
Chen YouTube transcript for a Bursa Malaysia event-impact research system.

Video title: {title}

Extract only claims that are explicitly supported by this transcript. Do not
use outside knowledge, do not forecast prices, and do not infer a sector merely
because it sounds plausible. Consolidate duplicate discussion into one record
per event–sector impact. It is acceptable to return no records.

For a sector_impact record:
- impacted_sector must be exactly one of: {sectors}
- impact_direction must be positive, negative, or mixed
- use sector_impact only when the transcript explicitly links the event to the
  performance, demand, costs, outlook, or valuation of that Bursa-relevant
  sector, or to a company in that sector. Mentioning a commodity, technology,
  or broad market alone is not enough to assign a Bursa sector.
- a discussion of energy prices, commodity prices, or an input cost is not by
  itself an effect on the Bursa Energy sector. The speaker must explicitly
  connect that discussion to affected energy-sector businesses or companies.
- reason must be a concise English causal mechanism grounded in the transcript
- impacted_industry is optional. Use it only when the source clearly identifies
  a more precise industry, such as Automotive or F&B. Do not invent one.
- evidence must contain one or more exact original-language excerpts and their
  timestamp range in HH:MM:SS–HH:MM:SS format. Use multiple excerpts when the
  claim relies on multiple parts of the video.
- the timestamp range is mandatory and must be copied from the timestamp labels
  in the supplied transcript. Never use a different format.
- include impacted_companies only if the speaker explicitly names and connects
  them to this impact; evidence_indexes are zero-based indexes into evidence.
- embedding_summary must be one concise English retrieval sentence.

For a company_impact record:
- use this, rather than sector_impact, if the speaker's positive/negative claim
  is specifically about one or more named companies or an IPO.
- impacted_sector must be null unless the speaker independently makes a
  sector-wide claim. A company's Bursa classification does not prove the whole
  sector has the same impact.
- impacted_industry may be included when explicitly stated by the source.
- impacted_companies must contain the explicitly named affected company or
  companies, and impact_direction must be present.

For broad market commentary that cannot responsibly be assigned to one official
Bursa sector, use market_context with impacted_sector and impact_direction null.
Do not turn global-market commentary into a sector record. Keep independent
events separate: do not combine two events into one trigger unless the speaker
explicitly says one causes the other.
Return only the required JSON structure. Do not provide chain-of-thought.
The JSON must have exactly one top-level key, `records`, containing an array of
objects with: knowledge_type, trigger_event, impacted_sector,
impacted_industry, impact_direction, reason, impacted_companies, evidence, and
embedding_summary.
Each evidence item has quote, translation (or null), and location. Each company
item has company_name, ticker (or null), and evidence_indexes.

Timestamped transcript:
{transcript}
"""


def is_quota_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(token in text for token in ("429", "resource_exhausted", "quota", "rate limit"))


def normalise_scope_fields(payload: dict) -> dict:
    """Prevent non-sector records from accidentally creating sector graph edges.

    Gemini may identify a company classification while extracting a
    company-specific claim. That classification is useful later, but it is not
    evidence that an entire sector is affected, so it is not retained here.
    """
    for record in payload.get("records", []):
        if record.get("impacted_companies") is None:
            record["impacted_companies"] = []
        if record.get("knowledge_type") == "company_impact":
            record["impacted_sector"] = None
        elif record.get("knowledge_type") == "market_context":
            record["impacted_sector"] = None
            record["impacted_industry"] = None
            record["impact_direction"] = None
    return payload


def materialise_evidence_quotes(payload: dict, segments: list[dict]) -> dict:
    """Replace LLM-written quotes with text copied directly from source captions.

    The LLM selects an evidence time range, but it can add punctuation or
    accidentally paraphrase Chinese text. The user-facing quote must instead be
    reconstructed from the original caption segments in that exact interval.
    """
    for record in payload.get("records", []):
        for evidence in record.get("evidence") or []:
            match = TIMESTAMP_RANGE_PATTERN.fullmatch(evidence.get("location") or "")
            if not match:
                raise ValueError(
                    "Evidence location must use the exact HH:MM:SS–HH:MM:SS timestamp format"
                )
            start = parse_timestamp(match.group("start"))
            end = parse_timestamp(match.group("end"))
            matching_segments = [
                segment["text_original"]
                for segment in segments
                if segment["end_seconds"] >= start and segment["start_seconds"] <= end
            ]
            if not matching_segments:
                raise ValueError(f"No source caption segments found for evidence range {evidence['location']}")
            evidence["quote"] = " ".join(matching_segments)
    return payload


def append_log(path: Path, entry: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def extract_one(client: genai.Client, model: str, source: dict) -> ExtractionBatch:
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(format_transcript(source["segments"]), source["video_title"]),
        config={
            "response_mime_type": "application/json",
            "temperature": 0,
        },
    )
    payload = normalise_scope_fields(json.loads(response.text))
    return ExtractionBatch.model_validate(materialise_evidence_quotes(payload, source["segments"]))


def save_draft(output_path: Path, source: dict, batch: ExtractionBatch, model: str) -> None:
    records = []
    for index, draft in enumerate(batch.records, start=1):
        record = draft.model_dump()
        record.update(
            {
                "knowledge_id": f"chen_{source['video_id']}_{index:02d}",
                "source_name": "Chen",
                "source_title": source["video_title"],
                "source_date": source.get("published_date") or "unknown",
                "source_link": source["video_url"],
                "extraction_model": model,
            }
        )
        records.append(record)
    output_path.write_text(
        json.dumps(
            {
                "source_video_id": source["video_id"],
                "source_caption_language": source["caption_language"],
                "source_caption_is_automatic": source["caption_is_automatic"],
                "transformed_at_utc": datetime.now(UTC).isoformat(),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, help="Maximum new transcripts to transform in this run.")
    parser.add_argument(
        "--video-id",
        action="append",
        help="Transform only this video ID. Repeat this option for a focused review batch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate drafts that already exist. Use after changing the extraction prompt.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=int,
        default=DEFAULT_DELAY_SECONDS,
        help="Wait between API requests. Keep this conservative on a free-tier key.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing from the local .env file.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "transformation_log.jsonl"
    source_paths = sorted(args.input_dir.glob("*.json"))
    if args.video_id:
        wanted_ids = set(args.video_id)
        source_paths = [path for path in source_paths if path.stem in wanted_ids]
        missing_ids = wanted_ids - {path.stem for path in source_paths}
        if missing_ids:
            raise SystemExit(f"No collected transcript found for: {', '.join(sorted(missing_ids))}")
    pending = [
        path
        for path in source_paths
        if args.overwrite or not (args.output_dir / path.name).exists()
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"Pending transcripts in this run: {len(pending)}", flush=True)
    client = genai.Client(api_key=api_key)
    for position, source_path in enumerate(pending, start=1):
        output_path = args.output_dir / source_path.name
        source = json.loads(source_path.read_text(encoding="utf-8"))
        try:
            batch = extract_one(client, args.model, source)
            save_draft(output_path, source, batch, args.model)
            status = "draft_created"
            detail = f"{len(batch.records)} record(s)"
        except Exception as error:
            status = "quota_stop" if is_quota_error(error) else "error"
            detail = str(error)

        append_log(
            log_path,
            {
                "checked_at_utc": datetime.now(UTC).isoformat(),
                "source_file": source_path.name,
                "video_id": source.get("video_id"),
                "model": args.model,
                "status": status,
                "detail": detail,
            },
        )
        print(f"[{position}/{len(pending)}] {source['video_id']}: {status} ({detail})", flush=True)
        if status == "quota_stop":
            print("Stopping safely after a quota/rate-limit response. Resume later; completed drafts are skipped.", flush=True)
            break
        if position < len(pending):
            time.sleep(args.delay_seconds)


if __name__ == "__main__":
    main()
