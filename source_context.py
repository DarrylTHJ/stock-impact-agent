"""Resolve selected knowledge records back to complete local Chen transcripts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from models import KnowledgeRecord


TRANSCRIPT_DIR = Path(__file__).parent / "data" / "chen_extracted"


@dataclass(frozen=True)
class ChenTranscript:
    video_id: str
    title: str
    url: str
    segments: list[dict]

    def as_prompt_text(self) -> str:
        lines = []
        for segment in self.segments:
            start = float(segment.get("start_seconds", 0))
            end = float(segment.get("end_seconds", start))
            text = str(segment.get("text_original", "")).strip()
            if text:
                lines.append(f"[{format_seconds(start)}-{format_seconds(end)}] {text}")
        return "\n".join(lines)

    def exact_text(self, start_seconds: float, end_seconds: float) -> str:
        selected = []
        for segment in self.segments:
            start = float(segment.get("start_seconds", 0))
            end = float(segment.get("end_seconds", start))
            if end >= start_seconds and start <= end_seconds:
                text = str(segment.get("text_original", "")).strip()
                if text:
                    selected.append(text)
        return " ".join(selected)

    def locate_exact_quote(self, quote: str) -> tuple[int, int] | None:
        """Find an exact quote and derive its timestamps from source segments.

        Gemini selects the quote, but it is never trusted to supply the final
        timestamps. Whitespace is ignored because Chinese caption services may
        split the same sentence differently across adjacent segments.
        """
        wanted = "".join(quote.split())
        if not wanted:
            return None

        transcript_text: list[str] = []
        character_segments: list[int] = []
        for segment_index, segment in enumerate(self.segments):
            segment_text = "".join(str(segment.get("text_original", "")).split())
            transcript_text.extend(segment_text)
            character_segments.extend([segment_index] * len(segment_text))

        joined = "".join(transcript_text)
        start_character = joined.find(wanted)
        if start_character < 0:
            return None
        end_character = start_character + len(wanted) - 1
        start_segment = self.segments[character_segments[start_character]]
        end_segment = self.segments[character_segments[end_character]]
        start_seconds = max(0, int(float(start_segment.get("start_seconds", 0))))
        end_seconds = max(
            start_seconds + 1,
            math.ceil(float(end_segment.get("end_seconds", start_seconds + 1))),
        )
        if end_seconds - start_seconds > 40:
            return None
        return start_seconds, end_seconds


def format_seconds(value: float) -> str:
    seconds = max(0, int(value))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def record_video_id(record: KnowledgeRecord) -> str | None:
    if record.source_video_id:
        return record.source_video_id
    if record.source_link:
        parsed = urlparse(record.source_link)
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        if video_id:
            return video_id
        if parsed.netloc.endswith("youtu.be"):
            return parsed.path.strip("/") or None
    return None


def load_direct_chen_transcripts(
    records: list[KnowledgeRecord],
) -> tuple[dict[str, ChenTranscript], list[str]]:
    """Load each unique complete transcript referenced by a Direct Chen record."""
    transcripts: dict[str, ChenTranscript] = {}
    warnings: list[str] = []
    for record in records:
        if record.source_name != "Chen":
            continue
        video_id = record_video_id(record)
        if not video_id or video_id in transcripts:
            if not video_id:
                warnings.append(f"{record.knowledge_id}: no source video ID")
            continue
        path = TRANSCRIPT_DIR / f"{video_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            segments = payload.get("segments") or []
            if not segments:
                raise ValueError("transcript has no segments")
            transcripts[video_id] = ChenTranscript(
                video_id=video_id,
                title=payload.get("video_title") or record.source_title,
                url=payload.get("video_url") or record.source_link or "",
                segments=segments,
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            warnings.append(f"{record.knowledge_id}: {error}")
    return transcripts, warnings


def timestamped_video_url(url: str, start_seconds: float) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}t={max(0, int(start_seconds))}s"
