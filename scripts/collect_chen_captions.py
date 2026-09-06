"""Collect publicly available timestamped captions from Alfred Chen's YouTube channel.

This collector does not use an LLM and does not create knowledge records.  Its
output is the auditable source material used by the later ingestion stage.
Only one preferred caption track is saved for each video. Videos without a
usable Chinese or English caption track are recorded in the collection log.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import yt_dlp


DEFAULT_CHANNEL_URL = "https://www.youtube.com/@AlfredChenOfficial/videos"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "chen_transcripts"
REQUEST_TIMEOUT_SECONDS = 30
PAUSE_BETWEEN_VIDEOS_SECONDS = 1

# Prefer Chinese because it is the language of the original source. English is
# a fallback only when a video has no Chinese captions.
LANGUAGE_PRIORITY = (
    "zh-Hans",
    "zh-Hant",
    "zh-CN",
    "zh-TW",
    "zh",
    "cmn",
    "yue",
    "en",
    "en-US",
    "en-GB",
)
FORMAT_PRIORITY = ("json3", "vtt")


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", "", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_json3(payload: str) -> list[dict[str, Any]]:
    data = json.loads(payload)
    segments: list[dict[str, Any]] = []
    for event in data.get("events", []):
        text = clean_text("".join(part.get("utf8", "") for part in event.get("segs", [])))
        if not text:
            continue
        start_ms = event.get("tStartMs")
        if start_ms is None:
            continue
        duration_ms = event.get("dDurationMs", 0)
        segments.append(
            {
                "start_seconds": round(start_ms / 1000, 3),
                "end_seconds": round((start_ms + duration_ms) / 1000, 3),
                "text_original": text,
            }
        )
    return segments


def timestamp_to_seconds(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_vtt(payload: str) -> list[dict[str, Any]]:
    lines = payload.replace("\r\n", "\n").split("\n")
    segments: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if "-->" not in line:
            index += 1
            continue
        start, end = (part.strip().split(" ")[0] for part in line.split("-->", 1))
        index += 1
        text_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            text_lines.append(lines[index])
            index += 1
        text = clean_text(" ".join(text_lines))
        if text:
            segments.append(
                {
                    "start_seconds": round(timestamp_to_seconds(start), 3),
                    "end_seconds": round(timestamp_to_seconds(end), 3),
                    "text_original": text,
                }
            )
        index += 1
    return segments


def choose_caption_track(info: dict[str, Any], include_automatic: bool) -> tuple[str, str, str, bool] | None:
    sources: list[tuple[dict[str, Any], bool]] = [(info.get("subtitles") or {}, False)]
    if include_automatic:
        sources.append((info.get("automatic_captions") or {}, True))

    for tracks, is_automatic in sources:
        for language in LANGUAGE_PRIORITY:
            for track in tracks.get(language, []):
                if track.get("ext") in FORMAT_PRIORITY and track.get("url"):
                    return language, track["ext"], track["url"], is_automatic
    return None


def append_log(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def collect_video(
    video_url: str,
    output_dir: Path,
    session: requests.Session,
    include_automatic: bool,
) -> tuple[str, str]:
    options = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(video_url, download=False)

    video_id = info["id"]
    output_path = output_dir / f"{video_id}.json"
    if output_path.exists():
        return video_id, "already_collected"

    selected = choose_caption_track(info, include_automatic)
    if not selected:
        return video_id, "no_supported_caption"

    language, extension, caption_url, is_automatic = selected
    response = session.get(caption_url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    segments = parse_json3(response.text) if extension == "json3" else parse_vtt(response.text)
    if not segments:
        return video_id, "empty_caption"

    record = {
        "video_id": video_id,
        "video_title": info.get("title"),
        "video_url": info.get("webpage_url") or video_url,
        "published_date": info.get("upload_date"),
        "channel_name": info.get("channel") or info.get("uploader"),
        "caption_language": language,
        "caption_is_automatic": is_automatic,
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "segments": segments,
    }
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return video_id, "collected"


def iter_video_urls(channel_url: str, limit: int | None) -> list[str]:
    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": limit,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        playlist = downloader.extract_info(channel_url, download=False)
    return [entry["url"] for entry in playlist.get("entries", []) if entry and entry.get("url")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-url", default=DEFAULT_CHANNEL_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, help="For a small test run, collect at most this many videos.")
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="Exclude automatic captions. By default they are collected when no manual track exists.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "collection_log.jsonl"
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (compatible; FYPResearchCollector/1.0)"

    video_urls = iter_video_urls(args.channel_url, args.limit)
    print(f"Found {len(video_urls)} video(s). Saving captions to {output_dir}")
    counts: dict[str, int] = {}
    for position, video_url in enumerate(video_urls, start=1):
        try:
            video_id, status = collect_video(video_url, output_dir, session, not args.manual_only)
        except Exception as error:  # Keep a failed video from ending the whole batch.
            video_id, status = "unknown", "error"
            error_text = str(error)
        else:
            error_text = None

        counts[status] = counts.get(status, 0) + 1
        append_log(
            log_path,
            {
                "checked_at_utc": datetime.now(UTC).isoformat(),
                "position": position,
                "video_url": video_url,
                "video_id": video_id,
                "status": status,
                "error": error_text,
            },
        )
        print(f"[{position}/{len(video_urls)}] {video_id}: {status}")
        time.sleep(PAUSE_BETWEEN_VIDEOS_SECONDS)

    print("Completed:", ", ".join(f"{status}={count}" for status, count in sorted(counts.items())))


if __name__ == "__main__":
    main()
