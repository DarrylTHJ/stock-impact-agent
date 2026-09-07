"""Run Chen ingestion from source captions through final verification safely.

The coordinator processes one video at a time and never overlaps Gemini calls.
It resumes from files already created in each named pipeline stage. It stops on
quota, network, or schema errors instead of retrying aggressively.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "data" / "chen_source_captions"
CANDIDATE_DIR = PROJECT_DIR / "data" / "chen_candidate_records"
INITIAL_DIR = PROJECT_DIR / "data" / "chen_initial_support_checks"
RECOVERED_DIR = PROJECT_DIR / "data" / "chen_recovered_evidence"
FINAL_DIR = PROJECT_DIR / "data" / "chen_final_verified_records"
LOG_DIR = PROJECT_DIR / "data" / "chen_pipeline_logs"
DEFAULT_DELAY_SECONDS = 45


def append_log(entry: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "coordinator_log.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps({"at_utc": datetime.now(UTC).isoformat(), **entry}, ensure_ascii=False) + "\n")


def is_service_error(output: str) -> bool:
    text = output.lower()
    return any(token in text for token in ("429", "quota", "resource_exhausted", "getaddrinfo", "connection", "timeout"))


def run_script(arguments: list[str], delay_seconds: int) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=PROJECT_DIR,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    append_log({"command": arguments, "return_code": result.returncode, "output": output[-4000:]})
    if result.returncode != 0 or is_service_error(output):
        return False, output
    # Only Gemini-calling scripts invoke this helper. Keep a conservative gap
    # before any next model request, even when a video has several stages.
    time.sleep(delay_seconds)
    return True, output


def initial_is_final(video_id: str) -> bool:
    data = json.loads((INITIAL_DIR / f"{video_id}.json").read_text(encoding="utf-8"))
    return not any(item["status"] == "needs_more_evidence" for item in data["decisions"])


def copy_initial_to_final(video_id: str) -> None:
    initial = json.loads((INITIAL_DIR / f"{video_id}.json").read_text(encoding="utf-8"))
    initial["finalised_at_utc"] = datetime.now(UTC).isoformat()
    initial["finalisation_route"] = "initial_support_check_approved_all_records"
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / f"{video_id}.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Process at most this many unfinished source videos.")
    parser.add_argument("--delay-seconds", type=int, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--video-id", action="append", help="Process one named video; repeatable.")
    args = parser.parse_args()

    source_paths = sorted(SOURCE_DIR.glob("*.json"))
    if args.video_id:
        wanted = set(args.video_id)
        source_paths = [path for path in source_paths if path.stem in wanted]
        missing = wanted - {path.stem for path in source_paths}
        if missing:
            raise SystemExit(f"Source caption not found: {', '.join(sorted(missing))}")
    unfinished = [path for path in source_paths if not (FINAL_DIR / path.name).exists()]
    if args.limit is not None:
        unfinished = unfinished[: args.limit]
    print(f"Unfinished videos in this run: {len(unfinished)}", flush=True)

    for position, source_path in enumerate(unfinished, start=1):
        video_id = source_path.stem
        candidate_path = CANDIDATE_DIR / source_path.name
        initial_path = INITIAL_DIR / source_path.name
        final_path = FINAL_DIR / source_path.name
        try:
            if not candidate_path.exists():
                ok, output = run_script(
                    ["scripts/transform_chen_transcripts.py", f"--video-id={video_id}"], args.delay_seconds
                )
                if not ok or not candidate_path.exists():
                    raise RuntimeError(output or "Candidate transformation did not produce a file")

            audit = subprocess.run(
                [sys.executable, "scripts/audit_chen_drafts.py", f"--video-id={video_id}"],
                cwd=PROJECT_DIR,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
            )
            audit_output = (audit.stdout + "\n" + audit.stderr).strip()
            append_log({"command": ["audit", video_id], "return_code": audit.returncode, "output": audit_output[-4000:]})
            if audit.returncode != 0:
                print(f"[{position}/{len(unfinished)}] {video_id}: audit_failed", flush=True)
                continue

            if not initial_path.exists():
                ok, output = run_script(
                    ["scripts/verify_chen_drafts.py", f"--video-id={video_id}"], args.delay_seconds
                )
                if not ok or not initial_path.exists():
                    raise RuntimeError(output or "Initial support check did not produce a file")

            if initial_is_final(video_id):
                copy_initial_to_final(video_id)
                print(f"[{position}/{len(unfinished)}] {video_id}: final_verified (no recovery needed)", flush=True)
                continue

            ok, output = run_script(
                ["scripts/recover_chen_evidence.py", f"--video-id={video_id}"], args.delay_seconds
            )
            if not ok or not (RECOVERED_DIR / source_path.name).exists():
                raise RuntimeError(output or "Evidence recovery did not produce a file")
            ok, output = run_script(
                [
                    "scripts/verify_chen_drafts.py",
                    "--final",
                    "--input-dir", "data/chen_recovered_evidence",
                    "--output-dir", "data/chen_final_verified_records",
                    f"--video-id={video_id}",
                ],
                args.delay_seconds,
            )
            if not ok or not final_path.exists():
                raise RuntimeError(output or "Final verification did not produce a file")
            print(f"[{position}/{len(unfinished)}] {video_id}: final_verified (after recovery)", flush=True)
        except RuntimeError as error:
            message = str(error)
            append_log({"video_id": video_id, "status": "stopped", "detail": message[-4000:]})
            print(f"[{position}/{len(unfinished)}] {video_id}: STOPPED - {message[-250:]}", flush=True)
            break


if __name__ == "__main__":
    main()
