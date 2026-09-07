"""Run Chen ingestion through final verification with batched checks.

Extraction and full-transcript recovery stay per video because each uses one
complete source. Initial and final support checks are batched across videos to
reduce Gemini calls. Only pipeline version 3 outputs are final/promotable.
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
SOURCE_DIR = PROJECT_DIR / "data" / "chen_extracted"
CANDIDATE_DIR = PROJECT_DIR / "data" / "chen_transformed_initial"
INITIAL_DIR = PROJECT_DIR / "data" / "chen_transformed_validated"
RECOVERED_DIR = PROJECT_DIR / "data" / "chen_transformed_revamped"
FINAL_DIR = PROJECT_DIR / "data" / "chen_transformed_final"
LOG_DIR = PROJECT_DIR / "data" / "chen_pipeline_logs"
DEFAULT_VIDEO_BATCH_SIZE = 6
DEFAULT_DELAY_SECONDS = 45


def append_log(entry: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "coordinator_log.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps({"at_utc": datetime.now(UTC).isoformat(), **entry}, ensure_ascii=False) + "\n")


def service_error(output: str) -> bool:
    text = output.lower()
    return any(token in text for token in ("429", "quota", "resource_exhausted", "getaddrinfo", "connection", "timeout"))


def run(arguments: list[str], delay_seconds: int, gemini: bool) -> tuple[bool, str]:
    result = subprocess.run([sys.executable, *arguments], cwd=PROJECT_DIR, text=True, encoding="utf-8", errors="replace", capture_output=True)
    output = (result.stdout + "\n" + result.stderr).strip()
    append_log({"command": arguments, "return_code": result.returncode, "output": output[-4000:]})
    if result.returncode != 0 or (gemini and service_error(output)):
        return False, output
    if gemini:
        time.sleep(delay_seconds)
    return True, output


def is_current(path: Path) -> bool:
    return path.exists()


def flags(ids: list[str]) -> list[str]:
    return [f"--video-id={video_id}" for video_id in ids]


def decisions(directory: Path, video_id: str) -> list[dict]:
    return json.loads((directory / f"{video_id}.json").read_text(encoding="utf-8"))["decisions"]


def copy_initial_final(video_id: str) -> None:
    data = json.loads((INITIAL_DIR / f"{video_id}.json").read_text(encoding="utf-8"))
    data.update({"finalised_at_utc": datetime.now(UTC).isoformat(), "finalisation_route": "initial_support_check_approved_all_records"})
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / f"{video_id}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--video-batch-size", type=int, default=DEFAULT_VIDEO_BATCH_SIZE)
    parser.add_argument("--delay-seconds", type=int, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--video-id", action="append")
    args = parser.parse_args()

    sources = sorted(SOURCE_DIR.glob("*.json"))
    if args.video_id:
        wanted = set(args.video_id)
        sources = [path for path in sources if path.stem in wanted]
    pending = [path for path in sources if not is_current(FINAL_DIR / path.name)]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"Unfinished videos: {len(pending)}", flush=True)

    for start in range(0, len(pending), args.video_batch_size):
        batch = pending[start : start + args.video_batch_size]
        ids = [path.stem for path in batch]
        print(f"Batch {start // args.video_batch_size + 1}: {', '.join(ids)}", flush=True)
        try:
            eligible: list[str] = []
            for source in batch:
                video_id = source.stem
                candidate = CANDIDATE_DIR / source.name
                if not is_current(candidate):
                    ok, output = run(["scripts/transform_chen_transcripts.py", "--overwrite", f"--video-id={video_id}"], args.delay_seconds, True)
                    if not ok or not is_current(candidate):
                        raise RuntimeError(f"{video_id}: transformation stopped: {output[-300:]}")
                ok, _ = run(["scripts/audit_chen_drafts.py", f"--video-id={video_id}"], args.delay_seconds, False)
                if ok:
                    eligible.append(video_id)
                else:
                    print(f"{video_id}: audit_failed", flush=True)

            initial_ids = [video_id for video_id in eligible if not is_current(INITIAL_DIR / f"{video_id}.json")]
            if initial_ids:
                ok, output = run(["scripts/verify_chen_drafts.py", "--overwrite", *flags(initial_ids)], args.delay_seconds, True)
                if not ok:
                    raise RuntimeError(f"initial support check stopped: {output[-300:]}")

            recovery_ids: list[str] = []
            for video_id in eligible:
                if not is_current(INITIAL_DIR / f"{video_id}.json"):
                    raise RuntimeError(f"{video_id}: no current initial support output")
                if any(item["status"] == "needs_more_evidence" for item in decisions(INITIAL_DIR, video_id)):
                    recovery_ids.append(video_id)
                else:
                    copy_initial_final(video_id)
                    print(f"{video_id}: final_verified (initial check)", flush=True)

            for video_id in recovery_ids:
                recovered = RECOVERED_DIR / f"{video_id}.json"
                if not is_current(recovered):
                    ok, output = run(["scripts/recover_chen_evidence.py", "--overwrite", f"--video-id={video_id}"], args.delay_seconds, True)
                    if not ok or not is_current(recovered):
                        raise RuntimeError(f"{video_id}: recovery stopped: {output[-300:]}")

            if recovery_ids:
                ok, output = run(["scripts/verify_chen_drafts.py", "--final", "--overwrite", "--input-dir", "data/chen_transformed_revamped", "--output-dir", "data/chen_transformed_final", *flags(recovery_ids)], args.delay_seconds, True)
                if not ok:
                    raise RuntimeError(f"final verification stopped: {output[-300:]}")
                for video_id in recovery_ids:
                    if not is_current(FINAL_DIR / f"{video_id}.json"):
                        raise RuntimeError(f"{video_id}: no current final verification output")
                    print(f"{video_id}: final_verified (after recovery)", flush=True)
        except RuntimeError as error:
            append_log({"status": "stopped", "detail": str(error)})
            print(f"STOPPED: {error}", flush=True)
            break


if __name__ == "__main__":
    main()
