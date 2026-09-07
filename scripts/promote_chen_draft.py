"""Promote reviewed Chen extraction drafts into the local ChromaDB knowledge store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from knowledge_store import LOCAL_DATA_FILE, load_records
from models import KnowledgeRecord
from vector_store import index_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True, help="Video ID of an already reviewed draft.")
    args = parser.parse_args()

    draft_path = PROJECT_DIR / "data" / "chen_final_verified_records" / f"{args.video_id}.json"
    if not draft_path.exists():
        raise SystemExit(
            f"Final verified draft not found: {draft_path}. Run the initial verification, "
            "evidence recovery, and final verification steps first."
        )
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    approved = [KnowledgeRecord.model_validate(record).model_dump(mode="json") for record in draft["records"]]

    existing: list[dict] = []
    if LOCAL_DATA_FILE.exists():
        existing = json.loads(LOCAL_DATA_FILE.read_text(encoding="utf-8"))
    approved_ids = {record["knowledge_id"] for record in approved}
    merged = [record for record in existing if record.get("knowledge_id") not in approved_ids]
    merged.extend(approved)
    LOCAL_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA_FILE.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    # The app also indexes on retrieval, but indexing here makes the promoted
    # records available immediately for a direct local test.
    index_records(load_records())
    print(f"Promoted and indexed {len(approved)} record(s) from {args.video_id}.")


if __name__ == "__main__":
    main()
