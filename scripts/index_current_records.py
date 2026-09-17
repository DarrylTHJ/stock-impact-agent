"""Build the current test corpus and ChromaDB index from both initial folders.

This is an explicit offline Load step. The Streamlit query path never modifies
the vector database. Initial records are suitable for current RAG testing and
can later be replaced with fully validated records using the same command.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from knowledge_store import LOCAL_DATA_FILE
from models import KnowledgeRecord
from vector_store import rebuild_index


INPUT_DIRS = (
    PROJECT_DIR / "data" / "chen_transformed_initial",
    PROJECT_DIR / "data" / "hlib_transformed_initial",
)


def main() -> None:
    records_by_id: dict[str, KnowledgeRecord] = {}
    skipped_files: list[str] = []
    source_files = 0

    for input_dir in INPUT_DIRS:
        for path in sorted(input_dir.glob("*.json")):
            source_files += 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                for raw_record in payload["records"]:
                    raw_record = dict(raw_record)
                    if payload.get("source_video_id"):
                        raw_record.setdefault("source_video_id", payload["source_video_id"])
                    if payload.get("source_file"):
                        raw_record.setdefault("source_file", payload["source_file"])
                    if payload.get("source_category"):
                        raw_record.setdefault("source_category", payload["source_category"])
                    if payload.get("source_report_type"):
                        raw_record.setdefault("source_report_type", payload["source_report_type"])
                    record = KnowledgeRecord.model_validate(raw_record)
                    records_by_id[record.knowledge_id] = record
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as error:
                skipped_files.append(f"{path.name}: {error}")

    records = list(records_by_id.values())
    serialized = [record.model_dump(mode="json") for record in records]
    LOCAL_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = LOCAL_DATA_FILE.with_suffix(".json.tmp")
    temporary_file.write_text(
        json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_file.replace(LOCAL_DATA_FILE)
    rebuild_index(records)

    counts: dict[str, int] = {}
    for record in records:
        counts[record.source_name] = counts.get(record.source_name, 0) + 1
    print(f"Rebuilt corpus and ChromaDB from {source_files} source file(s).")
    for source_name, count in sorted(counts.items()):
        print(f"  {source_name}: {count} record(s)")
    print(f"  Total: {len(records)} record(s)")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} invalid source file(s):")
        print("\n".join(skipped_files))


if __name__ == "__main__":
    main()
