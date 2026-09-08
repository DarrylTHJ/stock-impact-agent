"""Build a temporary Chen test index from the initial transformation folder.

This is for UI/retrieval testing only. Re-run it at any time: it replaces only
Chen records in the local knowledge JSON and ChromaDB, leaving HLIB records
alone. Final validated data can later replace this temporary test index.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from knowledge_store import LOCAL_DATA_FILE
from models import KnowledgeRecord
from vector_store import get_collection, index_records

INITIAL_DIR = PROJECT_DIR / "data" / "chen_transformed_initial"


def main() -> None:
    records: list[dict] = []
    skipped: list[str] = []
    for path in sorted(INITIAL_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.extend(
                KnowledgeRecord.model_validate(record).model_dump(mode="json")
                for record in payload["records"]
            )
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as error:
            skipped.append(f"{path.name}: {error}")

    existing: list[dict] = []
    if LOCAL_DATA_FILE.exists():
        existing = json.loads(LOCAL_DATA_FILE.read_text(encoding="utf-8"))
    # Initial Chen output is intentionally temporary; replace only that source.
    merged = [record for record in existing if record.get("source_name") != "Chen"]
    merged.extend(records)
    LOCAL_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA_FILE.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    # Remove stale Chen vectors before adding the current snapshot. HLIB vectors
    # remain untouched.
    collection = get_collection()
    collection.delete(where={"source_name": "Chen"})
    if records:
        index_records([KnowledgeRecord.model_validate(record) for record in records])

    print(f"Indexed {len(records)} Chen record(s) from {len(list(INITIAL_DIR.glob('*.json')))} initial file(s).")
    if skipped:
        print(f"Skipped {len(skipped)} unreadable/incomplete file(s):")
        print("\n".join(skipped))


if __name__ == "__main__":
    main()
