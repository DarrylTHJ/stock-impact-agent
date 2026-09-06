"""Read and filter local evidence records.

Vector retrieval will be added after the ingestion format is confirmed. Keeping
this layer separate means the UI never needs to know where records came from.
"""

import json
import re
from pathlib import Path

from models import KnowledgeRecord


DATA_FILE = Path(__file__).parent / "sample_data" / "knowledge_records.json"


def load_records() -> list[KnowledgeRecord]:
    raw_records = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    return [KnowledgeRecord.model_validate(record) for record in raw_records]


def retrieve_records(search_queries: list[str], source_name: str, limit: int = 5) -> list[KnowledgeRecord]:
    """Temporary transparent keyword scorer; replace with Chroma retrieval next."""
    query_words = set(re.findall(r"[a-z]+", " ".join(search_queries).lower()))
    scored: list[tuple[int, KnowledgeRecord]] = []

    for record in load_records():
        if record.source_name != source_name:
            continue
        searchable = f"{record.trigger_event} {record.embedding_summary}".lower()
        score = len(query_words.intersection(re.findall(r"[a-z]+", searchable)))
        if score:
            scored.append((score, record))

    return [record for _, record in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]
