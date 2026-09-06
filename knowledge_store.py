"""Read and filter local evidence records.

Vector retrieval will be added after the ingestion format is confirmed. Keeping
this layer separate means the UI never needs to know where records came from.
"""

import json
from pathlib import Path

from models import KnowledgeRecord
from vector_store import index_records, search_knowledge_ids


SAMPLE_DATA_FILE = Path(__file__).parent / "sample_data" / "knowledge_records.json"
LOCAL_DATA_FILE = Path(__file__).parent / "data" / "knowledge_records.json"


def load_records() -> list[KnowledgeRecord]:
    raw_records = json.loads(SAMPLE_DATA_FILE.read_text(encoding="utf-8"))
    if LOCAL_DATA_FILE.exists():
        raw_records.extend(json.loads(LOCAL_DATA_FILE.read_text(encoding="utf-8")))
    by_id = {record["knowledge_id"]: record for record in raw_records}
    return [KnowledgeRecord.model_validate(record) for record in by_id.values()]


def retrieve_records(search_queries: list[str], source_name: str, limit: int = 5) -> list[KnowledgeRecord]:
    """Find semantically similar evidence records through ChromaDB."""
    records = load_records()
    index_records(records)
    by_id = {record.knowledge_id: record for record in records}
    knowledge_ids = search_knowledge_ids(search_queries, source_name, limit)
    return [by_id[knowledge_id] for knowledge_id in knowledge_ids if knowledge_id in by_id]
