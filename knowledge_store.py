"""Load the local source of truth and retrieve records from its vector index."""

import json
from dataclasses import dataclass
from pathlib import Path

from models import KnowledgeRecord
from vector_store import search_knowledge_ids, search_knowledge_matches


LOCAL_DATA_FILE = Path(__file__).parent / "data" / "knowledge_records.json"


@dataclass(frozen=True)
class RetrievalCandidate:
    record: KnowledgeRecord
    cosine_distance: float
    matched_queries: tuple[str, ...] = ()

    @property
    def similarity(self) -> float:
        return max(0.0, min(1.0, 1.0 - self.cosine_distance))


def load_records() -> list[KnowledgeRecord]:
    raw_records: list[dict] = []
    if LOCAL_DATA_FILE.exists():
        raw_records = json.loads(LOCAL_DATA_FILE.read_text(encoding="utf-8"))
    by_id = {record["knowledge_id"]: record for record in raw_records}
    return [KnowledgeRecord.model_validate(record) for record in by_id.values()]


def retrieve_records(search_queries: list[str], source_name: str, limit: int = 10) -> list[KnowledgeRecord]:
    """Find semantically similar evidence records through ChromaDB."""
    records = load_records()
    by_id = {record.knowledge_id: record for record in records}
    knowledge_ids = search_knowledge_ids(search_queries, source_name, limit)
    return [by_id[knowledge_id] for knowledge_id in knowledge_ids if knowledge_id in by_id]


def retrieve_candidates(
    search_queries: list[str], source_name: str, limit: int = 10
) -> list[RetrievalCandidate]:
    """Return the nearest candidates for an LLM relevance review.

    No distance threshold is applied here. Weak semantic matches must remain
    visible to the review stage so it can explicitly reject and explain them.
    """
    records = load_records()
    by_id = {record.knowledge_id: record for record in records}
    matches = search_knowledge_matches(
        search_queries, source_name, limit, apply_distance_threshold=False
    )
    return [
        RetrievalCandidate(by_id[knowledge_id], distance, tuple(search_queries))
        for knowledge_id, distance in matches
        if knowledge_id in by_id
    ]


def retrieve_diverse_candidates(
    search_queries: list[str],
    source_name: str,
    *,
    per_query_limit: int = 4,
    total_limit: int = 20,
) -> list[RetrievalCandidate]:
    """Retrieve across each event theme, then deduplicate and rank candidates."""
    records = load_records()
    by_id = {record.knowledge_id: record for record in records}
    best_distance: dict[str, float] = {}
    query_matches: dict[str, list[str]] = {}

    unique_queries = dict.fromkeys(query.strip() for query in search_queries if query.strip())
    for query in unique_queries:
        matches = search_knowledge_matches(
            [query], source_name, per_query_limit, apply_distance_threshold=False
        )
        for knowledge_id, distance in matches:
            if knowledge_id not in by_id:
                continue
            best_distance[knowledge_id] = min(
                distance, best_distance.get(knowledge_id, float("inf"))
            )
            query_matches.setdefault(knowledge_id, []).append(query)

    ordered_ids = sorted(best_distance, key=best_distance.get)[:total_limit]
    return [
        RetrievalCandidate(
            by_id[knowledge_id],
            best_distance[knowledge_id],
            tuple(dict.fromkeys(query_matches[knowledge_id])),
        )
        for knowledge_id in ordered_ids
    ]
