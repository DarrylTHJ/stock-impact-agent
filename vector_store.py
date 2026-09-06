"""ChromaDB semantic index for knowledge-record search summaries.

Chroma holds vectors and light filtering metadata only. The complete evidence
record remains in the local JSON source of truth and is loaded by knowledge_id.
"""

import os
from functools import lru_cache
from pathlib import Path

# The model is downloaded once during setup; subsequent app starts use the
# local cache instead of making an unnecessary network request.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

from models import KnowledgeRecord


DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "knowledge_records"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_COSINE_DISTANCE = 0.80



class LocalEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(list(input), normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def get_embedding_function() -> LocalEmbeddingFunction:
    """Keep the local model in memory for the duration of the app process."""
    return LocalEmbeddingFunction()


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def index_records(records: list[KnowledgeRecord]) -> None:
    """Upsert keeps the semantic index current when local knowledge changes."""
    collection = get_collection()
    collection.upsert(
        ids=[record.knowledge_id for record in records],
        documents=[record.embedding_summary for record in records],
        metadatas=[
            {
                "knowledge_id": record.knowledge_id,
                "source_name": record.source_name,
                "knowledge_type": record.knowledge_type,
            }
            for record in records
        ],
    )


def search_knowledge_ids(
    search_queries: list[str], source_name: str, limit: int = 5
) -> list[str]:
    collection = get_collection()
    results = collection.query(
        query_texts=search_queries,
        n_results=limit,
        where={"source_name": source_name},
        include=["distances"],
    )

    closest_distances: dict[str, float] = {}
    for id_group, distance_group in zip(results.get("ids", []), results.get("distances", [])):
        for knowledge_id, distance in zip(id_group or [], distance_group or []):
            if distance <= MAX_COSINE_DISTANCE:
                closest_distances[knowledge_id] = min(
                    distance, closest_distances.get(knowledge_id, float("inf"))
                )

    return [
        knowledge_id
        for knowledge_id, _ in sorted(closest_distances.items(), key=lambda item: item[1])[:limit]
    ]
