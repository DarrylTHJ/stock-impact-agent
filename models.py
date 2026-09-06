"""The controlled data contract for extracted source knowledge."""

from typing import Literal
from pydantic import BaseModel, Field, HttpUrl


BURSA_SECTORS = [
    "Construction", "Consumer Products & Services", "Energy",
    "Financial Services", "Healthcare", "Industrial Products & Services",
    "Plantation", "Property", "REITs", "Technology",
    "Telecommunications & Media", "Transportation & Logistics", "Utilities",
]


class EvidenceItem(BaseModel):
    quote: str = Field(min_length=1)
    translation: str | None = None
    location: str = Field(description="YouTube timestamp range or PDF page reference")


class ImpactedCompany(BaseModel):
    company_name: str
    ticker: str | None = None
    evidence_indexes: list[int] = Field(default_factory=list)


class KnowledgeRecord(BaseModel):
    knowledge_id: str
    knowledge_type: Literal["sector_impact", "market_context"]
    trigger_event: str
    impacted_sector: str | None = None
    impact_direction: Literal["positive", "negative", "mixed"] | None = None
    reason: str
    impacted_companies: list[ImpactedCompany] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(min_length=1)
    embedding_summary: str
    source_name: Literal["Chen", "HLIB Research"]
    source_title: str
    source_date: str
    source_link: str | None = None
    extraction_model: str | None = None

    def validate_for_graph(self) -> bool:
        """Market context is useful narrative, but cannot create a graph edge."""
        return (
            self.knowledge_type == "sector_impact"
            and self.impacted_sector in BURSA_SECTORS
            and self.impact_direction is not None
        )
