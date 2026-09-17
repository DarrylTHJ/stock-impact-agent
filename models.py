"""The controlled data contract for extracted source knowledge."""

from typing import Literal
from pydantic import BaseModel, Field, model_validator


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
    knowledge_type: Literal["sector_impact", "company_impact", "market_context"]
    trigger_event: str
    impacted_sector: str | None = None
    impacted_industry: str | None = Field(
        default=None,
        description="Optional source-named industry, such as Automotive or F&B.",
    )
    impact_direction: Literal["positive", "negative", "mixed"] | None = None
    reason: str
    impacted_companies: list[ImpactedCompany] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(min_length=1)
    embedding_summary: str
    source_name: Literal["Chen", "HLIB Research"]
    source_title: str
    source_date: str
    source_link: str | None = None
    source_video_id: str | None = None
    source_file: str | None = None
    source_category: str | None = None
    source_report_type: str | None = None
    extraction_model: str | None = None

    @model_validator(mode="after")
    def enforce_impact_scope(self) -> "KnowledgeRecord":
        if self.knowledge_type == "sector_impact":
            if self.impacted_sector not in BURSA_SECTORS or self.impact_direction is None:
                raise ValueError("sector_impact requires an official Bursa sector and impact direction")
        elif self.knowledge_type == "company_impact":
            if not self.impacted_companies or self.impact_direction is None:
                raise ValueError("company_impact requires an explicitly impacted company and direction")
        elif self.impacted_industry is not None:
            raise ValueError("market_context must not have an impacted industry")
        return self

    def validate_for_graph(self) -> bool:
        """Only a source-supported sector impact creates a sector graph edge."""
        return (
            self.knowledge_type == "sector_impact"
            and self.impacted_sector in BURSA_SECTORS
            and self.impact_direction is not None
        )

    def validate_for_company_graph(self) -> bool:
        """Company edges require a source-supported company impact."""
        return self.knowledge_type == "company_impact" and bool(self.impacted_companies)
