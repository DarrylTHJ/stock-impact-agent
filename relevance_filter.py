"""LLM review of Chroma candidates before any result becomes a graph edge."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

from knowledge_store import RetrievalCandidate


class RelevanceDecision(BaseModel):
    knowledge_id: str
    verdict: Literal["direct", "applicable_rule", "general_background", "irrelevant"]
    explanation: str = Field(
        min_length=8,
        description="Concise explanation tied to the event and candidate record.",
    )


class RelevanceResponse(BaseModel):
    decisions: list[RelevanceDecision]


@dataclass(frozen=True)
class ReviewedCandidate:
    candidate: RetrievalCandidate
    verdict: str
    explanation: str

    @property
    def selected(self) -> bool:
        return self.verdict != "irrelevant"

    @property
    def direct(self) -> bool:
        return self.verdict == "direct"

    @property
    def applicable_rule(self) -> bool:
        return self.verdict == "applicable_rule"


@dataclass(frozen=True)
class RelevanceReview:
    candidates: list[ReviewedCandidate]
    used_fallback: bool = False
    status_message: str | None = None


def _candidate_payload(candidate: RetrievalCandidate) -> dict:
    record = candidate.record
    target = record.impacted_sector
    if record.impacted_companies:
        target = ", ".join(company.company_name for company in record.impacted_companies)
    evidence_preview = ""
    if record.evidence:
        item = record.evidence[0]
        evidence_preview = (item.translation or item.quote)[:500]
    return {
        "knowledge_id": record.knowledge_id,
        "source": record.source_name,
        "knowledge_type": record.knowledge_type,
        "trigger_event": record.trigger_event,
        "target": target,
        "impact_direction": record.impact_direction,
        "reason": record.reason,
        "embedding_summary": record.embedding_summary,
        "evidence_preview": evidence_preview,
        "vector_similarity": round(candidate.similarity, 3),
        "matched_queries": list(candidate.matched_queries),
    }


def review_relevance(
    event_text: str,
    event_summary: str,
    retrieval_queries: list[str],
    candidates: list[RetrievalCandidate],
) -> RelevanceReview:
    """Review both sources in one Gemini call and return every decision."""
    if not candidates:
        return RelevanceReview([])

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY2")
    if not api_key:
        return _fallback_review(candidates, "Gemini relevance review was unavailable: API key missing.")

    prompt = f"""
You are the relevance gate for an evidence-grounded Bursa Malaysia event-impact system.

User event:
{event_text[:10000]}

Normalised event summary:
{event_summary}

Retrieval phrases:
{json.dumps(retrieval_queries, ensure_ascii=False)}

Review EVERY candidate below exactly once. Do not invent knowledge IDs, impacts,
companies, evidence, or facts. Judge whether the stored causal rule can genuinely
help analyse this specific event—not whether it merely shares words.

Verdicts:
- direct: the source record explicitly discusses this same event, announcement,
  policy, company development, or materially identical occurrence.
- applicable_rule: the source does not discuss this exact occurrence, but it states
  a clear cause-to-impact rule whose cause is explicitly present in the user event.
  Use this only when the entity/domain and causal mechanism match and no important
  unstated assumption is needed. Example: event says Petronas increased capex, while
  the record says increased Petronas capex creates OGSE project opportunities.
- general_background: genuinely helpful context for understanding the event, but
  it must not determine direction, severity, the unified conclusion, or graph edges.
- irrelevant: lexical similarity, a different policy/company/cause, or too remote
  to help the reader understand this event.

Direct must be rare and event-specific. Applicable_rule must also be high precision:
the event must explicitly satisfy the stored rule's trigger. Mere topic similarity,
historical correlation, or a plausible missing link is general_background or
irrelevant. Complete transcripts are loaded only for Direct Chen records.

Company-specific records are irrelevant unless the event names that company or
clearly concerns it. Be strict. For every verdict, give one concise user-facing
explanation describing the actual match or mismatch. This is a decision trace,
not hidden chain-of-thought.

Candidates:
{json.dumps([_candidate_payload(c) for c in candidates], ensure_ascii=False, indent=2)}
"""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": RelevanceResponse,
                "temperature": 0,
            },
        )
        parsed = RelevanceResponse.model_validate_json(response.text)
        allowed_ids = {candidate.record.knowledge_id for candidate in candidates}
        by_id = {
            decision.knowledge_id: decision
            for decision in parsed.decisions
            if decision.knowledge_id in allowed_ids
        }
        reviewed = []
        for candidate in candidates:
            decision = by_id.get(candidate.record.knowledge_id)
            if decision is None:
                reviewed.append(
                    ReviewedCandidate(
                        candidate,
                        "irrelevant",
                        "Rejected because the relevance review did not return a valid decision for this record.",
                    )
                )
            else:
                reviewed.append(
                    ReviewedCandidate(candidate, decision.verdict, decision.explanation)
                )
        return RelevanceReview(reviewed)
    except Exception as error:
        return _fallback_review(
            candidates,
            f"Gemini relevance review was unavailable ({type(error).__name__}: {error}); semantic threshold fallback used.",
        )


def _fallback_review(
    candidates: list[RetrievalCandidate], status_message: str
) -> RelevanceReview:
    reviewed = [
        ReviewedCandidate(
            candidate,
            "irrelevant",
            "Not selected because the LLM relevance review was unavailable.",
        )
        for candidate in candidates
    ]
    return RelevanceReview(reviewed, used_fallback=True, status_message=status_message)
