"""Create one evidence-grounded answer per source from supported records."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

from models import BURSA_SECTORS, KnowledgeRecord
from source_context import ChenTranscript, record_video_id


class TranscriptEvidenceRange(BaseModel):
    video_id: str
    # Gemini supplies only approximate hints here. The trusted timestamps are
    # recalculated later by locating quote_original in the saved transcript, so
    # imperfect hints must not invalidate the complete synthesis response.
    start_seconds: int = 0
    end_seconds: int = 0
    quote_original: str = Field(
        min_length=2,
        description=(
            "Exact continuous original-language text copied from the supplied "
            "timestamped transcript."
        ),
    )
    translation: str

class UnifiedCompany(BaseModel):
    company_name: str
    ticker: str | None = None


class UnifiedSectorImpact(BaseModel):
    impacted_sector: str
    impacted_industry: str | None = None
    impact_direction: Literal["positive", "negative", "mixed"]
    support_level: Literal["direct", "source_grounded_inference"] = "direct"
    conclusion: str
    causal_steps: list[str] = Field(min_length=1, max_length=4)
    impacted_companies: list[UnifiedCompany] = Field(default_factory=list)
    supporting_knowledge_ids: list[str] = Field(default_factory=list)
    transcript_evidence: list[TranscriptEvidenceRange] = Field(default_factory=list)


class UnifiedCompanyImpact(BaseModel):
    company_name: str
    ticker: str | None = None
    impact_direction: Literal["positive", "negative", "mixed"]
    support_level: Literal["direct", "source_grounded_inference"] = "direct"
    conclusion: str
    causal_steps: list[str] = Field(min_length=1, max_length=4)
    supporting_knowledge_ids: list[str] = Field(default_factory=list)
    transcript_evidence: list[TranscriptEvidenceRange] = Field(default_factory=list)


class UnifiedMarketContext(BaseModel):
    summary: str
    supporting_knowledge_ids: list[str] = Field(default_factory=list)


class SourceAnalysis(BaseModel):
    source_name: Literal["Chen", "HLIB Research"]
    no_supported_conclusion: bool
    source_summary: str
    sector_impacts: list[UnifiedSectorImpact] = Field(default_factory=list)
    company_impacts: list[UnifiedCompanyImpact] = Field(default_factory=list)
    market_context: list[UnifiedMarketContext] = Field(default_factory=list)


class UnifiedSynthesisResponse(BaseModel):
    analyses: list[SourceAnalysis]


@dataclass(frozen=True)
class SynthesisResult:
    analyses: dict[str, SourceAnalysis]
    used_fallback: bool = False
    status_message: str | None = None


def _record_payload(record: KnowledgeRecord) -> dict:
    return {
        "knowledge_id": record.knowledge_id,
        "source_name": record.source_name,
        "knowledge_type": record.knowledge_type,
        "trigger_event": record.trigger_event,
        "impacted_sector": record.impacted_sector,
        "impacted_industry": record.impacted_industry,
        "impact_direction": record.impact_direction,
        "reason": record.reason,
        "impacted_companies": [company.model_dump() for company in record.impacted_companies],
        "existing_evidence": [item.model_dump() for item in record.evidence],
        "source_title": record.source_title,
        "source_date": record.source_date,
        "source_video_id": record_video_id(record),
        "source_file": record.source_file,
    }


def synthesize_impacts(
    event_text: str,
    event_summary: str,
    direct_records: list[KnowledgeRecord],
    applicable_rule_records: list[KnowledgeRecord],
    background_records: list[KnowledgeRecord],
    transcripts: dict[str, ChenTranscript],
) -> SynthesisResult:
    """Use all Direct evidence and full Direct Chen transcripts in one final call."""
    supported_records = direct_records + applicable_rule_records
    if not supported_records:
        return _empty_synthesis()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY2")
    if not api_key:
        return _fallback_synthesis(
            supported_records,
            {record.knowledge_id for record in direct_records},
            "Gemini synthesis unavailable: API key missing.",
        )

    transcript_payload = [
        {
            "video_id": transcript.video_id,
            "title": transcript.title,
            "complete_timestamped_transcript": transcript.as_prompt_text(),
        }
        for transcript in transcripts.values()
    ]
    background_payload = [
        {
            "knowledge_id": record.knowledge_id,
            "source": record.source_name,
            "knowledge_type": record.knowledge_type,
            "trigger_event": record.trigger_event,
            "target": record.impacted_sector,
            "direction": record.impact_direction,
            "reason": record.reason,
        }
        for record in background_records
    ]

    prompt = f"""
You produce the final evidence-grounded result for a Bursa Malaysia event-impact
research application. Analyse Chen and HLIB Research independently. Never merge
their evidence into one source conclusion.

USER EVENT:
{event_text[:12000]}

NORMALISED EVENT:
{event_summary}

DIRECT KNOWLEDGE RECORDS:
{json.dumps([_record_payload(record) for record in direct_records], ensure_ascii=False, indent=2)}

APPLICABLE CAUSAL RULES:
{json.dumps([_record_payload(record) for record in applicable_rule_records], ensure_ascii=False, indent=2)}

GENERAL BACKGROUND (context only; forbidden from determining graph impacts):
{json.dumps(background_payload, ensure_ascii=False, indent=2)}

COMPLETE TRANSCRIPTS FOR DIRECT CHEN VIDEOS ONLY:
{json.dumps(transcript_payload, ensure_ascii=False)}

Rules:
1. Produce separate Chen and HLIB Research analyses. Always return both sources.
2. Use all applicable Direct records and Applicable causal rules. Consolidate duplicate
   sector records into ONE result per official Bursa sector and duplicate company
   records into ONE result per company. Official sectors: {json.dumps(BURSA_SECTORS)}.
   An Applicable rule is a transparent inference: the submitted event supplies the
   trigger and the source supplies the trigger-to-impact rule. Do not claim that the
   source discussed the submitted occurrence itself.
3. The graph causal_steps must explain this user event specifically. Do not simply
   repeat stored record wording.
4. Use mixed only when supported evidence contains both positive and negative effects.
   Uncertainty or dependency alone is not automatically mixed; describe conditions.
5. General background may be mentioned outside the main conclusion later, but must
   not create or change a sector/company impact. Put only genuinely useful background
   into market_context with its supplied knowledge ID.
6. You may recover a missing detail or sector impact from a complete Direct transcript
   only when the speaker explicitly connects it to this event. Cite an exact video_id
   and copy exact original Chinese. Never invent original Chinese quote text.
7. For every Chen sector impact, return the strongest short transcript ranges that
   collectively support its causal chain, plus faithful English translations. Each
   range should align with the supplied transcript lines. Copy an exact continuous
   piece of the original Chinese into quote_original.
   Do not paraphrase, translate, correct, or modernise quote_original. If you cannot
   copy matching original text, omit that evidence range. The application will locate
   quote_original in the saved transcript and calculate the final timestamp itself;
   your start/end values are only approximate hints.
8. supporting_knowledge_ids must contain only supplied Direct or Applicable-rule IDs. A transcript-only
   recovered impact may have no knowledge ID but must have transcript evidence.
9. Do not predict stock prices or add outside knowledge.
10. A company_impact record may create only a company result, never a sector-wide
    result unless a separate supplied sector_impact record supports it.
11. If a source lacks both Direct evidence and Applicable rules, return
    no_supported_conclusion=true, empty sector_impacts and company_impacts lists,
    and plainly state that the source does not support a directional conclusion.
"""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": UnifiedSynthesisResponse,
                "temperature": 0,
            },
        )
        parsed = UnifiedSynthesisResponse.model_validate_json(response.text)
        analyses = _validated_analyses(
            parsed,
            direct_records,
            applicable_rule_records,
            background_records,
            transcripts,
        )
        return SynthesisResult(analyses)
    except Exception as error:
        return _fallback_synthesis(
            supported_records,
            {record.knowledge_id for record in direct_records},
            f"Gemini synthesis unavailable ({type(error).__name__}: {error}).",
        )


def _validated_analyses(
    response: UnifiedSynthesisResponse,
    direct_records: list[KnowledgeRecord],
    applicable_rule_records: list[KnowledgeRecord],
    background_records: list[KnowledgeRecord],
    transcripts: dict[str, ChenTranscript],
) -> dict[str, SourceAnalysis]:
    allowed_ids = {
        record.knowledge_id: record
        for record in direct_records + applicable_rule_records + background_records
    }
    directional_ids = {
        record.knowledge_id for record in direct_records + applicable_rule_records
    }
    direct_ids = {record.knowledge_id for record in direct_records}
    analyses: dict[str, SourceAnalysis] = {}
    for analysis in response.analyses:
        valid_impacts = []
        for impact in analysis.sector_impacts:
            if impact.impacted_sector not in BURSA_SECTORS:
                continue
            impact.supporting_knowledge_ids = [
                knowledge_id
                for knowledge_id in impact.supporting_knowledge_ids
                if knowledge_id in allowed_ids
                and knowledge_id in directional_ids
                and allowed_ids[knowledge_id].source_name == analysis.source_name
            ]
            verified_transcript_evidence = []
            for item in impact.transcript_evidence:
                transcript = transcripts.get(item.video_id)
                if transcript is None:
                    continue
                located_range = transcript.locate_exact_quote(item.quote_original)
                if located_range is None:
                    continue
                item.start_seconds, item.end_seconds = located_range
                verified_transcript_evidence.append(item)
            impact.transcript_evidence = verified_transcript_evidence
            if impact.supporting_knowledge_ids or impact.transcript_evidence:
                impact.support_level = (
                    "direct"
                    if impact.transcript_evidence
                    or any(item in direct_ids for item in impact.supporting_knowledge_ids)
                    else "source_grounded_inference"
                )
                valid_impacts.append(impact)
        analysis.sector_impacts = valid_impacts

        valid_company_impacts = []
        for impact in analysis.company_impacts:
            impact.supporting_knowledge_ids = [
                knowledge_id
                for knowledge_id in impact.supporting_knowledge_ids
                if knowledge_id in allowed_ids
                and knowledge_id in directional_ids
                and allowed_ids[knowledge_id].source_name == analysis.source_name
                and allowed_ids[knowledge_id].knowledge_type == "company_impact"
                and any(
                    company.company_name.casefold() == impact.company_name.casefold()
                    for company in allowed_ids[knowledge_id].impacted_companies
                )
            ]
            verified_transcript_evidence = []
            for item in impact.transcript_evidence:
                transcript = transcripts.get(item.video_id)
                if transcript is None:
                    continue
                located_range = transcript.locate_exact_quote(item.quote_original)
                if located_range is None:
                    continue
                item.start_seconds, item.end_seconds = located_range
                verified_transcript_evidence.append(item)
            impact.transcript_evidence = verified_transcript_evidence
            if impact.supporting_knowledge_ids or impact.transcript_evidence:
                impact.support_level = (
                    "direct"
                    if impact.transcript_evidence
                    or any(item in direct_ids for item in impact.supporting_knowledge_ids)
                    else "source_grounded_inference"
                )
                valid_company_impacts.append(impact)
        analysis.company_impacts = valid_company_impacts

        valid_context = []
        for context in analysis.market_context:
            context.supporting_knowledge_ids = [
                knowledge_id
                for knowledge_id in context.supporting_knowledge_ids
                if knowledge_id in allowed_ids
                and allowed_ids[knowledge_id].source_name == analysis.source_name
            ]
            if context.supporting_knowledge_ids:
                valid_context.append(context)
        analysis.market_context = valid_context
        analysis.no_supported_conclusion = not bool(valid_impacts or valid_company_impacts)
        analyses[analysis.source_name] = analysis

    for source in ("Chen", "HLIB Research"):
        analyses.setdefault(
            source,
            SourceAnalysis(
                source_name=source,
                no_supported_conclusion=True,
                source_summary=f"No {source} evidence or applicable causal rule supports a conclusion for this event.",
                sector_impacts=[],
            ),
        )
    return analyses


def _empty_synthesis() -> SynthesisResult:
    return SynthesisResult(
        analyses={
            source: SourceAnalysis(
                source_name=source,
                no_supported_conclusion=True,
                source_summary=(
                    f"No {source} evidence or applicable causal rule supports a conclusion for this event."
                ),
                sector_impacts=[],
            )
            for source in ("Chen", "HLIB Research")
        }
    )


def _fallback_synthesis(
    supported_records: list[KnowledgeRecord],
    direct_ids: set[str],
    status_message: str,
) -> SynthesisResult:
    analyses: dict[str, SourceAnalysis] = {}
    for source in ("Chen", "HLIB Research"):
        source_records = [record for record in supported_records if record.source_name == source]
        grouped: dict[str, list[KnowledgeRecord]] = {}
        for record in source_records:
            if record.validate_for_graph() and record.impacted_sector:
                grouped.setdefault(record.impacted_sector, []).append(record)
        impacts = []
        company_impacts = []
        for sector, records in grouped.items():
            directions = {record.impact_direction for record in records}
            direction = next(iter(directions)) if len(directions) == 1 else "mixed"
            impacts.append(
                UnifiedSectorImpact(
                    impacted_sector=sector,
                    impacted_industry=next(
                        (record.impacted_industry for record in records if record.impacted_industry),
                        None,
                    ),
                    impact_direction=direction,
                    support_level=(
                        "direct"
                        if any(record.knowledge_id in direct_ids for record in records)
                        else "source_grounded_inference"
                    ),
                    conclusion=" ".join(dict.fromkeys(record.reason for record in records)),
                    causal_steps=list(dict.fromkeys(record.reason for record in records))[:4],
                    impacted_companies=[
                        UnifiedCompany(company_name=company.company_name, ticker=company.ticker)
                        for record in records
                        for company in record.impacted_companies
                    ],
                    supporting_knowledge_ids=[record.knowledge_id for record in records],
                )
            )
        for record in source_records:
            if not record.validate_for_company_graph():
                continue
            for company in record.impacted_companies:
                company_impacts.append(
                    UnifiedCompanyImpact(
                        company_name=company.company_name,
                        ticker=company.ticker,
                        impact_direction=record.impact_direction or "mixed",
                        support_level=(
                            "direct"
                            if record.knowledge_id in direct_ids
                            else "source_grounded_inference"
                        ),
                        conclusion=record.reason,
                        causal_steps=[record.reason],
                        supporting_knowledge_ids=[record.knowledge_id],
                    )
                )
        analyses[source] = SourceAnalysis(
            source_name=source,
            no_supported_conclusion=not bool(impacts or company_impacts),
            source_summary=(
                f"{len(impacts)} sector conclusion(s) assembled from supported records without final LLM synthesis."
                if impacts
                else f"No {source} evidence or applicable causal rule supports a conclusion for this event."
            ),
            sector_impacts=impacts,
            company_impacts=company_impacts,
        )
    return SynthesisResult(analyses, used_fallback=True, status_message=status_message)
