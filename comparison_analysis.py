"""Compare the two source verdicts without changing or extending them."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

from impact_synthesis import SourceAnalysis


class ComparisonResponse(BaseModel):
    overall_summary: str
    agreements: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    evidence_comparison: str


@dataclass(frozen=True)
class ComparisonResult:
    comparison: ComparisonResponse
    used_fallback: bool = False
    status_message: str | None = None


def compare_source_results(
    event_summary: str, analyses: dict[str, SourceAnalysis]
) -> ComparisonResult:
    """Use only the two validated source outputs for a descriptive comparison."""
    fallback = _fallback_comparison(analyses)
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY2")
    if not api_key:
        return ComparisonResult(
            fallback,
            used_fallback=True,
            status_message="Gemini comparison unavailable: API key missing.",
        )

    source_payload = {
        source: analysis.model_dump(mode="json") for source, analysis in analyses.items()
    }
    prompt = f"""
You compare two already-validated interpretations of one Bursa Malaysia event.
Do not perform a new investment analysis. Do not add any sector, company,
direction, causal mechanism, fact, or outside knowledge that is absent below.
Do not decide that one source is correct.

EVENT:
{event_summary}

VALIDATED SOURCE RESULTS:
{json.dumps(source_payload, ensure_ascii=False, indent=2)}

Explain concisely:
- where Chen and HLIB agree;
- where their sectors, companies, directions, causal emphasis, or coverage differ;
- how their evidence strength differs, especially Direct versus source-grounded inference;
- an overall descriptive takeaway. If one or both sources have no directional
  result, state that as a difference in coverage rather than inventing agreement.
"""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": ComparisonResponse,
                "temperature": 0,
            },
        )
        return ComparisonResult(ComparisonResponse.model_validate_json(response.text))
    except Exception as error:
        return ComparisonResult(
            fallback,
            used_fallback=True,
            status_message=(
                f"Gemini comparison unavailable ({type(error).__name__}: {error})."
            ),
        )


def _fallback_comparison(
    analyses: dict[str, SourceAnalysis],
) -> ComparisonResponse:
    chen = analyses["Chen"]
    hlib = analyses["HLIB Research"]
    chen_targets = _targets(chen)
    hlib_targets = _targets(hlib)
    agreements = []
    differences = []
    for target in sorted(set(chen_targets) & set(hlib_targets)):
        if chen_targets[target] == hlib_targets[target]:
            agreements.append(
                f"Both sources identify {target} as {chen_targets[target]}."
            )
        else:
            differences.append(
                f"For {target}, Chen is {chen_targets[target]} while HLIB Research is {hlib_targets[target]}."
            )
    only_chen = sorted(set(chen_targets) - set(hlib_targets))
    only_hlib = sorted(set(hlib_targets) - set(chen_targets))
    if only_chen:
        differences.append(f"Only Chen identifies: {', '.join(only_chen)}.")
    if only_hlib:
        differences.append(f"Only HLIB Research identifies: {', '.join(only_hlib)}.")
    if not chen_targets:
        differences.append("Chen has no supported directional conclusion.")
    if not hlib_targets:
        differences.append("HLIB Research has no supported directional conclusion.")
    return ComparisonResponse(
        overall_summary=(
            "The comparison is limited to the validated source results shown above."
        ),
        agreements=agreements,
        differences=differences,
        evidence_comparison=(
            "Evidence strength is shown on each pathway as Direct source support or a source-grounded inference."
        ),
    )


def _targets(analysis: SourceAnalysis) -> dict[str, str]:
    targets = {
        impact.impacted_sector: impact.impact_direction
        for impact in analysis.sector_impacts
    }
    targets.update(
        {
            impact.company_name: impact.impact_direction
            for impact in analysis.company_impacts
        }
    )
    return targets


def build_graph_data(
    event_summary: str, analyses: dict[str, SourceAnalysis]
) -> dict:
    """Build a stable centre-out comparison map from validated results."""
    event_label = _short_words(event_summary, 8)
    nodes = [{"id": "event", "label": event_label, "type": "event",
              "detail": event_label, "x_ratio": .5, "y_ratio": .5}]
    edges: list[dict] = []
    for source, analysis in analyses.items():
        key = "chen" if source == "Chen" else "hlib"
        left = source == "Chen"
        impacts = list(analysis.sector_impacts) + list(analysis.company_impacts)
        count = max(1, len(impacts))
        target_groups: dict[tuple[str, str, str, str], dict] = {}
        for index, impact in enumerate(impacts):
            is_sector = hasattr(impact, "impacted_sector")
            target = impact.impacted_sector if is_sector else impact.company_name
            target_type = "sector" if is_sector else "company"
            industry = (impact.impacted_industry or "") if is_sector else ""
            target_key = (target_type, target, industry, impact.impact_direction)
            if target_key not in target_groups:
                target_groups[target_key] = {
                    "id": f"{key}:target:{len(target_groups)}",
                    "ys": [],
                    "details": [],
                    "companies": {},
                }
            target_group = target_groups[target_key]
            y = .22 + (.56 * (index + .5) / count)
            target_group["ys"].append(y)
            target_group["details"].append(impact.conclusion)
            mechanism_id = f"{key}:mechanism:{index}"
            mechanism_label = _causal_label(impact)
            nodes.append({"id": mechanism_id, "label": mechanism_label,
                          "type": "mechanism", "source": source, "detail": mechanism_label,
                          "x_ratio": .27 if left else .73, "y_ratio": y})
            common = {"source": source, "support": impact.support_level,
                      "knowledge_ids": impact.supporting_knowledge_ids}
            edges.append({"from": "event", "to": mechanism_id,
                          "direction": "neutral", **common})
            edges.append({"from": mechanism_id, "to": target_group["id"],
                          "direction": impact.impact_direction, **common})
            if is_sector:
                for company in impact.impacted_companies:
                    target_group["companies"].setdefault(company.company_name, common)

        for target_index, ((target_type, target, industry, direction), group) in enumerate(target_groups.items()):
            target_y = sum(group["ys"]) / len(group["ys"])
            target_label = f"{target}\n({industry})" if industry else target
            nodes.append({"id": group["id"], "label": target_label,
                          "type": target_type, "source": source,
                          "direction": direction,
                          "detail": _short_sentence(" ".join(dict.fromkeys(group["details"]))),
                          "x_ratio": .075 if left else .925, "y_ratio": target_y})
            for company_index, (company_name, common) in enumerate(group["companies"].items()):
                company_id = f"{key}:sector-company:{target_index}:{company_index}"
                nodes.append({"id": company_id, "label": company_name,
                              "type": "company", "source": source,
                              "direction": direction,
                              "detail": f"Supported company within {target}.",
                              "x_ratio": .08 if left else .92,
                              "y_ratio": min(.94, target_y + .08 * (company_index + 1))})
                edges.append({"from": group["id"], "to": company_id,
                              "direction": direction, **common})
        for index, context in enumerate(analysis.market_context):
            context_id = f"{key}:context:{index}"
            context_label = _short_words(context.summary, 8)
            nodes.append({"id": context_id, "label": context_label,
                          "type": "context", "source": source, "detail": _short_sentence(context.summary),
                          "x_ratio": .38 if left else .62, "y_ratio": .1 + index * .08})
            edges.append({"from": "event", "to": context_id, "source": source,
                          "support": "context", "direction": "neutral",
                          "knowledge_ids": context.supporting_knowledge_ids})
    return {"nodes": nodes, "edges": edges}


def _short_words(text: str, limit: int) -> str:
    """Keep graph labels compact while leaving full explanations in Stage 5."""
    compact = " ".join(text.split()).strip(" .")
    compact = re.sub(r"\bartificial intelligence\b", "AI", compact, flags=re.IGNORECASE)
    words = compact.split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(",;:") + "…"


def _short_sentence(text: str, limit: int = 18) -> str:
    first_sentence = text.split(". ", 1)[0].strip()
    shortened = _short_words(first_sentence, limit)
    return shortened if shortened.endswith((".", "…")) else shortened + "."


def _causal_label(impact) -> str:
    """Choose one compact mechanism while retaining the full chain in node details."""
    steps = [step.strip() for step in impact.causal_steps if step.strip()]
    if len(steps) >= 3:
        mechanism = steps[-2]
    elif len(steps) == 2:
        mechanism = steps[1]
    elif steps:
        mechanism = steps[0]
    else:
        mechanism = impact.conclusion
    return _short_words(mechanism, 6)
