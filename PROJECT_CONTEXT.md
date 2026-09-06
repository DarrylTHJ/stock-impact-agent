# Stock Impact Agent - Project Context

## Purpose

An evidence-grounded LLM application that analyses how local or international events may affect Bursa Malaysia sectors and explicitly supported listed companies. It does not predict stock prices or provide financial advice.

## Knowledge sources

- **Chen:** professional analyst-led investment education and market commentary from Alfred Chen's YouTube channel.
- **HLIB Research:** institutional equity research reports downloaded locally by the student.

The UI must show these analyses side by side, not merge them into one combined result.

## User inputs

The eventual application accepts typed English text, a news/article URL, or an uploaded PDF. Each becomes event text for LLM event analysis/query normalisation. The submitted event source is distinct from Chen/HLIB evidence used to interpret its impact.

## Result design

For Chen and HLIB separately, show evidence-backed sector impacts; positive, negative, or mixed direction; reason; exact evidence excerpts; source title, date, timestamp/page, and link where available. Market context can appear as narrative but cannot create a graph edge.

Do not display raw Chain of Thought. Use a concise evidence-backed decision trace instead.

## Knowledge records

One `sector_impact` record represents one source-supported impact on one official Bursa sector. A source mentioning several sectors produces separate records.

Core fields:

```text
knowledge_id
knowledge_type: sector_impact | market_context
trigger_event
impacted_sector: official Bursa sector only for sector_impact
impact_direction: positive | negative | mixed
reason
impacted_companies: only when explicitly supported by source evidence
evidence: one or more exact quotes plus timestamp/page
embedding_summary
source_name, source_title, source_date, source_link
```

Use the 13 official Bursa sectors: Construction; Consumer Products & Services; Energy; Financial Services; Healthcare; Industrial Products & Services; Plantation; Property; REITs; Technology; Telecommunications & Media; Transportation & Logistics; Utilities.

Broad-market statements are stored as `market_context`, usable for narrative and caveats but not sector/company graph nodes.

## Offline ingestion plan

```text
Chen transcript / local HLIB PDF
→ extract complete source text
→ LLM extracts structured knowledge records with evidence
→ validate records
→ store records locally
→ embed each embedding_summary and index it in ChromaDB
```

When possible, use the whole transcript/report for claim extraction so the LLM can cite multiple sections that jointly support a claim. Use chunks only when a source is too long.

## Online analysis plan

```text
User event
→ LLM event analysis/query normalisation
→ separate Chen and HLIB semantic retrieval
→ evidence-backed first-layer sector impacts
→ optional second-layer retrieval using a causal link from the first-layer analysis
→ separate results/graphs/evidence panels
```

The LLM may normalise a vague event into retrieval phrases, but cannot create an output impact without retrieved evidence.

## Graph rule

The graph may eventually represent:

```text
Event → causal link/reason → Bursa sector → explicitly supported company
```

For domino effects, retrieve second-layer evidence only after first-layer impacts are consolidated. Do not launch a second retrieval for every raw initial result. Maximum initial graph depth: two sector layers.

## Evaluation direction

Evaluate source faithfulness, not investment returns: citation correctness, citation completeness, attribution accuracy, graph/explanation consistency, and unsupported-claim rate.

## Current implementation state

- Clean separate codebase created in `stock-impact-agent/`.
- Data model with controlled Bursa sectors implemented.
- Fictional sample Chen/HLIB records added.
- Basic Streamlit comparison UI implemented.
- Current retrieval is temporary keyword matching.
- Next technical step: install dependencies, run UI, then add LLM event analysis and ChromaDB retrieval.
