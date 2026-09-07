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
knowledge_type: sector_impact | company_impact | market_context
trigger_event
impacted_sector: official Bursa sector only for sector_impact
impacted_industry: optional source-named precision, e.g. Automotive or F&B
impact_direction: positive | negative | mixed
reason
impacted_companies: only when explicitly supported by source evidence
evidence: one or more exact quotes plus timestamp/page
embedding_summary
source_name, source_title, source_date, source_link
```

Use the 13 official Bursa sectors: Construction; Consumer Products & Services; Energy; Financial Services; Healthcare; Industrial Products & Services; Plantation; Property; REITs; Technology; Telecommunications & Media; Transportation & Logistics; Utilities.

Broad-market statements are stored as `market_context`, usable for narrative and caveats but not sector/company graph nodes.

Use `company_impact` for a source-supported effect on an explicitly named company or IPO. Do not present this as a whole-sector impact. `impacted_industry` can later be shown in the graph as an industry node connected neutrally to the official Bursa main sector.

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
- ChromaDB semantic retrieval is implemented with a local multilingual embedding model. It embeds `embedding_summary` values and returns only matching `knowledge_id` values; full evidence continues to come from the local knowledge records.
- Current ChromaDB test corpus contains only **one fictional data-centre/Utilities record per source**. It proves the technical retrieval flow, but is far too small to evaluate retrieval quality. A broad query such as "boom in AI" may produce different Chen and HLIB outcomes because their two embedding summaries are worded differently.
- Semantic retrieval returns the nearest source-specific record first, then rejects it if its cosine distance is above the current `0.80` acceptance limit. In a tested broad-AI query, Chen's nearest record scored `0.833` (rejected) while HLIB's scored `0.772` (accepted). Do not loosen this merely to force a result: it would increase unsupported source claims. This is a useful example of why the real corpus needs multiple, well-extracted knowledge records.
- Gemini event analysis/query normalisation is implemented. It uses `GEMINI_API_KEY` from a local `.env` file and falls back to the original text if the API is unavailable.
- Gemini connectivity was verified directly. If the Streamlit UI reports that Gemini was unavailable, that specific server process lacks outbound network access; use the network-enabled server instance rather than interpreting it as a wrong API key or failed analysis design.
- Typed text, a public article URL, and a selectable-text PDF are accepted as event inputs. All are converted to event text before Gemini analysis. Scanned PDFs and paywalled/JavaScript-only URLs are not supported yet.
- A public YouTube caption collector now exists at `scripts/collect_chen_captions.py`. It stores one ignored local JSON file per Chen video in `data/chen_extracted/`, including video metadata and timestamped original-language caption segments. It records unavailable/error cases in `collection_log.jsonl`, and uses automatic captions only if no manual Chinese/English caption is available. A one-video test successfully saved a manual Chinese transcript with 621 segments.
- The completed collection saved 430 captioned Chen videos; 100 videos without supported captions were logged and excluded.
- `scripts/transform_chen_transcripts.py` sends one full timestamped transcript at a time to Gemini 3.1 Flash-Lite and creates initial knowledge records in `data/chen_transformed_initial/`. It uses a 45-second delay, checkpoints completed work, and stops on quota/rate-limit errors.
- `scripts/audit_chen_drafts.py` checks every draft's schema, source metadata, timestamp format, evidence-company links, and that each displayed quote is traceable to its source transcript. The transformer reconstructs displayed evidence quotes directly from source caption segments selected by the LLM's timestamp range; it does not trust model-written Chinese quotes. Automated checks do not replace a human review of whether the selected evidence truly supports the interpretation.
- The Chen stages are: `chen_extracted` (raw source), `chen_transformed_initial` (initial transformation), `chen_transformed_validated` (first LLM validation), `chen_transformed_revamped` (records supplied with additional evidence), and `chen_transformed_final` (records ready for promotion). The immediate priority is filling `chen_transformed_initial`; validation begins only after that stage is complete.
- `scripts/promote_chen_draft.py` promotes reviewed local drafts into `data/knowledge_records.json` and indexes them in ChromaDB. One Automotive test video is promoted only for local UI testing; its three records cover MBMR/Betamek company context, automotive TIV saturation, and EV disruption.
- Next technical step: run the collector for the available-caption channel videos, then add real Chen knowledge-record extraction after finalising the extraction prompt and source metadata.
