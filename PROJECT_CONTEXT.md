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

- `scripts/index_current_records.py` is the explicit offline Load step. It combines the current Chen and HLIB initial records, replaces `data/knowledge_records.json`, and fully rebuilds ChromaDB. The online query path never ingests or modifies the database.
- Current indexed test corpus: 1,560 records (1,097 Chen and 463 HLIB Research). Chen records retain `source_video_id`; HLIB records retain `source_file` and report metadata.
- ChromaDB embeds each record's concise `embedding_summary`. Complete evidence remains in the JSON source of truth; complete Chen transcripts remain in `data/chen_extracted/` and are not embedded as hundreds of transcript chunks.
- The Streamlit UI has a fixed light theme and three tabs: `Impact comparison`, `Evidence & sources`, and `Reasoning log`. Chen and HLIB remain separate and visible side by side.
- Typed text, a public readable article URL, and a selectable-text PDF are supported as event inputs. Scanned PDFs and paywalled/JavaScript-only pages are not yet supported.
- The single-page UI uses a compact sticky header and six expandable live stages:

```text
1. Analyse the user query
2. Retrieve knowledge records from Chen and HLIB
3. Determine relevance with explanations
4. Retrieve further context from high-similarity sources
5. Generate separate Chen and HLIB verdicts
6. Compare both agents and render an interactive causal map
```

- Retrieval runs separately for every event theme, takes up to four nearest results per theme, deduplicates them, and retains up to 10 candidates per source for LLM filtering. This reduces the chance that one wording/theme crowds out other relevant causal mechanisms without making the review unnecessarily large.
- `Direct` means the source discusses the same event. `Applicable rule` means the event explicitly satisfies a high-precision causal rule stated by the source; it may create a graph edge but is labelled `Source-grounded inference`. `General background` is useful context but cannot determine direction, severity, conclusions, or impact edges. `Irrelevant` is excluded but remains visible in the audit.
- If the relevance LLM is unavailable, candidates are conservatively marked Irrelevant instead of accepting weak vector matches and creating unsupported graphs.
- Only complete transcripts belonging to Direct Chen records are loaded. Transformed records therefore act as the searchable structured index; the complete transcript supplies deeper context only after a very high-precision match.
- Final synthesis considers Direct records and Applicable rules separately for each source. It consolidates duplicate sector results and company-specific results; company records cannot be promoted to sector-wide conclusions. Relevant market context is visible but non-directional.
- Stage 6 makes a separate constrained comparison call using only the validated Chen and HLIB outputs. It describes agreements, differences, causal emphasis, coverage and evidence strength without introducing new impacts or deciding which source is correct.
- The interactive dependency-free causal map is built deterministically from validated Stage 5 outputs. Its stable centre-out layout fixes the event in the middle, Chen pathways on the left and HLIB pathways on the right. Each impact uses one concise mechanism node, while click-details retain the full causal chain. Source colours identify pathways; target-node colours identify positive, negative or mixed direction. It supports dragging, zooming, panning, source filters, Applicable-rule and market-context filters. Solid edges are Direct, dashed edges are inferences and dotted edges are context.
- Selecting a Stage 3 record opens its evidence and raw metadata in a dialog. Completed results are kept in Streamlit session state, so inspecting records does not rerun retrieval or consume Gemini calls.
- A Petronas-capex integration test classified two Chen and two HLIB records as Applicable rules, generated separate positive Energy/OGSE source-grounded inferences, compared their different causal emphasis, and built a nine-node/nine-edge graph.
- `chrome_extension/` contains a local Manifest V3 quick-launch extension. Clicking its toolbar icon, or choosing its page/link context-menu item, opens `http://localhost:8501` with the current article URL safely encoded. Streamlit automatically selects URL mode and runs that URL once per new session. The extension contains no Gemini key and requires the local Python/Chroma application to be running.
- For Chen evidence, Gemini must copy exact original Chinese from the complete transcript. The application locates that quote in the saved transcript and derives the final timestamp deterministically. A quote that cannot be found exactly is rejected. This prevents plausible translations being attached to invented or incorrect timestamps.
- For HLIB evidence, the UI shows report title, date, relevant page location, and provides the original local PDF for download.
- A real integration test using the Malaysia/Huawei AI-chip event retrieved two Direct Chen records, loaded only their two complete transcripts, and produced one consolidated negative `Technology / Data Centre` result. Both displayed Chinese excerpts matched the stored transcripts and had system-derived timestamps. HLIB correctly returned no Direct conclusion rather than borrowing Chen evidence.
- The reasoning log is an evidence-backed processing audit, not hidden chain of thought. It shows candidate source/ID, similarity, matched retrieval themes, relevance verdict, and concise explanation.
