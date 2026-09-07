# Stock Impact Agent

An evidence-grounded FYP prototype for comparing how Chen's market commentary and HLIB research interpret an event's potential impact on Bursa Malaysia sectors and companies.

## First build target

- Typed English event input
- Article URL and PDF event input
- LLM-assisted event analysis/query normalisation
- ChromaDB semantic retrieval using a local multilingual embedding model
- Separate Chen and HLIB analysis panels
- Evidence-linked sector results
- Sample knowledge records, before real ingestion is added

## Run later

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Data in `data/` is local-only and must not be committed. `sample_data/` contains fictional records solely to exercise the application structure. URL extraction works only for publicly readable, static article pages; uploaded scanned PDFs require OCR, which is not included yet.

## Collect Chen transcripts

The collector saves one local JSON file per video with metadata and timestamped
caption segments. It does not call an LLM or create knowledge records.

First, run a one-video test:

```powershell
.\.venv\Scripts\python.exe scripts\collect_chen_captions.py --limit 1
```

Then collect the available caption tracks for the full channel:

```powershell
.\.venv\Scripts\python.exe scripts\collect_chen_captions.py
```

Files are written to `data/chen_source_captions/`. The accompanying
`collection_log.jsonl` identifies videos without suitable captions or videos
that need another attempt. Automatic captions are included only when a manual
Chinese/English track is unavailable; use `--manual-only` to exclude them.

## Transform Chen transcripts into review drafts

This step sends a **complete timestamped transcript** to Gemini and creates
draft knowledge records. It does not add anything to the application database
until the drafts have been reviewed. The default model is the lightweight
`gemini-3.1-flash-lite`, with one request every 45 seconds. The run stops on a
quota/rate-limit response and safely resumes later without repeating completed
videos.

Start with two videos:

```powershell
.\.venv\Scripts\python.exe scripts\transform_chen_transcripts.py --limit 2
```

Extracted knowledge records and their transformation log are stored in `data/chen_knowledge_records/`.
The extractor distinguishes `sector_impact` (a source-supported effect on an
entire Bursa sector), `company_impact` (an effect only on an explicitly named
company), and `market_context` (useful narrative that does not create a graph
impact edge).

## Verify drafts before promotion

Verification is a separate gate: it checks whether each candidate's fixed
source quote supports the claimed target, direction, and reason. Unsupported
sector/company candidates are demoted to grounded `market_context` only when
the evidence supports that context; otherwise they are rejected. It can verify
up to 12 candidate records in one Gemini request.

```powershell
.\.venv\Scripts\python.exe scripts\verify_chen_drafts.py --video-id 5mxB_Pzjlng
```

Initial support-check outputs are written to `data/chen_initial_support_checks/`.
After full-transcript recovery and final verification, only files in
`data/chen_final_verified_records/` can be promoted into the local knowledge
store and ChromaDB.

## Run the complete Chen pipeline

To proceed from source captions all the way to final verification without
overlapping Gemini requests, run the coordinator. It resumes from completed
stages, pauses 45 seconds between model calls, and stops safely on quota or
network errors.

```powershell
.\.venv\Scripts\python.exe scripts\run_chen_pipeline.py
```

Its progress log is `data/chen_pipeline_logs/coordinator_log.jsonl`.
