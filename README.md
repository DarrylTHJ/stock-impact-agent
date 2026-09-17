# Bursa Malaysia Event-Impact Explorer

An LLM-based dual-source system for analysing how a news event may affect Bursa Malaysia sectors and listed companies. It retrieves and compares evidence from two separate knowledge bases:

- Alfred Chen's market commentary
- HLIB Research reports

The system is a research prototype. Its output is not investment advice.

## What the system does

1. Accepts a typed event, article URL, or PDF upload.
2. Uses Gemini to create a factual summary and up to five neutral retrieval queries.
3. Searches a local ChromaDB vector database for similar Chen and HLIB knowledge records.
4. Classifies each candidate as Direct, Applicable Rule, General Background, or Irrelevant.
5. Retrieves surrounding transcript context for Direct Alfred Chen records.
6. Produces separate source analyses, then compares them and displays a causal graph.

## Requirements

- Python 3.10 or newer
- A Gemini API key
- Internet access for Gemini calls and article-URL extraction

## Setup

Open PowerShell in the project folder and create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your Gemini API key:

```env
GEMINI_API_KEY2=your_gemini_api_key_here
```

Do not commit or submit the `.env` file.

## Start the application

```powershell
streamlit run app.py
```

Streamlit will display a local URL, normally `http://localhost:8501`. Open it in a browser.

## Included knowledge base

The submitted project includes:

- `data/knowledge_records.json` - complete structured knowledge records
- `chroma_db/` - the corresponding semantic-search index
- `data/chen_extracted/` - timestamped Chen transcripts used for Direct-record context
- `data/hlib_source/` - original HLIB PDFs used as report evidence

These folders must remain in place for the application to retrieve knowledge and show evidence.

## Rebuild the vector database

Only run this when the transformed source records have changed:

```powershell
.\.venv\Scripts\python.exe scripts\index_current_records.py
```

This rebuilds `data/knowledge_records.json` and `chroma_db/` from the transformed Chen and HLIB records.

## Offline ETL scripts

The scripts below construct the knowledge base. They are not required to run the already-indexed application.

| Script | Purpose |
|---|---|
| `scripts/collect_chen_captions.py` | Collects Alfred Chen video metadata, captions, and timestamps. |
| `scripts/extract_hlib_pdfs.py` | Extracts HLIB report metadata, text, and page references. |
| `scripts/transform_chen_transcripts.py` | Uses Gemini to turn Chen transcripts into structured financial knowledge records. |
| `scripts/transform_hlib_reports.py` | Uses Gemini to turn HLIB reports into structured financial knowledge records. |
| `scripts/index_current_records.py` | Loads transformed records and rebuilds the ChromaDB index. |

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface and six-stage workflow controller. |
| `event_input.py` | Handles typed events, article URLs, and PDF uploads. |
| `event_analysis.py` | Creates factual event summaries and retrieval queries. |
| `vector_store.py` | Embedding model and ChromaDB search functions. |
| `relevance_filter.py` | Relevance classification of retrieved records. |
| `source_context.py` | Direct Chen transcript-context retrieval. |
| `impact_synthesis.py` | Independent Chen and HLIB impact analyses. |
| `comparison_analysis.py` | Source comparison and causal-graph data. |
| `interactive_graph.py` | Interactive causal-graph rendering. |
| `models.py` | Pydantic data models for structured knowledge records. |

## Notes

- The system keeps Alfred Chen and HLIB Research evidence separate until the comparison stage.
- Semantic similarity finds candidates; it does not prove that evidence applies to the event.
- The system may return **No Supported Conclusion** when it does not find suitable evidence.
