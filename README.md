# Bursa Malaysia Event-Impact Explorer

An LLM-based dual-source research prototype that analyses how a news event may affect Bursa Malaysia sectors and listed companies. It retrieves and compares evidence from Alfred Chen's market commentary and HLIB Research reports.

The output is not investment advice.

## What the system does

1. Accepts a typed event, article URL, or PDF upload.
2. Uses Gemini to create a factual summary and up to five neutral retrieval queries.
3. Searches a local ChromaDB vector database for Chen and HLIB knowledge records.
4. Classifies candidates as Direct, Applicable Rule, General Background, or Irrelevant.
5. Loads surrounding transcript context for Direct Alfred Chen records.
6. Produces separate source analyses, compares them, and displays a causal graph.

## Files supplied separately

The GitHub repository contains the source code. The populated local knowledge base is supplied separately as:

`stock-impact-agent-runtime-data.zip`

The ZIP contains the following paths in the correct project structure:

- `data/knowledge_records.json` - 1,560 structured knowledge records
- `data/chen_extracted/` - timestamped Chen transcripts used for context enrichment
- `data/hlib_source/` - original HLIB PDFs used for source-evidence downloads
- `chroma_db/` - the corresponding semantic-search index

Do not rename these folders after extraction. Do not publish the runtime bundle publicly unless you have permission to redistribute all included source reports.

## Fresh-clone setup on Windows

### 1. Clone the repository

```powershell
git clone https://github.com/DarrylTHJ/stock-impact-agent.git
cd stock-impact-agent
```

### 2. Add the runtime data

Place `stock-impact-agent-runtime-data.zip` anywhere on the computer, then extract it into the cloned project root. For example:

```powershell
Expand-Archive -LiteralPath "C:\path\to\stock-impact-agent-runtime-data.zip" -DestinationPath . -Force
```

After extraction, these checks should all return `True`:

```powershell
Test-Path .\data\knowledge_records.json
Test-Path .\data\chen_extracted
Test-Path .\data\hlib_source
Test-Path .\chroma_db
```

### 3. Create the Python environment

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the Gemini API key

Create a file named `.env` in the project root:

```env
GEMINI_API_KEY2=your_gemini_api_key_here
```

The `.env` file is ignored by Git and must not be committed. Each user should supply their own Gemini API key.

### 5. Start the application

```powershell
streamlit run app.py
```

Streamlit normally opens `http://localhost:8501` automatically.

On the first run, `sentence-transformers` downloads `paraphrase-multilingual-MiniLM-L12-v2`. This requires an internet connection and may take several minutes. Later runs use the locally cached model. Gemini analysis and article-URL extraction also require internet access.

## Minimum runtime requirements

| Item | Purpose |
|---|---|
| `data/knowledge_records.json` | Loads the complete evidence records. |
| `chroma_db/` | Performs semantic vector retrieval. |
| `data/chen_extracted/` | Loads complete transcripts for Direct Chen evidence. |
| `data/hlib_source/` | Enables original HLIB PDF downloads in the interface. |
| `.env` with `GEMINI_API_KEY2` | Enables Gemini query interpretation, relevance assessment, and analysis. |
| MiniLM model cache | Creates query embeddings. It downloads automatically on first use. |

If the runtime ZIP has not been extracted, the application displays a clear missing-data message and stops before analysis.

## Main project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface and six-stage workflow controller. |
| `event_input.py` | Handles typed events, article URLs, and PDF uploads. |
| `event_analysis.py` | Creates a factual event summary and retrieval queries. |
| `knowledge_store.py` | Loads full records and coordinates candidate retrieval. |
| `vector_store.py` | Loads the embedding model and searches ChromaDB. |
| `relevance_filter.py` | Classifies retrieved records by relevance. |
| `source_context.py` | Retrieves full Chen transcript context. |
| `impact_synthesis.py` | Generates and validates separate Chen and HLIB analyses. |
| `comparison_analysis.py` | Compares source results and constructs graph data. |
| `interactive_graph.py` | Renders the interactive causal graph. |
| `models.py` | Defines the structured knowledge-record schema. |

## Offline ETL scripts

The application does not run these scripts during normal use. They document and reproduce the offline knowledge-construction process when the full source and transformed datasets are available.

| Script | Purpose |
|---|---|
| `scripts/collect_chen_captions.py` | Collects Chen video metadata, captions, and timestamps. |
| `scripts/extract_hlib_pdfs.py` | Extracts HLIB report metadata, text, and page references. |
| `scripts/transform_chen_transcripts.py` | Transforms Chen transcripts into structured knowledge. |
| `scripts/transform_hlib_reports.py` | Transforms HLIB reports into structured knowledge. |
| `scripts/index_current_records.py` | Rebuilds `knowledge_records.json` and ChromaDB from transformed records. |

The minimal runtime ZIP does not contain every intermediate ETL folder. Rebuilding the complete database requires the full transformed datasets retained by the project author.

## Important behaviour

- Chen and HLIB evidence remain separate until the comparison stage.
- Semantic similarity identifies candidates but does not prove relevance.
- General Background can provide context but cannot independently justify a directional impact.
- The system returns **No Supported Conclusion** when it lacks Direct or Applicable Rule evidence.
