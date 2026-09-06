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
