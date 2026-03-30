# CompliBot MVP

CompliBot MVP is a Streamlit-based compliance document assistant for pharma and regulated workflows.

## Features

- Upload SOP / quality / compliance PDFs
- Ingest and index document content
- Ask grounded compliance questions
- Receive citation-backed answers
- View supporting evidence excerpts
- View a compliance-oriented review note

## Tech Stack

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- ChromaDB

## Run Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py