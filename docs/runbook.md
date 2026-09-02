# CompliBot MVP Runbook

## Purpose

This runbook describes how to set up, run, test, verify, reset, and safely maintain the standalone CompliBot MVP.

The repository is retained as a pharmaceutical compliance-retrieval reference/evolution project.

## Requirements

Verified environment:

```text
Python 3.11
```

## Create the Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Verify:

```bash
which python
python --version
```

The executable should resolve inside the repository's `.venv`.

## Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Verify:

```bash
python -m pip check
```

Expected:

```text
No broken requirements found.
```

## Compatibility Pins

The frozen MVP uses:

```text
chromadb==0.5.5
torch==2.2.2
numpy==1.26.4
posthog==4.8.0
```

These versions were verified together during final cleanup.

## Create Runtime Directories

```bash
mkdir -p \
  data/uploads \
  data/runtime/vector_store \
  logs
```

These directories must remain untracked.

## Start the Application

```bash
streamlit run app.py
```

Stop Streamlit with:

```text
Control + C
```

Do not leave the server suspended with `Control + Z`.

## Typical Workflow

1. Upload one or more PDFs.
2. Click `Ingest Documents`.
3. Allow the pipeline to extract and chunk text.
4. Generate local embeddings.
5. Store chunks in ChromaDB.
6. Ask a compliance-related question.
7. Review the structured answer and supporting evidence.

Example:

```text
What does this SOP say about deviation handling?
```

## Run Tests

Primary:

```bash
python -m pytest -q
```

Alternative:

```bash
pytest -q
```

Expected final result:

```text
7 passed
```

## Compile Check

```bash
python -m compileall \
  app.py \
  compli_pipeline.py \
  tests
```

## Smoke Test

```bash
mkdir -p \
  data/uploads \
  data/runtime/vector_store \
  logs

bash scripts/smoke_test.sh
```

Expected:

```text
Running smoke checks...
Smoke checks passed.
```

## Dependency Check

```bash
python -m pip check
```

Expected:

```text
No broken requirements found.
```

## ChromaDB Startup Check

```bash
python - <<'PY'
import tempfile
import chromadb

with tempfile.TemporaryDirectory() as directory:
    client = chromadb.PersistentClient(path=directory)
    collection = client.get_or_create_collection("startup_check")
    print("collection:", collection.name)
    print("count:", collection.count())

print("CHROMA STARTUP CHECK PASSED")
PY
```

Expected:

```text
collection: startup_check
count: 0
CHROMA STARTUP CHECK PASSED
```

## Rebuild the Vector Store

Stop Streamlit first.

Then remove the derived index:

```bash
rm -rf data/runtime/vector_store/chroma_store
```

Restart the application and ingest the intended PDFs again.

## Clear Runtime Uploads

Use:

```bash
find data/uploads \
  -mindepth 1 \
  -maxdepth 1 \
  -exec rm -rf {} +
```

This handles both files and directories.

Only remove runtime copies that are disposable.

## Check for Tracked PDFs

```bash
git ls-files \
  | grep -Ei '\.pdf$' \
  || echo "NO PDF FILES TRACKED"
```

Expected:

```text
NO PDF FILES TRACKED
```

## Check for Runtime Data

```bash
git ls-files \
  | grep -E '^data/(uploads|runtime/vector_store)/' \
  || echo "NO RUNTIME DATA TRACKED"
```

Expected:

```text
NO RUNTIME DATA TRACKED
```

## Check for Environment Files

```bash
git ls-files \
  | grep -E '(^|/)\.env($|\.local$|\..*\.local$)' \
  || echo "NO REAL ENV FILES TRACKED"
```

Expected:

```text
NO REAL ENV FILES TRACKED
```

## Verify Active Python and pytest

```bash
which python
which pytest
head -1 "$(which pytest)"
```

Both should resolve inside this repository's `.venv`.

If zsh cached an older executable:

```bash
hash -r 2>/dev/null || true
rehash 2>/dev/null || true
```

## Model Initialization

Constructing:

```python
CompliBotPipeline()
```

loads:

```text
all-MiniLM-L6-v2
```

If the model is not cached, SentenceTransformers may retrieve it.

The deterministic unit tests avoid full model initialization where unnecessary.

## Scanned PDFs

OCR is not implemented.

Image-only PDFs may therefore produce little or no extractable text.

## Casual-Chat Regression

Token-boundary-aware matching prevents:

```text
hi
```

from matching inside:

```text
this
```

This behavior is covered by a regression test.

## Full Verification

Before committing or releasing:

```bash
python -m pytest -q
pytest -q
python -m pip check

python -m compileall \
  app.py \
  compli_pipeline.py \
  tests

mkdir -p \
  data/uploads \
  data/runtime/vector_store \
  logs

bash scripts/smoke_test.sh

git diff --check
git status --short
```

Expected:

```text
7 passed
7 passed
No broken requirements found.
Smoke checks passed.
```

## Operational Boundary

CompliBot MVP is not a validated pharmaceutical decision system.

Outputs should be checked against approved source documents, applicable SOPs, QA procedures, and appropriate quality/regulatory review processes.

It does not provide legal, clinical, regulatory, or quality approval.

## Project Status

**Reference / evolution MVP — feature-frozen after cleanup and verification.**

Further production-oriented development belongs in the Pharma AI Platform.
