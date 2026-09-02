---
title: CompliBot MVP
emoji: 📋
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.39.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# CompliBot MVP

CompliBot MVP is an early pharmaceutical compliance-document retrieval prototype focused on SOP, deviation, CAPA, training, approval, and quality-process questions.

It demonstrates a local retrieval workflow using PDF extraction, compliance-oriented text processing, SentenceTransformer embeddings, ChromaDB vector search, deterministic retrieval heuristics, grounding checks, structured extractive answers, and source/chunk evidence.

This repository is retained as an **architectural evolution/reference project**.

The broader production-oriented successor is the **Pharma AI Platform**.

## What the MVP Demonstrates

Users can:

- upload SOP, CAPA, deviation, quality, guideline, and compliance PDFs
- extract text with PyPDF
- normalize extracted document text
- split content using compliance-oriented section labels
- create sentence-aware chunks
- generate local SentenceTransformer embeddings
- persist embeddings in ChromaDB
- classify compliance question types
- infer SOP versus guideline retrieval preference
- retrieve semantically related chunks
- apply deterministic compliance-oriented reranking
- evaluate whether retrieved evidence is sufficiently grounded
- produce deterministic structured answer fields
- display document and chunk-level supporting evidence

## Retrieval Pipeline

```text
Compliance PDF
      ↓
PyPDF extraction
      ↓
Text normalization
      ↓
Section-aware splitting
      ↓
Sentence-aware chunking
      ↓
SentenceTransformer embeddings
      ↓
ChromaDB vector retrieval
      ↓
Document-group preference
      ↓
Heuristic reranking
      ↓
Grounding checks
      ↓
Rule-based structured synthesis
      ↓
Source + chunk evidence
```

## Question Classification

The MVP uses deterministic keyword rules to classify questions into:

- Definition
- Training
- Escalation / Quality Event
- Policy / Requirement
- Procedure
- General Compliance Question

For example:

```text
How should a deviation be escalated?
```

is classified as:

```text
Escalation / Quality Event
```

Specific compliance categories are evaluated before broad procedure terms such as `how`.

## Document Grouping

Uploaded documents are heuristically grouped as:

```text
sop
guideline
quality_doc
general
```

Examples:

```text
SOP_Deviation_Handling.pdf → sop
ICH_Guideline.pdf          → guideline
Quality_Manual.pdf         → quality_doc
```

This classification is lightweight retrieval metadata based primarily on filenames. It is not authoritative regulatory-document classification.

## Retrieval Preference

Questions mentioning concepts such as:

- SOP
- deviation
- CAPA
- approval process
- review process
- document review

prefer SOP-classified chunks.

Questions mentioning concepts such as:

- guideline
- regulatory
- FDA
- ICH
- GCP
- GVP

prefer guideline-classified chunks.

If preferred-group retrieval returns insufficient evidence, the pipeline falls back to the broader collection.

## Heuristic Reranking

The standalone MVP uses a deterministic reranker rather than a learned cross-encoder.

Ranking signals include:

- ChromaDB vector distance
- preferred document group
- repeated evidence from the same source
- penalties for less-preferred document groups

These values are experimental retrieval heuristics, not calibrated relevance probabilities.

## Grounding Checks

Retrieved evidence is evaluated using signals including:

- vector distance
- keyword overlap
- pharmaceutical/compliance domain-term overlap
- support from multiple chunks

Possible internal states are:

```text
strongly_grounded
weakly_grounded
not_grounded
```

These labels are heuristic and should not be interpreted as calibrated factuality or confidence scores.

Casual-chat matching uses token boundaries. This prevents the greeting:

```text
hi
```

from incorrectly matching inside a normal word such as:

```text
this
```

A regression test protects this behavior.

## Structured Output

The standalone MVP does **not** use a generative LLM.

It constructs output deterministically from retrieved document sentences.

The response structure includes:

```text
answer_summary
procedure_guidance
key_requirements
evidence
source
compliance_note
question_type
```

Evidence records preserve:

```text
source
chunk_index
distance
text
```

## Technology

- Python 3.11
- Streamlit
- PyPDF
- SentenceTransformers
- `all-MiniLM-L6-v2`
- ChromaDB
- PyTorch
- NumPy
- pytest
- GitHub Actions

## Dependency Compatibility

The frozen MVP uses conservative compatibility pins:

```text
chromadb==0.5.5
torch==2.2.2
numpy==1.26.4
posthog==4.8.0
```

`torch==2.2.2` was verified with Python 3.11 on Intel macOS.

`posthog==4.8.0` avoids the telemetry API mismatch observed when the historical ChromaDB release is combined with newer PostHog releases.

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Create runtime directories:

```bash
mkdir -p \
  data/uploads \
  data/runtime/vector_store \
  logs
```

Run the application:

```bash
streamlit run app.py
```

## Testing

Run:

```bash
python -m pytest -q
```

or:

```bash
pytest -q
```

The final deterministic unit suite contains seven tests covering:

- question classification
- document grouping
- question-to-document preference
- casual-chat boundary handling
- grounding behavior
- chunking behavior
- structured answer synthesis

Run smoke checks:

```bash
bash scripts/smoke_test.sh
```

Check installed dependency consistency:

```bash
python -m pip check
```

## Continuous Integration

GitHub Actions verifies:

```text
dependency installation
      ↓
pip check
      ↓
Python compilation
      ↓
pytest
      ↓
smoke checks
```

CI uses Python 3.11.

## Runtime Data Policy

The following runtime paths are excluded from Git:

```text
data/uploads/
data/runtime/vector_store/
logs/
```

Local environment files are also ignored:

```text
.env
.env.local
.env.*.local
```

Uploaded PDFs and local ChromaDB indexes should not be committed.

## Scope and Limitations

This repository does **not** claim:

- validated pharmaceutical regulatory decision-making
- legal or regulatory advice
- clinical decision support
- production GxP validation
- 21 CFR Part 11 compliance
- calibrated factuality or confidence scoring
- OCR for scanned PDFs
- page-level citation verification
- BM25 hybrid retrieval
- cross-encoder reranking
- GraphRAG
- agentic workflows
- MCP integration
- enterprise RBAC
- automated compliance approval
- production-scale distributed retrieval

Document grouping, question routing, reranking, grounding, and answer synthesis are prototype deterministic heuristics.

## Relationship to Pharma AI Platform

CompliBot MVP represents an earlier stage in the evolution of the compliance-retrieval capability.

The standalone prototype demonstrates:

```text
PDF
 ↓
Compliance-aware processing
 ↓
Local embeddings
 ↓
Vector retrieval
 ↓
Compliance heuristics
 ↓
Grounding checks
 ↓
Structured extractive output
```

The **Pharma AI Platform** supersedes this standalone application with shared document preparation, routing, evidence handling, review workflows, auditability, and optional grounded LLM synthesis.

During the portfolio audit, this MVP was compared directly with the flagship implementation.

Its section-aware chunking was distinct, but there was no benchmark demonstrating superior retrieval quality. The historical grounding logic also contained a false-positive bug that was corrected in this reference repository.

Therefore no wholesale MVP logic was promoted into the flagship.

## Project Status

**Reference / evolution MVP — feature-frozen after cleanup and verification.**

Future production-oriented development belongs in the Pharma AI Platform rather than this standalone repository.
