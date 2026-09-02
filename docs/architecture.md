# CompliBot MVP Architecture

## Purpose

CompliBot MVP is an early pharmaceutical compliance-document retrieval prototype.

It demonstrates local retrieval over SOP, deviation, CAPA, training, guideline, and quality-process documents.

The standalone repository is retained as an architectural evolution/reference project. Production-oriented development belongs in the Pharma AI Platform.

## System Overview

```text
Streamlit UI
     ↓
CompliBotPipeline
     ↓
PDF Extraction
     ↓
Text Cleanup
     ↓
Section / Sentence Chunking
     ↓
SentenceTransformer Embeddings
     ↓
ChromaDB
     ↓
Question Classification
     ↓
Preferred Document Retrieval
     ↓
Heuristic Reranking
     ↓
Grounding Evaluation
     ↓
Deterministic Structured Synthesis
     ↓
Source + Chunk Evidence
```

## Main Components

### `app.py`

Provides the Streamlit interface for:

- PDF upload
- ingestion
- collection reset
- question entry
- question-type display
- structured answer rendering
- evidence rendering
- retrieval debug information

Retrieval behavior is delegated to `CompliBotPipeline`.

### `compli_pipeline.py`

Implements:

- ChromaDB initialization
- SentenceTransformer initialization
- PDF text extraction
- text normalization
- section splitting
- sentence splitting
- chunking
- document grouping
- embedding generation
- semantic retrieval
- preferred-group filtering
- heuristic reranking
- question classification
- grounding evaluation
- structured answer construction
- evidence snippet generation

## Runtime Storage

Uploads:

```text
data/uploads/
```

Persistent local ChromaDB data:

```text
data/runtime/vector_store/chroma_store/
```

Logs:

```text
logs/
```

These locations are runtime state and are ignored by Git.

## Document Processing

PDF text is extracted with PyPDF.

The MVP assumes documents contain extractable text and does not implement OCR.

Text cleanup includes:

- null-byte removal
- whitespace normalization
- punctuation cleanup
- filename-noise removal

## Section Processing

The pipeline recognizes compliance-oriented labels including:

- Purpose
- Scope
- Policy
- Procedure
- Responsibilities
- Approval
- Review
- Definitions
- CAPA
- Deviation
- Training
- Investigation
- Closure
- Quality Review
- Final Sign-off

Section detection is heuristic rather than schema-aware document parsing.

## Chunking

Default behavior uses approximately:

```text
chunk size: 700 characters
sentence overlap: 1 sentence
```

Short sections may remain intact. Longer sections are split at sentence boundaries.

## Embeddings

Model:

```text
all-MiniLM-L6-v2
```

Embeddings are generated locally through SentenceTransformers.

## Vector Store

ChromaDB collection:

```text
complibot_documents
```

Chunk metadata includes:

```text
source
chunk_index
doc_group
```

## Document Grouping

Documents are heuristically assigned to:

```text
sop
guideline
quality_doc
general
```

The current implementation relies primarily on filenames.

This metadata supports retrieval preference only and is not authoritative regulatory classification.

## Question Classification

Supported deterministic categories:

```text
Definition
Training
Escalation / Quality Event
Policy / Requirement
Procedure
General Compliance Question
```

Specific compliance categories are checked before generic procedure terms.

## Document Preference

The pipeline infers:

```text
sop
guideline
any
```

SOP-oriented questions include concepts such as deviation, CAPA, approval, review, and SOP procedures.

Guideline-oriented questions include concepts such as FDA, ICH, GCP, GVP, regulatory, and guideline.

If filtered retrieval returns insufficient results, the pipeline falls back to the broader collection.

## Retrieval and Reranking

The question is embedded using the same local SentenceTransformer model used for ingestion.

ChromaDB retrieves semantic candidates.

The deterministic reranker considers:

- vector distance
- preferred document group
- repeated evidence from the same source
- penalties for less-preferred groups

It is not a learned cross-encoder.

## Grounding

Grounding evaluation considers:

- vector distance
- question/document keyword overlap
- compliance-domain overlap
- support from multiple chunks

Internal statuses are:

```text
strongly_grounded
weakly_grounded
not_grounded
```

These are heuristic labels, not calibrated confidence scores.

## Casual-Chat Regression

The historical implementation used substring matching for casual-chat terms.

That caused:

```text
hi
```

to match inside:

```text
this
```

The final reference version uses token-boundary-aware matching and includes a regression test for this case.

## Structured Synthesis

No generative LLM is used in the standalone MVP.

Retrieved sentences are selected and formatted into:

```text
answer_summary
procedure_guidance
key_requirements
evidence
source
compliance_note
question_type
```

Evidence preserves source document, chunk index, distance, and text.

## Dependency Compatibility

Key frozen pins:

```text
chromadb==0.5.5
torch==2.2.2
numpy==1.26.4
posthog==4.8.0
```

The PostHog pin avoids a telemetry API incompatibility with the historical ChromaDB version.

## Testing

Seven deterministic unit tests cover:

- question classification
- document grouping
- document-preference inference
- casual-chat boundary handling
- grounding
- chunking
- structured synthesis

Tests avoid full embedding-model initialization when the behavior under test does not depend on embeddings.

## Continuous Integration

GitHub Actions performs:

```text
checkout
 ↓
Python 3.11
 ↓
dependency installation
 ↓
pip check
 ↓
compileall
 ↓
pytest
 ↓
smoke checks
```

## Security Boundary

This local prototype does not implement:

- authentication
- RBAC
- enterprise identity
- secrets management
- audit-grade change control
- electronic signatures
- Part 11 validation

Uploaded PDFs and vector indexes are excluded from Git.

## Relationship to Pharma AI Platform

The standalone MVP represents an earlier architectural stage.

The Pharma AI Platform supersedes it with shared document preparation, routing, evidence handling, review workflows, auditability, and optional grounded LLM synthesis.

The audit found no evidence that the MVP's distinct chunking behavior was superior enough to replace the flagship implementation.

No wholesale code was harvested.

## Non-Goals

This MVP does not claim:

- production regulatory validation
- clinical decision support
- legal advice
- automated compliance approval
- calibrated confidence scoring
- OCR
- BM25 hybrid retrieval
- cross-encoder reranking
- GraphRAG
- agents
- MCP
- enterprise governance
- production-scale distributed infrastructure

## Project Status

**Reference / evolution MVP — feature-frozen after cleanup and verification.**

Further production-oriented development belongs in the Pharma AI Platform.
