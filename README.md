---
title: CompliBot MVP
emoji: 📋
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.39.0
python_version: 3.10
app_file: app.py
pinned: false
---

# CompliBot MVP

CompliBot MVP is a Streamlit-based compliance document assistant for pharma and regulated workflows.

## What it does
- Upload SOP, CAPA, deviation, and compliance PDFs
- Ingest and index uploaded documents
- Ask grounded compliance questions
- Retrieve supporting evidence
- Provide compliance-oriented answers

## Example questions
- What does this SOP say about deviation handling?
- What is the CAPA process?
- What is the document review and approval process?
- What training is required before performing this task?

## Tech stack
- Python
- Streamlit
- PyPDF
- ChromaDB
- Sentence Transformers

## Module role inside PharmaAI Platform
CompliBot is the compliance and SOP assistant module.
It is different from:
- **PharmaRAG**: general document-grounded pharma/regulatory Q&A
- **PharmaSummarizer**: document title extraction, summary, and highlights
- **CompliBot**: compliance/process/policy-oriented Q&A over SOP and quality documents

## Notes
This MVP is part of the broader PharmaAI Platform roadmap and focuses on grounded compliance document Q&A.