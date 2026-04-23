# Runbook

## Local Setup
1. Activate the local virtual environment if present
2. Install dependencies from `requirements.txt`
3. Start the application from the project root

## Operational Checks
- `app.py` exists
- `compli_pipeline.py` exists
- `data/uploads/` exists
- vector store path exists

## Recovery Notes
- if vector store is missing, rebuild the retrieval store
- never commit runtime data or secrets
