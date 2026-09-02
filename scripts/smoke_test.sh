#!/bin/bash

set -e

echo "Running smoke checks..."

test -f app.py
test -f compli_pipeline.py
test -f requirements.txt
test -f requirements-dev.txt
test -f README.md
test -f pytest.ini

test -d data/uploads
test -d data/runtime/vector_store

test -f docs/architecture.md
test -f docs/runbook.md

test -f tests/unit/test_compli_pipeline.py
test -f .github/workflows/ci.yml

echo "Smoke checks passed."
