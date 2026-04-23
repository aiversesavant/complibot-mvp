#!/bin/bash
set -e

echo "Running smoke checks..."

test -f app.py
test -f compli_pipeline.py
test -d data/uploads
test -d data/runtime/vector_store
test -f requirements.txt
test -f README.md
test -f .env.example

echo "Smoke checks passed."
