#!/usr/bin/env bash
# Local development startup script.
# Usage: ./start.sh [--reload]
#
# Prerequisites:
#   1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
#   2. uv sync          (installs dependencies into .venv/)
#   3. cp .env.example .env   (adjust values if needed)
#
# Flags:
#   --reload   Enable auto-reload on code changes (development mode)

set -euo pipefail

cd "$(dirname "$0")"

# Ensure .env exists
if [[ ! -f .env ]]; then
    echo "⚠  No .env file found. Copying .env.example → .env"
    cp .env.example .env
fi

# Parse args
RELOAD_FLAG=""
if [[ "${1:-}" == "--reload" ]]; then
    RELOAD_FLAG="--reload"
fi

echo "▶ Starting Product Similarity Search API (local)…"
echo "  Config: .env"
echo "  Docs:   http://localhost:${SAP_API_PORT:-8000}/docs"
echo ""

exec uv run uvicorn sap_cxii_tech_ex_01.api.app:app \
    --host "${SAP_API_HOST:-0.0.0.0}" \
    --port "${SAP_API_PORT:-8000}" \
    --log-level info \
    $RELOAD_FLAG
