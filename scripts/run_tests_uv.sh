#!/usr/bin/env bash
# Create/update a project-local .venv with uv and run EBiEOT-GMM unit tests.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install: https://docs.astral.sh/uv/installation/" >&2
    exit 1
fi

# Pin the environment to this repo (uses pyproject.toml + uv.lock when present)
export UV_PROJECT="${ROOT}"

uv venv --allow-existing
# [project] deps + dependency-groups dev (pytest), all into .venv
uv sync --group dev
uv run pytest tests/test_gmm_based.py -v
