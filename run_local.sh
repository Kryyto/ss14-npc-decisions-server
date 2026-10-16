#!/usr/bin/env bash
# Runs the THX-01 typed-decision API next to an SS14 server (Linux/macOS).
# First run creates .venv and installs torch (CPU wheel, or CUDA with --cuda) + requirements; later runs
# reuse them and reinstall only when requirements.txt or the torch flavour changes. Variables already set
# in the environment (or in .env) win over the defaults below.
#
# Usage: ./run_local.sh [--cuda] [host] [port] [threads]
set -euo pipefail
cd "$(dirname "$0")"

TORCH_INDEX="https://download.pytorch.org/whl/cpu"
if [ "${1:-}" = "--cuda" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu126"
    shift
fi

HOST="${1:-127.0.0.1}"
PORT="${2:-8095}"
THREADS="${3:-2}"
PYTHON=".venv/bin/python"
MARKER=".venv/.requirements.sha256"

if [ ! -x "$PYTHON" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

HASH="$(sha256sum requirements.txt | cut -d' ' -f1)|$TORCH_INDEX"
if [ "$(cat "$MARKER" 2>/dev/null || true)" != "$HASH" ]; then
    echo "Installing dependencies from $TORCH_INDEX (this can take a while)..."
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install torch --index-url "$TORCH_INDEX"
    "$PYTHON" -m pip install -r requirements.txt
    echo "$HASH" > "$MARKER"
fi

if [ -f .env.example ] && [ ! -f .env ]; then
    cp .env.example .env
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS}"

echo "Starting on http://$HOST:$PORT (first start downloads the ~650 MB model)..."
exec "$PYTHON" -m app.main --host "$HOST" --port "$PORT"
