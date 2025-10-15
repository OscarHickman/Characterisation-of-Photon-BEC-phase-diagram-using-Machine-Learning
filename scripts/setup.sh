#!/usr/bin/env bash
set -euo pipefail
ROOT=$(dirname "$(dirname "${BASH_SOURCE[0]}")")
cd "$ROOT"

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pre-commit ruff || true
python -m pre_commit install || true

echo "Setup complete. Activate with: source .venv/bin/activate"
*** End Patch