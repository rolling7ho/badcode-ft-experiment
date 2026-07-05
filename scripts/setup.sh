#!/usr/bin/env bash
# Bootstrap a local dev environment: create a virtualenv, install
# dependencies, and set up .env. Safe to re-run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv in $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt
"$VENV_DIR/bin/pip" install -e ".[dev]"

if [ ! -f ".env" ]; then
  echo "Creating .env from .env.example ..."
  cp .env.example .env
else
  echo ".env already exists, leaving it untouched."
fi

echo "Done. Activate the environment with: source $VENV_DIR/bin/activate"
