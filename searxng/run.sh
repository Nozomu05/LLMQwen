#!/bin/bash
# Runs the bare-metal SearXNG instance set up by searxng/setup.sh.
# Used by serve.sh's searxng-svc tmux window; safe to run standalone too.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$SCRIPT_DIR/.venv/bin/activate"
export SEARXNG_SETTINGS_PATH="$SCRIPT_DIR/settings.yml"
export SEARXNG_PORT="${SEARXNG_PORT:-8899}"
export SEARXNG_BIND_ADDRESS="${SEARXNG_BIND_ADDRESS:-127.0.0.1}"

cd "$SCRIPT_DIR/src"
python3 searx/webapp.py
