#!/bin/bash
# One-time setup for a bare-metal (no Docker) SearXNG instance used as this
# project's web search backend. Safe to re-run — clones/installs are skipped
# if already present.
#
# Why this is more involved than `pip install`:
#   - SearXNG needs Python >= 3.10. If your system's default `python3`
#     already satisfies that, this script just uses it directly — no extra
#     interpreter needed. Only machines where the default `python3` is too
#     old (e.g. 3.9, as on the machine this was first built on) need it to
#     hunt for a newer one (python3.10/3.11/3.12/3.13) instead.
#   - SearXNG's internal cache requires SQLite >= 3.35. On some machines the
#     system SQLite (linked into every local Python) is older than that, and
#     upgrading an OS library needs root. We work around it unconditionally
#     with `pysqlite3-binary` (bundles a modern, statically-linked SQLite, no
#     system dependency) plus a sitecustomize.py that substitutes it for the
#     stdlib sqlite3 before SearXNG (or anything else) imports it — this is
#     harmless even on machines where it wasn't strictly necessary.
#
# Run with:  bash searxng/setup.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d src ]; then
    echo "Cloning SearXNG..."
    git clone --depth 1 https://github.com/searxng/searxng.git src
else
    echo "searxng/src already present, skipping clone."
fi

# Find a Python >= 3.10 — prefer the system default; only look for a
# specific versioned binary if the default doesn't qualify.
find_suitable_python() {
    for candidate in python3 python3.13 python3.12 python3.11 python3.10; do
        if ! command -v "$candidate" >/dev/null 2>&1; then
            continue
        fi
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

if [ ! -d .venv ]; then
    PYTHON_BIN="$(find_suitable_python)" || {
        echo "ERROR: no Python >= 3.10 found (tried python3, python3.10, python3.11, python3.12, python3.13)." >&2
        echo "Install one (e.g. 'sudo apt install python3.11') and re-run this script." >&2
        exit 1
    }
    echo "Creating venv with $PYTHON_BIN ($("$PYTHON_BIN" --version))..."
    "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
pip install -q --upgrade pip
pip install -q wheel
pip install -q -r src/requirements.txt
pip install -q pysqlite3-binary

SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
cat > "$SITE_PACKAGES/sitecustomize.py" <<'EOF'
# SearXNG requires SQLite >= 3.35 for its internal cache; this machine's
# system SQLite is older and upgrading it needs root. pysqlite3-binary
# bundles a modern, statically-linked SQLite with no system dependency —
# substitute it in place of the stdlib sqlite3 before anything else imports it.
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
EOF

cd src
pip install -q -e . --no-build-isolation

echo "Done. Start it with: bash searxng/run.sh"
