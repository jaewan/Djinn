#!/bin/bash
# Djinn Server Startup Script
# Usage: ./start_server.sh [--node-id ID] [--control-port PORT] [--data-port PORT]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

# Use PYTHONPATH to ensure package imports work correctly
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

exec python -c "from djinn.server.server import main; main()" "$@"

