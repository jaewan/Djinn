#!/bin/bash
# Launch baseline servers for apples-to-apples comparison
# 
# SCIENTIFIC MODE: Servers run SEQUENTIALLY (one at a time)
# This ensures no resource contention and fair GPU allocation
#
# Usage:
#   bash launch_all_servers.sh [--parallel]     # Sequential by default
#   bash launch_all_servers.sh --parallel       # Run all 3 in parallel
#   bash launch_all_servers.sh --which SERVER   # Run only one server

set -e

# Configuration
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)
SCRIPTS_DIR="$REPO_ROOT/Evaluation/exp5_1_overhead/scripts"
CONFIG_FILE="$REPO_ROOT/Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml"
DEVICE="${DEVICE:-cuda:0}"
WORKLOADS="${WORKLOADS:-hf_tiny_gpt2}"
LOG_DIR="/tmp/djinn_servers"
PARALLEL_MODE=false
WHICH_SERVER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL_MODE=true
            shift
            ;;
        --which)
            WHICH_SERVER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--parallel] [--which SERVER]"
            exit 1
            ;;
    esac
done

# Activate venv if it exists
if [ -d "$REPO_ROOT/.venv" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
    echo "✓ Activated .venv"
fi

# Ensure PYTHONPATH includes repo root
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# Create log directory
mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Apples-to-Apples Baseline Servers${NC}"
echo -e "${YELLOW}========================================${NC}"

if [ "$PARALLEL_MODE" = true ]; then
    echo -e "${BLUE}Mode: PARALLEL (all 3 servers at once)${NC}"
else
    echo -e "${BLUE}Mode: SEQUENTIAL (one at a time - scientific)${NC}"
fi
echo ""

# Function to cleanup on exit
cleanup() {
    echo -e "${YELLOW}Shutting down servers...${NC}"
    pkill -f "native_server\|rpc_server\|djinn.server.server_main" 2>/dev/null || true
    sleep 1
}

trap cleanup EXIT

run_djinn_server() {
    echo -e "${GREEN}[Djinn] Starting server on port 5556...${NC}"
    (
        cd "$REPO_ROOT"
        export GENIE_VMU_SESSION_ARENA_MB=128
        export GENIE_ENABLE_PROFILING=true
        python -m djinn.server.server_main --host 0.0.0.0 --port 5556 --gpu 0
    ) > "$LOG_DIR/djinn_server.log" 2>&1
}

run_rpc_server() {
    echo -e "${GREEN}[RPC] Starting server on port 29500...${NC}"
    (
        cd "$REPO_ROOT"
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT=29500
        export RANK=0
        export WORLD_SIZE=2
        python Evaluation/exp5_1_overhead/scripts/rpc_server.py \
            --config Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml \
            --workloads $WORKLOADS \
            --device "$DEVICE"
    ) > "$LOG_DIR/rpc_server.log" 2>&1
}

run_native_server() {
    echo -e "${GREEN}[Native] Starting server on port 5557...${NC}"
    (
        cd "$REPO_ROOT"
        python Evaluation/exp5_1_overhead/scripts/native_server.py \
            --port 5557 \
            --device "$DEVICE" \
            --config Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml \
            --workloads $WORKLOADS
    ) > "$LOG_DIR/native_server.log" 2>&1
}

if [ "$PARALLEL_MODE" = true ]; then
    # PARALLEL MODE: Start all 3 servers in background
    echo -e "${YELLOW}[1/3] Starting Djinn Server (port 5556)...${NC}"
    run_djinn_server &
    DJINN_PID=$!
    sleep 2
    echo -e "${GREEN}✓ Djinn Server started (PID: $DJINN_PID)${NC}"

    echo -e "${YELLOW}[2/3] Starting RPC Server (port 29500)...${NC}"
    run_rpc_server &
    RPC_PID=$!
    sleep 2
    echo -e "${GREEN}✓ RPC Server started (PID: $RPC_PID)${NC}"

    echo -e "${YELLOW}[3/3] Starting Native PyTorch Server (port 5557)...${NC}"
    run_native_server &
    NATIVE_PID=$!
    sleep 2
    echo -e "${GREEN}✓ Native Server started (PID: $NATIVE_PID)${NC}"

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}All servers started in parallel!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    wait
else
    # SEQUENTIAL MODE: Run servers one at a time (SCIENTIFIC)
    if [ -z "$WHICH_SERVER" ] || [ "$WHICH_SERVER" = "djinn" ]; then
        echo -e "${YELLOW}Starting: DJINN SERVER${NC}"
        echo "To run experiments: export GENIE_SERVER_ADDRESS=localhost:5556"
        echo "Press Ctrl+C when done with this server..."
        echo ""
        run_djinn_server
        echo -e "${YELLOW}Djinn complete. Check: $LOG_DIR/djinn_server.log${NC}"
        read -p "Press Enter to continue or Ctrl+C to exit..."
        echo ""
    fi

    if [ -z "$WHICH_SERVER" ] || [ "$WHICH_SERVER" = "rpc" ]; then
        echo -e "${YELLOW}Starting: RPC SERVER${NC}"
        echo "To run experiments: export MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 RANK=1 WORLD_SIZE=2"
        echo "Press Ctrl+C when done with this server..."
        echo ""
        run_rpc_server
        echo -e "${YELLOW}RPC complete. Check: $LOG_DIR/rpc_server.log${NC}"
        read -p "Press Enter to continue or Ctrl+C to exit..."
        echo ""
    fi

    if [ -z "$WHICH_SERVER" ] || [ "$WHICH_SERVER" = "native" ]; then
        echo -e "${YELLOW}Starting: NATIVE PYTORCH SERVER${NC}"
        echo "To run experiments: point client to localhost:5557"
        echo "Press Ctrl+C when done with this server..."
        echo ""
        run_native_server
        echo -e "${YELLOW}Native complete. Check: $LOG_DIR/native_server.log${NC}"
    fi
fi

echo ""
echo -e "${YELLOW}Log files:${NC}"
echo "  • $LOG_DIR/djinn_server.log"
echo "  • $LOG_DIR/rpc_server.log"
echo "  • $LOG_DIR/native_server.log"

