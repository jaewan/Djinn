#!/bin/bash
# Launch Ray head node for baseline comparison in Exp 5.1
#
# This starts a Ray cluster with GPU support for the Ray Actor baseline.
# 
# Usage:
#   # Terminal 1: Start Ray head node
#   bash launch_ray_baseline.sh
#
#   # Terminal 2: Run experiments
#   export RAY_ADDRESS=127.0.0.1:6379
#   python Evaluation/exp5_1_overhead/scripts/run_overhead_sweep.py \
#     --config Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml \
#     --workloads hf_tiny_gpt2
#
#   # Terminal 3: Stop Ray
#   ray stop

set -e

# Configuration
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)

# Activate venv if available
if [ -d "$REPO_ROOT/.venv" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
    echo "✓ Activated .venv"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Ray Head Node for Baseline Comparison${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
echo ""

# Check if Ray is installed
if ! python -c "import ray" 2>/dev/null; then
    echo -e "${RED}✗ Ray not found. Install with:${NC}"
    echo "  pip install ray[tune]"
    exit 1
fi

echo -e "${BLUE}Ray Configuration:${NC}"
echo "  • Port: 6379"
echo "  • GPUs: 1"
echo "  • Dashboard: http://127.0.0.1:8265"
echo ""

# Check if Ray is already running
if ray status &>/dev/null; then
    echo -e "${YELLOW}⚠️  Ray is already running. Stopping first...${NC}"
    ray stop
    sleep 2
fi

echo -e "${GREEN}[1/2] Starting Ray head node...${NC}"
ray start --head --port=6379 --num-gpus=1 --log-style=record

echo ""
echo -e "${GREEN}[2/2] Ray cluster ready!${NC}"
echo ""

echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Ray Cluster Status${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
echo ""

ray status

echo ""
echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Next Steps${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "1. In another terminal, run experiments:"
echo ""
echo "   export RAY_ADDRESS=127.0.0.1:6379"
echo "   cd $REPO_ROOT"
echo "   python Evaluation/exp5_1_overhead/scripts/run_overhead_sweep.py \\"
echo "     --config Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml \\"
echo "     --workloads hf_tiny_gpt2"
echo ""
echo "2. Monitor in browser: http://127.0.0.1:8265"
echo ""
echo "3. Stop Ray:"
echo "   ray stop"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop Ray${NC}"
echo ""

# Wait indefinitely
tail -f /dev/null

