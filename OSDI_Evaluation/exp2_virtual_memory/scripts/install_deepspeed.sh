#!/bin/bash
# Install and validate DeepSpeed for Experiment 2

set -e

echo "=================================="
echo "Installing DeepSpeed"
echo "=================================="

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 not found"
    exit 1
fi

echo ""
echo "[1/3] Installing DeepSpeed..."
pip3 install deepspeed --no-cache-dir

echo ""
echo "[2/3] Verifying DeepSpeed installation..."
python3 << 'EOF'
import sys
try:
    import deepspeed
    print(f"✅ DeepSpeed version: {deepspeed.__version__}")
except ImportError:
    print("❌ DeepSpeed import failed")
    sys.exit(1)
EOF

echo ""
echo "[3/3] Testing DeepSpeed inference module..."
python3 << 'EOF'
import sys
try:
    from deepspeed.inference import get_quantize_config
    print("✅ DeepSpeed inference module available")
except ImportError as e:
    print(f"⚠️  DeepSpeed inference module not fully available: {e}")
    print("   (This is OK - will fall back to standard inference)")
EOF

echo ""
echo "=================================="
echo "✅ DeepSpeed installation complete"
echo "=================================="

