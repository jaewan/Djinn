# Testing Guide: Validating Ablation Fixes

## Quick Start

After the peer review fixes, follow this testing sequence to validate everything works:

---

## Test 1: Syntax & Imports (5 minutes)

**Goal**: Verify all scripts can be imported without errors

```bash
cd /home/ubuntu/Djinn

# Test all ablation scripts
python3 -m py_compile OSDI_Evaluation/ablation/scripts/ablation_os_tax.py
python3 -m py_compile OSDI_Evaluation/ablation/scripts/ablation_session_arena.py
python3 -m py_compile OSDI_Evaluation/ablation/scripts/ablation_plan_cache.py
python3 -m py_compile OSDI_Evaluation/ablation/scripts/ablation_semantic_signals.py
python3 -m py_compile OSDI_Evaluation/ablation/scripts/run_all_ablations.py

echo "✅ All scripts compile successfully"
```

**Expected Output**: No errors

---

## Test 2: Import Validation (10 minutes)

**Goal**: Verify critical imports exist

```bash
cd /home/ubuntu/Djinn

# Test remote_accelerator device
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/ubuntu/Djinn')

from djinn.core.device_compatibility import enable_remote_accelerator_device
print("✅ remote_accelerator device import OK")
EOF

# Test ghost loader
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/ubuntu/Djinn')

from djinn.core.ghost_loader import create_hf_ghost_model
print("✅ ghost_loader import OK")
EOF

# Test EnhancedModelManager
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/ubuntu/Djinn')

from djinn.core.enhanced_model_manager import EnhancedModelManager
print("✅ EnhancedModelManager import OK")
EOF

echo "✅ All critical imports available"
```

**Expected Output**: Three "import OK" messages

---

## Test 3: YAML Config Generation (10 minutes)

**Goal**: Verify ablations 2 & 4 can generate valid configs

```bash
cd /home/ubuntu/Djinn

python3 << 'EOF'
import sys
import yaml
sys.path.insert(0, '/home/ubuntu/Djinn')

from OSDI_Evaluation.ablation.scripts.ablation_session_arena import generate_config_for_arena

# Test arena config generation
config = generate_config_for_arena(arena_mb=64, use_signals=True)

print("Generated config for arena=64MB, mode=semantic:")
print(yaml.dump(config, default_flow_style=False))

# Verify key fields
assert 'experiment' in config, "Missing 'experiment' key"
assert 'workload' in config, "Missing 'workload' key"
assert config['workload']['total_agents'] == 80
print("✅ Arena config generation OK")
EOF

python3 << 'EOF'
import sys
import yaml
sys.path.insert(0, '/home/ubuntu/Djinn')

from OSDI_Evaluation.ablation.scripts.ablation_semantic_signals import generate_config_for_mode

# Test signal config generation
config = generate_config_for_mode(n_agents=40, mode='proactive', lambda_rate=0.2)

print("\nGenerated config for N=40, mode=proactive:")
print(yaml.dump(config, default_flow_style=False))

# Verify key fields
assert config['workload']['total_agents'] == 40
print("✅ Signal config generation OK")
EOF
```

**Expected Output**: Two YAML configs printed, both validation passes

---

## Test 4: Ablation 1 (OS Tax) - Micro Test (30 minutes)

**Goal**: Verify Ablation 1 runs end-to-end

⚠️ **REQUIRES**: Djinn server running on localhost:5556

```bash
# Start Djinn server in background (if not already running)
# cd /home/ubuntu/Djinn
# python3 djinn/server/server.py &

cd /home/ubuntu/Djinn
mkdir -p OSDI_Evaluation/ablation/test_results

# Run only micro_add operation (fastest)
python3 << 'EOF'
import sys
import json
sys.path.insert(0, '/home/ubuntu/Djinn')

# Try to import and check basic functionality
from OSDI_Evaluation.ablation.scripts.ablation_os_tax import OperationBenchmark

bench = OperationBenchmark('test', 'test description')
bench.record('native', 1.5)
bench.record('djinn_cold', 2.5)
bench.record('djinn_warm', 1.8)

stats = bench.get_stats('native')
print(f"✅ Benchmark recording works: {stats}")
assert stats['mean'] == 1.5
print("✅ Ablation 1 basic functionality OK")
EOF
```

**Expected Output**: "Ablation 1 basic functionality OK"

---

## Test 5: Ablation 3 (Plan Cache) Ghost Loader Test (20 minutes)

**Goal**: Verify ghost model loading works

⚠️ **NOTE**: This test requires HuggingFace model download (first time ~1.4GB)

```bash
cd /home/ubuntu/Djinn

python3 << 'EOF'
import sys
sys.path.insert(0, '/home/ubuntu/Djinn')

print("Testing ghost model loading...")

try:
    from djinn.core.ghost_loader import create_hf_ghost_model
    from transformers import AutoTokenizer
    
    print("Loading ghost model (this may take 1-2 minutes on first run)...")
    model = create_hf_ghost_model('gpt2')
    print(f"✅ Ghost model loaded: {type(model)}")
    
    # Check if model has expected attributes
    assert hasattr(model, 'generate'), "Model missing 'generate' method"
    assert hasattr(model, 'eval'), "Model missing 'eval' method"
    print("✅ Ghost model has expected methods")
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded")
    
    # Test tokenization
    prompt = "Hello, how are you?"
    prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    print(f"✅ Tokenization works: {prompt_ids.shape}")
    
    print("\n✅ Ablation 3 ghost loader OK")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
```

**Expected Output**: "Ablation 3 ghost loader OK"

---

## Test 6: Full Ablation Suite (6-8 hours)

**Goal**: Run complete ablation study

⚠️ **REQUIREMENTS**:
- Djinn server running
- CUDA GPU with 80GB (H100) or 24GB (L4)
- 6-8 hours wall-clock time
- LLaMA-2-13B model available

```bash
cd /home/ubuntu/Djinn

# Dry run to check command construction
python3 OSDI_Evaluation/ablation/scripts/run_all_ablations.py \
    --output-dir ./ablation_test_results \
    --skip-ablation 1 2 3 4  # Skip all (just test structure)

# If that passes, run single ablation
echo "Testing Ablation 1 (fastest)..."
timeout 600 python3 OSDI_Evaluation/ablation/scripts/ablation_os_tax.py \
    --output ./ablation_test_results/ablation_1_test.json

if [ $? -eq 0 ]; then
    echo "✅ Ablation 1 passed"
    # Verify output file exists and has data
    if [ -f ./ablation_test_results/ablation_1_test.json ]; then
        lines=$(wc -l < ./ablation_test_results/ablation_1_test.json)
        echo "✅ Output file created ($lines lines)"
    fi
else
    echo "❌ Ablation 1 failed"
fi

# If Ablation 1 passes, run full suite
echo ""
echo "Running full ablation suite (this will take 6-8 hours)..."
python3 OSDI_Evaluation/ablation/scripts/run_all_ablations.py \
    --output-dir ./ablation_final_results
```

**Expected Output**: JSON result files in output directory, LaTeX tables, PDF figures

---

## Test Checklist

Use this checklist to track test progress:

```
SYNTAX VALIDATION
[ ] ablation_os_tax.py compiles
[ ] ablation_session_arena.py compiles
[ ] ablation_plan_cache.py compiles
[ ] ablation_semantic_signals.py compiles
[ ] run_all_ablations.py compiles

IMPORT VALIDATION
[ ] remote_accelerator device import OK
[ ] ghost_loader import OK
[ ] EnhancedModelManager import OK

CONFIG GENERATION
[ ] Arena config generates valid YAML
[ ] Signal config generates valid YAML

FUNCTIONALITY TESTS
[ ] Ablation 1: Benchmark recording works
[ ] Ablation 3: Ghost model loads
[ ] Master runner: Command construction OK

FULL SUITE
[ ] Ablation 1 runs successfully
[ ] Ablation 2 runs successfully
[ ] Ablation 3 runs successfully
[ ] Ablation 4 runs successfully
[ ] Output files generated
[ ] LaTeX tables readable
[ ] PDF figures generated
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'djinn'"
**Fix**: Add path: `sys.path.insert(0, '/home/ubuntu/Djinn')`

### "remote_accelerator device not recognized"
**Fix**: Call `enable_remote_accelerator_device()` before using
```python
from djinn.core.device_compatibility import enable_remote_accelerator_device
enable_remote_accelerator_device()
```

### "Experiment timed out after 600s"
**Cause**: Djinn server slow or OOM
**Fix**: 
1. Check server logs
2. Ensure GPU has free memory
3. Reduce `total_agents` in config

### "No output JSON found"
**Cause**: run_poisson_experiment.py didn't write output
**Fix**:
1. Check run_poisson_experiment.py output directory exists
2. Review subprocess STDERR
3. Ensure config file is valid YAML

### "UnboundLocalError: metrics used before assignment"
**Cause**: Experiment failed to produce metrics
**Fix**: Check Djinn server is running and accessible

---

## Performance Notes

| Ablation | Duration | Notes |
|----------|----------|-------|
| Ablation 1 (OS Tax) | 10-15 min | Fastest, no OOM risk |
| Ablation 2 (Arena) | 2-3 hours | 4 arena × 2 modes × ~30 min each |
| Ablation 3 (Cache) | 15-20 min | Depends on model inference speed |
| Ablation 4 (Signals) | 2-3 hours | Binary search, many agent counts |
| **Total** | **6-8 hours** | Run overnight |

---

## Success Criteria

✅ **Pass if**:
- All scripts compile without syntax errors
- All imports resolve successfully
- Ablation 1 produces JSON with native/cold/warm metrics
- Ablation 2 produces JSON with arena size results
- Ablation 3 produces JSON with cache metrics
- Ablation 4 produces JSON with signal mode results
- LaTeX tables generate in output directory
- PDF figures generate in output directory
- All measurements are numerically valid (no NaN/Inf)

❌ **Fail if**:
- Any script has import errors
- Subprocesses exit with non-zero code
- Output files are empty or unparseable
- Measurements contain NaN or Inf
- LaTeX table generation fails

---

## Next Steps After Validation

1. **Review output quality**
   - Check JSON files for completeness
   - Verify LaTeX tables render correctly
   - Ensure PDF figures are readable

2. **Validate scientific findings**
   - Ablation 1: OS Tax shows <2x overhead amortization
   - Ablation 2: Arena size decomposition shows ~60% density contribution
   - Ablation 3: Plan cache shows >2x latency improvement
   - Ablation 4: Semantic signals show >1.5x density vs reactive

3. **Integrate into paper**
   - Insert LaTeX tables into Section 5.1
   - Include PDF figures in evaluation section
   - Add narrative explaining findings

4. **Polish for OSDI**
   - Add confidence intervals (if running n=3 repeats)
   - Add statistical significance tests
   - Add comprehensive figure captions

---

**Estimated time to validation**: 30 minutes (Tests 1-5) + 6-8 hours (Test 6)

**Recommendation**: Run Tests 1-5 first to catch issues before running full suite.
