# Djinn Ablation Studies

This directory contains OSDI-quality ablation studies to isolate and quantify the contribution of individual Djinn system components, including a **model size sweep** demonstrating overhead amortization.

## Overview

The ablation studies answer two critical scientific questions:

1. **OS Tax (Ablation 1)**: What is the latency cost of Djinn's runtime interposition layer? 
   - **Model Sweep**: Tested on GPT-2 (124M), Phi-2 (2.7B), Llama-2-7B to demonstrate amortization
2. **Plan Cache (Ablation 2)**: Does meta-simulation caching eliminate planning overhead?

## Prerequisites

### Hardware
- NVIDIA GPU (tested on H100 80GB and L4 24GB)
- CUDA-capable system

### Software
- Python 3.8+
- PyTorch with CUDA support
- Djinn installed and configured
- Required packages: `transformers`, `numpy`, `aiohttp`

## Quick Start

### 1. Start Djinn Server

In Terminal 1:
```bash
cd /home/ubuntu/Djinn
python3 -m djinn.server --port 5556 --gpu 0
```

Wait until you see:
```
✅ Server ready on 127.0.0.1:5556
```

### 2. Run All Ablations

In Terminal 2:
```bash
cd /home/ubuntu/Djinn/OSDI_Evaluation/ablation/scripts
python3 run_ablations.py
```

This will:
- Run both ablations sequentially
- Save results to `../results/`
- Generate LaTeX tables for the paper

**Expected runtime**: ~5-10 minutes total

### 3. View Results

Results are saved in `OSDI_Evaluation/ablation/results/`:

**Single Model Run**:
- `ablation_os_tax_gpt2.json` - OS Tax (GPT-2) raw data
- `ablation_plan_cache.json` - Plan Cache raw data
- `os_tax_table.tex` - LaTeX table for OS Tax
- `plan_cache_table.tex` - LaTeX table for Plan Cache
- `ablation_summary.json` - Summary of all ablations

**Model Sweep Run**:
- `ablation_os_tax_gpt2.json`, `ablation_os_tax_microsoft_phi_2.json`, `ablation_os_tax_meta_llama_Llama_2_7b_hf.json` - Per-model raw data
- `model_sweep_table.tex` - **Publication-ready table showing amortization**
- `model_sweep_latency.pdf`, `model_sweep_overhead.pdf` - Visualization plots
- `ablation_summary.json` - Summary with all models

## Individual Ablations

### Ablation 1: OS Tax

Measures the overhead of Djinn's interposition layer by comparing **identical model execution** on native PyTorch vs Djinn remote execution. This ensures an apples-to-apples comparison.

**Single Model (GPT-2 only)**:
```bash
python3 ablation_os_tax.py --server 127.0.0.1:5556 --warmup 100 --iters 1000
```

**Model Sweep (Recommended)**:
```bash
python3 ablation_os_tax.py --server 127.0.0.1:5556 --model_id gpt2 --model_name "GPT-2"
python3 ablation_os_tax.py --server 127.0.0.1:5556 --model_id microsoft/phi-2 --model_name "Phi-2"
python3 ablation_os_tax.py --server 127.0.0.1:5556 --model_id meta-llama/Llama-2-7b-hf --model_name "Llama-2-7B"
```

Or use the master runner:
```bash
python3 run_ablations.py --model-sweep --skip-plan-cache
```

**Parameters:**
- `--server`: Djinn server address (default: `127.0.0.1:5556`)
- `--model_id`: HuggingFace model ID (default: `gpt2`)
- `--model_name`: Display name for model (default: `GPT-2`)
- `--warmup`: Warmup iterations (default: 100)
- `--iters`: Measurement iterations (default: 1000)
- `--output`: Output directory (default: `../results/`)

**Methodology:**
1. Measures micro-operations (add, matmul, TinyModel) for context
2. **CRITICAL**: Measures target model inference natively (baseline)
3. Measures target model inference through Djinn (remote)
4. Computes OS Tax = Djinn - Native for the same model

**Results Summary:**
| Model | Native | Djinn | Overhead |
|-------|--------|-------|----------|
| GPT-2 (124M) | 7.31ms | 9.33ms | **27.8%** |
| Phi-2 (2.7B) | 24.65ms | 26.65ms | **8.1%** |
| Llama-2-7B | 28.58ms | 31.95ms | **11.8%** ✅ |

**Key Finding**: Djinn's overhead is approximately **constant (~2-3ms)**, validating the fixed-cost model. For production 7B models, overhead is <12%.

**Output:**
- `ablation_os_tax_<model>.json`: Per-model raw latency measurements
- `model_sweep_table.tex`: Publication-ready LaTeX table
- `model_sweep_latency.pdf`, `model_sweep_overhead.pdf`: Visualization plots

### Ablation 2: Plan Cache

Measures the effectiveness of meta-simulation plan caching by comparing cold (first request) vs warm (cached) execution latency.

```bash
python3 ablation_plan_cache.py --server 127.0.0.1:5556 --uniform 10 --varied 10 --trials 3
```

**Parameters:**
- `--server`: Djinn server address (default: `127.0.0.1:5556`)
- `--uniform`: Number of uniform workload requests (default: 10)
- `--varied`: Number of varied workload requests (default: 10)
- `--trials`: Number of trials for uniform workload (default: 3)
- `--output`: Output directory (default: `../results/`)

**Expected Results:**
- Cold (first request): ~20-50ms
- Warm (cached): ~0.5-1ms
- Speedup: 20-50× for uniform workloads
- Cache hit rate: >95% for uniform workloads

**Output:**
- `ablation_plan_cache.json`: Raw latency measurements and cache stats
- `plan_cache_table.tex`: LaTeX table for paper

## Methodology

### Statistical Rigor

All ablations follow these principles:
- **Warmup**: 100+ iterations excluded from measurement
- **Multiple trials**: 3+ trials for confidence intervals
- **GPU synchronization**: All timing measurements synchronized with CUDA
- **Metrics**: Report mean, median, P99, standard deviation
- **Reproducibility**: All results saved as JSON with configuration

### Scientific Validity

1. **OS Tax**: Compares identical operations (native vs Djinn) to isolate interposition overhead
2. **Plan Cache**: Uses controlled workloads (uniform vs varied) to measure cache effectiveness
3. **Server metrics**: Fetches actual cache statistics from server `/metrics/vmu` endpoint

### Fair Comparison

- Native baseline uses local GPU (no network overhead)
- Djinn path uses actual server execution (includes network + RPC + planning)
- All measurements include full end-to-end latency (no artificial exclusions)

## Troubleshooting

### Server Connection Failed

**Error**: `Failed to connect to server`

**Solution**:
1. Verify server is running: `ps aux | grep djinn.server`
2. Check server logs for errors
3. Verify port 5556 is not blocked: `netstat -an | grep 5556`

### CUDA Out of Memory

**Error**: `CUDA out of memory`

**Solution**:
1. Reduce batch size or model size in ablation scripts
2. Restart server to clear GPU memory
3. Use smaller GPU (L4) for testing

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'djinn'`

**Solution**:
```bash
cd /home/ubuntu/Djinn
pip install -e .
```

### Metrics Endpoint Not Found

**Error**: `Failed to fetch metrics: HTTP 404`

**Solution**:
1. Verify server version includes metrics endpoint
2. Check server is running with diagnostics enabled
3. Default metrics port is 9095

## Integration with Paper

The LaTeX tables and plots generated by these ablations can be directly included in the paper:

### Model Sweep Results (Recommended)
```latex
\input{OSDI_Evaluation/ablation/results/model_sweep_table.tex}
\includegraphics[width=\columnwidth]{OSDI_Evaluation/ablation/results/model_sweep_overhead.pdf}
```

### Single Model Results (If Space Constrained)
```latex
\input{OSDI_Evaluation/ablation/results/os_tax_table.tex}
\input{OSDI_Evaluation/ablation/results/plan_cache_table.tex}
```

For detailed interpretation and scientific context, see `MODEL_SWEEP_ANALYSIS.md`.

## Directory Structure

```
ablation/
├── scripts/
│   ├── ablation_os_tax.py              # Ablation 1: OS Tax (supports model sweep)
│   ├── ablation_plan_cache.py          # Ablation 2: Plan Cache
│   ├── run_ablations.py                # Master runner (supports --model-sweep)
│   ├── generate_model_sweep_table.py   # Analysis & LaTeX generation
│   ├── plot_model_sweep.py             # Visualization generation
│   └── validate_ablations.py           # Validation utility
├── results/                            # Output directory (created automatically)
│   ├── ablation_os_tax_gpt2.json                          # OS Tax (GPT-2)
│   ├── ablation_os_tax_microsoft_phi_2.json             # OS Tax (Phi-2)
│   ├── ablation_os_tax_meta_llama_Llama_2_7b_hf.json    # OS Tax (Llama-7B)
│   ├── ablation_plan_cache.json                          # Plan Cache
│   ├── model_sweep_table.tex                             # Main results table
│   ├── model_sweep_latency.pdf                           # Latency comparison plot
│   ├── model_sweep_overhead.pdf                          # Overhead amortization plot
│   ├── ablation_summary.json                             # Summary metadata
│   └── ... (additional LaTeX tables)
├── MODEL_SWEEP_ANALYSIS.md                 # Comprehensive analysis document
├── README.md                               # This file
└── __init__.py                             # Python package marker
```

## Citation

If you use these ablation studies, please cite:

```bibtex
@inproceedings{djinn2025,
  title={Djinn: A Tensor Operating System for Interactive AI Workloads},
  author={...},
  booktitle={OSDI},
  year={2025}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
