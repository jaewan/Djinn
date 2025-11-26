# Phase 4: Production Load Test with Real Workloads

## Executive Summary

**Decision**: Phase 4 load testing uses **real HuggingFace models** (ResNet-50, ViT-Base, CLIP) instead of synthetic workloads, aligned with OSDI evaluation requirements.

## Rationale

### Why Real Workloads for OSDI?

1. **Evaluation Plan Alignment**: `docs/EvaluationPlan.md` explicitly specifies:
   - ResNet-50 (real model)
   - ViT-Base (real model)
   - CLIP (real multimodal model)
   
   These are **real, well-known models** that reviewers recognize.

2. **Reproducibility**: Real HuggingFace models are:
   - Standardized and reproducible
   - Well-documented
   - Easy for reviewers to verify

3. **Practical Value**: Demonstrates Djinn works with **production models**, not just synthetic approximations.

4. **OSDI Standards**: Top-tier systems conferences expect:
   - Real workloads that matter to practitioners
   - Models that demonstrate practical value
   - Results that generalize to production use

### Synthetic vs Real Workloads

| Aspect | Synthetic Workloads | Real Workloads |
|--------|-------------------|----------------|
| **Use Case** | Development/testing | Production evaluation |
| **Purpose** | Quick iteration, understanding behavior | OSDI evaluation, validation |
| **Reproducibility** | Good (controlled) | Excellent (HuggingFace) |
| **Reviewer Recognition** | Low | High |
| **Practical Value** | Limited | High |

## Implementation

### Real Workloads Added

1. **HuggingFaceVisionWorkload** (`hf_vision`)
   - Supports: ResNet-50, ViT-Base, EfficientNet
   - Uses `AutoModelForImageClassification`
   - Proper image preprocessing

2. **HuggingFaceMultimodalWorkload** (`hf_multimodal`)
   - Supports: CLIP, BLIP
   - Uses `AutoModel` with multimodal inputs
   - Handles image + text pairs

### Load Test Configuration

**User Mix** (from redesign plan):
- 60% LLM users: GPT-2 (can scale to GPT-J/LLaMA)
- 30% Vision users: ResNet-50 (can use ViT-Base)
- 10% Multimodal users: CLIP

**SLA Targets**:
- P99 latency < 100ms
- Error rate < 1%
- GPU utilization > 80%

## Usage

### Verify Real Models Work

```bash
# Start Djinn server
python -m djinn.server.server --node-id test --control-port 5555 --data-port 5556 --gpus 0

# Verify models can register and execute
python Evaluation/phase4_load_test/scripts/verify_real_models.py --server localhost:5556
```

### Run Smoke Test (Quick Verification)

```bash
# Run 1-minute smoke test
python Evaluation/phase4_load_test/scripts/run_production_load_test.py \
    --config Evaluation/phase4_load_test/configs/smoke_test.yaml
```

### Run Full Production Load Test

```bash
# Run 1-hour production load test
python Evaluation/phase4_load_test/scripts/run_production_load_test.py \
    --config Evaluation/phase4_load_test/configs/production_load_test.yaml
```

### Use Real Models in Other Experiments

```yaml
# Example: exp5_1_overhead config
workloads:
  - name: resnet50_real
    implementation: hf_vision
    params:
      model_id: "microsoft/resnet-50"
      batch_size: 8
      image_size: 224

  - name: clip_real
    implementation: hf_multimodal
    params:
      model_id: "openai/clip-vit-base-patch32"
      batch_size: 4
```

## Features

### GPU Metrics Collection

The load test automatically collects GPU utilization metrics:
- Average GPU utilization percentage
- Average GPU memory usage
- Per-second samples for detailed analysis

### Per-Class Metrics

Metrics are aggregated per user class:
- LLM users: latency, throughput, error rate
- Vision users: latency, throughput, error rate
- Multimodal users: latency, throughput, error rate

### SLA Validation

Automatically validates:
- P99 latency < target
- Error rate < target
- GPU utilization > target

## Files Created

- `Evaluation/common/workloads.py`: Added `HuggingFaceVisionWorkload` and `HuggingFaceMultimodalWorkload`
- `Evaluation/phase4_load_test/configs/production_load_test.yaml`: Full load test configuration
- `Evaluation/phase4_load_test/configs/smoke_test.yaml`: Quick smoke test configuration
- `Evaluation/phase4_load_test/scripts/run_production_load_test.py`: Load test orchestrator with GPU metrics
- `Evaluation/phase4_load_test/scripts/verify_real_models.py`: Model verification script
- `Evaluation/phase4_load_test/README.md`: This document

## Next Steps

1. ✅ **Verify Real Models Work**: Test that ResNet-50, ViT-Base, and CLIP can register and execute remotely
2. ✅ **Add GPU Metrics**: Integrate GPU utilization monitoring into load test
3. ⏳ **Run Smoke Test**: Execute short smoke test to verify infrastructure
4. ⏳ **Scale Up**: Optionally use larger models (GPT-J, ViT-Large, CLIP-Large) for more realistic load
