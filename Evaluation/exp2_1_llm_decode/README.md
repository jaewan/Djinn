# Experiment 2.1 – LLM Decode with KV Cache

Hero experiment for `docs/EvaluationPlan.md §6.2`.  Objective: prove Djinn’s
semantic awareness transforms disaggregated execution during LLM decoding by
demonstrating 30–57× lower per-token latency, 10³–10⁵× less data transfer, and
dramatically higher GPU utilization relative to semantic-blind variants.

## Workload
- **Model:** default `meta-llama/Llama-2-7b-hf` (override via CLI flag)
- **Prompt:** 72-token prompt (see `configs/prompt.txt`)
- **Output:** 50 new tokens (`max_new_tokens=50`)
- **Batch size:** 1 (repeatable)
- **Transport:** Client CPU → Djinn server (A100-80GB) over 25 Gbps link

## Baselines
| Name | Location | Purpose | Config reference |
|------|----------|---------|------------------|
| `native_pytorch` | local GPU | Upper bound | `configs/baselines.yaml` |
| `semantic_blind` | Djinn, semantics disabled | Quantify worst-case | `configs/baselines.yaml` |
| `partially_aware` | Djinn, caching w/o KV semantics | Incremental | `configs/baselines.yaml` |
| `full_djinn` | Djinn, all semantics | Target system | `configs/baselines.yaml` |

Baseline toggles are documented in `configs/baselines.yaml`.  The local PyTorch
runner is implemented in `scripts/run_local_baseline.py`; Djinn variants will
reuse the shared harness in a follow-up patch once QoS + semantic knobs are wired
through config.

## Metrics
- Per-token latency (ms) and total decode time
- Data transferred host↔device per token (computed via payload logs)
- GPU SM utilization (%), memory BW utilization (%)
- Queue/compute/transfer breakdown (via telemetry timestamps)
- Energy / power samples (optional via `nvidia-smi --query-gpu=power.draw`)

## Data Capture Plan
1. Run 30 iterations per baseline (discard first 5 for warm-up).
2. Save raw run JSON under `results/<baseline>/<timestamp>.json` with schema:
   ```json
   {
     "baseline": "native_pytorch",
     "model_id": "meta-llama/Llama-2-7b-hf",
     "prompt_tokens": 72,
     "new_tokens": 50,
     "batch_size": 1,
     "runs": [
       {
         "run_id": 1,
         "total_ms": 812.4,
         "per_token_ms": 16.2,
         "tokens_generated": 50,
         "gpu_util_pct": 84.3,
         "memory_bw_gbps": 1.2,
         "host_to_device_mb": 0.8,
         "device_to_host_mb": 0.1
       }
     ],
     "aggregates": {
       "mean_total_ms": 812.4,
       "p95_total_ms": 830.1,
       "mean_per_token_ms": 16.2
     }
   }
   ```
3. `scripts/analyze_results.py` (placeholder) will merge per-baseline JSON into
   CSV for plotting.

## Instructions
1. Ensure Djinn server is running for remote baselines; for PyTorch baseline a
   single A100 is sufficient.
2. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   pip install transformers accelerate datasets pynvml
   ```
3. Run local baseline as smoke test:
   ```bash
   python Evaluation/exp2_1_llm_decode/scripts/run_local_baseline.py \
     --model-id meta-llama/Llama-2-7b-hf \
     --prompt-file Evaluation/exp2_1_llm_decode/configs/prompt.txt \
     --output-dir Evaluation/exp2_1_llm_decode/results/native_pytorch \
     --runs 3
   ```
4. Validate JSON output and aggregate stats.
5. Integrate telemetry hooks for Djinn baselines (pending).

## Remote Djinn Baselines
When the server is available, run Djinn backends by passing `--backend djinn`.
The device string switches to `privateuseone:<N>` so execution happens on the remote
accelerator; `GENIE_SERVER_ADDRESS` or `--djinn-server` controls the target port.

```
source .venv/bin/activate
export DJINN_ENABLE_PRIVATEUSE1_DEVICE=1
python Evaluation/exp2_1_llm_decode/scripts/run_local_baseline.py \
  --backend djinn \
  --djinn-server localhost:5556 \
  --model-id sshleifer/tiny-gpt2 \
  --prompt-file Evaluation/exp2_1_llm_decode/configs/prompt.txt \
  --runs 1 \
  --warmup-runs 1 \
  --tag djinn_remote_smoke \
  --output-dir Evaluation/exp2_1_llm_decode/results/djinn_remote
```

Tail the Djinn server logs to ensure transport/connectivity and compare the
`backend`, `djinn_server`, and latency fields in the resulting JSON.

## Ray Keep-Alive Baseline
To reproduce the "persistent actor" comparison, launch Ray locally (or connect to
an existing cluster) and run:

```bash
# For local testing: Start Ray first
ray start --head --num-gpus=1 --include-dashboard=false

# Then run the baseline
python Evaluation/exp2_1_llm_decode/scripts/run_ray_keepalive_agents.py \
  --model-id meta-llama/Llama-2-7b-hf \
  --prompt-file Evaluation/exp2_1_llm_decode/configs/prompt.txt \
  --agent-counts 1 2 4 8 \
  --iterations 1 \
  --sleep-seconds 10 \
  --output-dir Evaluation/exp2_1_llm_decode/results/ray_keepalive
```

Each agent receives a dedicated Ray actor that pins the full model + KV cache on
GPU memory for the duration of the experiment.  The output JSON matches the
structure used by other hero baselines, enabling direct comparison of per-stage
latency across concurrency levels.

**For distributed setup**, see the "Distributed Setup" section below.

## Ray Serverless Baseline
For the stateless/serverless comparison (load/unload per step), run:

```bash
# For local testing: Start Ray first
ray start --head --num-gpus=1 --include-dashboard=false

# Then run the baseline
python Evaluation/exp2_1_llm_decode/scripts/run_ray_serverless_agents.py \
  --model-id meta-llama/Llama-2-7b-hf \
  --prompt-file Evaluation/exp2_1_llm_decode/configs/prompt.txt \
  --agent-counts 1 2 4 8 \
  --iterations 1 \
  --sleep-seconds 10 \
  --output-dir Evaluation/exp2_1_llm_decode/results/ray_serverless
```

Each `Reason`/`Reflect` step executes as a fresh Ray task that reloads the
weights, recomputes the prompt, and discards state afterwards, capturing the
cold-start costs inherent to this design.

**For distributed setup**, see the "Distributed Setup" section below.

## Experiment 2.2: Agent Scaling (Hero Experiment)

This is the main hero figure for the paper. We compare three baselines on the
"Parking Lot" problem: running N concurrent agents that do Reason → Act (sleep) → Reflect.

### Baselines

1. **Ray Keep-Alive** (`run_ray_keepalive_agents.py`):
   - Each agent holds model + KV cache exclusively
   - Expected: Flat latency, but OOMs at low N (~2-3 agents)

2. **Ray Serverless** (`run_ray_serverless_agents.py`):
   - Each step reloads model + KV cache
   - Expected: Scales to N=32, but latency is high (~5-8s per step)

3. **Djinn** (`run_djinn_agents.py`):
   - Shared weights (Text Segment), per-session KV cache (Data Segment)
   - **True incremental decode**: REFLECT phase sends only new tokens, reusing KV cache
   - Expected: Scales to N=32 with low latency (~150ms per step)
   - Expected: REFLECT latency ~5-20x faster than REASON (proves KV reuse)

### Quick Start (Smoke Test)

**Option 1: Sequential Runner (Recommended for Local Testing)**

Run all three baselines sequentially with automatic server management:

```bash
# Ensure Ray cluster is running
ray start --head --num-gpus=1 --include-dashboard=false

# Run all baselines sequentially (server starts/stops automatically)
python Evaluation/exp2_1_llm_decode/scripts/run_all_baselines_sequential.py \
  --agent-counts 1 2 \
  --iterations 1 \
  --new-tokens 10 \
  --sleep-seconds 1.0 \
  --model-id sshleifer/tiny-gpt2 \
  --output-dir /tmp/test_agent_scaling
```

**Option 2: Manual Execution (Local)**

```bash
# Terminal 1: Start Djinn server
python -m djinn.server.server_main --gpu 0 --port 5556

# Terminal 2: Run smoke test (1-2 agents only)
# Ray Keep-Alive
python Evaluation/exp2_1_llm_decode/scripts/run_ray_keepalive_agents.py \
  --agent-counts 1 2 \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 1 \
  --output-dir Evaluation/exp2_1_llm_decode/results/ray_keepalive

# Ray Serverless
python Evaluation/exp2_1_llm_decode/scripts/run_ray_serverless_agents.py \
  --agent-counts 1 2 \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 1 \
  --output-dir Evaluation/exp2_1_llm_decode/results/ray_serverless

# Djinn (requires Djinn server running)
python Evaluation/exp2_1_llm_decode/scripts/run_djinn_agents.py \
  --agent-counts 1 2 \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 1 \
  --djinn-server localhost:5556 \
  --output-dir Evaluation/exp2_1_llm_decode/results/djinn_agents
```

### Distributed Setup (Server + Client Machines) (Real experiment)

For production experiments, run servers on GPU machines and clients on separate machines.

**⚠️ CRITICAL: Ray Workflow Difference**

Ray requires a different workflow than Djinn:
- **Djinn**: Client connects directly to server (no setup needed)
- **Ray**: Client must **join the Ray cluster as a worker node** before running scripts

**Correct Ray Workflow:**
1. Server: `ray start --head ...` (starts Ray head node)
2. Client: `ray start --address='SERVER_IP:6379'` (joins cluster as worker)
3. Client: Run scripts (auto-connect, no `--ray-address` flag needed)

**Wrong Ray Workflow (will fail):**
- Client: Run scripts with `--ray-address` without joining cluster first
- Error: `Can't find a node_ip_address.json file`

**⚠️ AWS Security Group Port Requirements (IMPORTANT):**

| Service | Ports Required | Notes |
|---------|---------------|-------|
| **Ray** | **6379** + **10001-10020** | GCS port (6379) + Raylet/Object Store range (10001-10020) |
| **Djinn** | 5555, 5556 | Control + Data planes |

**Common Error**: If you only opened ports 5000-6000, Ray will fail because it needs ports **10001-10020** for worker communication!

**Quick Fix for AWS:**
1. Go to EC2 → Security Groups → Select your server's security group
2. Edit Inbound Rules → Add:
   - Custom TCP, Port **6379**, Source: Client security group (or allow all from client SG)
   - Custom TCP, Port Range **10001-10020**, Source: Client security group
3. **For bidirectional communication**, also ensure outbound rules allow these ports, or allow all traffic between the security groups
4. Save rules and retry connection

#### Prerequisites

1. **Server Machine**: GPU-enabled (A100/H100) with CUDA installed
2. **Client Machine**: CPU-only is fine, needs network access to server
3. **Network**: Ensure firewall allows:
   - **Ray**: 
     - GCS port: 6379 (default, required)
     - Raylet/Object Store: 10001-10020 (required for worker communication, bidirectional)
     - Dashboard (optional): 8265
   - **Djinn**: Ports 5555 (control), 5556 (data)

**AWS Security Group Configuration:**

For Ray clusters, you need to open these ports in your AWS Security Group:

```
Type: Custom TCP
Port Range: 6379
Source: Client machine IP or security group

Type: Custom TCP  
Port Range: 10001-10020
Source: Client machine IP or security group

Type: Custom TCP (optional)
Port Range: 8265
Source: Your IP (for dashboard access)
```

**Important Notes:**
- Ray uses port **6379** by default (not 5379)
- Ports **10001-10020** are required for Raylet/Object Store communication (bidirectional)
- If you only opened ports 5000-6000, Ray will fail because it needs ports 10001-10020!
- For distributed setups, **allow all traffic between server and client security groups** to avoid port conflicts

**⚠️ CRITICAL: Check Server-Side Firewall (ufw)**

Even if AWS Security Group allows the ports, **ufw on the server machine might be blocking them**:

```bash
# On SERVER machine, check ufw status
sudo ufw status

# If ufw is active and blocking, allow the ports:
sudo ufw allow 6379/tcp
sudo ufw allow 10001:10020/tcp

# Or disable ufw if you're using AWS Security Groups:
sudo ufw disable  # (only if you trust AWS Security Groups)
```

**Common Issue**: Security group allows ports, but `ufw` on server blocks them → ports appear unreachable from client!

#### 1. Ray Baselines (Keep-Alive & Serverless)

**CRITICAL: Client Must Join Ray Cluster First**

For distributed Ray experiments, the **client machine must join the Ray cluster as a worker node** before running the scripts. This is different from Djinn, where the client connects directly.

**On Server Machine:**

```bash
# Step 1: Start Ray head node (replace SERVER_IP with actual server IP)
# Use the server's private IP (e.g., 172.31.33.48) or public IP
ray start --head \
  --node-ip-address=SERVER_IP \
  --port=6379 \
  --ray-client-server-port=10000 \
  --min-worker-port=10001 \
  --max-worker-port=10020 \
  --num-gpus=1 \
  --include-dashboard=false

# Verify Ray is running:
ray status
# Should show: "Active: 1 node_..."
```

**Port Configuration:**
- **GCS port**: 6379 (default, recommended)
- **Client server port**: 10000 (avoids conflict with worker ports)
- **Worker ports**: 10001-10020 (must be open in AWS Security Group)

**On Client Machine:**

```bash
# Step 2: Join the Ray cluster as a worker node
# This is REQUIRED - the client must be part of the cluster
ray start --address='SERVER_IP:6379'

# Verify connection:
ray status
# Should show: "Active: 2 nodes..." (server + client)

# Step 3: Run the scripts (NO --ray-address needed, auto-connects)
python Evaluation/exp2_1_llm_decode/scripts/run_ray_keepalive_agents.py \
  --agent-counts 1 2 4 8 \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 10 \
  --output-dir Evaluation/exp2_1_llm_decode/results/ray_keepalive

python Evaluation/exp2_1_llm_decode/scripts/run_ray_serverless_agents.py \
  --agent-counts 1 2 4 8 16 32 \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 10 \
  --output-dir Evaluation/exp2_1_llm_decode/results/ray_serverless
```

**Important Notes:**
- The client machine **must run `ray start --address='SERVER_IP:6379'`** before running scripts
- Scripts will auto-connect to the cluster (no `--ray-address` flag needed)
- If you see `Can't find a node_ip_address.json file`, the client hasn't joined the cluster yet
- Both machines must have ports 10001-10020 open in AWS Security Group (bidirectional communication)

#### 2. Djinn Baseline

**On Server Machine:**

```bash
# Start Djinn server (replace SERVER_IP with actual server IP)
# Bind to 0.0.0.0 to accept connections from client machines
# OSDI FIX: Increased default concurrency limits for high-concurrency experiments
python -m djinn.server.server_main \
  --gpu 0 \
  --port 5556 \
  --host 0.0.0.0 \
  --max-concurrent 64 \
  --max-vram-gb 80

# Server will listen on:
# - Control plane: SERVER_IP:5555
# - Data plane: SERVER_IP:5556
```

**Concurrency Configuration (OSDI Fixes):**

The server now supports high-concurrency experiments with configurable limits:

- `--max-concurrent` (default: 64): Maximum concurrent requests the server accepts
  - Set to match your agent count: `--max-concurrent 32` for 32 agents
  - Replaces hard-coded limit of 10 that was causing "Concurrency limit exceeded" errors at N>=10

- `--max-vram-gb` (default: 80): Maximum VRAM allocation per tenant
  - Set to your GPU capacity: `--max-vram-gb 80` for A100-80GB

For distributed experiments with 16-32 agents:

```bash
python -m djinn.server.server_main \
  --gpu 0 \
  --port 5556 \
  --host 0.0.0.0 \
  --max-concurrent 64 \
  --max-vram-gb 80
```

**On Client Machine:**

```bash
# Connect to Djinn server (replace SERVER_IP with actual server IP)
python Evaluation/exp2_1_llm_decode/scripts/run_djinn_agents.py \
  --agent-counts 1 2 4 8 16 32 \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 10 \
  --djinn-server SERVER_IP:5556 \
  --output-dir Evaluation/exp2_1_llm_decode/results/djinn_agents
```

**Verifying Connectivity:**

```bash
# From client machine, test Djinn server connection
nc -zv SERVER_IP 5556

# Test Ray cluster connection
ray status --address SERVER_IP:6379
```

#### 3. Sequential Runner (Distributed)

The sequential runner can also work in distributed mode, but requires manual server management:

**On Server Machine:**

```bash
# Terminal 1: Start Ray head
ray start --head --node-ip-address=SERVER_IP --port=6379 --num-gpus=1

# Terminal 2: Start Djinn server
python -m djinn.server.server_main --gpu 0 --port 5556 --host 0.0.0.0
```

**On Client Machine:**

```bash
# Note: Sequential runner currently assumes localhost for Djinn server
# For distributed setup, run baselines individually (see above)
# Or modify run_all_baselines_sequential.py to accept --djinn-server-ip
```

#### Troubleshooting Distributed Setup

**Connection Refused:**
- Verify server IP is correct (use `hostname -I` or `ip addr` on server)
- Check firewall: `sudo ufw allow 5556/tcp` (Djinn) or `sudo ufw allow 6379/tcp` (Ray)
- Ensure server binds to `0.0.0.0`, not `127.0.0.1`

**Ray Connection Error: "Can't find a node_ip_address.json file"**

If you see: `Can't find a node_ip_address.json file... Did you do ray start or ray.init on this host?`

**Root Cause**: The client machine hasn't joined the Ray cluster yet. Ray requires the client to be a worker node in the cluster, not just connect as a pure client.

**Solution:**

1. **On CLIENT machine, join the Ray cluster first:**
   ```bash
   # Stop any existing Ray instance
   ray stop
   
   # Join the remote Ray cluster as a worker node
   ray start --address='SERVER_IP:6379'
   
   # Verify connection
   ray status
   # Should show: "Active: 2 nodes..." (server + client)
   ```

2. **Then run the scripts (no --ray-address needed):**
   ```bash
   python Evaluation/exp2_1_llm_decode/scripts/run_ray_serverless_agents.py \
     --agent-counts 1 2 4 8 16 32 \
     ...
   ```

**If joining the cluster fails, check:**

1. **AWS Security Group Ports (CRITICAL):**
   - GCS port: 6379 (must be open)
   - Raylet/Object Store: 10001-10020 (must be open, bidirectional)
   - **Common mistake**: Opening only ports 5000-6000 will NOT work!
   
   **Fix**: In AWS Console → EC2 → Security Groups → Edit Inbound Rules:
   - Add rule: Custom TCP, Port 6379, Source: Client security group or IP
   - Add rule: Custom TCP, Port Range 10001-10020, Source: Client security group or IP
   - **For bidirectional communication**, also add outbound rules or allow all traffic between security groups

2. **Verify Ray head is running on server:**
   ```bash
   # On server machine
   ray status
   # Should show: "Active: 1 node_..."
   ```

3. **Test network connectivity:**
   ```bash
   # On client machine
   nc -zv SERVER_IP 6379      # GCS port
   nc -zv SERVER_IP 10001     # Raylet port (sample)
   ```

4. **Check firewall (ufw) on server:**
   ```bash
   # On server machine
   sudo ufw allow 6379/tcp
   sudo ufw allow 10001:10020/tcp
   # Or disable ufw if using AWS Security Groups
   ```

5. **Ray version mismatch:**
   ```bash
   # On both server and client
   ray --version
   # Versions should match (or be compatible)
   ```

**Djinn Connection Failed:**
- Check server logs for errors
- Verify model registration succeeds (check server stdout)
- Test with `curl SERVER_IP:5556` (should fail gracefully, not timeout)

**Concurrency Limit Exceeded (OSDI Fix):**

If you see: `Concurrency limit exceeded: X active requests >= Y limit`

**Solution**: Increase server concurrency when starting Djinn server:

```bash
# Adjust --max-concurrent based on your agent count
# For N agents, use --max-concurrent at least N+10 for headroom
python -m djinn.server.server_main \
  --gpu 0 \
  --port 5556 \
  --host 0.0.0.0 \
  --max-concurrent 64  # Increased from default 4
```

Or use environment variable:

```bash
export GENIE_QOS_MAX_CONCURRENCY=64
python -m djinn.server.server_main --gpu 0 --port 5556 --host 0.0.0.0
```

**Root Cause**: Prior to OSDI fixes, Djinn had hard-coded concurrency limits:
- QoS scheduler: 4 concurrent requests
- Tenant policy: 10 concurrent requests

These were insufficient for agent scaling experiments with N >= 10.

**Ray Module Import Error:**

If you see: `ModuleNotFoundError: No module named 'Evaluation'`

**Solution**: Ensure you're running from the repo root:

```bash
cd ~/Djinn  # Navigate to repo root
python Evaluation/exp2_1_llm_decode/scripts/run_ray_serverless_agents.py \
  --agent-counts 1 2 4 8 16 32 \
  ...
```

**Note**: If you've joined the Ray cluster with `ray start --address='SERVER_IP:6379'`, you don't need the `--ray-address` flag in the scripts.

### Full Experiment (Paper Results)

**Local Setup (Single Machine):**

```bash
# Ray Keep-Alive: Default --gpu-per-actor 0.01 forces physical OOM (not scheduler queuing)
python Evaluation/exp2_1_llm_decode/scripts/run_ray_keepalive_agents.py \
  --agent-counts 1 2 3 4 \
  --stop-on-oom \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 10

python Evaluation/exp2_1_llm_decode/scripts/run_ray_serverless_agents.py \
  --agent-counts 1 2 4 8 16 32 \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 10

python Evaluation/exp2_1_llm_decode/scripts/run_djinn_agents.py \
  --agent-counts 1 2 4 8 16 32 \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 10 \
  --djinn-server localhost:5556
```

**Distributed Setup (Server + Client):**

**Step 1: Start servers on the server machine:**

```bash
# Terminal 1: Start Ray head (replace SERVER_IP with actual server IP)
ray start --head \
  --node-ip-address=SERVER_IP \
  --port=6379 \
  --ray-client-server-port=10000 \
  --min-worker-port=10001 \
  --max-worker-port=10020 \
  --num-gpus=1 \
  --include-dashboard=false

# Terminal 2: Start Djinn server with high-concurrency settings (OSDI FIX)
python -m djinn.server.server_main \
  --gpu 0 \
  --port 5556 \
  --host 0.0.0.0 \
  --max-concurrent 64 \
  --max-vram-gb 80
```

**Step 2: Join Ray cluster from client machine:**

```bash
# On CLIENT machine - CRITICAL: Must join cluster first!
ray start --address='SERVER_IP:6379'

# Verify connection
ray status
# Should show: "Active: 2 nodes..." (server + client)
```

**Step 3: Run baselines from client machine:**

```bash
# Ray Keep-Alive (no --ray-address needed, auto-connects)
python Evaluation/exp2_1_llm_decode/scripts/run_ray_keepalive_agents.py \
  --agent-counts 1 2 3 4 \
  --stop-on-oom \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 10 \
  --output-dir Evaluation/exp2_1_llm_decode/results/ray_keepalive

# Ray Serverless (no --ray-address needed, auto-connects)
python Evaluation/exp2_1_llm_decode/scripts/run_ray_serverless_agents.py \
  --agent-counts 1 2 4 8 16 32 \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 10 \
  --output-dir Evaluation/exp2_1_llm_decode/results/ray_serverless

# Djinn (requires --djinn-server flag)
python Evaluation/exp2_1_llm_decode/scripts/run_djinn_agents.py \
  --agent-counts 1 2 4 8 16 32 \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 10 \
  --djinn-server SERVER_IP:5556 \
  --output-dir Evaluation/exp2_1_llm_decode/results/djinn_agents
```

**Note:** The default `--gpu-per-actor 0.01` for Ray Keep-Alive bypasses Ray's logical scheduler and forces physical GPU memory contention, ensuring OOM crashes (not infinite queuing) at low N. This properly demonstrates the "Parking Lot" problem.

### Analysis

Once all three baselines complete:
```bash
python Evaluation/exp2_1_llm_decode/scripts/analyze_agent_scaling.py
```

This generates:
- **Figure 6**: P99 Latency vs Number of Concurrent Agents
- **Table 2**: Summary statistics (mean, p50, p99, OOM thresholds)
- **Claims validation**: Tenant density and latency improvement ratios

### Validation Checklist

- [ ] Ray Keep-Alive OOMs at expected N (2-3 agents) with physical memory exhaustion (not scheduler queuing)
- [ ] Ray Serverless scales to N=32, latency grows >5000ms
- [ ] Djinn scales to N=32, latency stays <200ms
- [ ] Djinn REFLECT latency is significantly lower than REASON latency (proves incremental decode + KV reuse)
- [ ] Djinn session persistence verified (check server logs: "Reusing session")
- [ ] Analysis script correctly computes 20x density and 10x latency claims

### Implementation Notes (OSDI Quality Fixes)

The evaluation includes two critical fixes for scientific rigor:

1. **Incremental Decode for Djinn**: The REFLECT phase sends only new tokens (not full history) to enable true KV cache reuse. This is tracked via `kv_processed_len` in `run_djinn_agents.py`.

2. **Physical OOM for Ray Keep-Alive**: The default `--gpu-per-actor 0.01` bypasses Ray's logical scheduler, forcing physical GPU memory contention. This ensures OOM crashes (not infinite queuing) at low N, properly demonstrating the "Parking Lot" problem.

## Open Tasks
- [ ] Run full agent scaling experiment on production cluster
- [ ] Verify claims (20x density, 10x latency) match expected paper values
- [ ] Generate final figures for OSDI submission

