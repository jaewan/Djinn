# Plan: Wire Real Djinn Drivers for Experiments 4.1 and 4.2

Goal: Replace the synthetic drivers in `exp4_1_scalability` and `exp4_2_qos` so
the harnesses hit the actual Djinn coordinator (per §7 of `docs/0_OVERVIEW.md`
and §6.4 of `docs/EvaluationPlan.md`). Steps below assume a single dev server
with one Djinn server process pinned to GPU0 and the client harness sharing the
same box (loopback TCP).

---

## Shared prerequisites
1. **Server bring-up**
   - Launch `DjinnServer` with QoS enabled (`GENIE_ENABLE_QOS=1`) and the
     current BasicQoS scheduler configuration.
   - Ensure the preferred GPU is bound and that the pre-allocated VMU slabs are
     visible in logs (`ServerState.initialize` confirmation).
2. **Client environment**
   - Reuse `EnhancedModelManager` from `djinn/core/enhanced_model_manager.py`.
   - Instantiate once per harness, call `register_model` with the chosen HF
     model (LLaMA-7B for Exp4.1, GPT-J for Exp4.2).
   - Load prompts/tokenizer locally only for input prep; all execution goes via
     `execute_model(..., hints={'qos_class': class_name, 'deadline_ms': ...})`.
3. **Common instrumentation**
   - Wrap each remote execution with:
     - Wall-clock latency (already captured)
     - `manager.last_execution_metrics` for server-provided
       `plan_summary`, `queue_wait_ms`, etc.
   - Sample NVML (or `djinn/profiling/network_monitor`) on the server to get
     real GPU utilization for Figure 10/11 overlays.

---

## Experiment 4.1 – DjinnLoadDriver
1. **Driver structure**
   - Implement `DjinnLoadDriver` in `scripts/run_load_sweep.py`:
     ```python
     class DjinnLoadDriver:
         def __init__(self, cfg, args):
             self.manager = EnhancedModelManager()
             self.model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"]).to("cpu")
             self.fingerprint = await self.manager.register_model(self.model, ...)
     ```
   - `run_point` should:
     - Spawn `point.concurrency` asyncio tasks.
     - Each task loops until `duration_s` expires:
       ```python
       await asyncio.sleep(expovariate(rate))
       inputs = {"input_ids": prompt_ids.clone(), ...}
       result = await self.manager.execute_model(self.model, inputs, hints=qos)
       ```
     - Record per-request latency, queue time (`metrics["queue_wait_ms"]`), and
       generated tokens to compute throughput.
2. **Load control**
   - Enforce Poisson arrivals by drawing inter-arrival delays per user.
   - Use `asyncio.Semaphore(max_inflight)` if we need to cap outstanding
     requests to the server’s max concurrency.
3. **Metrics**
   - For each point, compute actual P50/P99 from request samples.
   - Capture server-reported GPU utilization (via NVML sampler thread running
     in parallel) so the Figure 10 curves show observed saturation instead of
     the analytic guess.

---

## Experiment 4.2 – DjinnQoSDriver
1. **Driver implementation**
   - Mirror the structure above but map workload mix to QoS hints:
     - Realtime → `{'qos_class': 'realtime', 'deadline_ms': 10}`
     - Interactive → `{'qos_class': 'interactive', 'deadline_ms': 100}`
     - Batch → `{'qos_class': 'batch'}` (no deadline)
   - Use per-class arrival generators per §6.4.2; each generator enqueues
     requests into an asyncio PriorityQueue tagged with class label.
2. **Policy comparison**
   - For FCFS: disable QoS on the server via config or run a second server
     instance with `GENIE_ENABLE_QOS=0`.
   - For Djinn QoS: leave scheduler enabled and rely on `BasicQosScheduler`.
3. **Measurements**
   - Per request, record `latency_ms`, `queue_wait_ms`, and SLA violation
     (latency > target). Aggregate counts exactly as Table 4 requires.
   - Persist queue depth traces by sampling `BasicQosScheduler.debug_snapshot()`
     (needs new API) every 100 ms during the run.

---

## Validation checklist
- [ ] 30+ samples per (policy, class) so we can compute CI.
- [ ] Server logs confirm real VMU allocations + QoS scheduling decisions.
- [ ] Harness output includes raw per-request records for reproducibility.
- [ ] Cross-validated against synthetic driver to ensure order-of-magnitude
      consistency before we discard the analytic placeholders.

