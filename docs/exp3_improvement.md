
**To:** Engineering Team
**From:** Research Lead
**Subject:** URGENT: Redesigning Experiment 3 (Interactivity) for OSDI

The current version of Experiment 3 is **at risk of rejection**. We are currently comparing our system ($\sys$) against "Vanilla PyTorch," showing that PyTorch crashes while we do not.

**The OSDI Problem:**
While "PyTorch crashes" is a valid functional argument, it is a **weak systems argument**. Reviewers will see this as a "Strawman Comparison." They know PyTorch isn't designed for this. They will ask the harder questions:
1.  *"Why not just re-run the code (Recompute)? It saves infinite memory."*
2.  *"Why not just manually move tensors to CPU (`x.to('cpu')`)? Why do I need a whole new OS?"*

To get an **Accept**, we must prove that $\sys$ is not just "possible," but that it is **faster than Recompute** for deep models and **as fast (or faster) than Manual Offload** without the user writing code.

---

### **Part 1: The New Experiment Design**

We will run a "Resume Latency vs. Network Depth" study.
**Scenario:** A user pauses execution at Layer $L$ (Breakpoint), waits, and then resumes.
**Metric:** **Resume Latency (ms).** How long from the moment the user clicks "Resume" until the GPU is ready to compute Layer $L+1$?

We will compare three baselines:

1.  **Vanilla PyTorch (The Default):**
    *   *Behavior:* Keep everything in VRAM.
    *   *Result:* 0ms Latency (Fastest), but crashes at $N=3$.
    *   *Why include it:* To show the "Ideal Latency" baseline and the "Memory Wall."

2.  **Stateless Recompute (The "Serverless" Approach):**
    *   *Behavior:* On pause, delete all intermediate tensors. On resume, re-run the model from Layer 0 to Layer $L$.
    *   *Hypothesis:* Fast for early layers, very slow for deep layers.

3.  **Manual CPU Offload (The "Smart Script" Approach):**
    *   *Behavior:* User manually writes `tensor.to('cpu')` on pause and `tensor.to('cuda')` on resume.
    *   *Hypothesis:* This represents the "Speed of Light" for PCIe transfers. $\sys$ should match this, but win on usability.

---

### **Part 2: Implementation Instructions (For Junior Engineer)**

**Setup:**
*   **Hardware:** Single A100 or H100 (or L4). Keep it consistent.
*   **Model:** Llama-2-13B (or 70B if available).
*   **Layers:** We will measure latency at breakpoints: Layer 1, 10, 20, 30, 40 (End).

#### **Task A: Implement `benchmark_recompute.py`**
1.  Load the model weights (keep them resident/pinned so we don't measure model loading time, only compute time).
2.  Loop through layers $L \in [1, 10, 20, 30, 40]$:
    *   Start Timer (`t0`).
    *   Run `model.forward()` from Input $\to$ Layer $L$.
    *   `torch.cuda.synchronize()` (**CRITICAL:** Must sync to measure real time).
    *   End Timer (`t1`).
    *   Record `t1 - t0` as "Resume Latency".

#### **Task B: Implement `benchmark_manual_offload.py`**
1.  Run the model up to Layer $L$. Get the activation tensor `act`.
2.  Move `act` to CPU: `cpu_act = act.to('cpu', non_blocking=True)`.
3.  **The Measurement Loop:**
    *   Start Timer (`t0`).
    *   Move back to GPU: `gpu_act = cpu_act.to('cuda', non_blocking=True)`
    *   `torch.cuda.synchronize()`
    *   End Timer (`t1`).
    *   Record `t1 - t0`.
4.  *Note:* Ensure `cpu_act` is in **Pinned Memory** (`tensor.pin_memory()`) for the fairest comparison. If we beat pinned memory, it's a huge win. If we only beat pageable, that's okay too, but be specific.

#### **Task C: Implement `benchmark_djinn.py` ($\sys$)**
1.  Run $\sys$ to Layer $L$.
2.  Trigger `IO_WAIT` (Force swap to host).
3.  **The Measurement Loop:**
    *   Start Timer (`t0`).
    *   Trigger `RESUME` signal.
    *   Wait for $\sys$ to signal that Layer $L$ data is resident.
    *   End Timer (`t1`).
    *   Record `t1 - t0`.

---

### **Part 3: The Expected Output & Analysis**

You will produce two plots for the paper.

#### **Plot 1: The "Crossover" Chart (Line Chart)**
*   **X-Axis:** Breakpoint Depth (Layer Index).
*   **Y-Axis:** Resume Latency (ms).
*   **Visuals:**
    *   **Recompute:** A diagonal line shooting up. (Layer 1 = 5ms, Layer 40 = 800ms).
    *   **Manual Offload:** A flat line near the bottom (approx 30-50ms, depending on activation size).
    *   **$\sys$:** A flat line overlapping (or slightly below) Manual Offload.
*   **The Argument:** "While Recompute is viable for shallow debugging, it scales linearly with depth. $\sys$ provides **O(1) resume latency**, making it $20\times$ faster for deep-layer debugging."

#### **Plot 2: The "Capabilities" Table (qualitative + quantitative)**
(Include Vanilla PyTorch here)

| Method | Resume Latency (Layer 40) | Max Concurrent Sessions | Supports State Editing? |
| :--- | :--- | :--- | :--- |
| **Vanilla PyTorch** | **0 ms** | 3 (Crash) | Yes |
| **Recompute** | ~800 ms | **50+** | **No** (Impossible) |
| **Manual Offload** | ~40 ms | **50+** | Yes (Hard to code) |
| **$\sys$ (Ours)** | **~40 ms** | **50+** | **Yes (Transparent)** |

**Why this wins OSDI:**
1.  **Scientific Honesty:** We admit PyTorch is faster (0ms) when memory is infinite.
2.  **Trade-off Analysis:** We show exactly where Recompute fails (Deep Layers).
3.  **Systems Win:** We match the "Speed of Light" (Manual Offload) without forcing the user to manage memory manually.