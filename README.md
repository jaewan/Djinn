# Djinn

## Part 1: What is Djinn?

**Djinn is a Tensor Operating System that makes remote GPU clusters feel like local GPUs by exploiting semantic knowledge about tensor lifecycles.**

### 1.1 The Problem Djinn Solves

Modern GPU clusters are like 1960s mainframes: expensive, centralized, and shared among many users. Current solutions fail in two ways:

| Approach | Problem |
|----------|---------|
| **Driver-level disaggregation** (rCUDA, vCUDA) | Semantically blind—can't distinguish weights from activations |
| **Application-specific** (DistServe) | Not general—requires rewriting for each model |

Djinn operates at the **framework layer** (PyTorch), which is the unique point where:
- We have **semantic knowledge** (what tensors mean)
- We have **generality** (works for any PyTorch model)

### 1.2 The Core Insight

```
Traditional OS: Manages bytes (pages, heap, stack)
Tensor OS:      Manages tensors (weights, KV cache, activations)

The key insight: Tensors have SEMANTICS that bytes don't.
- Weights are shared, read-only, long-lived
- KV caches are private, append-only, session-lived
- Activations are private, ephemeral, request-lived

By exploiting these semantics, we can build a simpler, more efficient memory system.
```

---

## Part 2: Why Djinn is a Tensor Operating System

### 2.1 The Five OS Properties

An operating system provides five core services. Here's how Djinn implements each:

| OS Property | Traditional OS (Linux) | Djinn (Tensor OS) | Implementation |
|-------------|----------------------|-------------------|----------------|
| **Virtual Addressing** | Virtual memory (page tables) | LazyTensor handles | Tensors have logical IDs; VMU maps to physical GPU memory |
| **Memory Protection** | Process isolation (address spaces) | Session isolation (arenas) | Each session has a private Data arena; cannot access others |
| **Resource Multiplexing** | Time-sharing (scheduler) | Stream multiplexing + QoS | Multiple sessions share GPU via CUDA streams |
| **Automatic Reclamation** | Process exit cleanup | Session GC | When session ends, all its memory is automatically reclaimed |
| **Abstraction** | System calls hide hardware | `device='remote'` hides cluster | User writes PyTorch; Djinn handles network, placement, memory |

### 2.2 The Key Difference: Semantic Awareness

**Why can't a driver-level OS do this?**

```python
# What the CUDA driver sees:
cudaMalloc(&ptr1, 12GB)  # "Allocate 12GB"
cudaMalloc(&ptr2, 12GB)  # "Allocate 12GB"
cudaMalloc(&ptr3, 2GB)   # "Allocate 2GB"

# The driver has NO IDEA that:
# - ptr1 = model weights (can be shared across all users!)
# - ptr2 = activations (can be discarded after forward pass!)
# - ptr3 = KV cache (must persist for this session only)
```

**What Djinn sees:**

```python
# Djinn intercepts at PyTorch level:
model.to('remote')       # "This is a model → weights go to Text segment (shared)"
output = model(input)    # "These are activations → Stack segment (ephemeral)"
kv_cache.append(new_kv)  # "This is KV cache → Data segment (session-private)"
```

**This semantic knowledge enables:**
- **Weight sharing**: 50 users, 1 copy (not 50 copies)
- **Zero fragmentation**: Activations use bump-pointer, reset after request
- **Precise GC**: Session ends → exactly that session's KV is freed

---

## Part 3: Memory Management in Detail

### 3.1 The Three-Segment Model

Djinn's VMU (Virtual Memory Unit) partitions GPU memory into three segments, each optimized for a specific tensor lifecycle:

```
┌─────────────────────────────────────────────────────────────────┐
│                        GPU VRAM (80 GB)                         │
├─────────────────────────────────────────────────────────────────┤
│ OS Reserve │     TEXT SEGMENT      │  DATA SEGMENT │   STACK   │
│   (2 GB)   │      (40 GB)          │    (28 GB)    │  (10 GB)  │
│            │                       │               │           │
│  CUDA      │  Model A weights      │ Session 1 KV  │ Current   │
│  Runtime   │  (12 GB)              │ (2 GB arena)  │ request   │
│            │                       │               │ activations│
│            │  Model B weights      │ Session 2 KV  │           │
│            │  (24 GB)              │ (3 GB arena)  │ [bump ptr]│
│            │                       │               │           │
│            │  [append-only →]      │ Session 3 KV  │ [watermark]│
│            │                       │ (1 GB arena)  │           │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Text Segment (Weights)

**Purpose**: Store model weights, shared across all users.

**Properties**:
- **Append-only**: Weights are added, never removed during operation
- **Read-only**: After loading, weights are never modified
- **Shared**: All sessions access the same copy
- **Reference-counted**: Track how many sessions use each model

**Implementation**:
```python
class TextSegment:
    def __init__(self, capacity: int):
        self.buffer = torch.empty(capacity, dtype=torch.uint8, device='cuda')
        self.offset = 0  # Append point
        self.models: Dict[str, ModelEntry] = {}
    
    def load_model(self, model_id: str, state_dict: Dict) -> Dict[str, Tensor]:
        """Load model weights, return views into shared buffer."""
        if model_id in self.models:
            self.models[model_id].refcount += 1
            return self.models[model_id].views  # Reuse existing!
        
        # Copy weights into buffer (one time only)
        views = {}
        for name, tensor in state_dict.items():
            size = tensor.numel() * tensor.element_size()
            dest = self.buffer[self.offset:self.offset + size]
            dest.copy_(tensor.flatten().view(torch.uint8))
            views[name] = dest.view(tensor.dtype).view(tensor.shape)
            self.offset += align_to(size, 256)
        
        self.models[model_id] = ModelEntry(views=views, refcount=1)
        return views
```

**Key Invariant**: Zero external fragmentation (append-only means no holes).

### 3.3 Data Segment (KV Cache / Session State)

**Purpose**: Store per-session persistent state (KV cache, conversation history).

**Properties**:
- **Per-session arenas**: Each session gets a contiguous region
- **Bump-pointer within arena**: KV cache grows monotonically
- **Session isolation**: Session A cannot access Session B's arena
- **Bulk reclaim**: When session ends, entire arena is freed

**Implementation**:
```python
class DataSegment:
    def __init__(self, capacity: int):
        self.buffer = torch.empty(capacity, dtype=torch.uint8, device='cuda')
        self.capacity = capacity
        self.next_offset = 0
        self.sessions: Dict[str, SessionArena] = {}
    
    def reserve_arena(self, session_id: str, max_bytes: int) -> SessionArena:
        """Reserve a contiguous arena for a session."""
        if self.next_offset + max_bytes > self.capacity:
            raise DataSegmentOOM(f"Cannot reserve {max_bytes} for {session_id}")
        
        arena = SessionArena(
            session_id=session_id,
            base_offset=self.next_offset,
            capacity=max_bytes,
            used=0
        )
        self.sessions[session_id] = arena
        self.next_offset += max_bytes
        return arena
    
    def gc_session(self, session_id: str):
        """Reclaim session's arena."""
        arena = self.sessions.pop(session_id, None)
        if arena and arena.base_offset + arena.capacity == self.next_offset:
            self.next_offset = arena.base_offset  # Reclaim if at end
        # Otherwise: gap forms (tracked as fragmentation)


@dataclass
class SessionArena:
    session_id: str
    base_offset: int
    capacity: int
    used: int = 0
    
    def alloc(self, size: int) -> int:
        """Bump-pointer allocation within arena."""
        if self.used + size > self.capacity:
            raise ArenaOOM(f"Session {self.session_id} exceeded quota")
        offset = self.base_offset + self.used
        self.used += size
        return offset
```

**Key Invariant**: Zero external fragmentation *within* each arena.

### 3.4 Stack Segment (Activations)

**Purpose**: Store ephemeral intermediate tensors during forward pass.

**Properties**:
- **Bump-pointer**: Allocations just increment a pointer (O(1))
- **Watermark reset**: At request end, reset pointer to watermark (O(1))
- **Zero fragmentation**: No holes can form (no individual frees)
- **Request-scoped**: All activations discarded after forward pass

**Implementation**:
```python
class StackSegment:
    def __init__(self, capacity: int):
        self.buffer = torch.empty(capacity, dtype=torch.uint8, device='cuda')
        self.capacity = capacity
        self.pointer = 0
        self.watermark = 0
        self.peak_usage = 0
    
    def begin_request(self):
        """Save watermark for current request."""
        self.watermark = self.pointer
    
    def alloc(self, size: int, alignment: int = 256) -> int:
        """Bump-pointer allocation. O(1)."""
        aligned_ptr = align_to(self.pointer, alignment)
        if aligned_ptr + size > self.capacity:
            raise StackOverflow(f"Stack overflow: need {size}, have {self.capacity - aligned_ptr}")
        
        self.pointer = aligned_ptr + size
        self.peak_usage = max(self.peak_usage, self.pointer)
        return aligned_ptr
    
    def end_request(self):
        """Reset to watermark. O(1) - reclaims ALL request memory."""
        self.pointer = self.watermark
```

**Key Invariant**: Zero external fragmentation (by construction—no holes possible).

### 3.5 VMU Initialization (Dynamic Sizing)

The VMU must size itself based on **actual free GPU memory**, not just total memory:

```python
def initialize_vmu(config: VMUConfig) -> VMU:
    # Query actual GPU state
    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory
    reserved_memory = torch.cuda.memory_reserved(0)
    free_memory = total_memory - reserved_memory
    
    # Compute safe capacity
    os_reserve = config.os_reserve_gb * (1024**3)
    safety_margin = config.safety_margin_gb * (1024**3)
    safe_capacity = min(total_memory - os_reserve, free_memory - safety_margin)
    
    # Fail fast if insufficient
    if safe_capacity < config.min_viable_gb * (1024**3):
        raise InsufficientMemoryError(
            f"Need {config.min_viable_gb} GB, have {safe_capacity/1e9:.1f} GB"
        )
    
    # Compute segment sizes (workload-aware ratios)
    ratios = config.get_ratios()  # e.g., (0.5, 0.35, 0.15) for LLM inference
    
    return VMU(
        text=TextSegment(int(safe_capacity * ratios[0])),
        data=DataSegment(int(safe_capacity * ratios[1])),
        stack=StackSegment(int(safe_capacity * ratios[2])),
    )
```

---

## Part 4: End-to-End Dry Run

Let's walk through a complete example: **User runs LLM inference with GPT-J through Djinn**.

### 4.0 Setup

```python
# User's code (simple!)
import djinn
import torch

# Connect to remote GPU cluster
djinn.init(server="gpu-cluster.example.com:8080")

# Load model (transparently uses remote GPU)
model = djinn.load_model("EleutherAI/gpt-j-6B", device="remote")

# Generate text
output = model.generate("Hello, world!", max_tokens=50)
print(output)
```

### 4.1 Step 1: Server Startup (VMU Initialization)

**What happens when `djinn-server` starts:**

```
[Server Startup]
│
├─► Query GPU memory
│   total_memory = 80 GB
│   reserved_memory = 5 GB (CUDA runtime, other processes)
│   free_memory = 75 GB
│
├─► Compute safe capacity
│   os_reserve = 2 GB
│   safety_margin = 1 GB
│   safe_capacity = min(80-2, 75-1) = 74 GB
│
├─► Check minimum viable
│   min_viable = 8 GB
│   74 GB >= 8 GB ✓ (proceed)
│
├─► Allocate segments (LLM inference ratios: 0.5, 0.35, 0.15)
│   Text:  37 GB (for model weights)
│   Data:  26 GB (for KV caches)
│   Stack: 11 GB (for activations)
│
└─► Log: "VMU initialized: 74 GB (Text=37, Data=26, Stack=11)"
```

**VMU State After Startup:**
```
Text:  [                    37 GB empty                    ]
       offset=0
       
Data:  [                    26 GB empty                    ]
       next_offset=0, sessions={}
       
Stack: [                    11 GB empty                    ]
       pointer=0, watermark=0
```

### 4.2 Step 2: Model Loading

**What happens when user calls `djinn.load_model("gpt-j-6B")`:**

```
[Client]                                    [Server]
    │                                           │
    ├─► User: model = djinn.load_model(...)     │
    │                                           │
    ├─► Client creates "ghost model"            │
    │   (nn.Module on 'meta' device, 0 bytes)   │
    │                                           │
    ├─► Client sends: LOAD_MODEL(model_id="gpt-j-6B")
    │                                           │
    │                              ┌────────────┤
    │                              │            │
    │                              │  Server checks: Is gpt-j-6B in Text?
    │                              │  No → Download weights from HuggingFace
    │                              │            │
    │                              │  Server calls: vmu.text.load_model(...)
    │                              │    - Copy weights to Text segment
    │                              │    - Create views for each parameter
    │                              │    - refcount["gpt-j-6B"] = 1
    │                              │            │
    │                              └────────────┤
    │                                           │
    ◄─────────────────────────────── OK(fingerprint="gpt-j-abc123")
    │                                           │
    └─► Client stores fingerprint               │
        (ghost model + fingerprint = ready)     │
```

**VMU State After Model Load:**
```
Text:  [  GPT-J weights (12 GB)  |      25 GB free        ]
       offset=12GB
       models={"gpt-j-6B": {refcount=1, offset=0, size=12GB}}
       
Data:  [                    26 GB empty                    ]
       next_offset=0, sessions={}
       
Stack: [                    11 GB empty                    ]
       pointer=0, watermark=0
```

### 4.3 Step 3: Session Creation

**What happens when user starts a conversation (implicit or explicit):**

```
[Client]                                    [Server]
    │                                           │
    ├─► User: model.generate("Hello")           │
    │                                           │
    ├─► Client creates new session              │
    │   session_id = "sess_12345"               │
    │   expected_tokens = 2048 (default)        │
    │                                           │
    ├─► Client sends: CREATE_SESSION(
    │       session_id="sess_12345",
    │       model_id="gpt-j-6B",
    │       expected_tokens=2048
    │   )                                       │
    │                              ┌────────────┤
    │                              │            │
    │                              │  Server computes KV budget:
    │                              │    kv_per_token = 256 KB (model-specific)
    │                              │    max_kv = 2048 * 256 KB = 512 MB
    │                              │    reserved = 512 MB * 1.2 = 614 MB (with buffer)
    │                              │            │
    │                              │  Server calls: vmu.data.reserve_arena(
    │                              │      session_id="sess_12345",
    │                              │      max_bytes=614 MB
    │                              │  )         │
    │                              │            │
    │                              │  Arena created at offset=0, capacity=614 MB
    │                              │            │
    │                              └────────────┤
    │                                           │
    ◄─────────────────────────────── OK(session_id="sess_12345")
```

**VMU State After Session Creation:**
```
Text:  [  GPT-J weights (12 GB)  |      25 GB free        ]
       offset=12GB
       
Data:  [ Session 12345 arena (614 MB) |    25.4 GB free   ]
       next_offset=614MB
       sessions={"sess_12345": {base=0, capacity=614MB, used=0}}
       
Stack: [                    11 GB empty                    ]
       pointer=0, watermark=0
```

### 4.4 Step 4: Forward Pass (Generate First Token)

**What happens when model processes the prompt:**

```
[Client]                                    [Server]
    │                                           │
    ├─► Send: EXECUTE(                          │
    │       session_id="sess_12345",            │
    │       fingerprint="gpt-j-abc123",         │
    │       inputs={"input_ids": [1, 2, 3...]}  │  (prompt tokens)
    │   )                                       │
    │                              ┌────────────┤
    │                              │            │
    │                              │  [BEGIN REQUEST]
    │                              │  vmu.stack.begin_request()
    │                              │  → watermark = 0
    │                              │            │
    │                              │  [GET MODEL WEIGHTS]
    │                              │  weights = vmu.text.get_views("gpt-j-6B")
    │                              │  → Returns views into Text segment
    │                              │  → NO COPY, just pointer arithmetic
    │                              │            │
    │                              │  [ALLOCATE ACTIVATIONS]
    │                              │  For each layer:
    │                              │    offset = vmu.stack.alloc(activation_size)
    │                              │    → Bump pointer: 0 → 100MB → 200MB → ...
    │                              │            │
    │                              │  [COMPUTE FORWARD PASS]
    │                              │  attention_scores = compute_attention(...)
    │                              │  hidden_states = compute_ffn(...)
    │                              │  → All intermediates in Stack segment
    │                              │            │
    │                              │  [STORE KV CACHE]
    │                              │  arena = vmu.data.sessions["sess_12345"]
    │                              │  kv_offset = arena.alloc(kv_size)
    │                              │  → arena.used: 0 → 50MB
    │                              │            │
    │                              │  [COMPUTE OUTPUT]
    │                              │  logits = final_layer(hidden_states)
    │                              │  next_token = argmax(logits)
    │                              │            │
    │                              │  [END REQUEST]
    │                              │  vmu.stack.end_request()
    │                              │  → pointer = watermark = 0
    │                              │  → ALL 500MB of activations reclaimed instantly!
    │                              │            │
    │                              └────────────┤
    │                                           │
    ◄─────────────────────────────── OK(next_token=42)
```

**VMU State After First Token:**
```
Text:  [  GPT-J weights (12 GB)  |      25 GB free        ]
       offset=12GB (unchanged - weights are read-only)
       
Data:  [ Sess 12345: [KV 50MB |  564 MB free ] |  25.4 GB  ]
       next_offset=614MB
       sessions={"sess_12345": {base=0, capacity=614MB, used=50MB}}
       
Stack: [                    11 GB empty                    ]
       pointer=0 (reset!), watermark=0, peak=500MB
       
Note: Stack had 500MB of activations during forward pass,
      but they're ALL reclaimed after end_request()!
```

### 4.5 Step 5: Decode Loop (Generate More Tokens)

**What happens for each subsequent token (simplified):**

```
For token 2, 3, 4, ... 50:
    
    [Server]
    │
    ├─► vmu.stack.begin_request()
    │   → watermark = 0
    │
    ├─► Allocate smaller activations (decode is smaller than prefill)
    │   → Stack pointer: 0 → 20MB
    │
    ├─► Read existing KV cache from Data segment
    │   → arena.used = 50MB + 50MB + ... (growing)
    │
    ├─► Append new KV to Data segment
    │   kv_offset = arena.alloc(1MB)
    │   → arena.used += 1MB per token
    │
    ├─► Compute next token
    │
    └─► vmu.stack.end_request()
        → Stack pointer reset to 0
        → KV cache persists in Data (arena.used = 100MB after 50 tokens)
```

**VMU State After 50 Tokens:**
```
Text:  [  GPT-J weights (12 GB)  |      25 GB free        ]
       offset=12GB (unchanged)
       
Data:  [ Sess 12345: [KV 100MB |  514 MB free] |  25.4 GB  ]
       next_offset=614MB
       sessions={"sess_12345": {base=0, capacity=614MB, used=100MB}}
       
Stack: [                    11 GB empty                    ]
       pointer=0 (always reset after each token)
       peak=500MB (from prefill), now at 0
```

### 4.6 Step 6: Session Termination

**What happens when user disconnects or session times out:**

```
[Client]                                    [Server]
    │                                           │
    ├─► User closes connection                  │
    │   (or session lease expires)              │
    │                              ┌────────────┤
    │                              │            │
    │                              │  Server detects: session_12345 ended
    │                              │            │
    │                              │  Server calls: vmu.data.gc_session("sess_12345")
    │                              │    → Remove arena from sessions dict
    │                              │    → Arena was at end, so:
    │                              │      next_offset = arena.base_offset = 0
    │                              │    → 614 MB reclaimed!
    │                              │            │
    │                              │  Server calls: vmu.text.release_model("gpt-j-6B")
    │                              │    → refcount["gpt-j-6B"] -= 1
    │                              │    → refcount = 0, but we keep weights cached
    │                              │      (eviction is future work)
    │                              │            │
    │                              └────────────┤
```

**VMU State After Session End:**
```
Text:  [  GPT-J weights (12 GB)  |      25 GB free        ]
       offset=12GB (weights stay cached for next user)
       models={"gpt-j-6B": {refcount=0}}  // Still cached!
       
Data:  [                    26 GB empty                    ]
       next_offset=0  // Fully reclaimed!
       sessions={}    // No active sessions
       
Stack: [                    11 GB empty                    ]
       pointer=0, watermark=0
```

### 4.7 Step 7: Second User (Weight Sharing in Action!)

**What happens when User B loads the same model:**

```
[User B]                                    [Server]
    │                                           │
    ├─► model = djinn.load_model("gpt-j-6B")    │
    │                                           │
    ├─► Client sends: LOAD_MODEL(model_id="gpt-j-6B")
    │                              ┌────────────┤
    │                              │            │
    │                              │  Server checks: Is gpt-j-6B in Text?
    │                              │  YES! Already loaded!
    │                              │            │
    │                              │  Server calls: vmu.text.load_model(...)
    │                              │    → Model exists, just increment refcount
    │                              │    → refcount["gpt-j-6B"] = 1
    │                              │    → Return existing views (NO COPY!)
    │                              │            │
    │                              └────────────┤
    │                                           │
    ◄─────────────────────────────── OK(fingerprint="gpt-j-abc123")
    │                                           │
    │  Time: ~1ms (just refcount increment)     │
    │  Memory: 0 bytes additional               │
```

**This is the power of the Tensor OS:**
- User A loaded GPT-J: 12 GB
- User B loaded GPT-J: 0 GB additional
- 50 users loading GPT-J: Still just 12 GB total!

---

## Part 5: Summary Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DJINN TENSOR OS LIFECYCLE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────────┐ │
│  │   CLIENT    │     │   NETWORK   │     │   SERVER    │     │    GPU    │ │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └─────┬─────┘ │
│         │                   │                   │                   │       │
│  1. User writes             │                   │                   │       │
│     PyTorch code            │                   │                   │       │
│         │                   │                   │                   │       │
│  2. LazyTensor captures     │                   │                   │       │
│     operations              │                   │                   │       │
│         │                   │                   │                   │       │
│         ├──────────────────►│                   │                   │       │
│         │    EXECUTE(       │                   │                   │       │
│         │    session_id,    │                   │                   │       │
│         │    inputs)        ├──────────────────►│                   │       │
│         │                   │                   │                   │       │
│         │                   │            3. VMU allocates:          │       │
│         │                   │               - Text: weights         │       │
│         │                   │               - Data: KV cache        │       │
│         │                   │               - Stack: activations    │       │
│         │                   │                   │                   │       │
│         │                   │                   ├──────────────────►│       │
│         │                   │                   │   4. Execute      │       │
│         │                   │                   │      forward()    │       │
│         │                   │                   │◄──────────────────┤       │
│         │                   │                   │                   │       │
│         │                   │            5. Stack.end_request()     │       │
│         │                   │               → Activations freed     │       │
│         │                   │               → KV persists           │       │
│         │                   │                   │                   │       │
│         │◄──────────────────┤◄──────────────────┤                   │       │
│         │    result         │                   │                   │       │
│         │                   │                   │                   │       │
│  6. User gets result        │                   │                   │       │
│     (tensor or skeleton)    │                   │                   │       │
│         │                   │                   │                   │       │
└─────────┴───────────────────┴───────────────────┴───────────────────┴───────┘

MEMORY LIFECYCLE:
                                                                              
  Text:   ████████████░░░░░░░░░░░░  (Weights: loaded once, shared forever)    
                                                                              
  Data:   ████░░░░░░░░░░░░░░░░░░░░  (KV: grows during session, freed at end)  
                                                                              
  Stack:  ░░░░░░░░░░░░░░░░░░░░░░░░  (Activations: used during request only)   
          ▲                                                                   
          │ Reset to 0 after EVERY request!                                   
          │ Zero fragmentation by construction.                               
```

---

## Part 6: Key Takeaways

### 6.1 Why This Is an Operating System

| Property | How Djinn Achieves It |
|----------|----------------------|
| **Virtual Addressing** | LazyTensor handles abstract physical GPU location |
| **Memory Protection** | Session arenas are spatially disjoint |
| **Resource Multiplexing** | Multiple sessions share Text segment, time-share Stack |
| **Automatic Reclamation** | Session GC frees all session memory; Stack resets after each request |
| **Abstraction** | User writes `device='remote'`; Djinn handles everything |

### 6.2 Why This Works

1. **Semantic knowledge enables simple design**: Because we know tensor lifecycles, we can use bump-pointer (Stack) and arenas (Data) instead of complex heaps.

2. **Zero external fragmentation**: Stack resets completely; Data arenas have no holes within.

3. **Efficient sharing**: N users share 1 copy of weights (Text segment).

4. **Predictable behavior**: Admission control + quotas prevent OOM during operation.
