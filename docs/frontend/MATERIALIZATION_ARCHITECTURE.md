# Materialization Architecture (v2.3)

**Status**: Production Ready (v2.3.10)
**Last Updated**: November 21, 2025

---

## Overview

Djinn uses a **lazy-by-default** materialization strategy. Operations are deferred until materialization is explicitly required. This enables optimal performance by only executing what's needed.

---

## Materialization Decision Logic

### What Materializes Immediately

**Only Python Protocol Operations** (operations that Python requires to return concrete values):

```python
PYTHON_PROTOCOL_OPS = {
    '__bool__',      # if tensor: ... (control flow)
    '__int__',       # int(tensor)
    '__float__',     # float(tensor)
    '__index__',     # list[tensor] (indexing)
    'item',          # tensor.item() → Python number
    'tolist',        # tensor.tolist() → Python list
    'numpy',         # tensor.numpy() → numpy array
}
```

**Why**: Python's type system requires concrete values for these operations. LazyTensor cannot be used in these contexts.

### What Stays Lazy

**Everything else** preserves laziness:

1. **Tuple Operations** → Return `LazyTuple`
   - `split()`, `chunk()`, `unbind()`, `topk()`, `sort()`, etc.
   - Elements materialize only when accessed

2. **Standard Operations** → Return `LazyTensor`
   - `+`, `-`, `*`, `matmul()`, `relu()`, etc.
   - Materialize when `.cpu()`, `.numpy()`, etc. called

---

## LazyTuple Architecture

### Purpose

Tuple-returning operations (like `split()`) need special handling because they return multiple tensors. LazyTuple preserves laziness for all chunks until individual elements are accessed.

### Implementation

```python
class LazyTuple(tuple):
    """
    Tuple of LazyTensors with deferred materialization.
    
    Behaves like Python tuple but preserves laziness until elements accessed.
    """
    
    def __getitem__(self, index):
        """Access element - returns LazyTensor (still lazy)."""
        return super().__getitem__(index)
    
    def materialize(self) -> Tuple[torch.Tensor, ...]:
        """Materialize entire tuple - returns concrete tensors."""
        return tuple(elem.materialize() if isinstance(elem, LazyTensor) else elem 
                    for elem in self)
```

### Usage Example

```python
with djinn.capture():
    x = torch.randn(1, 5, 900)
    
    # Returns LazyTuple (lazy, no execution)
    chunks = x.split(300, dim=2)
    
    # Unpacking works (still lazy)
    a, b, c = chunks
    
    # Materialization happens on demand
    result = a.cpu()  # Only chunk 0 materializes
    # Shape: torch.Size([1, 5, 300]) ✅
```

### Performance Benefits

**Before (Eager Materialization)**:
- `split()` immediately materializes entire tensor
- Network transfer for full tensor (e.g., 80MB)
- No optimization opportunities

**After (LazyTuple)**:
- `split()` returns LazyTuple (no execution)
- Only accessed chunks materialize
- Network transfer only for accessed chunks (e.g., 4KB)
- Can optimize across multiple chunks
- Enables fusion opportunities

---

## Materialization Control Module

**File**: `djinn/frontend/core/materialization_control.py`

### Simplified Architecture

```python
def should_materialize_immediately(func_name: str) -> Tuple[bool, str]:
    """
    Determine if operation requires immediate materialization.
    
    Returns:
        (should_materialize: bool, reason: str)
    """
    # Only Python protocol ops materialize immediately
    if func_name in PYTHON_PROTOCOL_OPS:
        return True, 'python_protocol'
    
    # Everything else is lazy
    return False, 'lazy_default'
```

### Key Design Decisions

1. **Minimal Materialization**: Only materialize when Python requires it
2. **LazyTuple for Tuples**: Tuple operations preserve laziness
3. **No Eager Mode**: Removed deprecated eager mode code (Month 3 cleanup)
4. **Simple Logic**: Single code path, no dual modes

---

## Integration Points

### LazyTensor Integration

**In `__torch_dispatch__` and `__torch_function__`**:
```python
# Check for tuple operations FIRST (before automatic dispatch)
if func_name in TUPLE_RETURNING_OPS:
    # Return LazyTuple (proper solution)
    return LazyTuple.from_split(...)
```

### Executor Integration

**In `executor.py`**:
```python
def _execute_split(self, lazy_tensor, inputs, kwargs):
    """Execute split operation with chunk extraction."""
    # Extract chunk_index from kwargs (LazyTuple passes this)
    chunk_index = kwargs.get('_chunk_index')
    split_result = torch.split(...)
    return split_result[chunk_index]  # Return specific chunk
```

### MaterializationOptimizer Integration

**In `materialization_optimizer.py`**:
- Disables interception during materialization
- Ensures factory operations return concrete tensors
- Verifies no LazyTensors in result_cache

---

## Benefits

1. **Performance**: Only materialize what's needed
2. **Network Efficiency**: Transfer only accessed chunks
3. **Optimization**: Enable fusion across chunks
4. **Simplicity**: Single code path, no dual modes
5. **Correctness**: Preserves laziness until required

---

## Migration Notes

**Removed in Month 3**:
- `eager_tuple_operations` flag
- Eager mode logic
- Performance tracking code
- Deprecation warnings

**Current State**:
- LazyTuple is the only option
- Cleaner, simpler codebase
- ~100 lines of deprecated code removed

