"""
Unit tests for Djinn extensibility interfaces.

Tests:
- SwappableState interface and HuggingFace adapter
- PhaseHandler interface and AgentPhaseHandler
- EvictionPolicy interface and implementations
- Configuration system
"""

import pytest
import torch
from typing import Dict, Any, List

# SwappableState Tests
def test_swappable_state_interface():
    """Test that SwappableState is properly abstract."""
    from djinn.interfaces import SwappableState
    
    # Should not be instantiable
    with pytest.raises(TypeError):
        SwappableState()


def test_huggingface_kv_cache_adapter():
    """Test HuggingFace KV cache adapter."""
    from djinn.backends.huggingface import HuggingFaceKVCache
    
    # Create mock KV cache (tuple format)
    k_tensor = torch.randn(2, 8, 1024, 64, dtype=torch.float16)  # batch, heads, seq, dim
    v_tensor = torch.randn(2, 8, 1024, 64, dtype=torch.float16)
    mock_cache = ((k_tensor, v_tensor),)  # One layer
    
    # Test adapter
    adapter = HuggingFaceKVCache(mock_cache)
    
    # Test to_host_format
    cpu_data, metadata = adapter.to_host_format()
    assert metadata['type'] in ['hf_dynamic_cache', 'hf_cache']
    assert cpu_data is not None
    
    # Test gpu_size_bytes
    size = adapter.gpu_size_bytes()
    assert size > 0
    assert size == k_tensor.numel() * 2 + v_tensor.numel() * 2  # 2 bytes per element (FP16)


# PhaseHandler Tests
def test_phase_handler_interface():
    """Test that PhaseHandler is properly abstract."""
    from djinn.interfaces import PhaseHandler
    
    # Should not be instantiable
    with pytest.raises(TypeError):
        PhaseHandler()


def test_agent_phase_handler():
    """Test AgentPhaseHandler implementation."""
    from djinn.handlers import AgentPhaseHandler
    
    handler = AgentPhaseHandler({'prefetch_margin_ms': 150})
    
    # Check supported phases
    phases = handler.supported_phases()
    assert 'IO_WAIT' in phases
    assert 'COMPUTE' in phases


# EvictionPolicy Tests
def test_eviction_policy_interface():
    """Test that EvictionPolicy is properly abstract."""
    from djinn.interfaces import EvictionPolicy
    
    # Should not be instantiable
    with pytest.raises(TypeError):
        EvictionPolicy()


def test_eviction_candidate_dataclass():
    """Test EvictionCandidate dataclass."""
    from djinn.interfaces import EvictionCandidate
    import time
    
    candidate = EvictionCandidate(
        session_id='test_session',
        last_access=time.time(),
        size_bytes=1024 * 1024,  # 1MB
        qos_class='interactive',
        is_signal_managed=True
    )
    
    assert candidate.session_id == 'test_session'
    assert candidate.size_bytes == 1024 * 1024


def test_signal_driven_policy():
    """Test signal-driven eviction policy."""
    from djinn.policies import SignalDrivenPolicy
    from djinn.interfaces import EvictionCandidate
    import time
    
    policy = SignalDrivenPolicy()
    
    # Create candidates
    now = time.time()
    candidates = [
        EvictionCandidate('sess_1', now - 10, 1024 * 1024, 'interactive', True),   # Signal-managed
        EvictionCandidate('sess_2', now - 5, 1024 * 1024, 'batch', False),          # Not signal-managed
        EvictionCandidate('sess_3', now - 2, 1024 * 1024, 'interactive', True),     # Signal-managed
    ]
    
    # Select victims needing 2MB
    victims = policy.select_victims(candidates, 2 * 1024 * 1024)
    
    # Should only select signal-managed sessions
    assert all(c.is_signal_managed for c in candidates if c.session_id in victims)
    assert 'sess_2' not in victims  # Not signal-managed
    assert len(victims) >= 2  # Need 2MB, each is 1MB


def test_lru_policy():
    """Test LRU eviction policy."""
    from djinn.policies import LRUPolicy
    from djinn.interfaces import EvictionCandidate
    import time
    
    policy = LRUPolicy()
    
    # Create candidates
    now = time.time()
    candidates = [
        EvictionCandidate('sess_1', now - 100, 1024 * 1024, 'batch', False),    # Oldest
        EvictionCandidate('sess_2', now - 50, 1024 * 1024, 'batch', False),
        EvictionCandidate('sess_3', now - 10, 1024 * 1024, 'interactive', False),
        EvictionCandidate('sess_4', now - 1, 1024 * 1024, 'realtime', False),   # Most recent, REALTIME
    ]
    
    # Select victims needing 2MB (should be LRU, skipping REALTIME)
    victims = policy.select_victims(candidates, 2 * 1024 * 1024)
    
    # Should not include REALTIME
    assert 'sess_4' not in victims
    # Should include oldest (sess_1)
    assert 'sess_1' in victims


# Configuration Tests
def test_configuration_defaults():
    """Test DjinnConfig defaults."""
    from djinn.config import DjinnConfig
    
    config = DjinnConfig()
    
    assert config.scheduler.max_concurrency == 64
    assert config.semantic.prefetch_margin_ms == 150
    assert config.plugins.phase_handler == "djinn.handlers.agent_handler.AgentPhaseHandler"
    assert config.plugins.eviction_policy == "djinn.policies.signal_driven.SignalDrivenPolicy"


def test_configuration_from_dict():
    """Test loading configuration from dict."""
    from djinn.config import SchedulerConfig, SemanticSchedulerConfig
    
    scheduler_data = {
        'max_concurrency': 32,
        'lifo_threshold_multiplier': 0.3,
        'escalation_delay_ms': 500.0,
    }
    
    config = SchedulerConfig.from_dict(scheduler_data)
    assert config.max_concurrency == 32
    assert config.lifo_threshold_multiplier == 0.3
    assert config.escalation_delay_ms == 500.0


def test_configuration_validation():
    """Test configuration validation."""
    from djinn.config import DjinnConfig
    
    # Valid config should not raise
    config = DjinnConfig()
    
    # Invalid config (max_concurrency <= 0) should raise
    from djinn.config import SchedulerConfig
    with pytest.raises(ValueError):
        DjinnConfig(scheduler=SchedulerConfig(max_concurrency=0))


# Integration Tests
def test_config_singleton():
    """Test that config uses singleton pattern."""
    from djinn.config import get_config, set_config, DjinnConfig
    
    # Get default
    config1 = get_config()
    config2 = get_config()
    
    # Should be same instance
    assert config1 is config2
    
    # Can set new config
    new_config = DjinnConfig()
    set_config(new_config)
    config3 = get_config()
    assert config3 is new_config


def test_interface_implementations():
    """Test that implementations satisfy their interfaces."""
    from djinn.interfaces import SwappableState, PhaseHandler, EvictionPolicy, InferenceBackend
    from djinn.backends.huggingface import HuggingFaceKVCache, HuggingFaceBackend
    from djinn.handlers import AgentPhaseHandler
    from djinn.policies import SignalDrivenPolicy, LRUPolicy
    
    # Check inheritance
    assert issubclass(HuggingFaceKVCache, SwappableState)
    assert issubclass(HuggingFaceBackend, InferenceBackend)
    assert issubclass(AgentPhaseHandler, PhaseHandler)
    assert issubclass(SignalDrivenPolicy, EvictionPolicy)
    assert issubclass(LRUPolicy, EvictionPolicy)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

