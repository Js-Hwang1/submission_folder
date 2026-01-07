"""
CircuitKV: Current-Flow Betweenness for KV Cache Eviction in LLMs.

This module exposes the high-performance CUDA implementation of CircuitKV,
which uses absorbing random walks on a sparse attention graph to identify
structurally important "bridge" tokens for KV cache retention.

Key Innovation:
- Simulates electrical current from Query (Source) to Context Start (Sink)
- Captures "bridge tokens" that connect Question to Context but have low degree
- Unlike PPR, these tokens accumulate current because they're on the path
"""

from circuit_kv.engine import (
    CircuitKVMonitor,
    CircuitKVConfig,
    LandmarkWalkerMonitor,
    LandmarkWalkerConfig,
)
from circuit_kv.utils import set_seed

__version__ = "0.2.0"
__all__ = [
    "CircuitKVMonitor",
    "CircuitKVConfig",
    "LandmarkWalkerMonitor",
    "LandmarkWalkerConfig",
    "set_seed",
]


def _load_extension():
    """Lazy import of the C++ extension to give better error messages."""
    try:
        from circuit_kv import _C
        return _C
    except ImportError as e:
        raise ImportError(
            "Could not load the CircuitKV CUDA extension. "
            "Make sure you have built it with `pip install -e .` "
            f"Original error: {e}"
        ) from e


def get_extension():
    """Get the C++ extension module."""
    return _load_extension()
