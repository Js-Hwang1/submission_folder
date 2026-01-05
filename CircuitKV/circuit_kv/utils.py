"""
CircuitKV Utilities.

Helper functions for reproducibility, profiling, and diagnostics.
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    This locks:
    - Python's random module
    - NumPy's random generator
    - PyTorch CPU and CUDA generators

    Args:
        seed: The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For full determinism (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate torch device.

    Args:
        device: Device string ('cuda', 'cpu', etc.) or None for auto-detect.

    Returns:
        torch.device object.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cuda_memory_stats() -> dict:
    """
    Get current CUDA memory statistics.

    Returns:
        Dictionary with memory stats in MB.
    """
    if not torch.cuda.is_available():
        return {}

    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def format_memory(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"
