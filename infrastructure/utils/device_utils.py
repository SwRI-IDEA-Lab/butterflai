"""
infrastructure/utils/device_utils.py
======================================
GPU/CPU device selection for ButterflAI.

Always use get_device() rather than hardcoding 'cuda' or 'cpu'.
Colab GPU availability is not guaranteed.
"""

import torch


def get_device() -> torch.device:
    """
    Return the best available torch device.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns
    -------
    torch.device
        Use as: model.to(get_device())

    Examples
    --------
    >>> device = get_device()
    >>> tensor = torch.randn(3, 4).to(device)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def device_summary() -> str:
    """Return a human-readable device summary string."""
    device = get_device()
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"CUDA ({name}, {vram:.1f} GB VRAM)"
    elif device.type == "mps":
        return "MPS (Apple Silicon)"
    else:
        return "CPU"
