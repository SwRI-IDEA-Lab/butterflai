"""
infrastructure/utils/reproducibility.py
=========================================
Centralized random seed management for ButterflAI.

Seeding philosophy: set seeds early and consistently.
All notebooks call setup() which calls set_all_seeds() automatically.
"""

import os
import random
import numpy as np


def set_all_seeds(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Parameters
    ----------
    seed : int
        Seed value. Default 42 is used program-wide for reproducibility.

    Notes
    -----
    CUDA determinism requires setting CUBLAS_WORKSPACE_CONFIG in the
    environment before importing torch. This function sets it if not already
    present, but it only takes effect on the *next* CUDA initialization.
    For full determinism in Colab, include this in the setup cell before
    any torch imports.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional: set CUBLAS determinism env var
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Note: torch.use_deterministic_algorithms(True) can break some ops;
        # leave off by default but document for students who want strict repro.
    except ImportError:
        pass  # torch not yet installed (early weeks)
