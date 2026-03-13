"""
infrastructure/utils/colab_setup.py
====================================
Standard ButterflAI Colab environment initializer.

Usage (always the first code cell in any weekly notebook):
    from infrastructure.utils.colab_setup import setup
    setup()
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional


def setup(
    seed: int = 42,
    mount_drive: bool = False,
    drive_data_path: Optional[str] = None,
    install_extra: Optional[list[str]] = None,
    verbose: bool = True,
) -> dict:
    """
    Initialize the ButterflAI Colab environment.

    Performs the following in order:
      1. Detects whether running in Colab or locally
      2. Installs any missing extra dependencies
      3. Optionally mounts Google Drive
      4. Sets global random seeds for reproducibility
      5. Configures Matplotlib defaults
      6. Returns an environment info dict

    Parameters
    ----------
    seed : int
        Global random seed for NumPy, Python random, and PyTorch.
    mount_drive : bool
        If True and running in Colab, mount Google Drive at /content/drive.
    drive_data_path : str, optional
        Path within Drive to symlink as /content/butterflai_data.
        E.g. 'MyDrive/ButterflAI/data'
    install_extra : list of str, optional
        Additional pip packages to install (e.g. ['astropy', 'sunpy']).
    verbose : bool
        Print environment summary on completion.

    Returns
    -------
    dict
        Keys: 'in_colab', 'device', 'seed', 'drive_mounted', 'data_path'
    """
    info = {
        "in_colab": _is_colab(),
        "device": None,
        "seed": seed,
        "drive_mounted": False,
        "data_path": None,
    }

    # --- Extra dependencies ---
    if install_extra:
        _pip_install(install_extra, verbose=verbose)

    # --- Google Drive ---
    if mount_drive and info["in_colab"]:
        info["drive_mounted"] = _mount_drive(drive_data_path, verbose=verbose)
        if drive_data_path:
            info["data_path"] = Path(f"/content/drive/{drive_data_path}")

    # --- Seeds ---
    from infrastructure.utils.reproducibility import set_all_seeds
    set_all_seeds(seed)

    # --- Device ---
    from infrastructure.utils.device_utils import get_device
    info["device"] = get_device()

    # --- Matplotlib defaults ---
    _configure_matplotlib()

    if verbose:
        _print_summary(info)

    return info


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_colab() -> bool:
    """Detect Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _pip_install(packages: list[str], verbose: bool = True) -> None:
    """Install pip packages quietly."""
    for pkg in packages:
        if verbose:
            print(f"  Installing {pkg}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            check=True,
            capture_output=not verbose,
        )


def _mount_drive(drive_data_path: Optional[str], verbose: bool = True) -> bool:
    """Mount Google Drive. Returns True on success."""
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        if drive_data_path:
            src = Path(f"/content/drive/{drive_data_path}")
            dst = Path("/content/butterflai_data")
            if src.exists() and not dst.exists():
                dst.symlink_to(src)
                if verbose:
                    print(f"  Data symlinked: {dst} → {src}")
        return True
    except Exception as e:
        print(f"  [Warning] Drive mount failed: {e}")
        return False


def _configure_matplotlib() -> None:
    """Apply ButterflAI standard Matplotlib settings."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120,
        "figure.figsize": (10, 4),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "image.cmap": "RdYlBu_r",   # Good default for solar data (activity maps)
    })


def _print_summary(info: dict) -> None:
    """Print a compact environment summary."""
    env = "Google Colab" if info["in_colab"] else "Local"
    print(f"🦋 ButterflAI environment ready")
    print(f"   Runtime  : {env}")
    print(f"   Device   : {info['device']}")
    print(f"   Seed     : {info['seed']}")
    if info["drive_mounted"]:
        print(f"   Drive    : mounted")
    if info["data_path"]:
        print(f"   Data     : {info['data_path']}")
