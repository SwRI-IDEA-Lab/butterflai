"""Shared plumbing for the Week-11 headless training scripts.

``11a_train.py`` (residual target) and ``11c_train_empirical.py`` (empirical
target) are script versions of the notebooks of the same name, meant to run
detached (``nohup`` / ``tmux`` / ``sbatch``) on a machine you are not
connected to. This module holds everything the two have in common: argument
parsing, GPU pinning, the ``sys.path`` bootstrap that disambiguates the
several ``conditioned_infrastructure.py`` copies in the repo, and the
experiment-queue runner.

Nothing here imports torch at module scope — ``--gpu`` has to reach
``CUDA_VISIBLE_DEVICES`` *before* the first CUDA import, so the heavy imports
are deferred into :func:`bootstrap`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence

# weeks/week_11/train_cli_common.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser(description: str, default_wandb_project: str) -> argparse.ArgumentParser:
    """Common argument parser for both training scripts.

    Parameters
    ----------
    description : str
        Shown in ``--help``.
    default_wandb_project : str
        Project used when ``--wandb-project`` is not given; mirrors the
        corresponding notebook's ``WANDB_PROJECT``.
    """
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "experiments", nargs="*", metavar="NAME",
        help="canonical experiment name(s) to train, e.g. base_cat "
             "base+opp_film_four. Order is the queue order.",
    )
    p.add_argument(
        "--all", action="store_true",
        help="train every experiment in the registry (in registry order).",
    )
    p.add_argument(
        "--list", action="store_true",
        help="print the registry (name, config, whether a checkpoint already "
             "exists) and exit without training.",
    )
    p.add_argument(
        "--gpu", type=int, default=None, metavar="N",
        help="physical CUDA index to pin to (matches nvidia-smi). Sets "
             "CUDA_VISIBLE_DEVICES=N before torch is imported, so the job "
             "sees exactly one GPU. Omit to let Lightning auto-pick.",
    )
    p.add_argument(
        "--cpu", action="store_true",
        help="force CPU training even if a GPU is visible.",
    )
    p.add_argument(
        "--wandb-project", default=default_wandb_project,
        help="wandb project name (default: %(default)s).",
    )
    p.add_argument(
        "--wandb-entity", default=None,
        help="wandb entity/team (default: your personal entity).",
    )
    p.add_argument(
        "--wandb-mode", default="online",
        choices=["online", "offline", "disabled"],
        help="sets WANDB_MODE. Use 'offline' on a box without outbound "
             "network (sync later with `wandb sync`), 'disabled' to skip "
             "wandb entirely and log to CKPT_DIR/csv_logs/ "
             "(default: %(default)s).",
    )
    p.add_argument(
        "--max-epochs", type=int, default=None, metavar="N",
        help="override cfg['max_epochs'] for every queued experiment "
             "(useful for a smoke test: --max-epochs 5).",
    )
    p.add_argument(
        "--eval-every", type=int, default=None, metavar="N",
        help="override cfg['eval_every_n_epochs'] (validation + checkpoint "
             "cadence) for every queued experiment.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="training seed (default: %(default)s).",
    )
    p.add_argument(
        "--parquet", default=None, metavar="FILE",
        help="v2 parquet to train on. Defaults to "
             "week_10/diffusion_windows_v2.parquet, the one 11_00_build_v2 "
             "writes.",
    )
    p.add_argument(
        "--ckpt-dir", default=None, metavar="DIR",
        help="where checkpoints and csv_logs go. Defaults to the week_10 "
             "directory, which is what 11b/11d/11e read — point it elsewhere "
             "for a smoke test you don't want mixed in with real runs.",
    )
    p.add_argument(
        "--keep-going", action="store_true",
        help="on failure, log the traceback and continue with the next "
             "experiment instead of aborting the queue. The script still "
             "exits non-zero if anything failed.",
    )
    p.add_argument(
        "--git-pull", action="store_true",
        help="git pull the repo before training (the notebooks do this at "
             "setup). Off by default: pulling mid-queue would swap the code "
             "under a running job.",
    )
    p.add_argument(
        "--install-requirements", action="store_true",
        help="let colab_setup.setup() pip-install requirements.txt. Off by "
             "default; on a prepared conda env this is just slow.",
    )
    return p


# --------------------------------------------------------------------------
# Environment / bootstrap
# --------------------------------------------------------------------------
def apply_gpu_env(args: argparse.Namespace) -> None:
    """Pin the process to one GPU and set WANDB_MODE.

    MUST be called before anything imports torch: ``CUDA_VISIBLE_DEVICES`` is
    read once, at CUDA-context creation. After this call the pinned GPU is
    always local index 0 inside the process.
    """
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["WANDB_MODE"] = args.wandb_mode
    # Detached runs get their stdout through a pipe, which is block-buffered;
    # unbuffer it so `tail -f nohup.out` shows progress as it happens.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def bootstrap(args: argparse.Namespace) -> Dict[str, object]:
    """Resolve the repo, fix ``sys.path``, and load the shared artifacts.

    Mirrors the setup cells of the 11a/11c notebooks:

    1. ``chdir`` to the repo root — ``find_week10_artifacts`` walks upward
       from the cwd, so a detached job launched from anywhere still resolves.
    2. Put ``week_10`` in front of ``week_09`` on ``sys.path`` (the Week-09
       ``conditioned_infrastructure`` is an older stub that would otherwise
       shadow the real one), then repoint to the ``week_11`` copy, which is
       where this notebook family's API actually lives.
    3. Locate the Week-10 artifacts and load the classical model.

    Returns
    -------
    dict
        ``paths`` (from ``find_week10_artifacts``), ``week10_dir``,
        ``parquet_v2``, ``ckpt_dir``, ``classical``, ``bin_centers``,
        ``bin_width``, ``alpha_np``, ``sigma_np``, ``T``, ``device``.
    """
    os.chdir(REPO_ROOT)

    if args.git_pull:
        try:
            subprocess.run(["git", "-C", str(REPO_ROOT), "pull"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"git pull skipped: {e}")

    sys.path.insert(0, str(REPO_ROOT))
    from infrastructure.utils.colab_setup import setup
    setup(seed=args.seed, install_requirements=args.install_requirements)

    week09_dir = str(REPO_ROOT / "weeks" / "week_09")
    week10_dir_guess = str(REPO_ROOT / "weeks" / "week_10")
    week11_dir = str(REPO_ROOT / "weeks" / "week_11")
    # week_10 inserted last -> resolves first; week_09 trails so its
    # conditioned_infrastructure stub never shadows, but its
    # unconditioned_infrastructure stays importable.
    for p in (week09_dir, week11_dir, week10_dir_guess):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    sys.modules.pop("conditioned_infrastructure", None)

    from conditioned_infrastructure import find_week10_artifacts
    paths = find_week10_artifacts(extra_required=[
        "data/composite_sunspot_groups_peak_area.csv",
    ])
    # find_week10_artifacts resolves conditioned_infrastructure to the week_10
    # copy (its home, where the parquet + checkpoints live) and evicts any
    # other cached copy. Repoint the *module* to the week_11 copy — that's
    # where this family's API lives — while keeping the week_10 artifact paths.
    if week11_dir in sys.path:
        sys.path.remove(week11_dir)
    sys.path.insert(0, week11_dir)
    sys.modules.pop("conditioned_infrastructure", None)
    sys.modules.pop("empirical_infrastructure", None)

    import numpy as np
    import torch
    from unconditioned_infrastructure import make_cosine_schedule
    from butterflAI_model import ButterflAIModel
    import conditioned_infrastructure as _ci

    week10_dir = paths["week10_dir"]
    parquet_v2 = (os.path.abspath(args.parquet) if args.parquet
                  else os.path.join(week10_dir, "diffusion_windows_v2.parquet"))
    ckpt_dir = os.path.abspath(args.ckpt_dir) if args.ckpt_dir else week10_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    lat_bins = np.linspace(0, 45, 16)
    bin_width = 3.0
    bin_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])

    T = 200
    alpha_np, sigma_np, _ = make_cosine_schedule(T=T, s=0.008)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        dev_name = "cpu"
    else:
        # --gpu N already restricted the process to that GPU, so the visible
        # device is always local index 0; the label reports the physical one.
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        physical = os.environ.get("CUDA_VISIBLE_DEVICES", "auto")
        dev_name = (f"cuda:0 ({torch.cuda.get_device_name(device)}, "
                    f"CUDA_VISIBLE_DEVICES={physical})")

    print("=" * 78)
    print(f"repo root        : {REPO_ROOT}")
    print(f"infrastructure   : {_ci.__file__}")
    print(f"week10_dir       : {week10_dir}")
    print(f"parquet_v2       : {parquet_v2}")
    print(f"ckpt_dir         : {ckpt_dir}")
    print(f"device           : {dev_name}")
    print(f"wandb mode       : {os.environ.get('WANDB_MODE')}")
    print("=" * 78, flush=True)

    return {
        "paths": paths,
        "week10_dir": week10_dir,
        "parquet_v2": parquet_v2,
        "ckpt_dir": ckpt_dir,
        "classical": ButterflAIModel(paths["classical_weights"]),
        "bin_centers": bin_centers,
        "bin_width": bin_width,
        "alpha_np": alpha_np,
        "sigma_np": sigma_np,
        "T": T,
        "device": device,
    }


def load_windows_v2(parquet_v2: str):
    """Load the prebuilt v2 parquet, train+val rows only.

    The scripts never rebuild it — that is ``11_00_build_v2.ipynb``'s job —
    and the test split stays reserved for the PI.
    """
    import pandas as pd

    if not os.path.isfile(parquet_v2):
        raise FileNotFoundError(
            f"{parquet_v2} not found — run 11_00_build_v2.ipynb first.")
    windows = pd.read_parquet(parquet_v2)
    windows = windows.loc[windows["split"].isin(["train", "val"])].reset_index(drop=True)
    print(f"v2 parquet (train+val): {len(windows)} rows, {len(windows.columns)} cols")
    print(f"  splits: {windows['split'].value_counts().sort_index().to_dict()}")
    print(f"  cycles: {sorted(windows['cycle'].unique())}", flush=True)
    return windows


# --------------------------------------------------------------------------
# Queue selection and running
# --------------------------------------------------------------------------
def select_experiments(args: argparse.Namespace,
                       registry: Dict[str, dict]) -> List[str]:
    """Resolve the CLI request into an ordered list of experiment names."""
    if args.all:
        if args.experiments:
            raise SystemExit("pass either --all or explicit names, not both.")
        return list(registry)
    unknown = [n for n in args.experiments if n not in registry]
    if unknown:
        raise SystemExit(
            f"unknown experiment(s) {unknown}.\nDefined: {list(registry)}")
    if not args.experiments:
        raise SystemExit(
            "no experiments requested. Pass names, or --all, or --list to see "
            "the registry.")
    return list(args.experiments)


def print_registry(registry: Dict[str, dict], ckpt_dir: str,
                   ckpt_prefix: str) -> None:
    """Print every spec plus whether its checkpoint is already on disk."""
    print(f"\n{len(registry)} experiments (checkpoints in {ckpt_dir}, "
          f"stem {ckpt_prefix}<name>.ckpt):\n")
    for name, cfg in registry.items():
        done = os.path.isfile(os.path.join(ckpt_dir, f"{ckpt_prefix}{name}.ckpt"))
        print(f"  [{'x' if done else ' '}] {name:26s} arch={cfg['arch']:6s} "
              f"consumed={cfg['consumed_keys']} fourier={cfg['fourier']} "
              f"cond_dropout_p={cfg['cond_dropout_p']} "
              f"max_epochs={cfg['max_epochs']}")
    print("\n[x] = checkpoint already exists; training would be skipped "
          "(delete it to force a retrain).", flush=True)


def apply_cfg_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Return a copy of ``cfg`` with the CLI overrides applied."""
    cfg = dict(cfg)
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.eval_every is not None:
        cfg["eval_every_n_epochs"] = args.eval_every
    return cfg


def run_queue(names: Sequence[str], registry: Dict[str, dict],
              args: argparse.Namespace,
              train_one: Callable[[str, dict], object]) -> int:
    """Train each queued experiment in turn; return a process exit code.

    ``train_one(name, cfg)`` does the actual training; the underlying
    ``train_experiment`` / ``train_empirical_experiment`` are idempotent, so
    re-running a finished queue is a no-op.
    """
    failures: List[str] = []
    t_queue = time.time()

    for i, name in enumerate(names, 1):
        cfg = apply_cfg_overrides(registry[name], args)
        print(f"\n{'=' * 78}\n[{i}/{len(names)}] training {name}\n"
              f"    cfg: {cfg}\n{'=' * 78}", flush=True)
        t0 = time.time()
        try:
            train_one(name, cfg)
        except Exception:
            import traceback
            traceback.print_exc()
            failures.append(name)
            print(f"[{name}] FAILED after {time.time() - t0:.1f}s", flush=True)
            if not args.keep_going:
                print("aborting queue (pass --keep-going to continue past "
                      "failures).", flush=True)
                break
        else:
            print(f"[{name}] done in {(time.time() - t0) / 60:.1f} min",
                  flush=True)

    print(f"\n{'=' * 78}")
    print(f"queue finished in {(time.time() - t_queue) / 60:.1f} min: "
          f"{len(names) - len(failures)}/{len(names)} succeeded")
    if failures:
        print(f"failed: {failures}")
    print("=" * 78, flush=True)
    return 1 if failures else 0
