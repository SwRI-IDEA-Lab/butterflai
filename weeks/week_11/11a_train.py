#!/usr/bin/env python
"""Week 11 — train the residual diffusion model (script version of 11a_train.ipynb).

Training only, headless: no plots, no wandb dependency, safe to detach. The
experiment registry and every spec below are a literal transcription of the
notebook's Task-65 cell, so the two stay comparable; the notebook's
``ENABLED_EXPERIMENTS`` list becomes the command line.

Model selection is on ``val_emd`` (mean Wasserstein-1 between generated and
empirical latitude densities), not the denoising ``val_loss`` — the latter
rises while sample quality improves. The selected checkpoint lands in
``CKPT_DIR/select/<name>/``; final-iterate weights in ``ckpt_<name>.ckpt``.
Both live in ``weeks/week_10/`` alongside the parquet.

Examples
--------
    # what is defined, and what is already trained
    python weeks/week_11/11a_train.py --list

    # one experiment on GPU 3, detached
    nohup python weeks/week_11/11a_train.py base_cat --gpu 3 \
        > logs/11a_base_cat.log 2>&1 &

    # the whole grid, don't stop at the first failure
    nohup python weeks/week_11/11a_train.py --all --gpu 3 --keep-going \
        > logs/11a_all.log 2>&1 &

See README_training_scripts.md for the full runbook.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_cli_common as common

# Mirrors WANDB_PROJECT in 11a_train.ipynb (Task 64).
DEFAULT_WANDB_PROJECT = "butterflai-w11-amunoz-metrics"

# Run names are GENERATED from the specs by canonical_experiment_name(), not
# typed by hand, so a name can never disagree with the config it labels:
#
#     <cond-set>_<arch>[_four][_guid][_h###][_L#]
#
#   cond-set : "base" plus added groups in the fixed order hemi < opp < traj,
#              joined by "+"  ("trajv" = traj group + its validity mask)
#   arch     : "cat" (concat) or "film" — ALWAYS stated
#   four     : fourier=True          guid : cond_dropout_p > 0
#   h###/L#  : appended only when capacity differs from _BASE_TEMPLATE
#
# e.g. base+opp_film_four_guid. Add a spec to _SPECS; the name follows.

_BASE_TEMPLATE = {
    "arch":           "concat",
    "consumed_keys":  ["cond_base"],
    "groups":         ["base"],
    "hidden_dim":     128,
    "n_layers":       3,
    "fourier":        False,
    "cond_dropout_p": 0.0,
    "max_epochs":     20000,
    "lr":             1e-3,
    "batch_size":     64,
    "lambda_mass":    1.0,
    "lambda_neg":     1.0,
    "eval_every_n_epochs": 200,
    "n_eval_windows": 512,
    "n_ensemble":     16,
}


def _spec(**overrides):
    d = dict(_BASE_TEMPLATE)
    d.update(overrides)
    return d


_SPECS = [
    # ── Base conditioning (4-D cond) × mechanism ────────────────────────────
    # The full 2×2×2 over arch, Fourier lifting and guidance, holding the
    # information content fixed: how much comes from mechanism alone?
    _spec(),                                                   # base_cat  (Week-10 baseline)
    _spec(arch="film"),                                        # base_film
    _spec(fourier=True),                                       # base_cat_four
    _spec(arch="film", fourier=True),                          # base_film_four
    _spec(cond_dropout_p=0.1),                                 # base_cat_guid
    _spec(arch="film", cond_dropout_p=0.1),                    # base_film_guid
    _spec(fourier=True, cond_dropout_p=0.1),                   # base_cat_four_guid
    _spec(arch="film", fourier=True, cond_dropout_p=0.1),      # base_film_four_guid

    # ── Level 1 — one new cond group at a time, concat arch ─────────────────
    # Mechanism held fixed: does more information alone move val_emd?
    _spec(consumed_keys=["cond_base", "cond_cyclehemi"],       # base+hemi_cat
          groups=["base", "cyclehemi"]),
    _spec(consumed_keys=["cond_base", "cond_opp"],             # base+opp_cat
          groups=["base", "opp"]),
    _spec(consumed_keys=["cond_base", "cond_traj"],            # base+traj_cat
          groups=["base", "traj"]),

    # ── Levels 3–5 — best L1 cond group (opp) × mechanism ───────────────────
    _spec(arch="film",                                         # base+opp_film
          consumed_keys=["cond_base", "cond_opp"],
          groups=["base", "opp"]),
    _spec(arch="film",                                         # base+opp_film_guid
          consumed_keys=["cond_base", "cond_opp"],
          groups=["base", "opp"],
          cond_dropout_p=0.1),
    _spec(arch="film",                                         # base+opp_film_four
          consumed_keys=["cond_base", "cond_opp"],
          groups=["base", "opp"],
          fourier=True),
    _spec(arch="film",                                         # base+opp_film_four_guid
          consumed_keys=["cond_base", "cond_opp"],
          groups=["base", "opp"],
          fourier=True,
          cond_dropout_p=0.1),

    # ── Trajectory conditioning, and the opp × traj pair ────────────────────
    _spec(arch="film",                                         # base+traj_film_four
          consumed_keys=["cond_base", "cond_traj"],
          groups=["base", "traj"],
          fourier=True),
    _spec(arch="film",                                         # base+opp+traj_film_four
          consumed_keys=["cond_base", "cond_opp", "cond_traj"],
          groups=["base", "opp", "traj"],
          fourier=True),
    # Same as base+traj_film_four plus the window-validity mask — the "v".
    _spec(arch="film",                                         # base+trajv_film_four
          consumed_keys=["cond_base", "cond_traj", "cond_traj_valid"],
          groups=["base", "traj"],
          fourier=True),
]


def main(argv=None) -> int:
    parser = common.build_parser(
        description=__doc__.split("\n\n")[0],
        default_wandb_project=DEFAULT_WANDB_PROJECT,
    )
    args = parser.parse_args(argv)

    # Must precede every torch import — see apply_gpu_env's docstring.
    common.apply_gpu_env(args)
    env = common.bootstrap(args)

    from conditioned_infrastructure import build_experiment_registry, train_experiment

    # Keys the registry by canonical name; raises if two specs collide, which
    # catches both duplicates and knobs the naming scheme doesn't yet encode.
    experiments = build_experiment_registry(_SPECS, _BASE_TEMPLATE)

    if args.list:
        common.print_registry(experiments, env["ckpt_dir"], ckpt_prefix="ckpt_")
        return 0

    names = common.select_experiments(args, experiments)
    windows_aug = common.load_windows_v2(env["parquet_v2"])

    def train_one(name: str, cfg: dict) -> None:
        train_experiment(
            name=name, cfg=cfg,
            windows_aug=windows_aug, ckpt_dir=env["ckpt_dir"],
            wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
            alpha_np=env["alpha_np"], sigma_np=env["sigma_np"], T=env["T"],
            bin_centers=env["bin_centers"], bin_width=env["bin_width"],
            seed=args.seed, device=env["device"],
        )

    rc = common.run_queue(names, experiments, args, train_one)
    print("val_emd-selected checkpoints under",
          Path(env["ckpt_dir"]) / "select", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
