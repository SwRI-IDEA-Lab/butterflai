#!/usr/bin/env python
"""Week 11 — train the empirical-distribution diffusion (script version of 11c_train_empirical.ipynb).

Training only, headless: no plots, no wandb dependency, safe to detach. The
target is the *empirical* per-window latitude distribution rather than the
residual against the parametric classical; the classical density acts as a
prior through a KL regularizer. Non-negativity and integration to 1 hold by
construction — training happens in a standardized logit space and sampling
decodes via softmax.

The registry is the same 18-variant grid as the residual pair (11a/11b) — the
empirical target changes the loss, not the ablation grid — plus three
target-specific knobs (``lambda_kl``, ``alpha_smooth``, ``n_pseudo_obs``).
Checkpoints land at ``ckpt_emp_<name>.ckpt`` so they never collide with the
residual family's ``ckpt_<name>.ckpt``; the val_emd-selected checkpoint goes
to ``CKPT_DIR/select/<name>/``. Evaluation and the decoded-density diagnostic
stay in ``11d_evaluate_empirical.ipynb``.

Examples
--------
    # what is defined, and what is already trained
    python weeks/week_11/11c_train_empirical.py --list

    # one experiment on GPU 2, detached
    nohup python weeks/week_11/11c_train_empirical.py base_cat --gpu 2 \
        > logs/11c_base_cat.log 2>&1 &

    # the whole grid, don't stop at the first failure
    nohup python weeks/week_11/11c_train_empirical.py --all --gpu 2 --keep-going \
        > logs/11c_all.log 2>&1 &

See README_training_scripts.md for the full runbook.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_cli_common as common

# Mirrors WANDB_PROJECT in 11c_train_empirical.ipynb — a sibling project to the
# residual one so the two dashboards stay separable.
DEFAULT_WANDB_PROJECT = "butterflai-w10ext-amunoz-emp"

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

_BASE_TEMPLATE = {
    "arch":           "concat",
    "consumed_keys":  ["cond_base"],
    "groups":         ["base"],
    "hidden_dim":     128,
    "n_layers":       3,
    "fourier":        False,
    "cond_dropout_p": 0.0,
    "max_epochs":     5000,
    "lr":             1e-3,
    "batch_size":     64,
    # Empirical-target specific knobs.
    "lambda_kl":      0.1,     # weight on KL(p_model || p_classical)
    "alpha_smooth":   1.0,     # Laplace smoothing of the histogram target
    "n_pseudo_obs":   30.0,    # pseudo-count mass behind that smoothing
}


def _spec(**overrides):
    d = dict(_BASE_TEMPLATE)
    d.update(overrides)
    return d


_SPECS_EMP = [
    # ── Base conditioning (4-D cond) × mechanism ────────────────────────────
    # The full 2×2×2 over arch, Fourier lifting and guidance, holding the
    # information content fixed: how much comes from mechanism alone?
    _spec(),                                                   # base_cat  (baseline)
    _spec(arch="film"),                                        # base_film
    _spec(fourier=True),                                       # base_cat_four
    _spec(arch="film", fourier=True),                          # base_film_four
    _spec(cond_dropout_p=0.1),                                 # base_cat_guid
    _spec(arch="film", cond_dropout_p=0.1),                    # base_film_guid
    _spec(fourier=True, cond_dropout_p=0.1),                   # base_cat_four_guid
    _spec(arch="film", fourier=True, cond_dropout_p=0.1),      # base_film_four_guid

    # ── Level 1 — one new cond group at a time, concat arch ─────────────────
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
    # Target-specific override, in the same spirit as --max-epochs.
    parser.add_argument(
        "--lambda-kl", type=float, default=None, metavar="W",
        help="override cfg['lambda_kl'], the weight on "
             "KL(p_model || p_classical), for every queued experiment. 0 "
             "removes the classical anchor; raise toward 1–3 if generated "
             "distributions drift unphysically far from classical.",
    )
    args = parser.parse_args(argv)

    # Must precede every torch import — see apply_gpu_env's docstring.
    common.apply_gpu_env(args)
    env = common.bootstrap(args)

    from conditioned_infrastructure import build_experiment_registry
    from empirical_infrastructure import (
        train_empirical_experiment,
        EMP_CKPT_PREFIX,
    )

    # Keys the registry by canonical name; raises if two specs collide, which
    # catches both duplicates and knobs the naming scheme doesn't yet encode.
    experiments = build_experiment_registry(_SPECS_EMP, _BASE_TEMPLATE)
    if args.lambda_kl is not None:
        experiments = {n: dict(c, lambda_kl=args.lambda_kl)
                       for n, c in experiments.items()}

    if args.list:
        common.print_registry(experiments, env["ckpt_dir"],
                              ckpt_prefix=EMP_CKPT_PREFIX)
        return 0

    names = common.select_experiments(args, experiments)
    windows_aug = common.load_windows_v2(env["parquet_v2"])

    def train_one(name: str, cfg: dict) -> None:
        # train_empirical_experiment has no `device` argument — it builds its
        # Trainer with accelerator="auto". --gpu already pinned the process via
        # CUDA_VISIBLE_DEVICES, so "auto" resolves to the GPU you asked for.
        train_empirical_experiment(
            name=name, cfg=cfg, windows_aug=windows_aug,
            classical=env["classical"], bin_centers=env["bin_centers"],
            ckpt_dir=env["ckpt_dir"],
            alpha_np=env["alpha_np"], sigma_np=env["sigma_np"], T=env["T"],
            bin_width=env["bin_width"], seed=args.seed,
            enable_progress_bar=False,
            wandb_project=(None if args.wandb_mode == "disabled"
                           else args.wandb_project),
            wandb_entity=args.wandb_entity,
        )

    rc = common.run_queue(names, experiments, args, train_one)
    print("val_emd-selected checkpoints under",
          Path(env["ckpt_dir"]) / "select", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
