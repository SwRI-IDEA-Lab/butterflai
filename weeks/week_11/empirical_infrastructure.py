"""
empirical_infrastructure.py — Week 11 reverse diffusion on *empirical* per-window
latitude distributions (not on residuals against the parametric classical).

Why a separate module
---------------------
The residual framing (``hist_emp − hist_par``) hides most of cond's predictive
signal behind the parametric Gaussian — the diffusion is asked to learn a small,
noisy correction using the same inputs the classical already exploits. Week-11
ablations and the oracle gap analysis pointed at the target framing rather than
the architecture, so this module switches the target to the empirical
distribution itself, with the classical density acting as a *prior* (via a KL
regularizer) rather than a fixed pedestal.

What it provides
----------------
- Logit-space DDIM on per-bin probability mass, with softmax decode at sampling
  → simplex constraints (non-negative, integrates to 1) hold *by construction*.
- Laplace smoothing of the empirical histogram to defang single-bin spikes from
  low-count windows.
- KL(p_model || p_classical) regularizer in the training loss; controlled by
  ``lambda_kl``.
- The same per-group cond plumbing as ``ExtendedConditionalResidualDataset`` so
  every existing ``EXPERIMENTS`` knob (concat/film, fourier, cond_dropout_p,
  consumed_keys) carries over unchanged.

Reuses (no edits)
-----------------
``ConditionalDiffusionMLP``, ``build_model``, ``_consumed_cond_dim``,
``block_cond_concat``, ``discover_experiment_checkpoints`` from
``conditioned_infrastructure``.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy.stats import norm as sp_norm
from typing import Dict, List, Optional, Sequence, Tuple

import pytorch_lightning as pl

from conditioned_infrastructure import (
    build_model,
    _consumed_cond_dim,
    discover_experiment_checkpoints,
)
from unconditioned_infrastructure import SampleQualityCallback


# ---------------------------------------------------------------------------
# Histogram utilities
# ---------------------------------------------------------------------------

def laplace_smooth_mass(
    mass: np.ndarray,
    alpha: float = 1.0,
    n_pseudo: float = 30.0,
) -> np.ndarray:
    """Dirichlet-prior smoothing on a per-row probability vector.

    The v2 parquet stores normalized empirical densities, not raw counts, so a
    plain count-based Laplace formula has nothing to count. We instead treat
    each row as a normalized histogram from ``n_pseudo`` observations and apply

        m_smooth = (n_pseudo * mass + alpha) / (n_pseudo + K * alpha)

    With ``alpha=1`` and ``n_pseudo=30`` (typical window size noted in the
    parquet build code) this is equivalent to a ~1/3 convex blend against a
    uniform prior, mild enough to keep real bimodality but enough to dampen
    single-observation spikes that dominate low-count windows.
    """
    K = mass.shape[-1]
    return (n_pseudo * mass + alpha) / (n_pseudo + K * alpha)


def classical_mass_at_bins(
    classical,
    mu: float,
    sigma: float,
    bin_centers: np.ndarray,
    bin_width: float,
) -> np.ndarray:
    """Classical parametric mass at the 15 latitude bin centers.

    Returns a normalized (sums to 1) vector. Falls back to a uniform prior when
    ``sigma <= 0`` or the classical gate would otherwise produce all-zero mass,
    so the KL term stays finite for windows the classical can't represent.
    """
    K = bin_centers.shape[0]
    if not np.isfinite(sigma) or sigma <= 0:
        return np.full(K, 1.0 / K, dtype=np.float32)
    pdf = sp_norm.pdf(bin_centers, loc=mu, scale=sigma).astype(np.float32)
    m = pdf * bin_width
    s = float(m.sum())
    if not np.isfinite(s) or s <= 0:
        return np.full(K, 1.0 / K, dtype=np.float32)
    return (m / s).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmpiricalDistributionDataset(torch.utils.data.Dataset):
    """Per-window empirical distributions in *standardized logit space*, with
    a classical prior per window for the KL regularizer. Same per-group cond
    plumbing as ``ExtendedConditionalResidualDataset`` so the same EXPERIMENTS
    structure (consumed_keys, groups, fourier, cond_dropout_p, …) carries over.

    ``__getitem__(idx)`` returns:

        {
            "r_clean":          (15,) standardized mean-centered logits,
            "m_classical":      (15,) classical Dirichlet prior for KL term,
            "cond_<g>":         per-group standardized cond tensor,
            "cond_<g>_valid":   per-group 0/1 mask, if applicable,
        }

    The ``r_clean`` key name is preserved so ``ConditionalDiffusionMLP`` and the
    forward-diffusion code path (which expects "r_clean" in the batch) read it
    without modification — semantically it now carries logits, not residuals.
    """

    GROUP_COLS: Dict[str, List[str]] = {
        "base":       ["area_smoothed", "mu_universal", "model_sigma", "amplitude"],
        "cyclehemi":  ["cycle_norm", "hemi_id"],
        "opp":        ["opp_area_smoothed", "opp_mu_universal", "opp_amplitude"],
    }
    GROUP_VALIDITY: Dict[str, str] = {
        "opp":  "opp_valid",
        "traj": "traj_valid",
    }
    HIST_COLS = [f"hist_emp_{j:02d}" for j in range(15)]

    def __init__(
        self,
        df,
        split: str,
        classical,
        bin_centers: np.ndarray,
        bin_width: float = 3.0,
        groups: Sequence[str] = ("base",),
        group_stats: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        logit_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        alpha_smooth: float = 1.0,
        n_pseudo_obs: float = 30.0,
        logit_eps: float = 1e-8,
    ):
        self.groups = list(groups)
        self.bin_width = float(bin_width)
        self.bin_centers = np.asarray(bin_centers, dtype=np.float32)
        unknown = set(self.groups) - (set(self.GROUP_COLS) | {"traj"})
        if unknown:
            raise ValueError(f"unknown conditioning group(s): {unknown}")

        self._group_cols: Dict[str, List[str]] = {}
        for g in self.groups:
            if g == "traj":
                self._group_cols[g] = sorted(
                    [c for c in df.columns if c.startswith("area_lag")],
                    key=lambda c: int(c.replace("area_lag", "")),
                )
                if not self._group_cols[g]:
                    raise ValueError("'traj' group requested but no area_lag* "
                                     "columns found in df")
            else:
                self._group_cols[g] = list(self.GROUP_COLS[g])

        df_split = df.loc[df["split"] == split].reset_index(drop=True)

        # Empirical density -> mass -> Laplace-smoothed mass.
        hist_density = df_split[self.HIST_COLS].to_numpy(dtype=np.float32)
        mass = hist_density * self.bin_width
        row_sum = mass.sum(axis=1, keepdims=True)
        row_sum = np.clip(row_sum, 1e-12, None)
        mass = mass / row_sum
        # Normalized empirical density (1/deg), kept for the distributional
        # validation metrics (the model's softmax output is compared to it).
        self.emp_dens = (mass / self.bin_width).astype(np.float32)
        mass_smooth = laplace_smooth_mass(mass, alpha=alpha_smooth,
                                          n_pseudo=n_pseudo_obs)

        # Mean-centered logits (removes softmax's translation invariance so
        # per-bin standardization is meaningful).
        log_mass = np.log(mass_smooth + logit_eps)
        logits = log_mass - log_mass.mean(axis=1, keepdims=True)
        logits_t = torch.from_numpy(logits.astype(np.float32))

        if logit_stats is None:
            self.logit_means = logits_t.mean(dim=0)
            self.logit_stds  = logits_t.std(dim=0).clamp(min=1e-6)
        else:
            self.logit_means = torch.as_tensor(logit_stats[0], dtype=torch.float32)
            self.logit_stds  = torch.as_tensor(logit_stats[1], dtype=torch.float32).clamp(min=1e-6)
        self._logits_std = (logits_t - self.logit_means) / self.logit_stds

        # Classical prior per window for KL regularizer.
        m_cl = np.zeros((len(df_split), 15), dtype=np.float32)
        if "tau_center" not in df_split.columns:
            raise ValueError("df must carry 'tau_center' for the KL prior")
        if "amplitude" not in df_split.columns:
            raise ValueError("df must carry 'amplitude' for the KL prior")
        taus = df_split["tau_center"].to_numpy(dtype=np.float32)
        amps = df_split["amplitude"].to_numpy(dtype=np.float32)
        for i in range(len(df_split)):
            tau_i = float(taus[i])
            A_i   = float(amps[i])
            mu_i    = float(classical.mu(tau_i))
            sigma_i = float(classical.sigma(mu_i, A_i))
            m_cl[i] = classical_mass_at_bins(classical, mu_i, sigma_i,
                                             self.bin_centers, self.bin_width)
        self._m_classical = torch.from_numpy(m_cl)

        # Per-group standardization — train-split-only when not provided.
        if group_stats is None:
            df_train = df.loc[df["split"] == "train"]
            self.group_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            for g in self.groups:
                cols = self._group_cols[g]
                raw  = torch.from_numpy(df_train[cols].to_numpy(dtype=np.float32))
                means = raw.mean(dim=0)
                stds  = raw.std(dim=0).clamp(min=1e-6)
                self.group_stats[g] = (means, stds)
        else:
            self.group_stats = {
                g: (torch.as_tensor(group_stats[g][0], dtype=torch.float32),
                    torch.as_tensor(group_stats[g][1], dtype=torch.float32))
                for g in self.groups
            }

        self._cond: Dict[str, torch.Tensor] = {}
        self._valid: Dict[str, torch.Tensor] = {}
        for g in self.groups:
            cols = self._group_cols[g]
            raw  = torch.from_numpy(df_split[cols].to_numpy(dtype=np.float32))
            means, stds = self.group_stats[g]
            self._cond[g] = (raw - means) / stds
            if g in self.GROUP_VALIDITY:
                col = self.GROUP_VALIDITY[g]
                if col not in df_split.columns:
                    raise ValueError(f"group '{g}' requires validity column "
                                     f"'{col}' in df")
                v = torch.from_numpy(df_split[col].to_numpy(dtype=np.float32))
                self._valid[g] = v.unsqueeze(-1)

    @property
    def group_dims(self) -> Dict[str, int]:
        return {g: self._cond[g].shape[1] for g in self.groups}

    def __len__(self) -> int:
        return self._logits_std.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {
            "r_clean":     self._logits_std[idx],
            "m_classical": self._m_classical[idx],
        }
        for g in self.groups:
            item[f"cond_{g}"] = self._cond[g][idx]
            if g in self._valid:
                item[f"cond_{g}_valid"] = self._valid[g][idx]
        return item


# ---------------------------------------------------------------------------
# Lightning module: DDIM on standardized logits + KL anchor to classical
# ---------------------------------------------------------------------------

class ExtendedConditionalEmpiricalDiffusionLightning(pl.LightningModule):
    """Logit-space DDIM that decodes to a probability density via softmax.

    Same noise-prediction backbone (``ConditionalDiffusionMLP`` / FiLM /
    ConcatExtended via ``build_model``) and CFG plumbing as
    ``ExtendedConditionalDiffusionLightning``. The only loss-side difference is
    an additive KL(p_model || p_classical) term controlled by ``lambda_kl``,
    computed by recovering the predicted clean logits, softmaxing, and
    comparing to the per-window classical mass carried in the batch as
    ``m_classical``.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha,
        sigma,
        T: int,
        consumed_keys: Sequence[str],
        group_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        total_cond_dim: int,
        logit_means=None,
        logit_stds=None,
        lambda_kl: float = 0.1,
        bin_centers=None,
        bin_width: float = 3.0,
        lambda_band: float = 0.0,
        band_lo: float = 5.0,
        band_hi: float = 40.0,
        lr: float = 1e-3,
        scheduler: str = "cosine",
        weight_decay: float = 1e-4,
        cond_dropout_p: float = 0.0,
        kl_floor_eps: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "model", "alpha", "sigma",
            "logit_means", "logit_stds",
            "group_stats",
        ])

        self.model          = model
        self.T              = int(T)
        self.lr             = float(lr)
        self.scheduler_kind = str(scheduler)
        self.weight_decay   = float(weight_decay)
        self.cond_dropout_p = float(cond_dropout_p)
        self.consumed_keys  = list(consumed_keys)
        self.total_cond_dim = int(total_cond_dim)
        self.lambda_kl      = float(lambda_kl)
        self.kl_floor_eps   = float(kl_floor_eps)
        self.bin_width      = float(bin_width)
        self.lambda_band    = float(lambda_band)   # mass outside Spoerer band
        self.band_lo        = float(band_lo)
        self.band_hi        = float(band_hi)

        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.register_buffer("sigma", torch.as_tensor(sigma, dtype=torch.float32))

        lm = (torch.zeros(15, dtype=torch.float32) if logit_means is None
              else torch.as_tensor(logit_means, dtype=torch.float32))
        ls = (torch.ones(15, dtype=torch.float32)  if logit_stds  is None
              else torch.as_tensor(logit_stds,  dtype=torch.float32))
        self.register_buffer("logit_means", lm)
        self.register_buffer("logit_stds",  ls)

        K = lm.shape[0]
        bc = ((torch.arange(K, dtype=torch.float32) + 0.5) * self.bin_width
              if bin_centers is None
              else torch.as_tensor(bin_centers, dtype=torch.float32))
        in_band = ((bc >= self.band_lo) & (bc <= self.band_hi)).to(torch.float32)
        self.register_buffer("bin_centers", bc, persistent=False)
        self.register_buffer("band_mask", in_band, persistent=False)

        self._group_names: List[str] = []
        for name, (means, stds) in group_stats.items():
            self._group_names.append(name)
            self.register_buffer(f"cond_{name}_means",
                                 torch.as_tensor(means, dtype=torch.float32))
            self.register_buffer(f"cond_{name}_stds",
                                 torch.as_tensor(stds,  dtype=torch.float32))

        self.null_cond = nn.Parameter(torch.zeros(self.total_cond_dim))

    def _concat_cond(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([batch[k] for k in self.consumed_keys], dim=-1)

    def _shared_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        r_clean     = batch["r_clean"]
        m_classical = batch["m_classical"]
        cond        = self._concat_cond(batch)
        B           = r_clean.shape[0]

        if self.cond_dropout_p > 0.0:
            drop = (torch.rand(B, device=cond.device) < self.cond_dropout_p)
            null = self.null_cond.to(cond.dtype).expand_as(cond)
            cond = torch.where(drop.unsqueeze(-1), null, cond)

        t       = torch.randint(0, self.T, (B,), device=r_clean.device)
        eps     = torch.randn_like(r_clean)
        alpha_t = rearrange(self.alpha[t], "b -> b 1")
        sigma_t = rearrange(self.sigma[t], "b -> b 1")
        r_t     = alpha_t * r_clean + sigma_t * eps

        eps_hat  = self.model(r_t, t, cond)
        mse_loss = F.mse_loss(eps_hat, eps)

        loss = mse_loss

        # Recover the predicted clean density once if any density-space term
        # (KL anchor or Spoerer-band penalty) is active. The softmax output is
        # already a normalized, non-negative probability mass, so — unlike the
        # residual model — no mass-conservation/non-negativity terms are needed.
        if self.lambda_kl > 0.0 or self.lambda_band > 0.0:
            x0_std  = (r_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)
            logits  = x0_std * self.logit_stds + self.logit_means
            p_model = F.softmax(logits, dim=-1)

            if self.lambda_kl > 0.0:
                kl_per_sample = (
                    p_model * (torch.log(p_model + self.kl_floor_eps)
                               - torch.log(m_classical + self.kl_floor_eps))
                ).sum(dim=-1)
                kl_loss = kl_per_sample.mean()
                loss    = loss + self.lambda_kl * kl_loss
                self.log("mse_loss", mse_loss, on_step=False, on_epoch=True)
                self.log("kl_loss",  kl_loss,  on_step=False, on_epoch=True)

            if self.lambda_band > 0.0:
                oob = (p_model * (1.0 - self.band_mask)).sum(dim=-1)  # mass
                self.log("out_of_band_mass", oob.mean(),
                         on_step=False, on_epoch=True)
                loss = loss + self.lambda_band * (oob ** 2).mean()

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        decay_params, no_decay_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or name.endswith(".bias"):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )

        if self.scheduler_kind == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-4,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sched, "monitor": "val_loss", "interval": "epoch"},
            }
        if self.scheduler_kind == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                eta_min=1e-4,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sched, "interval": "step"},
            }
        if self.scheduler_kind == "none":
            return optimizer
        raise ValueError(f"unknown scheduler {self.scheduler_kind!r}")


# ---------------------------------------------------------------------------
# Sampling: DDIM in standardized-logit space, decode via softmax
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_empirical_extended(
    lightning_module: ExtendedConditionalEmpiricalDiffusionLightning,
    cond: torch.Tensor,
    guidance_w: float = 0.0,
    data_dim: int = 15,
    bin_width: float = 3.0,
    device=None,
) -> torch.Tensor:
    """DDIM sampler analogous to ``sample_conditional_extended`` but decoding
    each sampled clean state to a probability density via softmax.

    Returns a tensor of shape (B, 15) holding per-bin density (non-negative,
    sums to 1/bin_width when summed across bins → integrates to 1 in physical
    units).
    """
    if device is None:
        device = next(lightning_module.parameters()).device
    lightning_module.eval()
    lightning_module.to(device)
    cond = cond.to(device)

    alpha = lightning_module.alpha
    sigma = lightning_module.sigma
    T_loc = lightning_module.T

    B = cond.shape[0]
    r_t = torch.randn(B, data_dim, device=device)

    use_cfg = float(guidance_w) > 0.0
    if use_cfg:
        null = lightning_module.null_cond.to(device).expand_as(cond)

    for t in range(T_loc - 1, -1, -1):
        t_batch  = torch.full((B,), t, dtype=torch.long, device=device)
        eps_cond = lightning_module.model(r_t, t_batch, cond)
        if use_cfg:
            eps_null = lightning_module.model(r_t, t_batch, null)
            eps_hat  = (1.0 + guidance_w) * eps_cond - guidance_w * eps_null
        else:
            eps_hat  = eps_cond

        alpha_t = alpha[t]
        sigma_t = sigma[t]
        r_0_hat = (r_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)

        if t > 0:
            r_t = alpha[t - 1] * r_0_hat + sigma[t - 1] * eps_hat
        else:
            r_t = r_0_hat

    logits  = r_t * lightning_module.logit_stds.to(device) + lightning_module.logit_means.to(device)
    p_mass  = torch.softmax(logits, dim=-1)
    p_dens  = p_mass / float(bin_width)
    return p_dens


# ---------------------------------------------------------------------------
# Train / load helpers (parallel to train_experiment / load_trained_experiment)
# ---------------------------------------------------------------------------

EMP_CKPT_PREFIX = "ckpt_emp_"


class EmpSampleQualityCallback(SampleQualityCallback):
    """Distributional validation metrics for the logit/softmax model.

    Mirrors ``CondSampleQualityCallback`` but draws from
    ``sample_empirical_extended`` (whose output is already a normalized
    density, so the model density needs no ``par`` reconstruction). When
    ``val_emp``/``val_cond`` are supplied it logs ``val_emd``/``val_crps_mu``/…
    in ``on_validation_epoch_end`` for model selection.
    """

    def __init__(self, train_samples, cond_reference,
                 val_emp=None, val_cond=None, val_tau=None,
                 n_eval_windows: int = 64, n_ensemble: int = 16, **kw):
        super().__init__(train_samples, **kw)
        self._cond_ref = cond_reference.detach().clone()
        self._val_emp  = None if val_emp  is None else np.asarray(val_emp,  dtype=np.float32)
        self._val_cond = None if val_cond is None else val_cond.detach().clone()
        self._val_tau  = None if val_tau  is None else np.asarray(val_tau,  dtype=np.float32)
        self._n_eval_windows = int(n_eval_windows)
        self._n_ensemble     = int(n_ensemble)
        self._eval_idx = None
        if self._val_cond is not None:
            n = self._val_cond.shape[0]
            k = min(self._n_eval_windows, n)
            self._eval_idx = np.random.default_rng(0).choice(n, size=k, replace=False)

    def _sample(self, pl_module):
        device = next(pl_module.parameters()).device
        return sample_empirical_extended(
            pl_module, self._cond_ref, guidance_w=0.0,
            bin_width=self.bin_width, device=device,
        ).cpu().numpy()

    def _distributional_eval(self, trainer, pl_module):
        if self._val_cond is None or self._val_emp is None:
            return
        from distribution_metrics import distributional_report

        device = next(pl_module.parameters()).device
        idx = self._eval_idx
        M = self._n_ensemble
        cond_rep = self._val_cond[idx].repeat_interleave(M, dim=0)
        model_dens = sample_empirical_extended(
            pl_module, cond_rep, guidance_w=0.0,
            bin_width=self.bin_width, device=device,
        ).cpu().numpy().reshape(len(idx), M, -1)            # (W, M, K) density

        emp_dens = self._val_emp[idx]
        tau = None if self._val_tau is None else self._val_tau[idx]
        rep = distributional_report(
            model_dens, emp_dens, self.bin_centers,
            bin_width=self.bin_width, tau=tau,
        )
        for key in ("emd", "energy", "mu_mae", "sigma_mae", "skew_mae",
                    "crps_mu", "crps_sigma"):
            pl_module.log(f"val_{key}", float(rep[key]),
                          on_step=False, on_epoch=True)
        pl_module.train()


def train_empirical_experiment(
    name: str,
    cfg: Dict,
    windows_aug,
    classical,
    bin_centers: np.ndarray,
    ckpt_dir: str,
    alpha_np,
    sigma_np,
    T: int,
    bin_width: float = 3.0,
    seed: int = 42,
    enable_progress_bar: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
) -> str:
    """Train one empirical-target experiment to completion and save its
    checkpoint as ``ckpt_emp_<name>.ckpt``. Idempotent on existing checkpoints.

    If ``wandb_project`` is provided, training logs to WandB with the
    experiment ``name`` as the run name; falls back to a local ``CSVLogger``
    if WandB is unavailable. This mirrors ``train_experiment`` in
    ``conditioned_infrastructure.py``.
    """
    from pytorch_lightning.loggers import WandbLogger, CSVLogger
    try:
        import wandb
    except ImportError:
        wandb = None
    from infrastructure.utils.reproducibility import set_all_seeds

    ckpt_path = os.path.join(ckpt_dir, f"{EMP_CKPT_PREFIX}{name}.ckpt")
    if os.path.isfile(ckpt_path):
        print(f"[{name}] checkpoint exists at {ckpt_path}; skipping training.")
        return ckpt_path

    set_all_seeds(seed)

    alpha_smooth = float(cfg.get("alpha_smooth", 1.0))
    n_pseudo_obs = float(cfg.get("n_pseudo_obs", 30.0))

    train_ds = EmpiricalDistributionDataset(
        windows_aug, "train", classical=classical,
        bin_centers=bin_centers, bin_width=bin_width,
        groups=cfg["groups"],
        alpha_smooth=alpha_smooth, n_pseudo_obs=n_pseudo_obs,
    )
    val_ds = EmpiricalDistributionDataset(
        windows_aug, "val", classical=classical,
        bin_centers=bin_centers, bin_width=bin_width,
        groups=cfg["groups"],
        group_stats=train_ds.group_stats,
        logit_stats=(train_ds.logit_means, train_ds.logit_stds),
        alpha_smooth=alpha_smooth, n_pseudo_obs=n_pseudo_obs,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0,
    )

    total_dim = sum(_consumed_cond_dim(train_ds, k)
                    for k in cfg["consumed_keys"])

    model = build_model(cfg, total_cond_dim=total_dim)
    lit = ExtendedConditionalEmpiricalDiffusionLightning(
        model=model, alpha=alpha_np, sigma=sigma_np, T=T,
        consumed_keys=cfg["consumed_keys"],
        group_stats={g: train_ds.group_stats[g] for g in cfg["groups"]
                     if f"cond_{g}" in cfg["consumed_keys"]},
        total_cond_dim=total_dim,
        logit_means=train_ds.logit_means, logit_stds=train_ds.logit_stds,
        lambda_kl=float(cfg.get("lambda_kl", 0.1)),
        bin_centers=bin_centers, bin_width=bin_width,
        lambda_band=float(cfg.get("lambda_band", 0.0)),
        lr=cfg["lr"], scheduler="cosine", weight_decay=1e-4,
        cond_dropout_p=cfg.get("cond_dropout_p", 0.0),
    )

    # Distributional validation metrics + selection on val EMD (not val_loss).
    cond_ref = torch.cat(
        [torch.stack([train_ds[i][k] for i in range(min(500, len(train_ds)))])
         for k in cfg["consumed_keys"]], dim=-1,
    )
    val_cond = torch.cat(
        [torch.stack([val_ds[i][k] for i in range(len(val_ds))])
         for k in cfg["consumed_keys"]], dim=-1,
    )
    eval_every = cfg.get("eval_every_n_epochs", 200)
    cb = EmpSampleQualityCallback(
        train_samples=train_ds.emp_dens, cond_reference=cond_ref,
        val_emp=val_ds.emp_dens, val_cond=val_cond,
        n_eval_windows=cfg.get("n_eval_windows", 64),
        n_ensemble=cfg.get("n_ensemble", 16),
        every_n_epochs=eval_every, n_compare=cond_ref.shape[0],
        bin_centers=bin_centers, bin_width=bin_width,
    )
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(ckpt_dir, "select", name),
        filename="best-{epoch}-{val_emd:.3f}",
        monitor="val_emd", mode="min", save_top_k=1,
        every_n_epochs=eval_every, save_last=True,
    )

    if wandb_project is not None:
        try:
            logger = WandbLogger(
                project=wandb_project, entity=wandb_entity, name=f"emp_{name}",
                save_dir=os.path.join(ckpt_dir, "wandb_logs"),
            )
        except Exception as _e:
            print(f"[{name}] WandB unavailable ({_e}); falling back to CSVLogger.")
            logger = CSVLogger(os.path.join(ckpt_dir, "csv_logs"), name=f"emp_{name}")
    else:
        logger = CSVLogger(os.path.join(ckpt_dir, "csv_logs"), name=f"emp_{name}")

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"], logger=logger,
        accelerator="auto", devices="auto",
        log_every_n_steps=10, enable_progress_bar=enable_progress_bar,
        callbacks=[cb, ckpt_cb],
        check_val_every_n_epoch=eval_every,
    )
    trainer.fit(lit, train_loader, val_loader)
    # Final-iterate checkpoint; the val_emd-best lives under select/<name>/.
    trainer.save_checkpoint(ckpt_path)
    if wandb is not None and wandb_project is not None:
        try:
            wandb.finish()
        except Exception:
            pass
    print(f"[{name}] saved {ckpt_path}")
    return ckpt_path


def load_trained_empirical_experiment(
    name: str,
    cfg: Dict,
    windows_aug,
    classical,
    bin_centers: np.ndarray,
    ckpt_dir: str,
    alpha_np,
    sigma_np,
    bin_width: float = 3.0,
) -> Tuple[ExtendedConditionalEmpiricalDiffusionLightning,
           EmpiricalDistributionDataset,
           EmpiricalDistributionDataset,
           int]:
    """Rebuild the empirical datasets and load the matching checkpoint."""
    alpha_smooth = float(cfg.get("alpha_smooth", 1.0))
    n_pseudo_obs = float(cfg.get("n_pseudo_obs", 30.0))

    train_ds = EmpiricalDistributionDataset(
        windows_aug, "train", classical=classical,
        bin_centers=bin_centers, bin_width=bin_width,
        groups=cfg["groups"],
        alpha_smooth=alpha_smooth, n_pseudo_obs=n_pseudo_obs,
    )
    val_ds = EmpiricalDistributionDataset(
        windows_aug, "val", classical=classical,
        bin_centers=bin_centers, bin_width=bin_width,
        groups=cfg["groups"],
        group_stats=train_ds.group_stats,
        logit_stats=(train_ds.logit_means, train_ds.logit_stds),
        alpha_smooth=alpha_smooth, n_pseudo_obs=n_pseudo_obs,
    )
    total_dim = sum(_consumed_cond_dim(train_ds, k)
                    for k in cfg["consumed_keys"])

    model = build_model(cfg, total_cond_dim=total_dim)
    ckpt_path = os.path.join(ckpt_dir, f"{EMP_CKPT_PREFIX}{name}.ckpt")
    lit = ExtendedConditionalEmpiricalDiffusionLightning.load_from_checkpoint(
        ckpt_path,
        model=model, alpha=alpha_np, sigma=sigma_np,
        group_stats={g: train_ds.group_stats[g] for g in cfg["groups"]
                     if f"cond_{g}" in cfg["consumed_keys"]},
        total_cond_dim=total_dim, consumed_keys=cfg["consumed_keys"],
    )
    return lit, train_ds, val_ds, total_dim


def discover_emp_experiment_checkpoints(ckpt_dir: str) -> Dict[str, str]:
    """Discover ``ckpt_emp_<name>.ckpt`` files. Thin wrapper around the
    residual side's discover function with the empirical prefix.
    """
    return discover_experiment_checkpoints(
        ckpt_dir, prefix=EMP_CKPT_PREFIX, suffix=".ckpt",
    )


# ---------------------------------------------------------------------------
# NLL primitive for direct density (no `+ p_classical`, no `+ log Z`)
# ---------------------------------------------------------------------------

def hard_nll_direct(
    model,
    hcs,
    p_density_by_block,
    bin_width: float = 3.0,
    eps: float = 1e-6,
):
    """Score per-bin densities at observed latitudes, gated by the classical
    model's ``mu_0(A)`` so the included-block set matches ``hard_nll_combined``
    exactly (apples-to-apples vs the residual experiments and the classical
    baseline). Signature matches ``hard_nll_combined`` so it can be passed
    through ``k_run_combined`` unchanged.
    """
    total, included, candidate = 0.0, 0, 0
    n_floored, n_lats = 0, 0
    for hc in hcs:
        A    = hc["amplitude"]
        mu0A = float(model.mu_0(A))
        for blk in hc["blocks"]:
            candidate += 1
            mu = float(model.mu(blk["tau"]))
            if mu > mu0A:
                continue
            sigma = float(model.sigma(mu, A))
            if sigma <= 0:
                continue
            key = (hc["cycle"], hc["hemisphere"], blk["center_decimal"])
            if key not in p_density_by_block:
                continue
            p_density = p_density_by_block[key]

            lats   = blk["lats"]
            bin_ix = np.clip(np.floor(lats / bin_width).astype(int), 0, 14)
            p_lat  = p_density[bin_ix]
            p_safe = np.maximum(eps, p_lat)
            n_floored += int((p_lat < eps).sum())
            n_lats    += len(lats)

            ll = np.log(p_safe).mean()
            if not np.isfinite(ll):
                continue
            total    -= ll
            included += 1
    nll = total / included if included > 0 else float("inf")
    return nll, {"included": included, "candidate": candidate,
                 "coverage": included / max(candidate, 1),
                 "floor_fraction": n_floored / max(n_lats, 1)}
