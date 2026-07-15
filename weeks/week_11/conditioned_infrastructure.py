"""
conditioned_infrastructure.py — ButterflAI Week 09 *conditional* diffusion machinery.

This script contains the new class and function *definitions* introduced in
Week 09. The classes mirror their Week 08 unconditional counterparts; the
only conceptual differences are (a) the Dataset returns a dict
``{"r_clean", "cond"}`` rather than a bare residual tensor, (b) the MLP
concatenates ``cond`` alongside ``r_t`` and the timestep embedding at
the input layer, (c) the LightningModule reads from the dict batch and
persists the train-set cond standardization stats as buffers, and
(d) the sampler accepts a per-sample ``cond`` argument.

Everything that does not change between the unconditional and conditional
pipelines — the cosine schedule, the sinusoidal timestep embedding, the
TimestepEmbedding wrapper module, the SampleQualityCallback — is imported
unchanged from `unconditioned_infrastructure.py` rather than re-implemented here.
This is deliberate: the only differences worth seeing are the differences
conditioning introduces.

Contents
--------
ConditionalResidualDataset       : clean residuals + normalized (area, mu) covariates.
ConditionalDiffusionMLP          : (r_t, t, cond) → ε̂, raw input-concatenation.
ConditionalDiffusionLightning    : training/validation with conditioning batches.
sample_conditional               : DDIM sampler that accepts per-sample conditioning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import pytorch_lightning as pl

# Reuse Week 08 machinery unchanged — conditioning does not affect any of these.
from unconditioned_infrastructure import (
    TimestepEmbedding,
    make_cosine_schedule,         # re-exported for callers that want one source of truth
    SampleQualityCallback,        # re-exported (the training notebook reuses it)
)


# ─── Task 45 — ConditionalResidualDataset ─────────────────────────────────

class ConditionalResidualDataset(torch.utils.data.Dataset):
    """Indexed clean residuals (hist_emp − hist_par) plus normalized
    conditioning covariates (area_smoothed, mu_universal, model_sigma,
    amplitude) for one split.

    `__getitem__(idx)` returns a **dictionary**:

        {
            "r_clean": torch.float32 tensor, shape (15,),
            "cond":    torch.float32 tensor, shape (4,),
        }

    Per-bin residual standardization is identical to the Week 08 dataset:
    each split computes its own bin_means / bin_stds at construction time
    and exposes them so the sampler can de-standardize at inference time.

    Conditioning standardization uses **train-split-only** statistics. When
    `cond_means` / `cond_stds` are not supplied, the Dataset computes them
    from the train rows of the supplied DataFrame regardless of which
    split this Dataset itself represents.
    """

    HIST_COLS = [f"hist_emp_{j:02d}" for j in range(15)]
    PAR_COLS  = [f"hist_par_{j:02d}" for j in range(15)]
    COND_COLS = ["area_smoothed", "mu_universal", "model_sigma", "amplitude"]

    def __init__(self, df, split: str, cond_means=None, cond_stds=None):
        df_split = df.loc[df["split"] == split].reset_index(drop=True)

        # Residuals + per-bin standardization (same convention as Week 08).
        emp = df_split[self.HIST_COLS].to_numpy(dtype=np.float32)
        par = df_split[self.PAR_COLS].to_numpy(dtype=np.float32)
        residuals = torch.from_numpy(emp - par)                          # (N, 15)
        self.bin_means = residuals.mean(dim=0)                           # (15,)
        self.bin_stds  = residuals.std(dim=0).clamp(min=1e-6)            # (15,)
        self._residuals = (residuals - self.bin_means) / self.bin_stds   # (N, 15)

        # Conditioning standardization: train-split-only stats unless the
        # caller supplied pre-computed values.
        if cond_means is None or cond_stds is None:
            df_train = df.loc[df["split"] == "train"]
            cond_train = torch.from_numpy(
                df_train[self.COND_COLS].to_numpy(dtype=np.float32)
            )
            self.cond_means = cond_train.mean(dim=0)
            self.cond_stds  = cond_train.std(dim=0).clamp(min=1e-6)
        else:
            self.cond_means = torch.as_tensor(cond_means, dtype=torch.float32)
            self.cond_stds  = torch.as_tensor(cond_stds,  dtype=torch.float32)

        cond_raw  = torch.from_numpy(
            df_split[self.COND_COLS].to_numpy(dtype=np.float32)
        )                                                                # (N, 4)
        self._cond = (cond_raw - self.cond_means) / self.cond_stds       # (N, 4)

    def __len__(self):
        return self._residuals.shape[0]

    def __getitem__(self, idx):
        return {
            "r_clean": self._residuals[idx],
            "cond":    self._cond[idx],
        }


# ─── Task 48 — ConditionalDiffusionMLP ─────────────────────────────────────

class ConditionalDiffusionMLP(nn.Module):
    """ε-predictor MLP: (r_t, t, cond) → ε̂.

    Concatenates r_t, the timestep embedding (from the TimestepEmbedding
    module reused from Week 08), and the conditioning vector at the input
    layer. Pure input-concatenation — no separate ConditionEmbedding
    wrapper.
    """

    def __init__(
        self,
        data_dim: int = 15,
        cond_dim: int = 4,
        hidden_dim: int = 128,
        t_embed_dim: int = 64,
        t_hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.cond_dim = cond_dim

        self.t_embedding = TimestepEmbedding(t_embed_dim, t_hidden_dim)
        in_dim = data_dim + t_hidden_dim + cond_dim

        layers = []
        prev = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.SiLU())
            prev = hidden_dim
        layers.append(nn.Linear(prev, data_dim))                         # no activation on output
        self.net = nn.Sequential(*layers)

    def forward(self, r_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_embedding(t)                                      # (B, t_hidden_dim)
        x = torch.cat([r_t, t_emb, cond], dim=-1)                        # (B, data_dim + t_hidden_dim + cond_dim)
        return self.net(x)


# ─── Task 50 — ConditionalDiffusionLightning ───────────────────────────────

class ConditionalDiffusionLightning(pl.LightningModule):
    """Conditional version of DiffusionLightning.

    Each training/validation batch arrives as a **dict**
    ``{"r_clean": (B, 15), "cond": (B, 4)}``. The model call gets an extra
    ``cond`` argument; everything else (schedule buffers, AdamW with
    no-decay split, cosine/plateau LR schedulers, save_hyperparameters)
    is unchanged from Week 08.

    The train-set conditioning statistics (``cond_means``, ``cond_stds``)
    are persisted as buffers on the module alongside the existing
    ``bin_means`` / ``bin_stds`` so that ``sample_conditional`` can read
    all four statistics from a loaded checkpoint.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha,
        sigma,
        T: int,
        lr: float = 1e-3,
        scheduler: str = "cosine",
        weight_decay: float = 1e-4,
        bin_means=None,
        bin_stds=None,
        cond_means=None,
        cond_stds=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "model", "alpha", "sigma",
            "bin_means", "bin_stds",
            "cond_means", "cond_stds",
        ])

        self.model          = model
        self.T              = int(T)
        self.lr             = float(lr)
        self.scheduler_kind = str(scheduler)
        self.weight_decay   = float(weight_decay)

        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.register_buffer("sigma", torch.as_tensor(sigma, dtype=torch.float32))

        # Register all four standardization statistics as buffers. Use
        # zeros/ones placeholders when None is supplied so the state_dict
        # entries always exist — load_from_checkpoint overwrites them with
        # the saved values.
        bin_means_t = (torch.zeros(15, dtype=torch.float32) if bin_means is None
                       else torch.as_tensor(bin_means, dtype=torch.float32))
        bin_stds_t  = (torch.ones(15, dtype=torch.float32)  if bin_stds  is None
                       else torch.as_tensor(bin_stds,  dtype=torch.float32))
        cond_means_t = (torch.zeros(4, dtype=torch.float32) if cond_means is None
                        else torch.as_tensor(cond_means, dtype=torch.float32))
        cond_stds_t  = (torch.ones(4, dtype=torch.float32)  if cond_stds  is None
                        else torch.as_tensor(cond_stds,  dtype=torch.float32))
        self.register_buffer("bin_means",  bin_means_t)
        self.register_buffer("bin_stds",   bin_stds_t)
        self.register_buffer("cond_means", cond_means_t)
        self.register_buffer("cond_stds",  cond_stds_t)

    # ── training / validation share the same forward corruption ───────────
    def _shared_step(self, batch):
        r_clean = batch["r_clean"]                                       # (B, 15)
        cond    = batch["cond"]                                          # (B, 4)
        B       = r_clean.shape[0]
        t       = torch.randint(0, self.T, (B,), device=r_clean.device)
        eps     = torch.randn_like(r_clean)

        alpha_t = rearrange(self.alpha[t], "b -> b 1")                   # (B, 1)
        sigma_t = rearrange(self.sigma[t], "b -> b 1")                   # (B, 1)
        r_t     = alpha_t * r_clean + sigma_t * eps

        eps_hat = self.model(r_t, t, cond)
        return F.mse_loss(eps_hat, eps)

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    # ── AdamW with no-decay bias/1-D group + optional LR scheduler ────────
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
                optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-4
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


# ─── Task 52 — sample_conditional ──────────────────────────────────────────

@torch.no_grad()
def sample_conditional(
    lightning_module: ConditionalDiffusionLightning,
    cond: torch.Tensor,
    data_dim: int = 15,
    device=None,
) -> torch.Tensor:
    """Conditional DDIM sampling, deterministic.

    Parameters
    ----------
    lightning_module : ConditionalDiffusionLightning
        The trained module. Buffers ``alpha``, ``sigma``, ``bin_means``,
        ``bin_stds`` are read for the reverse step and de-standardization.
        ``cond_means`` / ``cond_stds`` are *not* read here — the caller is
        responsible for passing already-normalized ``cond`` (which is what
        ``ConditionalResidualDataset`` returns).
    cond : torch.Tensor of shape (B, cond_dim)
        Already-normalized conditioning vectors. The batch size B determines
        how many residuals are generated.
    data_dim : int, default 15.
    device : optional torch device. If None, inferred from the module.

    Returns
    -------
    torch.Tensor of shape (B, data_dim), in *physical* residual units
    (de-standardized using the module's ``bin_means`` / ``bin_stds`` buffers).
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
    for t in range(T_loc - 1, -1, -1):
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)
        eps_hat = lightning_module.model(r_t, t_batch, cond)

        alpha_t = alpha[t]
        sigma_t = sigma[t]
        r_0_hat = (r_t - sigma_t * eps_hat) / alpha_t

        if t > 0:
            r_t = alpha[t - 1] * r_0_hat + sigma[t - 1] * eps_hat
        else:
            r_t = r_0_hat

    return r_t * lightning_module.bin_stds.to(device) + lightning_module.bin_means.to(device)


# ══════════════════════════════════════════════════════════════════════════
# Week-10 extension — dict-of-groups conditioning, FiLM, CFG, factory.
#
# Everything above this banner is the as-shipped Week 10 reference and is
# load-bearing for notebooks 10a/10b/10c — do not edit those classes. The
# additions below are used by 10d/10e (the ablation study). They are
# strictly additive: importing this module under the old names still
# resolves the legacy classes byte-for-byte.
# ══════════════════════════════════════════════════════════════════════════

from typing import Optional, Sequence, Dict, Tuple, List


# ─── Task 60-helper — dict-of-groups dataset ──────────────────────────────

class ExtendedConditionalResidualDataset(torch.utils.data.Dataset):
    """Conditional residual dataset where each conditioning *group* is its
    own dict entry. Adding a new group never breaks anything that already
    consumes the dataset — it just adds new keys to the returned dict.

    `__getitem__(idx)` returns a dictionary with these keys (subset
    determined by ``groups``):

        {
            "r_clean":          float32 tensor (15,),
            "cond_base":        float32 tensor (4,),     "base" in groups
            "cond_cyclehemi":   float32 tensor (2,),     "cyclehemi"
            "cond_opp":         float32 tensor (3,),     "opp"
            "cond_opp_valid":   float32 tensor (1,),     "opp"
            "cond_traj":        float32 tensor (K,),     "traj"
            "cond_traj_valid":  float32 tensor (1,),     "traj"
        }

    Each cond group is standardized using **train-split-only** statistics.
    Validity flags ("..._valid") are NOT standardized; they are passed
    through as float32 0/1 so a downstream model can use them as a mask.
    """

    GROUP_COLS: Dict[str, List[str]] = {
        "base":       ["area_smoothed", "mu_universal", "model_sigma", "amplitude"],
        "cyclehemi":  ["cycle_norm", "hemi_id"],
        "opp":        ["opp_area_smoothed", "opp_mu_universal", "opp_amplitude"],
        # "traj" is dynamic (K columns) — resolved at __init__ time.
    }
    GROUP_VALIDITY: Dict[str, str] = {
        "opp":  "opp_valid",
        "traj": "traj_valid",
    }

    HIST_COLS = [f"hist_emp_{j:02d}" for j in range(15)]
    PAR_COLS  = [f"hist_par_{j:02d}" for j in range(15)]

    def __init__(
        self,
        df,
        split: str,
        groups: Sequence[str] = ("base",),
        group_stats: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        self.groups = list(groups)
        unknown = set(self.groups) - (set(self.GROUP_COLS) | {"traj"})
        if unknown:
            raise ValueError(f"unknown conditioning group(s): {unknown}")

        # Resolve "traj" columns (area_lag1, area_lag2, …) from the df schema.
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

        # Residuals + per-bin standardization (same convention as Week 08).
        emp = df_split[self.HIST_COLS].to_numpy(dtype=np.float32)
        par = df_split[self.PAR_COLS].to_numpy(dtype=np.float32)
        residuals = torch.from_numpy(emp - par)
        self.bin_means = residuals.mean(dim=0)
        self.bin_stds  = residuals.std(dim=0).clamp(min=1e-6)
        self._residuals = (residuals - self.bin_means) / self.bin_stds

        # Keep the classical (parametric) density per window so the training
        # step can enforce non-negativity of the reconstructed empirical
        # density ``emp = par + residual``.
        self._par = torch.from_numpy(par)

        # Per-group standardization — train-split-only.
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

        # Pre-compute the normalized cond tensors and validity masks per group.
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
        return self._residuals.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {
            "r_clean": self._residuals[idx],
            "par":     self._par[idx],
        }
        for g in self.groups:
            item[f"cond_{g}"] = self._cond[g][idx]
            if g in self._valid:
                item[f"cond_{g}_valid"] = self._valid[g][idx]
        return item


# ─── Task 65-helper — Fourier-feature embedding of cond scalars ───────────

class FourierFeatureCondEmbedding(nn.Module):
    """Project a continuous cond vector into a higher-dimensional space
    via sin/cos at log-spaced frequencies, optionally concatenating the
    raw values back in. Useful when the residual structure varies
    nonlinearly with a continuous conditioning scalar.

    Input  : (B, cond_dim) float32
    Output : (B, cond_dim * (1 + 2 * n_freqs)) if include_raw else
             (B, cond_dim * 2 * n_freqs)
    """

    def __init__(
        self,
        cond_dim: int,
        n_freqs: int = 4,
        max_freq: float = 8.0,
        include_raw: bool = True,
    ):
        super().__init__()
        self.cond_dim     = int(cond_dim)
        self.n_freqs      = int(n_freqs)
        self.include_raw  = bool(include_raw)
        freqs = torch.logspace(0.0, np.log10(max_freq), n_freqs)
        self.register_buffer("freqs", freqs)                             # (n_freqs,)

    @property
    def out_dim(self) -> int:
        per = 2 * self.n_freqs + (1 if self.include_raw else 0)
        return self.cond_dim * per

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim) → (B, cond_dim, n_freqs)
        scaled = cond.unsqueeze(-1) * self.freqs                          # (B, D, F)
        feats  = torch.cat([scaled.sin(), scaled.cos()], dim=-1)          # (B, D, 2F)
        feats  = feats.reshape(cond.shape[0], -1)                         # (B, D*2F)
        if self.include_raw:
            feats = torch.cat([cond, feats], dim=-1)
        return feats


# ─── Task 66-helper — FiLM-conditioned ε-predictor MLP ────────────────────

class FiLMConditionalDiffusionMLP(nn.Module):
    """ε-predictor MLP that consumes (r_t, t, cond) like the Week 10
    baseline but routes ``cond`` through a small embedding MLP that
    produces *per-layer* (γ, β) modulation parameters. Each hidden layer
    applies ``h ← γ * h + β`` before the SiLU.

    The ``consumed_keys`` attribute tells the Lightning module which
    batch-dict entries to concatenate into the ``cond`` argument; the
    model itself only sees the concatenated vector.
    """

    def __init__(
        self,
        data_dim: int = 15,
        cond_dim: int = 4,
        hidden_dim: int = 128,
        t_embed_dim: int = 64,
        t_hidden_dim: int = 128,
        cond_hidden_dim: int = 64,
        n_layers: int = 3,
        consumed_keys: Sequence[str] = ("cond_base",),
        fourier: Optional[FourierFeatureCondEmbedding] = None,
    ):
        super().__init__()
        self.data_dim       = int(data_dim)
        self.cond_dim       = int(cond_dim)
        self.hidden_dim     = int(hidden_dim)
        self.n_layers       = int(n_layers)
        self.consumed_keys  = list(consumed_keys)

        self.t_embedding = TimestepEmbedding(t_embed_dim, t_hidden_dim)

        # Optional Fourier-feature embedding on the cond vector.
        self.fourier = fourier
        cond_in_dim  = fourier.out_dim if fourier is not None else cond_dim

        # Cond-embedding MLP: cond → (γ_1, β_1, γ_2, β_2, …, γ_L, β_L)
        # Output flat dim = 2 * hidden_dim * n_layers.
        film_out_dim = 2 * hidden_dim * n_layers
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_in_dim, cond_hidden_dim),
            nn.SiLU(),
            nn.Linear(cond_hidden_dim, film_out_dim),
        )

        # Input projection: r_t + t_emb → hidden_dim. cond is not concat
        # at the input; it enters via FiLM modulation on each layer.
        self.input_proj = nn.Linear(data_dim + t_hidden_dim, hidden_dim)

        # Hidden layers (just linear maps — modulation handled in forward).
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, r_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.fourier is not None:
            cond = self.fourier(cond)
        film_params = self.cond_embed(cond)                               # (B, 2*H*L)
        film_params = film_params.reshape(-1, self.n_layers, 2, self.hidden_dim)
        gammas = 1.0 + film_params[:, :, 0, :]                            # mean-1 init
        betas  =       film_params[:, :, 1, :]

        t_emb = self.t_embedding(t)                                       # (B, t_hidden_dim)
        h = self.input_proj(torch.cat([r_t, t_emb], dim=-1))              # (B, H)
        for layer_idx, lin in enumerate(self.layers):
            h = lin(h)
            h = gammas[:, layer_idx, :] * h + betas[:, layer_idx, :]
            h = F.silu(h)
        return self.output_proj(h)


# ─── Concat variant carrying consumed_keys (factory parity) ───────────────

class ExtendedConcatConditionalDiffusionMLP(ConditionalDiffusionMLP):
    """Raw-concat MLP equivalent to the Week 10 baseline, but carrying a
    ``consumed_keys`` attribute so the extended Lightning module knows
    which batch-dict entries to concat into the ``cond`` argument. Adds
    optional ``fourier`` lifting of the cond vector before the input
    concat; pass ``fourier=None`` for the exact Week 10 behavior.
    """

    def __init__(
        self,
        data_dim: int = 15,
        cond_dim: int = 4,
        hidden_dim: int = 128,
        t_embed_dim: int = 64,
        t_hidden_dim: int = 128,
        n_layers: int = 3,
        consumed_keys: Sequence[str] = ("cond_base",),
        fourier: Optional[FourierFeatureCondEmbedding] = None,
    ):
        # If Fourier lifting is requested, the parent class needs to know
        # the *lifted* cond dim so its input layer is wide enough.
        effective_cond_dim = fourier.out_dim if fourier is not None else cond_dim
        super().__init__(
            data_dim=data_dim, cond_dim=effective_cond_dim,
            hidden_dim=hidden_dim, t_embed_dim=t_embed_dim,
            t_hidden_dim=t_hidden_dim, n_layers=n_layers,
        )
        self.consumed_keys = list(consumed_keys)
        self.fourier = fourier
        # Track the *raw* cond dim separately so the Lightning module's
        # null embedding is sized correctly (it standardizes pre-Fourier).
        self.raw_cond_dim = int(cond_dim)

    def forward(self, r_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.fourier is not None:
            cond = self.fourier(cond)
        return super().forward(r_t, t, cond)


# ─── Task 66-helper — Extended Lightning module (dict batches + CFG) ──────

class ExtendedConditionalDiffusionLightning(pl.LightningModule):
    """Extended Week-10 LightningModule.

    Differences from `ConditionalDiffusionLightning`:

    1. Reads cond from a *dict* batch by concatenating
       ``batch[k] for k in model.consumed_keys`` (validity masks are
       routed through `consumed_keys` like any other group entry).
    2. Persists per-group standardization statistics as named buffers
       (``cond_<group>_means`` / ``cond_<group>_stds``) so the sampler
       can recover them from a loaded checkpoint without the dataset.
    3. Optional **classifier-free guidance** training via
       ``cond_dropout_p``: with that probability per sample, the
       concatenated cond is replaced by a learned null embedding
       (``self.null_cond``, shape (total_cond_dim,)). At sampling time,
       the same null vector is used for the unconditional branch.

    The cosine schedule, AdamW no-decay split, and LR schedulers are
    identical to the base class.
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
        lr: float = 1e-3,
        scheduler: str = "cosine",
        weight_decay: float = 1e-4,
        cond_dropout_p: float = 0.0,
        bin_means=None,
        bin_stds=None,
        bin_centers=None,
        bin_width: float = 3.0,
        lambda_mass: float = 0.0,
        lambda_neg: float = 0.0,
        lambda_band: float = 0.0,
        band_lo: float = 5.0,
        band_hi: float = 40.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "model", "alpha", "sigma",
            "bin_means", "bin_stds", "bin_centers",
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

        # Soft-constraint weights on the reconstructed empirical density
        # ``emp = par + residual`` (all default to 0 → behavior unchanged).
        self.bin_width   = float(bin_width)
        self.lambda_mass = float(lambda_mass)   # zero-integral / mass conservation
        self.lambda_neg  = float(lambda_neg)    # non-negativity of emp = par + r
        self.lambda_band = float(lambda_band)   # mass outside the Spoerer band
        self.band_lo     = float(band_lo)
        self.band_hi     = float(band_hi)

        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.register_buffer("sigma", torch.as_tensor(sigma, dtype=torch.float32))

        bin_means_t = (torch.zeros(15, dtype=torch.float32) if bin_means is None
                       else torch.as_tensor(bin_means, dtype=torch.float32))
        bin_stds_t  = (torch.ones(15, dtype=torch.float32)  if bin_stds  is None
                       else torch.as_tensor(bin_stds,  dtype=torch.float32))
        self.register_buffer("bin_means", bin_means_t)
        self.register_buffer("bin_stds",  bin_stds_t)

        K = bin_means_t.shape[0]
        bin_centers_t = ((torch.arange(K, dtype=torch.float32) + 0.5) * self.bin_width
                         if bin_centers is None
                         else torch.as_tensor(bin_centers, dtype=torch.float32))
        # Non-persistent: derived constants, kept out of the state_dict so
        # checkpoints trained before these constraints still load cleanly.
        self.register_buffer("bin_centers", bin_centers_t, persistent=False)
        in_band = ((bin_centers_t >= self.band_lo) & (bin_centers_t <= self.band_hi))
        self.register_buffer("band_mask", in_band.to(torch.float32), persistent=False)

        # Per-group standardization buffers, named by group.
        self._group_names: List[str] = []
        for name, (means, stds) in group_stats.items():
            self._group_names.append(name)
            self.register_buffer(f"cond_{name}_means",
                                 torch.as_tensor(means, dtype=torch.float32))
            self.register_buffer(f"cond_{name}_stds",
                                 torch.as_tensor(stds,  dtype=torch.float32))

        # Learned null embedding for CFG. Always registered (zero-init);
        # only updated/used when ``cond_dropout_p`` > 0 during training
        # or when the sampler is called with ``guidance_w`` > 0.
        self.null_cond = nn.Parameter(torch.zeros(self.total_cond_dim))

    # ── helpers ───────────────────────────────────────────────────────────
    def _concat_cond(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([batch[k] for k in self.consumed_keys], dim=-1)

    def _shared_step(self, batch: Dict[str, torch.Tensor],
                     stage: str = "train") -> torch.Tensor:
        r_clean = batch["r_clean"]
        cond    = self._concat_cond(batch)
        B       = r_clean.shape[0]

        # CFG: drop the cond with probability p, replacing with null.
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

        loss = mse_loss + self._constraint_terms(batch, r_t, sigma_t,
                                                  alpha_t, eps_hat, stage)
        return loss

    def _constraint_terms(self, batch, r_t, sigma_t, alpha_t, eps_hat,
                          stage: str) -> torch.Tensor:
        """Soft constraints + diagnostics on the reconstructed density.

        Recovers the predicted clean residual in *physical* units, forms the
        reconstructed empirical density ``emp = par + residual``, and returns
        the (possibly zero) penalty to add to the loss. The constraint
        diagnostics are always logged so they can be watched even when the
        penalty weights are 0.
        """
        # Predicted clean residual, de-standardized to physical density units.
        r0_std  = (r_t - sigma_t * eps_hat) / alpha_t.clamp(min=1e-6)
        r0_phys = r0_std * self.bin_stds + self.bin_means     # (B, K)

        # Mass added by the residual: sum_b r * bin_width. Proper residuals
        # integrate to zero (emp and par are both normalized densities).
        added_mass = (r0_phys * self.bin_width).sum(dim=-1)   # (B,)
        mass_penalty = (added_mass ** 2).mean()
        self.log(f"{stage}_residual_integral_abs",
                 added_mass.abs().mean(), on_step=False, on_epoch=True)

        penalty = self.lambda_mass * mass_penalty

        if "par" in batch:
            emp_hat = batch["par"] + r0_phys                  # (B, K)
            neg     = torch.clamp(-emp_hat, min=0.0)          # density < 0
            neg_penalty = (neg ** 2).mean()
            self.log(f"{stage}_neg_density_frac",
                     (emp_hat < 0).float().mean(), on_step=False, on_epoch=True)
            penalty = penalty + self.lambda_neg * neg_penalty

            # Mass placed outside the Spoerer band (±band_lo..band_hi deg).
            emp_pos = torch.clamp(emp_hat, min=0.0)
            oob = (emp_pos * (1.0 - self.band_mask) * self.bin_width).sum(dim=-1)
            self.log(f"{stage}_out_of_band_mass",
                     oob.mean(), on_step=False, on_epoch=True)
            penalty = penalty + self.lambda_band * (oob ** 2).mean()

        return penalty

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="train")
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="val")
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
                optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-4
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


# ─── Task 68-helper — CFG-capable sampler (dict batches) ──────────────────

@torch.no_grad()
def sample_conditional_extended(
    lightning_module: ExtendedConditionalDiffusionLightning,
    cond: torch.Tensor,
    guidance_w: float = 0.0,
    data_dim: int = 15,
    device=None,
    enforce_zero_integral: bool = False,
) -> torch.Tensor:
    """DDIM sampling with optional classifier-free guidance.

    ``cond`` is the already-concatenated, already-normalized conditioning
    tensor of shape (B, total_cond_dim). The caller is responsible for
    matching concat order to ``lightning_module.consumed_keys``.

    ``guidance_w == 0`` reproduces the legacy `sample_conditional`
    behavior. Positive values combine conditional + null-cond
    predictions as ``eps = (1 + w) * eps_cond − w * eps_null``.

    ``enforce_zero_integral`` projects each sampled residual onto the
    zero-integral hyperplane (per-sample demean) so the generated residual
    conserves mass exactly — ``emp = par + residual`` then integrates to the
    same total as ``par``. With uniform bins this L2-optimal projection is just
    subtracting the per-sample mean (cf. ``project_residuals_zero_integral`` in
    ``evaluation``).
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
        r_0_hat = (r_t - sigma_t * eps_hat) / alpha_t

        if t > 0:
            r_t = alpha[t - 1] * r_0_hat + sigma[t - 1] * eps_hat
        else:
            r_t = r_0_hat

    out = r_t * lightning_module.bin_stds.to(device) + lightning_module.bin_means.to(device)
    if enforce_zero_integral:
        out = out - out.mean(dim=-1, keepdim=True)
    return out


# ─── Task 65-helper — experiment-config-driven model factory ──────────────

def build_model(
    config: Dict,
    total_cond_dim: int,
    data_dim: int = 15,
) -> nn.Module:
    """Build the ε-predictor MLP described by an experiment ``config``.

    Recognized keys (with defaults):

        config["arch"]                : "concat" | "film"
        config["consumed_keys"]       : list[str], dict-batch entries to concat
        config["hidden_dim"]          : int, default 128
        config["n_layers"]            : int, default 3
        config["fourier"]             : bool, default False
        config["fourier_n_freqs"]     : int, default 4
        config["fourier_max_freq"]    : float, default 8.0
        config["fourier_include_raw"] : bool, default True

    Other keys (e.g. ``cond_dropout_p``) are ignored here; they belong
    to the Lightning module.
    """
    arch         = str(config.get("arch", "concat"))
    consumed     = list(config.get("consumed_keys", ["cond_base"]))
    hidden_dim   = int(config.get("hidden_dim", 128))
    n_layers     = int(config.get("n_layers", 3))
    use_fourier  = bool(config.get("fourier", False))

    fourier_mod = None
    if use_fourier:
        fourier_mod = FourierFeatureCondEmbedding(
            cond_dim=total_cond_dim,
            n_freqs=int(config.get("fourier_n_freqs", 4)),
            max_freq=float(config.get("fourier_max_freq", 8.0)),
            include_raw=bool(config.get("fourier_include_raw", True)),
        )

    if arch == "concat":
        return ExtendedConcatConditionalDiffusionMLP(
            data_dim=data_dim, cond_dim=total_cond_dim,
            hidden_dim=hidden_dim, n_layers=n_layers,
            consumed_keys=consumed, fourier=fourier_mod,
        )
    if arch == "film":
        return FiLMConditionalDiffusionMLP(
            data_dim=data_dim, cond_dim=total_cond_dim,
            hidden_dim=hidden_dim, n_layers=n_layers,
            consumed_keys=consumed, fourier=fourier_mod,
        )
    raise ValueError(f"unknown arch {arch!r}; expected 'concat' or 'film'")


# ══════════════════════════════════════════════════════════════════════════
# Notebook helpers — generic plumbing for 10d / 10e
# ══════════════════════════════════════════════════════════════════════════
# These promote engineering boilerplate out of the student-facing
# notebooks. Each helper preserves the behavior previously inlined in
# the notebook cells; the science (experiment design, oracle MLP, NLL
# math, plotting) stays in the notebook where students can author it.

import os
import sys


def find_week10_artifacts(extra_required: Sequence[str] = ()) -> Dict[str, object]:
    """Locate the standard Week-10 artifact paths and prep ``sys.path``.

    Always required:
      - unconditioned_infrastructure.py
      - conditioned_infrastructure.py
      - diffusion_windows.parquet
      - butterflAI_model.py
      - official_model.npz

    Names passed in ``extra_required`` (e.g. ``"diffusion_windows_v2.parquet"``
    or ``"data/composite_sunspot_groups_peak_area.csv"``) are also required.

    Side effects:
      - Inserts the repo root and the located ``week_08/09/10`` dirs on
        ``sys.path`` so subsequent imports resolve.
      - If ``conditioned_infrastructure`` was already imported from a
        different file (e.g. a Week-09 stub), it is evicted from
        ``sys.modules`` so the next import picks up the Week-10 file.

    Returns
    -------
    dict
        Keys: ``unconditioned_py``, ``conditioned_py``, ``parquet_v1``,
        ``parquet_v2`` (None if not found), ``classical_py``,
        ``classical_weights``, ``raw_csv`` (None if not found),
        ``ckpt_conditional`` (None if not found), ``week10_dir``.

    Raises
    ------
    FileNotFoundError
        If any required artifact (default + ``extra_required``) is missing.
    """
    _cwd = os.getcwd()
    _walk_bases = [_cwd] + [os.path.abspath(os.path.join(_cwd, *([".."] * i)))
                            for i in range(1, 6)]

    _week_dirs: List[str] = []
    for _base in _walk_bases:
        for _sub in [("weeks", "week_10"), ("weeks", "week_09"), ("weeks", "week_08")]:
            _cand = os.path.join(_base, *_sub)
            if os.path.isdir(_cand) and _cand not in _week_dirs:
                _week_dirs.append(_cand)
        if (any(w in _base for w in ("week_08", "week_09", "week_10"))
                and os.path.isdir(_base) and _base not in _week_dirs):
            _week_dirs.append(_base)

    def _find_simple(filename: str) -> Optional[str]:
        for d in _week_dirs:
            p = os.path.join(d, filename)
            if os.path.isfile(p):
                return p
        return None

    def _find_relative(rel: str) -> Optional[str]:
        for _base in _walk_bases:
            p = os.path.join(_base, rel)
            if os.path.isfile(p):
                return p
        return None

    def _find_any(name: str) -> Optional[str]:
        if "/" in name or "\\" in name:
            return _find_relative(name)
        return _find_simple(name)

    required = [
        "unconditioned_infrastructure.py",
        "conditioned_infrastructure.py",
        "diffusion_windows.parquet",
        "butterflAI_model.py",
        "official_model.npz",
    ] + list(extra_required)

    optional = [
        "diffusion_windows_v2.parquet",
        "data/composite_sunspot_groups_peak_area.csv",
        "ckpt_conditional.ckpt",
    ]

    located: Dict[str, str] = {}
    for name in list(required) + [n for n in optional if n not in required]:
        p = _find_any(name)
        if p is not None:
            located[name] = p

    missing = [n for n in required if n not in located]
    if missing:
        raise FileNotFoundError(
            f"Cannot locate {missing}. Searched: {_week_dirs}"
        )

    cond_py    = located["conditioned_infrastructure.py"]
    week10_dir = os.path.dirname(cond_py)
    repo_root  = os.path.abspath(os.path.join(week10_dir, "..", ".."))

    for _p in [repo_root, week10_dir,
               os.path.dirname(located["unconditioned_infrastructure.py"]),
               os.path.dirname(located["butterflAI_model.py"])]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    if "conditioned_infrastructure" in sys.modules:
        _existing = sys.modules["conditioned_infrastructure"]
        if getattr(_existing, "__file__", None) != cond_py:
            del sys.modules["conditioned_infrastructure"]

    return {
        "unconditioned_py":   located["unconditioned_infrastructure.py"],
        "conditioned_py":     cond_py,
        "parquet_v1":         located.get("diffusion_windows.parquet"),
        "parquet_v2":         located.get("diffusion_windows_v2.parquet"),
        "classical_py":       located["butterflAI_model.py"],
        "classical_weights":  located["official_model.npz"],
        "raw_csv":            located.get("data/composite_sunspot_groups_peak_area.csv"),
        "ckpt_conditional":   located.get("ckpt_conditional.ckpt"),
        "week10_dir":         week10_dir,
    }


class CondSampleQualityCallback(SampleQualityCallback):
    """Sample-quality callback that draws from the CFG-capable extended
    sampler. Each instance owns a fixed reference batch of cond vectors
    (already concatenated and train-set-normalized) so the per-epoch
    comparison is apples-to-apples across the run.

    When per-window validation targets are supplied (``val_par``, ``val_emp``,
    ``val_cond``), the callback also computes the full distributional suite
    (EMD, energy distance, CRPS, moment errors) on the validation split in
    ``on_validation_epoch_end`` and logs ``val_emd``/``val_crps_mu``/… as
    model-selection signals. For each evaluated window it draws ``n_ensemble``
    conditional samples and reconstructs model densities ``emp = par +
    residual``.
    """

    def __init__(self, train_samples, cond_reference,
                 val_par=None, val_emp=None, val_cond=None, val_tau=None,
                 n_eval_windows: int = 64, n_ensemble: int = 16, **kw):
        super().__init__(train_samples, **kw)
        self._cond_ref = cond_reference.detach().clone()

        self._val_par  = None if val_par  is None else np.asarray(val_par,  dtype=np.float32)
        self._val_emp  = None if val_emp  is None else np.asarray(val_emp,  dtype=np.float32)
        self._val_cond = None if val_cond is None else val_cond.detach().clone()
        self._val_tau  = None if val_tau  is None else np.asarray(val_tau,  dtype=np.float32)
        self._n_eval_windows = int(n_eval_windows)
        self._n_ensemble     = int(n_ensemble)

        # Fixed window subset so the metric is comparable across epochs.
        self._eval_idx = None
        if self._val_cond is not None:
            n = self._val_cond.shape[0]
            k = min(self._n_eval_windows, n)
            self._eval_idx = np.random.default_rng(0).choice(n, size=k, replace=False)

    def _sample(self, pl_module):
        device = next(pl_module.parameters()).device
        return sample_conditional_extended(
            pl_module, self._cond_ref, guidance_w=0.0, device=device,
        ).cpu().numpy()

    def _distributional_eval(self, trainer, pl_module):
        if self._val_cond is None or self._val_par is None or self._val_emp is None:
            return
        from distribution_metrics import distributional_report

        device = next(pl_module.parameters()).device
        idx = self._eval_idx
        M = self._n_ensemble

        # Tile each selected window M times → one batched sampler call.
        cond_sel = self._val_cond[idx]                       # (W, D)
        cond_rep = cond_sel.repeat_interleave(M, dim=0)      # (W*M, D)
        resid = sample_conditional_extended(
            pl_module, cond_rep, guidance_w=0.0, device=device,
        ).cpu().numpy().reshape(len(idx), M, -1)             # (W, M, K)

        par = self._val_par[idx][:, None, :]                 # (W, 1, K)
        model_dens = par + resid                             # emp = par + residual
        emp_dens = self._val_emp[idx]                        # (W, K)
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


def _consumed_cond_dim(train_ds, key: str) -> int:
    """Width contributed to the concatenated cond vector by one
    ``consumed_keys`` entry. Validity flags (``cond_<group>_valid``) are
    1-D 0/1 masks routed through ``consumed_keys`` like any other entry,
    but they are not standardized groups, so they have no ``group_dims``
    entry — they always contribute exactly one dimension.
    """
    if key.endswith("_valid"):
        return 1
    return train_ds.group_dims[key.replace("cond_", "")]


def train_experiment(
    name: str,
    cfg: Dict,
    windows_aug,
    ckpt_dir: str,
    wandb_project: str,
    wandb_entity: Optional[str],
    alpha_np,
    sigma_np,
    T: int,
    bin_centers,
    bin_width: float,
    seed: int = 42,
    device: Optional["torch.device"] = None,
) -> str:
    """Train one experiment to completion and save its checkpoint.

    Skips training (without raising) if ``ckpt_dir/ckpt_<name>.ckpt``
    already exists — the same idempotent discipline the 10d notebook
    used inline. Falls back from WandB to a local ``CSVLogger`` if WandB
    is unavailable.

    Parameters
    ----------
    name : str
        Experiment id; checkpoint file becomes ``ckpt_<name>.ckpt``.
    cfg : dict
        Experiment spec; see ``build_model`` for the recognized keys.
        Additionally consumes ``groups``, ``consumed_keys``,
        ``batch_size``, ``max_epochs``, ``lr``, ``cond_dropout_p``.
    windows_aug : pandas.DataFrame
        v2 parquet contents (train+val rows only).
    ckpt_dir, wandb_project, wandb_entity : configuration.
    alpha_np, sigma_np : diffusion schedule arrays.
    T : int
        Number of diffusion timesteps.
    bin_centers, bin_width
        Passed to the sample-quality callback for visual logging.
    seed : int
        Reproducibility seed.
    device : torch.device, optional
        Which device to train on. If ``None`` (default), Lightning's
        ``accelerator="auto", devices="auto"`` picks for you. Pass e.g.
        ``torch.device("cuda:1")`` to pin training to a specific GPU; the
        CUDA index selects which physical GPU Lightning uses.

    Returns
    -------
    str
        Absolute path to the saved checkpoint.
    """
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger, CSVLogger
    try:
        import wandb
    except ImportError:
        wandb = None
    from infrastructure.utils.reproducibility import set_all_seeds

    ckpt_path = os.path.join(ckpt_dir, f"ckpt_{name}.ckpt")
    if os.path.isfile(ckpt_path):
        print(f"[{name}] checkpoint exists at {ckpt_path}; skipping training.")
        return ckpt_path

    set_all_seeds(seed)

    train_ds = ExtendedConditionalResidualDataset(windows_aug, "train",
                                                  groups=cfg["groups"])
    val_ds   = ExtendedConditionalResidualDataset(windows_aug, "val",
                                                  groups=cfg["groups"],
                                                  group_stats=train_ds.group_stats)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0,
    )

    total_dim = sum(_consumed_cond_dim(train_ds, k)
                    for k in cfg["consumed_keys"])

    model = build_model(cfg, total_cond_dim=total_dim)
    lit = ExtendedConditionalDiffusionLightning(
        model=model, alpha=alpha_np, sigma=sigma_np, T=T,
        consumed_keys=cfg["consumed_keys"],
        group_stats={g: train_ds.group_stats[g] for g in cfg["groups"]
                     if f"cond_{g}" in cfg["consumed_keys"]},
        total_cond_dim=total_dim,
        bin_means=train_ds.bin_means, bin_stds=train_ds.bin_stds,
        bin_centers=bin_centers, bin_width=bin_width,
        lr=cfg["lr"], scheduler="cosine", weight_decay=1e-4,
        cond_dropout_p=cfg.get("cond_dropout_p", 0.0),
        lambda_mass=cfg.get("lambda_mass", 0.0),
        lambda_neg=cfg.get("lambda_neg", 0.0),
        lambda_band=cfg.get("lambda_band", 0.0),
    )

    bin_means_np = train_ds.bin_means.numpy()
    bin_stds_np  = train_ds.bin_stds.numpy()
    all_tr_std   = torch.stack([train_ds[i]["r_clean"]
                                for i in range(len(train_ds))]).numpy()
    all_tr_phys  = all_tr_std * bin_stds_np + bin_means_np
    all_va_std   = torch.stack([val_ds[i]["r_clean"]
                                for i in range(len(val_ds))]).numpy()
    all_va_phys  = all_va_std * bin_stds_np + bin_means_np
    cond_ref = torch.cat(
        [torch.stack([train_ds[i][k]
                      for i in range(min(500, len(train_ds)))])
         for k in cfg["consumed_keys"]],
        dim=-1,
    )

    # Per-window validation targets for the distributional selection metrics:
    # classical density (par), reconstructed empirical density (par+residual),
    # and the concatenated normalized cond, all aligned row-for-row.
    val_par  = torch.stack([val_ds[i]["par"] for i in range(len(val_ds))]).numpy()
    val_emp  = val_par + all_va_phys
    val_cond = torch.cat(
        [torch.stack([val_ds[i][k] for i in range(len(val_ds))])
         for k in cfg["consumed_keys"]],
        dim=-1,
    )

    cb = CondSampleQualityCallback(
        train_samples=all_tr_phys, val_samples=all_va_phys,
        cond_reference=cond_ref,
        val_par=val_par, val_emp=val_emp, val_cond=val_cond,
        n_eval_windows=cfg.get("n_eval_windows", 64),
        n_ensemble=cfg.get("n_ensemble", 16),
        every_n_epochs=cfg.get("eval_every_n_epochs", 200),
        n_compare=cond_ref.shape[0],
        bin_centers=bin_centers, bin_width=bin_width,
    )

    # Model selection on the distributional metric (val EMD), NOT the denoising
    # val_loss — the latter rises while sample quality improves. Checkpoint
    # cadence is aligned to the eval cadence so val_emd is present when checked.
    eval_every = cfg.get("eval_every_n_epochs", 200)
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(ckpt_dir, "select", name),
        filename="best-{epoch}-{val_emd:.3f}",
        monitor="val_emd", mode="min", save_top_k=1,
        every_n_epochs=eval_every, save_last=True,
    )

    try:
        logger = WandbLogger(
            project=wandb_project, entity=wandb_entity, name=name,
            save_dir=os.path.join(ckpt_dir, "wandb_logs"),
        )
    except Exception as _e:
        print(f"[{name}] WandB unavailable ({_e}); falling back to CSVLogger.")
        logger = CSVLogger(os.path.join(ckpt_dir, "csv_logs"), name=name)

    # Resolve the requested device into Lightning's (accelerator, devices)
    # pair. A specific CUDA device pins training to that GPU; otherwise let
    # Lightning auto-detect (the prior, unconditional behavior).
    if device is not None and torch.device(device).type == "cuda":
        _dev = torch.device(device)
        accelerator, devices = "gpu", [_dev.index if _dev.index is not None else 0]
    elif device is not None and torch.device(device).type == "cpu":
        accelerator, devices = "cpu", "auto"
    else:
        accelerator, devices = "auto", "auto"

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"], logger=logger,
        accelerator=accelerator, devices=devices,
        log_every_n_steps=10, enable_progress_bar=False,
        callbacks=[cb, ckpt_cb],
        check_val_every_n_epoch=eval_every,
    )
    trainer.fit(lit, train_loader, val_loader)
    # ``save_checkpoint`` writes the *final-iterate* weights; the val_emd-best
    # checkpoint lives under ckpt_dir/select/<name>/ for selection-based use.
    trainer.save_checkpoint(ckpt_path)
    if wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass
    print(f"[{name}] saved {ckpt_path}")
    return ckpt_path


def load_trained_experiment(
    name: str,
    cfg: Dict,
    windows_aug,
    ckpt_dir: str,
    alpha_np,
    sigma_np,
) -> Tuple[ExtendedConditionalDiffusionLightning, "ExtendedConditionalResidualDataset",
           "ExtendedConditionalResidualDataset", int]:
    """Rebuild the datasets and load the matching checkpoint into an
    ``ExtendedConditionalDiffusionLightning`` for evaluation or
    visualization. Returns ``(lit, train_ds, val_ds, total_cond_dim)``.
    """
    train_ds = ExtendedConditionalResidualDataset(windows_aug, "train",
                                                  groups=cfg["groups"])
    val_ds   = ExtendedConditionalResidualDataset(windows_aug, "val",
                                                  groups=cfg["groups"],
                                                  group_stats=train_ds.group_stats)
    total_dim = sum(_consumed_cond_dim(train_ds, k)
                    for k in cfg["consumed_keys"])

    model = build_model(cfg, total_cond_dim=total_dim)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_{name}.ckpt")
    lit = ExtendedConditionalDiffusionLightning.load_from_checkpoint(
        ckpt_path,
        model=model, alpha=alpha_np, sigma=sigma_np,
        group_stats={g: train_ds.group_stats[g] for g in cfg["groups"]
                     if f"cond_{g}" in cfg["consumed_keys"]},
        total_cond_dim=total_dim, consumed_keys=cfg["consumed_keys"],
    )
    return lit, train_ds, val_ds, total_dim


def block_cond_concat(
    hcs,
    lit: ExtendedConditionalDiffusionLightning,
    cfg: Dict,
    train_ds: "ExtendedConditionalResidualDataset",
) -> Tuple[List[Tuple[int, str, float]], torch.Tensor]:
    """Build the concatenated, train-set-normalized cond tensor for every
    block in ``hcs``, in ``cfg['consumed_keys']`` order.

    ``hcs`` is a list of hemicycle dicts of the shape produced by 10c's
    block builder; each block must carry a ``groups_raw`` dict keyed by
    group name. The per-group means/stds are read from the loaded
    checkpoint's buffers (``cond_<g>_means`` / ``cond_<g>_stds``) so the
    val data is normalized exactly the way training was.

    Returns ``(keys, cond_tensor)`` where ``keys`` is a list of
    ``(cycle, hemisphere, center_decimal)`` block ids and
    ``cond_tensor`` has shape ``(n_blocks, total_cond_dim)``.
    """
    keys: List[Tuple[int, str, float]] = []
    rows: List[torch.Tensor] = []
    for hc in hcs:
        for blk in hc["blocks"]:
            keys.append((hc["cycle"], hc["hemisphere"], blk["center_decimal"]))
            parts: List[torch.Tensor] = []
            for k in cfg["consumed_keys"]:
                g   = k.replace("cond_", "")
                raw = torch.tensor(blk["groups_raw"][g], dtype=torch.float32)
                if k.endswith("_valid"):
                    parts.append(raw)  # 0/1 mask, never standardized
                else:
                    means = getattr(lit, f"cond_{g}_means").cpu()
                    stds  = getattr(lit, f"cond_{g}_stds").cpu()
                    parts.append((raw - means) / stds)
            rows.append(torch.cat(parts, dim=-1))
    return keys, torch.stack(rows, dim=0)


def k_run_combined(
    hard_nll_combined,
    classical,
    hcs,
    keys,
    samples_NK15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Push K residual samples per block through ``hard_nll_combined`` and
    return per-K NLL and floor-fraction arrays.

    Parameters
    ----------
    hard_nll_combined : callable
        The hard-gated NLL primitive (defined in 10c / 10e), with the
        signature ``hard_nll_combined(model, hcs, residuals_by_block) -> (nll, detail)``.
    classical : ButterflAIModel
        Fixed model passed through to ``hard_nll_combined`` (the
        baseline whose hard gate is being preserved).
    hcs : list
        Hemicycle list whose ``blocks`` align with ``keys``.
    keys : list of (cycle, hemisphere, center_decimal)
        Block ids in the same row order as ``samples_NK15``.
    samples_NK15 : numpy.ndarray
        Array of shape ``(N_blocks, K, 15)`` of residual samples in
        physical units.

    Returns
    -------
    (nlls, floors) : tuple of numpy arrays, each shape ``(K,)``.
    """
    nlls, floors = [], []
    for k in range(samples_NK15.shape[1]):
        rbb = {key: samples_NK15[i, k] for i, key in enumerate(keys)}
        nll, det = hard_nll_combined(classical, hcs, rbb)
        nlls.append(nll)
        floors.append(det["floor_fraction"])
    return np.asarray(nlls), np.asarray(floors)


def discover_experiment_checkpoints(
    ckpt_dir: str,
    prefix: str = "ckpt_",
    suffix: str = ".ckpt",
) -> Dict[str, str]:
    """Discover trained experiment checkpoints in a directory.

    Scans ``ckpt_dir`` for files matching ``{prefix}<name>{suffix}`` and
    returns a mapping ``{<name>: <absolute path>}`` sorted by ``<name>``.
    Used by the eval phase of the merged notebook to pick up whichever
    experiments train mode has produced so far. Callers typically assert
    that each discovered ``<name>`` appears as a key in their
    ``EXPERIMENTS`` dict.
    """
    import glob as _glob
    paths = sorted(_glob.glob(os.path.join(ckpt_dir, f"{prefix}*{suffix}")))
    out: Dict[str, str] = {}
    for p in paths:
        base = os.path.basename(p)
        name = base[len(prefix):-len(suffix)] if suffix else base[len(prefix):]
        out[name] = p
    return out
