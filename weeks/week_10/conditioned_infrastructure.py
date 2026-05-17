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
    conditioning covariates (area_smoothed, mu_universal) for one split.

    `__getitem__(idx)` returns a **dictionary**:

        {
            "r_clean": torch.float32 tensor, shape (15,),
            "cond":    torch.float32 tensor, shape (2,),
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
    COND_COLS = ["area_smoothed", "mu_universal"]

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
        )                                                                # (N, 2)
        self._cond = (cond_raw - self.cond_means) / self.cond_stds       # (N, 2)

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
        cond_dim: int = 2,
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
    ``{"r_clean": (B, 15), "cond": (B, 2)}``. The model call gets an extra
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
        cond_means_t = (torch.zeros(2, dtype=torch.float32) if cond_means is None
                        else torch.as_tensor(cond_means, dtype=torch.float32))
        cond_stds_t  = (torch.ones(2, dtype=torch.float32)  if cond_stds  is None
                        else torch.as_tensor(cond_stds,  dtype=torch.float32))
        self.register_buffer("bin_means",  bin_means_t)
        self.register_buffer("bin_stds",   bin_stds_t)
        self.register_buffer("cond_means", cond_means_t)
        self.register_buffer("cond_stds",  cond_stds_t)

    # ── training / validation share the same forward corruption ───────────
    def _shared_step(self, batch):
        r_clean = batch["r_clean"]                                       # (B, 15)
        cond    = batch["cond"]                                          # (B, 2)
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
