"""
conditioned_infrastructure.py — ButterflAI Week 09 *conditional* diffusion machinery.

This script contains the new class and function *definitions* introduced in
Week 09, in template form. Students fill in the bodies of `__init__`,
`__len__`, `__getitem__`, `_shared_step`, `forward`, `configure_optimizers`,
and `sample_conditional`; the markdown tasks in `09c_conditioned_train.ipynb`
and `09d_conditioned_evaluate.ipynb` specify the contract for each.

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

    Dict-returning Datasets are more extensible than tuple-returning ones:
    when Week 10 or the experimentation week adds a third conditioning
    field, only the producer (here) and the consumers that actually use
    the new field need to change — every other site keeps reading the
    keys it already understands. PyTorch's default DataLoader collation
    handles dict-of-tensors natively, so a batch out of the loader is
    a dict whose values each have a leading batch dimension:

        batch = {"r_clean": (B, 15) float32, "cond": (B, 2) float32}

    Per-bin residual standardization is identical to the Week 08 dataset:
    each split computes its own bin_means / bin_stds at construction time
    and exposes them so the sampler can de-standardize at inference time.

    Conditioning standardization is different and important: the conditioning
    vector must be normalized using *train-split-only* statistics. The
    network learns to expect inputs in the train-set's normalized scale; if
    the validation dataset were standardized using its own statistics, the
    model would see distributionally different conditioning at evaluation
    time than it saw at training time, which silently corrupts the
    cond-sensitivity check.

    Two construction patterns:
      - When `cond_means` / `cond_stds` are not supplied, the Dataset
        computes them from the train split of the supplied DataFrame
        (regardless of which split this Dataset itself represents). This
        means you can construct the val dataset directly without
        needing to construct the train dataset first.
      - When `cond_means` / `cond_stds` *are* supplied, they are used as-is.
        Useful for downstream code that already has the statistics in hand
        (e.g. loaded from a checkpoint).
    """

    HIST_COLS = [f"hist_emp_{j:02d}" for j in range(15)]
    PAR_COLS  = [f"hist_par_{j:02d}" for j in range(15)]
    COND_COLS = ["area_smoothed", "mu_universal"]

    def __init__(self, df, split: str, cond_means=None, cond_stds=None):
        # 1. Filter to this split
        df_split = df.loc[df["split"] == split].reset_index(drop=True)

        # 2. Compute residuals: emp - par, shape (N, 15)
        emp = df_split[self.HIST_COLS].to_numpy(dtype=np.float32)
        par = df_split[self.PAR_COLS].to_numpy(dtype=np.float32)
        residuals = torch.from_numpy(emp - par)                        # (N, 15)

        # 3. Per-bin standardization
        self.bin_means = residuals.mean(dim=0)                         # (15,)
        self.bin_stds  = residuals.std(dim=0).clamp(min=1e-6)          # (15,)
        self._residuals = (residuals - self.bin_means) / self.bin_stds  # (N, 15)

        # 4. Conditioning standardization — always use train-split statistics
        if cond_means is None or cond_stds is None:
            df_train = df.loc[df["split"] == "train"]
            cond_raw_train = df_train[self.COND_COLS].to_numpy(dtype=np.float32)
            self.cond_means = torch.from_numpy(
                cond_raw_train.mean(axis=0).astype(np.float32))        # (2,)
            self.cond_stds  = torch.from_numpy(
                cond_raw_train.std(axis=0).astype(np.float32)
            ).clamp(min=1e-6)                                          # (2,)
        else:
            self.cond_means = torch.as_tensor(cond_means, dtype=torch.float32)
            self.cond_stds  = torch.as_tensor(cond_stds,  dtype=torch.float32)

        # 5. Extract and standardize conditioning for this split
        cond_raw = df_split[self.COND_COLS].to_numpy(dtype=np.float32)
        cond_tensor = torch.from_numpy(cond_raw)                       # (N, 2)
        self._cond = (cond_tensor - self.cond_means) / self.cond_stds  # (N, 2)

    def __len__(self):
        return self._residuals.shape[0]

    def __getitem__(self, idx):
        return {
            "r_clean": self._residuals[idx].float(),
            "cond":    self._cond[idx].float(),
        }


# ─── Task 48 — ConditionalDiffusionMLP ─────────────────────────────────────

class ConditionalDiffusionMLP(nn.Module):
    """ε-predictor MLP: (r_t, t, cond) → ε̂.

    Concatenates r_t, the timestep embedding (from the TimestepEmbedding
    module reused from Week 08), and the conditioning vector at the input
    layer. Pure input-concatenation — no separate ConditionEmbedding
    wrapper this week; we go minimal-machinery first. The experimentation
    week will revisit "does adding a ConditionEmbedding MLP help?"

    The forward signature `(r_t, t, cond) → eps_hat` is the contract the
    ConditionalDiffusionLightning training step relies on.
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

        # Input is r_t (data_dim) + timestep embedding (t_hidden_dim) + cond (cond_dim)
        in_dim = data_dim + t_hidden_dim + cond_dim

        layers = []
        prev = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.SiLU())
            prev = hidden_dim
        layers.append(nn.Linear(prev, data_dim))   # no activation on output
        self.net = nn.Sequential(*layers)

    def forward(self, r_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # r_t  shape: (B, data_dim)
        # t    shape: (B,)
        # cond shape: (B, cond_dim)
        t_emb = self.t_embedding(t)                        # (B, t_hidden_dim)
        x = torch.cat([r_t, t_emb, cond], dim=-1)         # (B, data_dim + t_hidden_dim + cond_dim)
        return self.net(x)                                 # (B, data_dim)


# ─── Task 50 — ConditionalDiffusionLightning ───────────────────────────────

class ConditionalDiffusionLightning(pl.LightningModule):
    """Conditional version of DiffusionLightning.

    The only conceptual difference from the Week 08 LightningModule is that
    each training/validation batch now arrives as a **dict**:

        batch = {"r_clean": (B, 15) tensor, "cond": (B, 2) tensor}

    The `_shared_step` reads `batch["r_clean"]` and `batch["cond"]` by key
    (no positional unpacking — that would defeat the point of the dict
    convention). The model call gets an extra `cond` argument. Everything
    else — schedule buffers, AdamW with no-decay split, cosine/plateau LR
    schedulers, `save_hyperparameters` for clean checkpoint loading — is
    unchanged.

    We also persist the train-set conditioning statistics (`cond_means`,
    `cond_stds`) as buffers on the module, alongside the existing
    `bin_means` / `bin_stds`. This means `sample_conditional` can read all
    four statistics from a loaded checkpoint without needing the original
    training dataset in scope — the same buffer-as-interface pattern that
    let Week 08's `sample` be Dataset-free.
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

        # Register all four statistics buffers, falling back to zero/one
        # placeholders when not supplied so load_from_checkpoint can overwrite.
        _bin_means  = bin_means  if bin_means  is not None else torch.zeros(15)
        _bin_stds   = bin_stds   if bin_stds   is not None else torch.ones(15)
        _cond_means = cond_means if cond_means is not None else torch.zeros(2)
        _cond_stds  = cond_stds  if cond_stds  is not None else torch.ones(2)

        self.register_buffer("bin_means",  torch.as_tensor(_bin_means,  dtype=torch.float32))
        self.register_buffer("bin_stds",   torch.as_tensor(_bin_stds,   dtype=torch.float32))
        self.register_buffer("cond_means", torch.as_tensor(_cond_means, dtype=torch.float32))
        self.register_buffer("cond_stds",  torch.as_tensor(_cond_stds,  dtype=torch.float32))

    def _shared_step(self, batch):
        r_clean = batch["r_clean"]   # (B, 15)
        cond    = batch["cond"]      # (B, 2)
        B       = r_clean.shape[0]

        t   = torch.randint(0, self.T, (B,), device=r_clean.device)
        eps = torch.randn_like(r_clean)

        alpha_t = rearrange(self.alpha[t], "b -> b 1")   # (B, 1)
        sigma_t = rearrange(self.sigma[t], "b -> b 1")   # (B, 1)
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
        The trained module. Buffers `alpha`, `sigma`, `bin_means`,
        `bin_stds` are read for the reverse step and the de-standardization.
        Note that `cond_means` / `cond_stds` are *not* read here — the
        caller is responsible for passing already-normalized `cond` (which
        is what `ConditionalResidualDataset` returns).
    cond : torch.Tensor of shape (B, cond_dim)
        Already-normalized conditioning vectors. The batch size B determines
        how many residuals are generated.
    data_dim : int, default 15.
    device : optional torch device. If None, inferred from the module.

    Returns
    -------
    torch.Tensor of shape (B, data_dim), in *physical* residual units
    (de-standardized using the module's bin_means / bin_stds buffers).
    """
    if device is None:
        device = next(lightning_module.parameters()).device
    lightning_module.eval()
    lightning_module.to(device)
    cond = cond.to(device)

    B      = cond.shape[0]
    T_loc  = lightning_module.T
    alpha  = lightning_module.alpha
    sigma  = lightning_module.sigma

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
