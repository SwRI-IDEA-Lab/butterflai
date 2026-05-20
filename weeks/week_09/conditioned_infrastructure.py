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
        # Filter `df` to rows with `split` column equal to `split`, reset_index(drop=True)
        df_split = df[df["split"] == split].reset_index(drop=True)
        
        # Extract the 15 emp and 15 par columns as float32 numpy arrays; 
        # compute residuals = emp - par as a torch.float32 tensor of shape (N, 15).
        emp = df_split[self.HIST_COLS].values.astype(np.float32)
        par = df_split[self.PAR_COLS].values.astype(np.float32)
        residuals = torch.from_numpy(emp - par)  # shape (N, 15)
        
        # Compute per-bin self.bin_means and self.bin_stds (clamp the
        # stds at 1e-6 so division is safe); store self._residuals = (residuals - bin_means) / bin_stds.
        self.bin_means = residuals.mean(dim=0)  # shape (15,)
        self.bin_stds = residuals.std(dim=0).clamp(min=1e-6)  # shape (15,)
        self._residuals = (residuals - self.bin_means) / self.bin_stds  # shape (N, 15)
        
        # Determine the conditioning standardization statistics: 
        # If cond_means is None or cond_stds is None: compute them from the TRAIN rows of the supplied 
        # `df` (so non-train datasets still get train-split-only normalization).
        # Otherwise: use the supplied values, converted to torch.float32 tensors. Store as self.cond_means, self.cond_stds.
        if cond_means is None or cond_stds is None:
            df_train = df[df["split"] == "train"]
            cond_train = df_train[self.COND_COLS].to_numpy().astype(np.float32)  # shape (N_train, 2)
            self.cond_means = torch.tensor(cond_train.mean(axis=0), dtype=torch.float32)  # shape (2,)
            self.cond_stds = torch.tensor(cond_train.std(axis=0, ddof=1), dtype=torch.float32).clamp(min=1e-6)        
        else:
            self.cond_means = torch.as_tensor(cond_means, dtype=torch.float32)  # shape (2,)
            self.cond_stds = torch.as_tensor(cond_stds, dtype=torch.float32)

        # Extract the COND_COLS from df_split, standardize using cond_means / cond_stds, store as self._cond
        # (shape (N, 2), float32).
        cond_vals = df_split[self.COND_COLS].to_numpy().astype(np.float32)  # shape (N, 2)
        cond_tensor = torch.from_numpy(cond_vals)  # shape (N, 2)
        self._cond = (cond_tensor - self.cond_means) / self.cond_stds  # shape (N, 2)

    def __len__(self):
        return self._residuals.shape[0]

    def __getitem__(self, idx):
        # Returns a dict:
        #   {"r_clean": torch.float32 tensor of shape (15,),
        #    "cond":    torch.float32 tensor of shape (2,)}
        
        return {
            "r_clean": self._residuals[idx],  # shape (15,)
            "cond": self._cond[idx],          # shape (2,)
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
        # Tasks:
        #   1. self.data_dim = data_dim, self.cond_dim = cond_dim.
        #   2. self.t_embedding = TimestepEmbedding(t_embed_dim, t_hidden_dim).
        #   3. Compute the main MLP's input dimension:
        #         in_dim = data_dim + t_hidden_dim + cond_dim
        #   4. Build the main MLP as `n_layers` hidden layers of
        #      `hidden_dim` units with SiLU activations between them, then
        #      a final Linear(hidden_dim, data_dim) with no activation on
        #      the output (ε can take any real value). Store as self.net.
        raise NotImplementedError("Implement ConditionalDiffusionMLP.__init__")

    def forward(self, r_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # r_t  shape: (B, data_dim)
        # t    shape: (B,)
        # cond shape: (B, cond_dim)
        # Returns eps_hat shape: (B, data_dim)
        # Tasks:
        #   1. t_emb = self.t_embedding(t)             # (B, t_hidden_dim)
        #   2. x = torch.cat([r_t, t_emb, cond], dim=-1)
        #   3. return self.net(x)
        raise NotImplementedError("Implement ConditionalDiffusionMLP.forward")


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
        # Tasks:
        #   1. self.save_hyperparameters(ignore=[
        #          "model", "alpha", "sigma",
        #          "bin_means", "bin_stds",
        #          "cond_means", "cond_stds",
        #      ])
        #   2. Store self.model, self.T, self.lr, self.scheduler_kind,
        #      self.weight_decay (cast lr/weight_decay to float, T to int,
        #      scheduler to str — same pattern as Week 08).
        #   3. Register `alpha`, `sigma`, `bin_means`, `bin_stds`,
        #      `cond_means`, `cond_stds` as float32 buffers. The four
        #      statistics buffers may be None at construction time (e.g.
        #      placeholders for load_from_checkpoint) — handle that by
        #      either registering zero/one placeholders or by setting the
        #      attribute to None. The cleanest pattern: register all four
        #      as buffers (use zeros / ones if None was passed) so they
        #      always exist as state_dict entries that get overwritten at
        #      load time.
        raise NotImplementedError("Implement ConditionalDiffusionLightning.__init__")

    def _shared_step(self, batch):
        # `batch` is a dict from the DataLoader (after default collation):
        #   batch["r_clean"] : (B, 15) float32, the standardized residuals.
        #   batch["cond"]    : (B, 2)  float32, the normalized conditioning.
        # Tasks:
        #   1. Read by key: r_clean = batch["r_clean"], cond = batch["cond"].
        #   2. Sample t = torch.randint(0, self.T, (B,), device=r_clean.device).
        #   3. Draw eps = torch.randn_like(r_clean).
        #   4. Look up alpha_t and sigma_t from self.alpha[t], self.sigma[t],
        #      reshape to (B, 1) via einops.rearrange.
        #   5. Compute r_t = alpha_t * r_clean + sigma_t * eps.
        #   6. Compute eps_hat = self.model(r_t, t, cond).
        #   7. Return F.mse_loss(eps_hat, eps).
        raise NotImplementedError("Implement _shared_step")

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # This method is *identical* to the Week 08 DiffusionLightning's
        # configure_optimizers: AdamW with biases and 1-D params in a
        # no-weight-decay group, plus a "cosine" or "plateau" LR scheduler
        # chosen via self.scheduler_kind. Copy your Week 08 implementation
        # here — same signature, same logic. Nothing about conditioning
        # changes how the optimizer is built.
        raise NotImplementedError("Copy from Week 08 DiffusionLightning.configure_optimizers")


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
    # Tasks:
    #   1. If device is None, infer from next(lightning_module.parameters()).device.
    #   2. lightning_module.eval(), .to(device); cond = cond.to(device).
    #   3. B = cond.shape[0].
    #   4. Initialize r_t = torch.randn(B, data_dim, device=device).
    #   5. Loop t from T-1 down to 0:
    #        - t_batch = torch.full((B,), t, dtype=torch.long, device=device).
    #        - eps_hat = lightning_module.model(r_t, t_batch, cond).
    #        - alpha_t, sigma_t = lightning_module.alpha[t], lightning_module.sigma[t].
    #        - r_0_hat = (r_t - sigma_t * eps_hat) / alpha_t.
    #        - if t > 0: r_t = alpha[t-1] * r_0_hat + sigma[t-1] * eps_hat.
    #          else:     r_t = r_0_hat.
    #   6. De-standardize: r_t * bin_stds + bin_means, return.
    raise NotImplementedError("Implement sample_conditional")
