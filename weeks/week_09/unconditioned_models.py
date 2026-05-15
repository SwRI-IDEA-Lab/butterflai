"""
unconditioned_models.py — ButterflAI Week 09 unconditional diffusion machinery.

Everything with a *definition* lives here: schedules, dataset, network modules,
LightningModule, sampler, callbacks. The two Week 09 notebooks
(09a_unconditioned_train.ipynb and 09b_unconditioned_evaluate.ipynb) import
from this script. Models trained in the training
notebook are loaded by the evaluation notebook through standard PyTorch
Lightning checkpointing — no global-scope dependencies on `train_dataset` or
any other notebook-level object.

Contents
--------
make_cosine_schedule          : Nichol & Dhariwal (2021) cosine α/σ arrays.
ResidualDataset               : per-bin-standardized clean residuals.
sinusoidal_embedding          : integer t → dense vector.
TimestepEmbedding             : sinusoidal embedding + small learnable MLP.
DiffusionMLP                  : (r_t, t) → ε̂, with t-blind ablation flag.
DiffusionLightning            : training + validation + buffered schedule.
sample                        : deterministic DDIM sampler, de-standardized.
SampleQualityCallback         : periodic + end-of-training sample-quality logging.
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


# ─── Schedule ──────────────────────────────────────────────────────────────

def make_cosine_schedule(T: int = 200, s: float = 0.008):
    """Cosine noise schedule, Nichol & Dhariwal (2021).

    Returns three numpy arrays of length T + 1:
        alpha[t]     = sqrt(alpha_bar[t])
        sigma[t]     = sqrt(1 - alpha_bar[t])
        alpha_bar[t] = ∏_{k≤t} (1 - β_k), with β_k clipped to [1e-8, 0.999]

    The training step looks up alpha[t] and sigma[t] only; alpha_bar is
    returned for any caller that wants SNR(t) = alpha_bar[t] / (1 - alpha_bar[t]).
    """
    t_arr = np.arange(T + 1, dtype=float)
    f0 = np.cos(np.pi / 2 * s / (1 + s)) ** 2
    alpha_bar_raw = np.cos(np.pi / 2 * (t_arr / T + s) / (1 + s)) ** 2 / f0
    alpha_bar_raw = np.clip(alpha_bar_raw, 0.0, 1.0)

    beta = np.zeros(T + 1)
    for t in range(1, T + 1):
        beta[t] = np.clip(1.0 - alpha_bar_raw[t] / alpha_bar_raw[t - 1], 1e-8, 0.999)

    alpha_bar = np.ones(T + 1)
    for t in range(1, T + 1):
        alpha_bar[t] = alpha_bar[t - 1] * (1.0 - beta[t])

    alpha = np.sqrt(alpha_bar)
    sigma = np.sqrt(np.clip(1.0 - alpha_bar, 0.0, 1.0))
    return alpha, sigma, alpha_bar


# ─── Dataset ───────────────────────────────────────────────────────────────

class ResidualDataset(torch.utils.data.Dataset):
    """Indexed collection of clean residuals (hist_emp − hist_par) for one split.

    Residuals are precomputed once at construction time and standardized
    per-bin to unit variance, so the network trains on inputs whose scale
    matches the N(0, I) noise injected by the forward process. Sampling
    de-standardizes through the bin_means / bin_stds attributes (also
    persisted as buffers on the LightningModule).
    """

    HIST_COLS = [f"hist_emp_{j:02d}" for j in range(15)]
    PAR_COLS  = [f"hist_par_{j:02d}" for j in range(15)]

    def __init__(self, df, split: str):
        df_split = df.loc[df["split"] == split].reset_index(drop=True)
        emp = df_split[self.HIST_COLS].to_numpy(dtype=np.float32)
        par = df_split[self.PAR_COLS].to_numpy(dtype=np.float32)
        residuals = torch.from_numpy(emp - par)                       # (N, 15)

        self.bin_means = residuals.mean(dim=0)                        # (15,)
        self.bin_stds  = residuals.std(dim=0).clamp(min=1e-6)         # (15,)
        self._residuals = (residuals - self.bin_means) / self.bin_stds

    def __len__(self):
        return self._residuals.shape[0]

    def __getitem__(self, idx):
        return self._residuals[idx]


# ─── Network modules ───────────────────────────────────────────────────────

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map integer timesteps to a dense (B, dim) vector via sin/cos at
    log-spaced frequencies. `dim` must be even.

    Uses einops.rearrange for the explicit broadcast — no unsqueeze.
    """
    assert dim % 2 == 0, "embedding dim must be even"
    half   = dim // 2
    i_vec  = torch.arange(half, dtype=torch.float32, device=t.device)
    freqs  = torch.exp(-math.log(10000.0) * i_vec / half)            # (half,)

    t_col  = rearrange(t.to(torch.float32), "b -> b 1")              # (B, 1)
    f_row  = rearrange(freqs,               "h -> 1 h")              # (1, half)
    args   = t_col * f_row                                           # (B, half)

    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)      # (B, dim)


class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding → small MLP that learns the t-representation."""

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self.embed_dim))


class DiffusionMLP(nn.Module):
    """ε-predictor MLP: (r_t, t) → ε̂. t-aware by default; t-blind for ablation.

    When `use_timestep_embedding=False` no TimestepEmbedding is built, the
    input layer accepts r_t alone, and `forward` ignores `t` entirely.
    """

    def __init__(
        self,
        data_dim: int = 15,
        hidden_dim: int = 128,
        t_embed_dim: int = 64,
        t_hidden_dim: int = 128,
        n_layers: int = 3,
        use_timestep_embedding: bool = True,
    ):
        super().__init__()
        self.data_dim                = data_dim
        self.use_timestep_embedding  = use_timestep_embedding

        if use_timestep_embedding:
            self.t_embedding = TimestepEmbedding(t_embed_dim, t_hidden_dim)
            in_dim = data_dim + t_hidden_dim
        else:
            self.t_embedding = None
            in_dim = data_dim

        layers = []
        prev = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.SiLU())
            prev = hidden_dim
        layers.append(nn.Linear(prev, data_dim))                     # no activation on output
        self.net = nn.Sequential(*layers)

    def forward(self, r_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.use_timestep_embedding:
            t_emb = self.t_embedding(t)
            x = torch.cat([r_t, t_emb], dim=-1)
        else:
            x = r_t
        return self.net(x)


# ─── LightningModule ───────────────────────────────────────────────────────

class DiffusionLightning(pl.LightningModule):
    """Stitch a DiffusionMLP, a (frozen) noise schedule, and an MSE-on-ε
    training step into a self-contained Lightning module.

    Constructor responsibilities
    ----------------------------
    - Hold the model and the schedule arrays α, σ as non-learnable buffers
      (so they ride to GPU automatically and are saved in the checkpoint).
    - Hold the per-bin standardization stats from the training dataset, also
      as buffers — the standalone `sample()` reads them at inference time to
      return samples in physical residual units without needing the original
      dataset object in scope.
    - Store hyperparameters for clean `load_from_checkpoint` (ignoring the
      large/numpy-typed kwargs that are training-time-only).
    """

    def __init__(
        self,
        model: nn.Module,
        alpha,
        sigma,
        T: int,
        lr: float = 1e-3,
        scheduler: str = "plateau",
        weight_decay: float = 1e-4,
        bin_means=None,
        bin_stds=None,
    ):
        super().__init__()
        # Persisted small-scalar hyperparameters for load_from_checkpoint.
        # `model` and the numpy schedule/bin arrays are excluded — they are
        # reconstructed (model) or restored from buffers (alpha/sigma/bins)
        # at load time. See 08_evaluate.ipynb for the loading pattern.
        self.save_hyperparameters(ignore=["model", "alpha", "sigma", "bin_means", "bin_stds"])

        self.model          = model
        self.T              = int(T)
        self.lr             = float(lr)
        self.scheduler_kind = str(scheduler)
        self.weight_decay   = float(weight_decay)

        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.register_buffer("sigma", torch.as_tensor(sigma, dtype=torch.float32))

        if bin_means is not None and bin_stds is not None:
            self.register_buffer("bin_means", torch.as_tensor(bin_means, dtype=torch.float32))
            self.register_buffer("bin_stds",  torch.as_tensor(bin_stds,  dtype=torch.float32))
        else:
            self.bin_means = None
            self.bin_stds  = None

    # ── training / validation share the same forward corruption ───────────
    def _shared_step(self, batch):
        r_clean = batch                                              # (B, 15)
        B       = r_clean.shape[0]
        t       = torch.randint(0, self.T, (B,), device=r_clean.device)
        eps     = torch.randn_like(r_clean)

        alpha_t = rearrange(self.alpha[t], "b -> b 1")               # (B, 1)
        sigma_t = rearrange(self.sigma[t], "b -> b 1")               # (B, 1)
        r_t     = alpha_t * r_clean + sigma_t * eps

        eps_hat = self.model(r_t, t)
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


# ─── Sampler ───────────────────────────────────────────────────────────────

@torch.no_grad()
def sample(
    lightning_module: DiffusionLightning,
    batch_size: int,
    data_dim: int = 15,
    device=None,
) -> torch.Tensor:
    """Deterministic DDIM sampling from r_T ~ N(0, I) down to r_0.

    Returns samples in *physical* residual units — the LightningModule's
    `bin_means`/`bin_stds` buffers (registered at training time) are applied
    to de-standardize. The function has no dependency on the original
    training dataset, so it works unchanged in 08_train.ipynb and
    08_evaluate.ipynb.
    """
    if device is None:
        device = next(lightning_module.parameters()).device
    lightning_module.eval()
    lightning_module.to(device)

    alpha = lightning_module.alpha
    sigma = lightning_module.sigma
    T_loc = lightning_module.T

    r_t = torch.randn(batch_size, data_dim, device=device)
    for t in range(T_loc - 1, -1, -1):
        t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
        eps_hat = lightning_module.model(r_t, t_batch)

        alpha_t = alpha[t]
        sigma_t = sigma[t]
        r_0_hat = (r_t - sigma_t * eps_hat) / alpha_t

        if t > 0:
            r_t = alpha[t - 1] * r_0_hat + sigma[t - 1] * eps_hat
        else:
            r_t = r_0_hat

    if lightning_module.bin_means is not None and lightning_module.bin_stds is not None:
        return r_t * lightning_module.bin_stds.to(device) + lightning_module.bin_means.to(device)
    return r_t


# ─── Callbacks ─────────────────────────────────────────────────────────────

class SampleQualityCallback(pl.Callback):
    """Distributional diagnostics during training and at end-of-training.

    During training, every `every_n_epochs` epochs, draws `n_compare` samples
    and logs as scalars:
        sample_std_mse   = MSE between sampled bin-wise std and training std
        sample_cov_frob  = Frobenius distance between sampled cov and training cov
    If `val_samples` is supplied, the same two metrics are logged against the
    validation residuals as `val_sample_std_mse` / `val_sample_cov_frob`.

    At end-of-training, if a WandbLogger is attached AND `bin_centers` was
    provided, logs two figures comparing the bin-wise mean and bin-wise std
    of training residuals vs. samples drawn from the final-iterate weights,
    as `eval/binwise_mean` and `eval/binwise_std`. These are the qualitative
    panels that close the WandB run for visual inspection.

    All training/val arrays are interpreted in *physical* residual units —
    callers pass in de-standardized arrays. Sampled tensors come out of
    `sample()` already in physical units too.
    """

    def __init__(
        self,
        train_samples,
        every_n_epochs: int = 100,
        n_compare: int = 500,
        val_samples=None,
        bin_centers=None,
        bin_width: float = 3.0,
    ):
        super().__init__()
        self.train_arr   = np.asarray(train_samples, dtype=np.float32)
        self.val_arr     = (None if val_samples is None
                            else np.asarray(val_samples, dtype=np.float32))
        self.bin_centers = (None if bin_centers is None
                            else np.asarray(bin_centers, dtype=np.float32))
        self.bin_width   = float(bin_width)
        self.every       = int(every_n_epochs)
        self.n           = int(n_compare)

    # ── periodic scalar diagnostics ────────────────────────────────────────
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every != 0:
            return
        samples = sample(pl_module, batch_size=self.n,
                         data_dim=self.train_arr.shape[1]).cpu().numpy()
        # sample() de-standardizes, so `samples` is in physical units.

        sample_std = samples.std(0)
        sample_cov = np.cov(samples, rowvar=False)

        std_mse  = float(np.mean((sample_std - self.train_arr.std(0)) ** 2))
        cov_dist = float(np.linalg.norm(sample_cov - np.cov(self.train_arr, rowvar=False)))
        pl_module.log("sample_std_mse",  std_mse)
        pl_module.log("sample_cov_frob", cov_dist)

        if self.val_arr is not None:
            val_std_mse  = float(np.mean((sample_std - self.val_arr.std(0)) ** 2))
            val_cov_dist = float(np.linalg.norm(sample_cov - np.cov(self.val_arr, rowvar=False)))
            pl_module.log("val_sample_std_mse",  val_std_mse)
            pl_module.log("val_sample_cov_frob", val_cov_dist)

        pl_module.train()

    # ── end-of-training figure panels (WandB only) ─────────────────────────
    def on_train_end(self, trainer, pl_module):
        if not isinstance(trainer.logger, WandbLogger):
            return
        if self.bin_centers is None:
            return
        try:
            import wandb
        except ImportError:
            return

        samples = sample(pl_module, batch_size=self.n,
                         data_dim=self.train_arr.shape[1]).cpu().numpy()

        rng      = np.random.default_rng(0)
        idx      = rng.integers(0, len(self.train_arr), size=self.n)
        train_np = self.train_arr[idx]

        w = self.bin_width * 0.4

        fig_mean, ax = plt.subplots(figsize=(9, 4))
        ax.bar(self.bin_centers - w / 2, train_np.mean(axis=0), width=w,
               color="C0", label="training",      edgecolor="black", linewidth=0.4)
        ax.bar(self.bin_centers + w / 2, samples.mean(axis=0),  width=w,
               color="C2", label="model samples", edgecolor="black", linewidth=0.4)
        ax.axhline(0, color="k", linewidth=0.4)
        ax.set_xlabel("|latitude| (°)"); ax.set_ylabel("mean residual")
        ax.set_title("End-of-training: bin-wise mean — training vs. model samples")
        ax.legend()
        fig_mean.tight_layout()

        fig_std, ax = plt.subplots(figsize=(9, 4))
        ax.bar(self.bin_centers - w / 2, train_np.std(axis=0), width=w,
               color="C0", label="training",      edgecolor="black", linewidth=0.4)
        ax.bar(self.bin_centers + w / 2, samples.std(axis=0),  width=w,
               color="C2", label="model samples", edgecolor="black", linewidth=0.4)
        ax.set_xlabel("|latitude| (°)"); ax.set_ylabel("std residual")
        ax.set_title("End-of-training: bin-wise std — training vs. model samples")
        ax.legend()
        fig_std.tight_layout()

        # Include a scalar in the same log call so the wandb media panel can
        # anchor the images on the step axis. Without this, the images land
        # at a wandb internal step one past the last scalar, and the panel's
        # default view shows "No matching media".
        trainer.logger.experiment.log({
            "eval/binwise_mean":       wandb.Image(fig_mean),
            "eval/binwise_std":        wandb.Image(fig_std),
            "eval/end_of_training":    float(trainer.current_epoch),
            "trainer/global_step":     trainer.global_step,
        })
        plt.close(fig_mean)
        plt.close(fig_std)
