"""Evaluation primitives for Week 11 ablation experiments.

Provides:
- NLL primitives (ported from 10c): hard_nll_classical, hard_nll_combined
- Per-window evaluation block construction: build_eval_hemicycles
- Oracle MLP diagnostic: OracleMLP, gaussian_nll, fit_oracle
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm as sp_norm
from typing import Dict, List, Optional, Sequence, Tuple

from conditioned_infrastructure import ExtendedConditionalResidualDataset


# ---------------------------------------------------------------------------
# NLL primitives (byte-identical logic from 10c)
# ---------------------------------------------------------------------------

def hard_nll_classical(model, hcs):
    """Classical Gaussian density on raw |latitude| values, hard-gated
    against ``model.mu_0(amplitude)``.

    Parameters
    ----------
    model : ButterflAIModel
    hcs : list of hemicycle dicts

    Returns
    -------
    (nll, details) where details has keys
    ``included``, ``candidate``, ``coverage``.
    """
    total, included, candidate = 0.0, 0, 0
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
            ll = sp_norm.logpdf(blk["lats"], loc=mu, scale=sigma).mean()
            if not np.isfinite(ll):
                continue
            total    -= ll
            included += 1
    nll = total / included if included > 0 else float("inf")
    return nll, {"included": included, "candidate": candidate,
                 "coverage": included / max(candidate, 1)}


def hard_nll_combined(model, hcs, residuals_by_block,
                      bin_width=3.0, eps=1e-6):
    """Classical + per-bin residual density, floored at *eps*.

    Parameters
    ----------
    model : ButterflAIModel
    hcs : list of hemicycle dicts
    residuals_by_block : dict
        ``(cycle, hemisphere, center_decimal) → (15,)`` array of residual
        densities.
    bin_width : float
        Width of each latitude bin (degrees).
    eps : float
        Floor to avoid log(0) when residual over-corrects.

    Returns
    -------
    (nll, details) with extra key ``floor_fraction``.
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
            if key not in residuals_by_block:
                continue
            residual = residuals_by_block[key]

            lats   = blk["lats"]
            p_cl   = sp_norm.pdf(lats, loc=mu, scale=sigma)
            bin_ix = np.clip(np.floor(lats / bin_width).astype(int), 0, 14)
            p_raw  = p_cl + residual[bin_ix]
            p_comb = np.maximum(eps, p_raw)
            n_floored += int((p_raw < eps).sum())
            n_lats    += len(lats)

            ll = np.log(p_comb).mean()
            if not np.isfinite(ll):
                continue
            total    -= ll
            included += 1
    nll = total / included if included > 0 else float("inf")
    return nll, {"included": included, "candidate": candidate,
                 "coverage": included / max(candidate, 1),
                 "floor_fraction": n_floored / max(n_lats, 1)}


# ---------------------------------------------------------------------------
# Per-window evaluation block construction
# ---------------------------------------------------------------------------

def build_eval_hemicycles(
    raw_csv_path: str,
    windows_v2: pd.DataFrame,
    classical,
    splits: Sequence[str] = ("train", "val"),
) -> Tuple[List[dict], Dict[str, List[str]]]:
    """Build per-window evaluation blocks tagged with ``groups_raw`` dicts.

    Parameters
    ----------
    raw_csv_path : str
        Path to the raw sunspot-group CSV.
    windows_v2 : pd.DataFrame
        The v2 parquet (augmented with all conditioning groups).
    classical : ButterflAIModel
        Fitted classical model (for ``to_tau``, ``lookup_t0``, etc.).
    splits : sequence of str
        Which splits to include (default train + val).

    Returns
    -------
    (hemicycles, group_cols) where ``hemicycles`` is a list of hemicycle
    dicts and ``group_cols`` maps group name → list of column names.
    """
    raw_df = pd.read_csv(raw_csv_path)
    raw_df["date"]       = pd.to_datetime(raw_df[["year", "month", "day"]])
    raw_df["abs_lat"]    = raw_df["latitude"].abs()
    raw_df["hemisphere"] = np.where(raw_df["latitude"] >= 0, "north", "south")
    raw_df = raw_df.dropna(subset=["CYCLE"]).copy()
    raw_df["CYCLE"]      = raw_df["CYCLE"].astype(int)

    # Resolve GROUP_COLS dynamically (traj columns from parquet schema).
    group_cols: Dict[str, List[str]] = dict(
        ExtendedConditionalResidualDataset.GROUP_COLS
    )
    traj_cols = sorted(
        [c for c in windows_v2.columns if c.startswith("area_lag")],
        key=lambda c: int(c.replace("area_lag", "")),
    )
    if traj_cols:
        group_cols["traj"] = traj_cols

    # Validity flags are routed through consumed_keys like any other group
    # entry (e.g. "cond_traj_valid"); expose them in groups_raw so
    # block_cond_concat can pass them through un-standardized.
    for _g, _vcol in ExtendedConditionalResidualDataset.GROUP_VALIDITY.items():
        if _g in group_cols and _vcol in windows_v2.columns:
            group_cols[f"{_g}_valid"] = [_vcol]

    # Per-(cycle, hemisphere) split and amplitude from the parquet.
    hc_split = (
        windows_v2.groupby(["cycle", "hemisphere"])["split"]
        .agg(lambda s: s.iloc[0]).to_dict()
    )
    amp_lookup = (
        windows_v2.groupby(["cycle", "hemisphere"])["amplitude"]
        .agg("first").to_dict()
    )

    # Convert parquet tau_center → calendar year.
    _t0_by_hc = {
        (int(c), str(h)): float(classical.lookup_t0(int(c), h))
        for (c, h) in classical.known_hemicycles()
    }
    _wdf = windows_v2.copy()
    _wdf = _wdf[_wdf.apply(
        lambda r: (int(r["cycle"]), str(r["hemisphere"])) in _t0_by_hc,
        axis=1,
    )]
    _wdf["year_center"] = _wdf.apply(
        lambda r: float(r["tau_center"])
                  + _t0_by_hc[(int(r["cycle"]), str(r["hemisphere"]))],
        axis=1,
    )

    # Build per-window groups_raw lookup.
    _groups_raw_lookup: Dict[tuple, Dict[str, np.ndarray]] = {}
    for _, r in _wdf.iterrows():
        key = (int(r["cycle"]), str(r["hemisphere"]), float(r["year_center"]))
        gr: Dict[str, np.ndarray] = {}
        for g, cols in group_cols.items():
            if all(c in r.index for c in cols):
                gr[g] = np.array([float(r[c]) for c in cols], dtype=np.float32)
        _groups_raw_lookup[key] = gr

    def _lookup(cyc, hemi, c_dec, tol=1e-2):
        best_v, best_d = None, tol
        for (c2, h2, yc), v in _groups_raw_lookup.items():
            if c2 != cyc or h2 != hemi:
                continue
            d = abs(float(yc) - c_dec)
            if d <= best_d:
                best_v, best_d = v, d
        return best_v

    def _build_per_window(cyc, hemi, df_hc):
        if len(df_hc) == 0:
            return []
        y0 = df_hc["date"].min().year
        y1 = df_hc["date"].max().year + 1
        bounds = sorted({pd.Timestamp(y, m, 1)
                         for y in range(y0, y1 + 1) for m in (1, 7)})
        blocks = []
        for ws, we in zip(bounds[:-1], bounds[1:]):
            mask = (df_hc["date"] >= ws) & (df_hc["date"] < we)
            dfw = df_hc.loc[mask]
            if len(dfw) < 20:
                continue
            c_ts  = ws + (we - ws) / 2
            c_dec = c_ts.year + c_ts.dayofyear / 365.25
            gr = _lookup(cyc, hemi, c_dec)
            if gr is None:
                continue
            if "base" in gr and not np.all(np.isfinite(gr["base"])):
                continue
            blocks.append({
                "center_decimal": float(c_dec),
                "tau":  float(classical.to_tau(cyc, hemi, c_dec)),
                "lats": dfw["abs_lat"].to_numpy(np.float32),
                "groups_raw": gr,
            })
        return blocks

    hemicycles: List[dict] = []
    for (cyc, hemi), split in hc_split.items():
        if split not in splits:
            continue
        try:
            t0 = float(classical.lookup_t0(cyc, hemi))
        except KeyError:
            continue
        A = float(amp_lookup[(cyc, hemi)])
        dfh = raw_df[
            (raw_df["CYCLE"] == int(cyc)) & (raw_df["hemisphere"] == hemi)
        ]
        blocks = _build_per_window(cyc, hemi, dfh)
        if not blocks:
            continue
        hemicycles.append({
            "cycle": int(cyc),
            "hemisphere": hemi,
            "amplitude": A,
            "t0": t0,
            "split": split,
            "blocks": blocks,
        })

    return hemicycles, group_cols


# ---------------------------------------------------------------------------
# Oracle MLP diagnostic
# ---------------------------------------------------------------------------

class OracleMLP(nn.Module):
    """Map a conditioning vector to per-bin Gaussian residual parameters.

    Output is ``(mean, log_std)``, each of shape ``(B, 15)``.
    """

    def __init__(self, cond_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 30),
        )

    def forward(self, cond: torch.Tensor):
        out = self.net(cond)
        mean    = out[:, :15]
        log_std = out[:, 15:].clamp(-7.0, 7.0)   # keep exp(2*log_std) finite
        return mean, log_std


def gaussian_nll(r: torch.Tensor, mean: torch.Tensor,
                 log_std: torch.Tensor) -> torch.Tensor:
    """Per-row Gaussian NLL of standardized residual *r* under the
    predicted ``(mean, log_std)``.  Sum over 15 bins, mean over batch.
    """
    nll = 0.5 * (np.log(2 * np.pi) + 2 * log_std
                 + (r - mean) ** 2 / torch.exp(2 * log_std))
    return nll.sum(dim=-1).mean()


def fit_oracle(
    cond_train: torch.Tensor,
    r_train: torch.Tensor,
    cond_val: torch.Tensor,
    r_val: torch.Tensor,
    max_epochs: int = 500,
    lr: float = 1e-2,
    hidden_dim: int = 64,
    seed: int = 0,
) -> Tuple[nn.Module, float]:
    """Fit an :class:`OracleMLP` on ``(cond, r)`` pairs and return the
    checkpoint with the best validation Gaussian NLL.

    Parameters
    ----------
    cond_train, r_train : torch.Tensor
        Training conditioning vectors and standardized residuals (15-D).
    cond_val, r_val : torch.Tensor
        Validation counterparts.
    max_epochs : int
        Maximum number of training epochs.
    lr : float
        Adam learning rate.
    hidden_dim : int
        Hidden layer width for the oracle MLP.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (best_model, best_val_nll)
    """
    torch.manual_seed(seed)
    cond_dim = cond_train.shape[1]
    model = OracleMLP(cond_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Seed best_state with the initial weights so a diverging run (NaN val
    # loss on every epoch) still returns a usable module.
    best_val_nll = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(max_epochs):
        model.train()
        mean, log_std = model(cond_train)
        loss = gaussian_nll(r_train, mean, log_std)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            v_mean, v_log_std = model(cond_val)
            val_nll = gaussian_nll(r_val, v_mean, v_log_std).item()
        if np.isfinite(val_nll) and val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val_nll
