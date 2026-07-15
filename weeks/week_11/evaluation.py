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
from distribution_metrics import (
    distributional_report,
    emd_density,
    energy_distance_to_point,
    density_moments,
    crps_ensemble,
)


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


def hard_nll_combined_normalized(model, hcs, residuals_by_block, bin_centers,
                                 bin_width=3.0, eps=1e-6):
    """Renormalized counterpart of :func:`hard_nll_combined`.

    Scores the same pointwise combined density ``p_cl + residual`` at the
    observed latitudes, but divides it by its total mass on the 15-bin
    support so the result is a *proper* (normalized) density NLL. This
    adds ``+log(Z)`` per block, where

        Z = sum_b max(eps, p_cl_bin[b] + residual[b]) * bin_width

    and ``p_cl_bin`` is the classical density at the bin centers.

    At guidance weight ``w == 0`` the generated residual integrates to ~0
    and ``p_cl_bin`` cancels the classical part of the residual target
    (``residual = emp - par``), so ``Z ~ 1`` and this reduces to
    :func:`hard_nll_combined`. Under classifier-free guidance the residual
    is amplified, ``Z`` grows, and ``+log(Z)`` cancels the artificial gain
    that the un-normalized metric rewards (the leak that drives E6/E8 NLL
    negative).

    Parameters
    ----------
    model : ButterflAIModel
    hcs : list of hemicycle dicts
    residuals_by_block : dict
        ``(cycle, hemisphere, center_decimal) -> (15,)`` residual densities.
    bin_centers : array_like, shape (15,)
        Latitude bin centers in degrees (e.g. ``[1.5, 4.5, ..., 43.5]``).
    bin_width : float
        Width of each latitude bin (degrees).
    eps : float
        Floor for both the pointwise density and the normalizer.

    Returns
    -------
    (nll, details) with the keys of :func:`hard_nll_combined` plus
    ``mean_added_mass`` (mean ``sum(residual)*bin_width``) and ``mean_logZ``.
    """
    bin_centers = np.asarray(bin_centers, dtype=float)
    total, included, candidate = 0.0, 0, 0
    n_floored, n_lats = 0, 0
    sum_added_mass, sum_logZ = 0.0, 0.0
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

            # Normalizer: total mass of the (floored) binned combined density.
            p_cl_bin = sp_norm.pdf(bin_centers, loc=mu, scale=sigma)
            comb_bin = np.maximum(eps, p_cl_bin + residual)
            Z        = max(eps, float((comb_bin * bin_width).sum()))
            logZ     = float(np.log(Z))

            ll = np.log(p_comb).mean()
            if not np.isfinite(ll) or not np.isfinite(logZ):
                continue
            total          -= (ll - logZ)
            included       += 1
            sum_added_mass += float((residual * bin_width).sum())
            sum_logZ       += logZ
    nll = total / included if included > 0 else float("inf")
    return nll, {"included": included, "candidate": candidate,
                 "coverage": included / max(candidate, 1),
                 "floor_fraction": n_floored / max(n_lats, 1),
                 "mean_added_mass": sum_added_mass / max(included, 1),
                 "mean_logZ": sum_logZ / max(included, 1)}


def project_residuals_zero_integral(residuals_by_block):
    """Return a new residuals dict with each (15,) residual demeaned so its
    integral over the 15 uniform bins is zero.

    With uniform ``bin_width`` the constraint ``sum(r) * bin_width = 0`` is
    equivalent to ``mean(r) = 0``, and the L2-optimal projection onto that
    hyperplane is per-sample demeaning.
    """
    return {k: (v - v.mean()) for k, v in residuals_by_block.items()}


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


# ---------------------------------------------------------------------------
# Metric-integrity & physical-plausibility diagnostics
# ---------------------------------------------------------------------------

def sample_diagnostics(samples_NK15, bin_stds_phys, bin_width=3.0):
    """Metric-integrity diagnostics for a set of generated residual samples.

    Both summaries use **robust** statistics (median, IQR) because
    classifier-free guidance can produce a small fraction of extreme
    samples whose magnitudes (~1e15) would dominate mean/std and make the
    panel unreadable while saying nothing about typical behavior.

    Parameters
    ----------
    samples_NK15 : numpy.ndarray, shape (N_blocks, K, 15)
        Generated residual samples in physical (density) units.
    bin_stds_phys : array_like, shape (15,)
        Empirical per-bin residual spread (``train_ds.bin_stds``); used as
        the natural reference scale for "healthy" sample diversity. The
        Gaussian relation ``IQR ≈ 1.349·σ`` converts it to an IQR scale.
    bin_width : float
        Latitude bin width (degrees).

    Returns
    -------
    dict with
    ``added_mass`` : median over blocks*K of ``sum_b residual*bin_width``.
        A faithful residual adds ~0 net mass; classifier-free guidance
        inflates it (the leak the un-normalized NLL rewards).
    ``diversity`` : median over blocks*bins of (across-K residual IQR ÷
        empirical IQR scale). ~1 is healthy; ->0 signals mode collapse
        under strong guidance.
    """
    s = np.asarray(samples_NK15, dtype=float)               # (N, K, 15)
    added = float(np.median(s.sum(axis=-1) * bin_width))
    per_bin_iqr = np.percentile(s, 75, axis=1) - np.percentile(s, 25, axis=1)
    bin_stds = np.asarray(bin_stds_phys, dtype=float)
    diversity = float(np.median(
        per_bin_iqr / np.maximum(bin_stds * 1.349, 1e-12)
    ))
    return {"added_mass": added, "diversity": diversity}


def distributional_scorecard(model_dens_by_block, emp_dens_by_block,
                             bin_centers, bin_width=3.0, tau_by_block=None):
    """Distributional goodness-of-fit scorecard over a set of windows.

    Model-agnostic companion to the NLL primitives above: it takes
    already-built **densities** (so it works identically for the classical
    Gaussian, the residual model ``par + residual``, and the logit/softmax
    model) and runs the full :mod:`distribution_metrics` suite — EMD, energy
    distance, moment errors, CRPS, and the mean-latitude rank histogram. Pair
    it with :func:`hard_nll_combined_normalized` for a fair NLL-vs-shape
    comparison.

    Parameters
    ----------
    model_dens_by_block : dict
        ``block_key -> (M, 15)`` ensemble of model densities (1/deg). For a
        deterministic model (e.g. the classical Gaussian) pass ``M = 1``.
    emp_dens_by_block : dict
        ``block_key -> (15,)`` empirical density for the same windows.
    bin_centers : ndarray, shape (15,)
    bin_width : float
    tau_by_block : dict, optional
        ``block_key -> tau`` for the drift arrays in the report.

    Returns
    -------
    dict
        The :func:`distribution_metrics.distributional_report` output over the
        blocks present in both dicts, plus ``n_blocks``.
    """
    keys = [k for k in model_dens_by_block if k in emp_dens_by_block]
    if not keys:
        raise ValueError("no overlapping block keys between model and empirical densities")

    M = max(np.atleast_2d(model_dens_by_block[k]).shape[0] for k in keys)
    ens = np.stack([
        np.broadcast_to(np.atleast_2d(model_dens_by_block[k]), (M, len(bin_centers)))
        for k in keys
    ]).astype(np.float64)                                    # (N, M, K)
    emp = np.stack([np.asarray(emp_dens_by_block[k], dtype=np.float64) for k in keys])
    tau = (None if tau_by_block is None
           else np.array([tau_by_block[k] for k in keys], dtype=np.float64))

    report = distributional_report(ens, emp, bin_centers,
                                   bin_width=bin_width, tau=tau)
    report["n_blocks"] = len(keys)
    return report


def assemble_butterfly(model, hc, mean_residual_by_block, bin_centers,
                       eps=1e-6):
    """Assemble a latitude-vs-time density map for one hemicycle.

    Each window/block contributes one latitude column: the hard-gated
    classical binned density plus the mean (over K) generated residual,
    floored at *eps*. Blocks failing the hard gate (``mu > mu_0``) or
    lacking a residual are skipped.

    Parameters
    ----------
    model : ButterflAIModel
    hc : dict
        One hemicycle entry from :func:`build_eval_hemicycles`.
    mean_residual_by_block : dict
        ``(cycle, hemisphere, center_decimal) -> (15,)`` mean residual.
    bin_centers : array_like, shape (15,)
        Latitude bin centers (degrees).
    eps : float

    Returns
    -------
    (density_map, times) where ``density_map`` is ``(n_windows, 15)`` in
    density units and ``times`` is ``(n_windows,)`` decimal years, both
    sorted by time. Empty arrays if no block passes the gate.
    """
    bin_centers = np.asarray(bin_centers, dtype=float)
    A    = hc["amplitude"]
    mu0A = float(model.mu_0(A))
    cols, times = [], []
    for blk in hc["blocks"]:
        mu = float(model.mu(blk["tau"]))
        if mu > mu0A:
            continue
        sigma = float(model.sigma(mu, A))
        if sigma <= 0:
            continue
        key = (hc["cycle"], hc["hemisphere"], blk["center_decimal"])
        if key not in mean_residual_by_block:
            continue
        p_cl_bin = sp_norm.pdf(bin_centers, loc=mu, scale=sigma)
        cols.append(np.maximum(eps, p_cl_bin + mean_residual_by_block[key]))
        times.append(float(blk["center_decimal"]))
    if not cols:
        return np.empty((0, len(bin_centers))), np.empty((0,))
    order = np.argsort(times)
    return np.asarray(cols)[order], np.asarray(times)[order]


def butterfly_physical_checks(density_map, times, bin_centers,
                              lat_lo=5.0, lat_hi=40.0):
    """Physical-plausibility checks on an assembled butterfly map.

    Parameters
    ----------
    density_map : numpy.ndarray, shape (n_windows, 15)
        Per-window latitude densities (output of :func:`assemble_butterfly`).
    times : array_like, shape (n_windows,)
        Decimal years per window.
    bin_centers : array_like, shape (15,)
        Latitude bin centers (degrees).
    lat_lo, lat_hi : float
        Spörer-zone bounds (deg) for the in-band mass fraction.

    Returns
    -------
    dict with
    ``sporer_slope`` : centroid-latitude slope vs time (deg/yr); negative
        means equatorward drift, the physically expected sign.
    ``sporer_ok`` : bool, ``sporer_slope < 0``.
    ``in_band_fraction`` : mean fraction of per-window mass within
        ``[lat_lo, lat_hi]``.
    ``centroid_start`` / ``centroid_end`` : first/last window centroid lat.
    """
    bin_centers = np.asarray(bin_centers, dtype=float)
    dm = np.asarray(density_map, dtype=float)
    if dm.shape[0] < 2:
        return {"sporer_slope": float("nan"), "sporer_ok": False,
                "in_band_fraction": float("nan"),
                "centroid_start": float("nan"), "centroid_end": float("nan")}
    mass = dm.sum(axis=1)
    centroid = (dm * bin_centers).sum(axis=1) / np.maximum(mass, 1e-12)
    slope = float(np.polyfit(np.asarray(times, dtype=float), centroid, 1)[0])
    in_band = (bin_centers >= lat_lo) & (bin_centers <= lat_hi)
    in_band_frac = float(
        (dm[:, in_band].sum(axis=1) / np.maximum(mass, 1e-12)).mean()
    )
    return {"sporer_slope": slope, "sporer_ok": slope < 0.0,
            "in_band_fraction": in_band_frac,
            "centroid_start": float(centroid[0]),
            "centroid_end": float(centroid[-1])}


def hemispheric_symmetry(checks_by_hc):
    """Compare north/south plausibility for cycles present in both hemispheres.

    Parameters
    ----------
    checks_by_hc : dict
        ``(cycle, hemisphere) -> butterfly_physical_checks(...) dict``.

    Returns
    -------
    dict ``cycle -> {slope_diff, in_band_diff}`` for cycles that have both
    a 'north' and a 'south' entry. Empty if no cycle is paired (the val
    split here is hemisphere-mixed, so pairing is often partial).
    """
    by_cycle = {}
    for (cyc, hemi), d in checks_by_hc.items():
        by_cycle.setdefault(cyc, {})[hemi] = d
    out = {}
    for cyc, hemis in by_cycle.items():
        if "north" in hemis and "south" in hemis:
            n, s = hemis["north"], hemis["south"]
            out[cyc] = {
                "slope_diff": abs(n["sporer_slope"] - s["sporer_slope"]),
                "in_band_diff": abs(n["in_band_fraction"]
                                    - s["in_band_fraction"]),
            }
    return out
