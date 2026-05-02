#!/usr/bin/env python3
"""
08_timeshifts.py

Joint optimisation of the 8-parameter butterfly-wing model **and**
per-hemisphere-cycle timeshifts via a warm-started L-BFGS-B.

The timeshifts Δt_i correct the t₀ reference epochs that were previously
fixed after the two-pass 15°-crossing / residual-minimisation pre-processing.
Freeing them allows the global NLL to close any remaining misalignment.

Parameter layout  (8 + N_hc  scalars in one vector):
    Global  [0:8]
        [0]  a_mupeak  — slope     of μ_peak(A) = a_mupeak·A + b_mupeak
        [1]  b_mupeak  — intercept                               [degrees]
        [2]  a_mi      — slope     of m_i(A) = a_mi·A + b_mi
        [3]  b_mi      — intercept
        [4]  m_shared  — universal equatorward σ-line slope      [°/°]
        [5]  b_shared  — universal equatorward σ-line intercept  [°]
        [6]  a_mu      — mean-path amplitude                     [degrees]
        [7]  b_mu      — mean-path e-folding time                [years]
    Per-cycle  [8 : 8+N_hc]
        Δt_i — timeshift correction for hemisphere-cycle i       [years]
               (added to the pre-computed t0_refined value)

Optimisation stages
-------------------
S1  4p   Nelder-Mead   amplitude params only; path, line, Δt fixed
S2  8p   L-BFGS-B      all global params free; Δt fixed at 0
S3  8+N  L-BFGS-B      all params free, Δt bounded ±2 yr
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm as sp_norm
from scipy.optimize import curve_fit, minimize_scalar, minimize

# ── Paths and constants ────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "composite_sunspot_groups_peak_area.csv"

# ── Fitting mode ───────────────────────────────────────────────────────────
# True  → match notebook 06 exactly:
#           • amplitude smoothing uses correctedArea > 50 MSH
#           • σ(μ) curve_fit is unbounded; validity requires sL, sR < 20°
# False → broader settings that retain more cycles:
#           • amplitude smoothing uses correctedArea > 30 MSH (same as lat data)
#           • σ(μ) curve_fit uses explicit bounds; validity allows sL ≤ 100°, sR ≤ 40°
MATCH_NOTEBOOK: bool = False

# Dividing amplitudes by A_REF keeps all slope coefficients O(1),
# preventing L-BFGS-B finite-difference gradients from drowning in noise.
A_REF: float = 1000.0

# Temperature for the soft μ₀ sigmoid gate  [degrees].
# weight(μ) = 1 / (1 + exp((μ − μ₀) / T))
# At T = 0.5°: weight = 0.99 at μ₀ − 2.3°, 0.5 at μ₀, 0.01 at μ₀ + 2.3°.
# Makes the NLL differentiable everywhere — required for L-BFGS-B in S3.
MU0_SIGMOID_T: float = 0.5


# ══════════════════════════════════════════════════════════════════════════
# Section 1 — Primitive model functions
# ══════════════════════════════════════════════════════════════════════════

def exponential_decay(tau: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Universal mean-latitude path.

    Parameters
    ----------
    tau : ndarray — time since cycle reference epoch  [years]
    a   : float   — initial latitude at τ = 0  [degrees]
    b   : float   — e-folding time  [years]
    """
    return a * np.exp(-tau / b)


def split_normal_amplitude(
    mu: np.ndarray,
    A: float,
    mu_peak: float,
    sigma_L: float,
    sigma_R: float,
) -> np.ndarray:
    """
    Split-normal envelope  σ(μ)  used only for the per-cycle warm-start fit.

    Parameters
    ----------
    mu      : ndarray — mean emergence latitude  [degrees]
    A       : float   — peak spread amplitude  [degrees]
    mu_peak : float   — latitude at maximum spread  [degrees]
    sigma_L : float   — poleward half-width  [degrees]
    sigma_R : float   — equatorward half-width  [degrees]
    """
    return np.where(
        mu >= mu_peak,
        A * np.exp(-0.5 * ((mu - mu_peak) / sigma_L) ** 2),
        A * np.exp(-0.5 * ((mu - mu_peak) / sigma_R) ** 2),
    )


def piecewise_linear_sigma(
    mu: np.ndarray,
    m_shared: float,
    b_shared: float,
    mu_peak: float,
    m_i: float,
) -> np.ndarray:
    """
    Piecewise-linear model  σ(μ): universal equatorward line joined at μ_peak
    to a per-cycle poleward line, with continuity enforced.

    Parameters
    ----------
    mu       : ndarray — mean emergence latitude  [degrees]
    m_shared : float   — equatorward slope  [°/°]
    b_shared : float   — equatorward intercept  [°]
    mu_peak  : float   — latitude of the σ peak for this cycle  [°]
    m_i      : float   — poleward slope for this cycle  [°/°]
    """
    sigma_at_peak = m_shared * mu_peak + b_shared
    b_poleward    = sigma_at_peak - m_i * mu_peak
    return np.clip(
        np.where(mu <= mu_peak,
                 m_shared * mu + b_shared,
                 m_i      * mu + b_poleward),
        0.0, None,
    )


def linear_fit(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Simple affine function  y = a·x + b."""
    return a * x + b


# ══════════════════════════════════════════════════════════════════════════
# Section 2 — Parameter vector helpers
# ══════════════════════════════════════════════════════════════════════════

def pack_global(fit_results: dict,
                m_shared: float, b_shared: float,
                a_mu: float, b_mu: float) -> np.ndarray:
    """Pack the 8 global parameters into a 1-D array."""
    return np.array([
        fit_results["mu_peak"][0], fit_results["mu_peak"][1],
        fit_results["m_i"][0],    fit_results["m_i"][1],
        m_shared, b_shared,
        a_mu, b_mu,
    ])


def pack_4p(fit_results: dict) -> np.ndarray:
    """Pack amplitude-dependent coefficients (4 params) into a 1-D array."""
    return np.array([
        fit_results["mu_peak"][0], fit_results["mu_peak"][1],
        fit_results["m_i"][0],    fit_results["m_i"][1],
    ])


def unpack_4p(x: np.ndarray) -> dict:
    """Unpack 4-parameter vector → dict with 'mu_peak' and 'm_i' tuples."""
    return {
        "mu_peak": (float(x[0]), float(x[1])),
        "m_i":     (float(x[2]), float(x[3])),
    }


def unpack_8p(x: np.ndarray) -> tuple:
    """
    Unpack the 8-parameter global vector.

    Returns
    -------
    fit_results, m_shared, b_shared, a_mu, b_mu
    """
    return (
        {"mu_peak": (float(x[0]), float(x[1])),
         "m_i":     (float(x[2]), float(x[3]))},
        float(x[4]), float(x[5]),
        float(x[6]), float(x[7]),
    )


def unpack_full(x: np.ndarray, n_hc: int) -> tuple:
    """
    Unpack the full (8 + N_hc)-parameter vector.

    Returns
    -------
    fit_results, m_shared, b_shared, a_mu, b_mu, delta_t0s
    """
    fit_results, m_sh, b_sh, a_mu, b_mu = unpack_8p(x)
    return fit_results, m_sh, b_sh, a_mu, b_mu, x[8:8 + n_hc]


# ══════════════════════════════════════════════════════════════════════════
# Section 3 — NLL objective
# ══════════════════════════════════════════════════════════════════════════

def compute_nll(
    cycle_data: list,
    fit_results: dict,
    m_shared: float,
    b_shared: float,
    a_mu: float,
    b_mu: float,
    a_mu0: float,
    b_mu0: float,
    delta_t0s: np.ndarray | None = None,
) -> float:
    """
    Negative log-likelihood for the butterfly-wing model.

    Everything is computed from scratch each call — no pre-cached derived
    quantities.  For each hemisphere-cycle i:

        t0_eff  = t0_refined + Δt_i
        τ       = year_center − t0_eff
        μ(τ)    = a_mu · exp(−τ / b_mu)
        μ₀(A)   = a_mu0 · A + b_mu0          ← fixed from regression
        gate    : skip year if μ(τ) > μ₀(A)  ← whole wing, not just τ ≥ 0
        σ(μ)    = piecewise_linear_sigma(...)

    The μ₀ gate is a soft sigmoid so the NLL is differentiable everywhere:

        weight(μ) = 1 / (1 + exp((μ − μ₀(A)) / MU0_SIGMOID_T))

    Blocks well below μ₀ contribute with weight ≈ 1; blocks above μ₀ with
    weight ≈ 0.  The NLL is weighted-averaged over effective block weight.

    Parameters
    ----------
    cycle_data  : list of (amplitude, t0_refined, [(year_center, lats), ...])
    fit_results : dict — 'mu_peak' and 'm_i' as (slope, intercept) tuples
    m_shared    : float — equatorward σ-line slope
    b_shared    : float — equatorward σ-line intercept
    a_mu        : float — mean-path amplitude  [degrees]
    b_mu        : float — mean-path e-folding time  [years]
    a_mu0       : float — slope of μ₀(A) = a_mu0·A + b_mu0  (fixed, never optimised)
    b_mu0       : float — intercept of μ₀(A)  [degrees]      (fixed, never optimised)
    delta_t0s   : ndarray shape (N_hc,) or None — per-cycle timeshift corrections

    Returns
    -------
    float — weighted-mean NLL per block (nats).  Returns 1e6 if no valid blocks.
    """
    a_mupeak, b_mupeak = fit_results["mu_peak"]
    a_mi,     b_mi     = fit_results["m_i"]
    total, n = 0.0, 0.0

    for i, (amplitude, t0_ref, year_blocks) in enumerate(cycle_data):
        mu0_p    = a_mu0 * amplitude + b_mu0
        mupeak_p = a_mupeak * amplitude + b_mupeak
        mi_p     = a_mi     * amplitude + b_mi
        dt       = float(delta_t0s[i]) if delta_t0s is not None else 0.0
        t0_eff   = t0_ref + dt

        for year_center, lats in year_blocks:
            tau    = year_center - t0_eff
            mu     = a_mu * np.exp(-tau / b_mu)
            weight = 1.0 / (1.0 + np.exp((mu - mu0_p) / MU0_SIGMOID_T))
            if weight < 1e-4:       # negligible contribution — skip for speed
                continue
            sigma = piecewise_linear_sigma(mu, m_shared, b_shared, mupeak_p, mi_p)
            if sigma <= 0.0:
                continue
            total -= weight * sp_norm.logpdf(lats, loc=mu, scale=sigma).mean()
            n     += weight

    return total / n if n > 0.0 else 1e6


def compute_nll_percycle(
    cycle_data: list,
    hc_index: list,
    percycle_params: dict,
    m_shared: float,
    b_shared: float,
    a_mu: float,
    b_mu: float,
) -> float:
    """
    Baseline NLL using per-cycle mu_peak, m_i, and mu0 directly (no amplitude regression).

    Parameters
    ----------
    cycle_data      : list of (amplitude, t0_refined, [(year_center, lats), ...])
    hc_index        : list of (cycle, hemi) in the same order as cycle_data
    percycle_params : dict mapping (cycle, hemi) → (mu_peak, m_i, mu0)
    m_shared        : float — equatorward σ-line slope
    b_shared        : float — equatorward σ-line intercept
    a_mu            : float — mean-path amplitude  [degrees]
    b_mu            : float — mean-path e-folding time  [years]

    Returns
    -------
    float — weighted-mean NLL per block (nats).
    """
    total, n = 0.0, 0.0
    for i, (_, t0_ref, year_blocks) in enumerate(cycle_data):
        mu_peak_i, mi_i, mu0_i = percycle_params[hc_index[i]]
        for year_center, lats in year_blocks:
            tau    = year_center - t0_ref
            mu     = a_mu * np.exp(-tau / b_mu)
            weight = 1.0 / (1.0 + np.exp((mu - mu0_i) / MU0_SIGMOID_T))
            if weight < 1e-4:
                continue
            sigma = piecewise_linear_sigma(mu, m_shared, b_shared, mu_peak_i, mi_i)
            if sigma <= 0.0:
                continue
            total -= weight * sp_norm.logpdf(lats, loc=mu, scale=sigma).mean()
            n     += weight
    return total / n if n > 0.0 else 1e6


def compute_nll_hard_gate(
    cycle_data: list,
    fit_results: dict,
    m_shared: float,
    b_shared: float,
    a_mu: float,
    b_mu: float,
    a_mu0: float,
    b_mu0: float,
    delta_t0s: np.ndarray | None = None,
) -> float:
    """
    NLL with a hard μ₀ gate — identical scoring to notebook 06's
    ``compute_global_nll``.

    Years where μ(τ) > μ₀(A) are excluded entirely (no sigmoid weighting).
    Each surviving year-block counts as 1 in the normalisation denominator,
    matching the notebook's ``n_terms`` accumulator.

    Parameters
    ----------
    cycle_data  : list of (amplitude, t0_refined, [(year_center, lats), ...])
    fit_results : dict — 'mu_peak' and 'm_i' as (slope, intercept) tuples
    m_shared    : float — equatorward σ-line slope
    b_shared    : float — equatorward σ-line intercept
    a_mu        : float — mean-path amplitude  [degrees]
    b_mu        : float — mean-path e-folding time  [years]
    a_mu0       : float — slope of μ₀(A) = a_mu0·A + b_mu0
    b_mu0       : float — intercept of μ₀(A)  [degrees]
    delta_t0s   : ndarray shape (N_hc,) or None — per-cycle timeshift corrections

    Returns
    -------
    float — mean NLL per block (nats).  Returns 1e6 if no valid blocks.
    """
    a_mupeak, b_mupeak = fit_results["mu_peak"]
    a_mi,     b_mi     = fit_results["m_i"]
    total, n = 0.0, 0.0

    for i, (amplitude, t0_ref, year_blocks) in enumerate(cycle_data):
        mu0_p    = a_mu0 * amplitude + b_mu0
        mupeak_p = a_mupeak * amplitude + b_mupeak
        mi_p     = a_mi     * amplitude + b_mi
        dt       = float(delta_t0s[i]) if delta_t0s is not None else 0.0
        t0_eff   = t0_ref + dt

        for year_center, lats in year_blocks:
            tau = year_center - t0_eff
            mu  = a_mu * np.exp(-tau / b_mu)
            if mu > mu0_p:
                continue
            sigma = piecewise_linear_sigma(mu, m_shared, b_shared, mupeak_p, mi_p)
            if sigma <= 0.0:
                continue
            total -= sp_norm.logpdf(lats, loc=mu, scale=sigma).mean()
            n     += 1.0

    return total / n if n > 0.0 else 1e6


def compute_nll_percycle_hard_gate(
    cycle_data: list,
    hc_index: list,
    percycle_params: dict,
    m_shared: float,
    b_shared: float,
    a_mu: float,
    b_mu: float,
) -> float:
    """
    Per-cycle baseline NLL with a hard μ₀ gate — notebook-06 scoring.

    Parameters
    ----------
    cycle_data      : list of (amplitude, t0_refined, [(year_center, lats), ...])
    hc_index        : list of (cycle, hemi) in the same order as cycle_data
    percycle_params : dict mapping (cycle, hemi) → (mu_peak, m_i, mu0)
    m_shared        : float — equatorward σ-line slope
    b_shared        : float — equatorward σ-line intercept
    a_mu            : float — mean-path amplitude  [degrees]
    b_mu            : float — mean-path e-folding time  [years]

    Returns
    -------
    float — mean NLL per block (nats).  Returns 1e6 if no valid blocks.
    """
    total, n = 0.0, 0.0
    for i, (_, t0_ref, year_blocks) in enumerate(cycle_data):
        mu_peak_i, mi_i, mu0_i = percycle_params[hc_index[i]]
        for year_center, lats in year_blocks:
            tau = year_center - t0_ref
            mu  = a_mu * np.exp(-tau / b_mu)
            if mu > mu0_i:
                continue
            sigma = piecewise_linear_sigma(mu, m_shared, b_shared, mu_peak_i, mi_i)
            if sigma <= 0.0:
                continue
            total -= sp_norm.logpdf(lats, loc=mu, scale=sigma).mean()
            n     += 1.0
    return total / n if n > 0.0 else 1e6


# ══════════════════════════════════════════════════════════════════════════
# Section 4 — Data preparation helpers
# ══════════════════════════════════════════════════════════════════════════

def find_15deg_crossing(years: np.ndarray, means: np.ndarray) -> float | None:
    """
    Linearly interpolate the decimal year at which the yearly mean absolute
    latitude crosses 15° from above.

    Parameters
    ----------
    years : ndarray — calendar years (integer-valued)
    means : ndarray — yearly mean absolute latitudes  [degrees]

    Returns
    -------
    float — decimal year of the 15° crossing, or None if not found.
    """
    below = means < 15.0
    if not below.any() or below.all():
        return None
    idx = int(np.argmax(below))
    if idx == 0:
        return None
    y0, mu0 = years[idx - 1], means[idx - 1]
    y1, mu1 = years[idx],     means[idx]
    return float(y0 + (15.0 - mu0) / (mu1 - mu0))


def bin_latitudes(
    tau_values: np.ndarray,
    lat_values: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin τ into equal-width intervals and fit a Gaussian mean to each.

    Parameters
    ----------
    tau_values : ndarray — cycle-aligned times  [years]
    lat_values : ndarray — absolute latitudes   [degrees]
    n_bins     : int     — number of time bins

    Returns
    -------
    bin_tau : ndarray — bin-center τ for bins with ≥ 10 spots
    bin_mu  : ndarray — Gaussian mean for each qualifying bin  [degrees]
    """
    bins        = np.linspace(tau_values.min(), tau_values.max(), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bt, bm = [], []
    for i in range(n_bins):
        mask = (tau_values >= bins[i]) & (tau_values < bins[i + 1])
        lats = lat_values[mask]
        if len(lats) < 10:
            continue
        mu_fit, _ = sp_norm.fit(lats)
        bt.append(bin_centers[i])
        bm.append(mu_fit)
    return np.array(bt), np.array(bm)


def bin_sigma(
    tau_values: np.ndarray,
    lat_values: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin τ into equal-width intervals and return (mean, spread) per bin.

    Parameters
    ----------
    tau_values : ndarray — cycle-aligned times  [years]
    lat_values : ndarray — absolute latitudes   [degrees]
    n_bins     : int     — number of time bins

    Returns
    -------
    bin_mu    : ndarray — fitted mean latitudes (sorted)  [degrees]
    bin_sigma : ndarray — fitted spreads  [degrees]
    """
    bins        = np.linspace(tau_values.min(), tau_values.max(), n_bins + 1)
    bm, bs = [], []
    for i in range(n_bins):
        mask = (tau_values >= bins[i]) & (tau_values < bins[i + 1])
        lats = lat_values[mask]
        if len(lats) < 10:
            continue
        mu_fit, sigma_fit = sp_norm.fit(lats)
        bm.append(mu_fit)
        bs.append(sigma_fit)
    bm_arr = np.array(bm)
    bs_arr = np.array(bs)
    order  = np.argsort(bm_arr)
    return bm_arr[order], bs_arr[order]


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:

    # ── 1. Load and filter data ───────────────────────────────────────────
    print("Loading data ...")
    df = pd.read_csv(DATA_PATH, parse_dates=[[0, 1, 2]], keep_date_col=False)
    df.rename(columns={"year_month_day": "date"}, inplace=True)
    df = df[df["latitude"].notna()].copy()

    df["hemisphere"]   = np.where(df["latitude"] >= 0, "north", "south")
    df["abs_latitude"] = df["latitude"].abs()
    df["year"]         = df["date"].dt.year
    df["decimal_year"] = df["date"].dt.year + df["date"].dt.dayofyear / 365.25

    df = df[df["correctedArea"] > 30].copy()
    cycles    = sorted(df["CYCLE"].dropna().unique())
    cycles_13 = [c for c in cycles if c >= 12]

    # ── 2. Compute t₀ at the 15° crossing ────────────────────────────────
    print("Computing cycle reference epochs (15° crossing) ...")
    t0_lookup: dict[tuple, float] = {}
    for (cyc, hemi), group in df.groupby(["CYCLE", "hemisphere"]):
        yearly_mean = group.groupby("year")["abs_latitude"].mean().sort_index()
        years, means = yearly_mean.index.values, yearly_mean.values
        t0 = find_15deg_crossing(years, means)
        if t0 is not None:
            t0_lookup[(cyc, hemi)] = t0

    all_hc = [(int(cyc), hemi) for cyc in cycles_13 for hemi in ["north", "south"]]
    no_t0  = [(c, h) for c, h in all_hc if (c, h) not in t0_lookup]
    print(f"  {len(t0_lookup)}/{len(all_hc)} pairs have a 15° crossing.")
    if no_t0:
        print(f"  No crossing: {no_t0}")

    df["t0"]  = df.apply(lambda r: t0_lookup.get((r["CYCLE"], r["hemisphere"]), np.nan), axis=1)
    df["tau"] = df["decimal_year"] - df["t0"]

    # ── 3. Fit universal mean path and refine t₀ ─────────────────────────
    print("Fitting universal mean path μ(τ) ...")
    N_BINS_MU = 20
    all_tau_bins, all_mu_bins = [], []
    hemicycle_bins: dict[tuple, tuple] = {}

    drop_s3: dict = {}
    for cyc in cycles_13:
        for hemi in ["north", "south"]:
            if (cyc, hemi) not in t0_lookup:
                continue
            mask   = (df["CYCLE"] == cyc) & (df["hemisphere"] == hemi) & df["tau"].notna()
            df_sel = df[mask]
            if len(df_sel) < 50:
                drop_s3[(int(cyc), hemi)] = f"sparse ({len(df_sel)} spots)"
                continue
            bt, bm = bin_latitudes(
                df_sel["tau"].values, df_sel["abs_latitude"].values, N_BINS_MU
            )
            if len(bt) < 5:
                drop_s3[(int(cyc), hemi)] = f"too few τ-bins ({len(bt)})"
                continue
            hemicycle_bins[(cyc, hemi)] = (bt, bm)
            all_tau_bins.extend(bt)
            all_mu_bins.extend(bm)

    popt_path, _ = curve_fit(
        exponential_decay, all_tau_bins, all_mu_bins, p0=[15.0, 5.0]
    )
    a_mu_univ, b_mu_univ = popt_path
    print(f"  μ(τ) = {a_mu_univ:.2f}° · exp(−τ / {b_mu_univ:.2f} yr)")

    t0_refined: dict[tuple, float] = {}
    for (cyc, hemi), (bt, bm) in hemicycle_bins.items():
        def residual(dt, _bt=bt, _bm=bm):
            return np.sum((_bm - exponential_decay(_bt - dt, a_mu_univ, b_mu_univ)) ** 2)
        res = minimize_scalar(residual, bounds=(-4, 4), method="bounded")
        t0_refined[(cyc, hemi)] = t0_lookup[(cyc, hemi)] + res.x

    df["t0_refined"]  = df.apply(
        lambda r: t0_refined.get((r["CYCLE"], r["hemisphere"]), np.nan), axis=1
    )
    df["tau_refined"] = df["decimal_year"] - df["t0_refined"]
    print(f"  {len(hemicycle_bins)}/{len(t0_lookup)} pairs survived binning.")
    if drop_s3:
        for key, reason in sorted(drop_s3.items()):
            print(f"    dropped {key}: {reason}")

    # ── 4. Per-cycle σ(μ) envelope fits ──────────────────────────────────
    print("Fitting per-cycle σ(μ) envelopes ...")
    N_BINS_SIGMA = 20
    sigma_fits: list[dict] = []
    drop_s4: dict = {}

    for cyc in cycles_13:
        for hemi in ["north", "south"]:
            if (cyc, hemi) not in t0_refined:
                continue
            mask   = (df["CYCLE"] == cyc) & (df["hemisphere"] == hemi) & df["tau_refined"].notna()
            df_sel = df[mask]
            if len(df_sel) < 50:
                drop_s4[(int(cyc), hemi)] = f"sparse ({len(df_sel)} spots)"
                continue
            bm_arr, bs_arr = bin_sigma(
                df_sel["tau_refined"].values, df_sel["abs_latitude"].values, N_BINS_SIGMA
            )
            if len(bm_arr) < 5:
                drop_s4[(int(cyc), hemi)] = f"too few bins ({len(bm_arr)})"
                continue
            try:
                p0 = [bs_arr.max(), bm_arr[np.argmax(bs_arr)], 5.0, 4.0]
                if MATCH_NOTEBOOK:
                    popt, _ = curve_fit(split_normal_amplitude, bm_arr, bs_arr,
                                        p0=p0, maxfev=10_000)
                else:
                    _bounds = ([0.5, 2.0, 0.5, 0.5], [20.0, 38.0, 100.0, 40.0])
                    popt, _ = curve_fit(split_normal_amplitude, bm_arr, bs_arr,
                                        p0=p0, bounds=_bounds, maxfev=10_000)
                A_f, mu_peak_f, sL_f, sR_f = popt
            except RuntimeError:
                drop_s4[(int(cyc), hemi)] = "curve_fit failed"
                continue
            sL_max, sR_max = (20.0, 20.0) if MATCH_NOTEBOOK else (100.0, 40.0)
            if not (0.5 < A_f < 20 and 2 < mu_peak_f < 38
                    and 0.5 < sL_f < sL_max and 0.5 < sR_f < sR_max):
                drop_s4[(int(cyc), hemi)] = (
                    f"implausible  A={A_f:.2f} μpk={mu_peak_f:.2f} "
                    f"σL={sL_f:.2f} σR={sR_f:.2f}"
                )
                continue
            sigma_fits.append(dict(
                cycle=cyc, hemisphere=hemi,
                A=A_f, mu_peak=mu_peak_f, sL=sL_f, sR=sR_f,
                bin_mu=bm_arr, bin_sigma=bs_arr,
            ))

    print(f"  {len(sigma_fits)} hemisphere-cycles with valid σ(μ) fits.")
    if drop_s4:
        for key, reason in sorted(drop_s4.items()):
            print(f"    dropped {key}: {reason}")

    # ── 5. Universal piecewise-linear σ(μ) envelope ──────────────────────
    print("Fitting universal piecewise-linear σ(μ) envelope ...")
    eq_mu    = [x for r in sigma_fits for x in r["bin_mu"][r["bin_mu"] <= r["mu_peak"]]]
    eq_sigma = [s for r in sigma_fits
                for s, m in zip(r["bin_sigma"], r["bin_mu"]) if m <= r["mu_peak"]]
    m_init, b_init = np.polyfit(eq_mu, eq_sigma, 1)
    m_i_init       = np.mean([-r["A"] / (2 * max(r["sL"], 1.0)) for r in sigma_fits])

    n_sf = len(sigma_fits)

    def piecewise_residuals(x: np.ndarray) -> float:
        m_sh, b_sh = x[0], x[1]
        total = 0.0
        for i, r in enumerate(sigma_fits):
            sigma_pred = piecewise_linear_sigma(
                r["bin_mu"], m_sh, b_sh, x[2 + 2 * i], x[3 + 2 * i]
            )
            total += np.sum((r["bin_sigma"] - sigma_pred) ** 2)
        return total

    x0_pl    = np.array([m_init, b_init] + [v for r in sigma_fits
                                              for v in (r["mu_peak"], m_i_init)])
    bounds_pl = ([(0.0, 2.0), (-5.0, 5.0)] + [(2.0, 38.0), (-5.0, 0.0)] * n_sf)
    opt_pl   = minimize(piecewise_residuals, x0_pl, method="L-BFGS-B", bounds=bounds_pl)

    m_shared_fit = float(opt_pl.x[0])
    b_shared_fit = float(opt_pl.x[1])
    print(f"  σ_eq(μ) = {m_shared_fit:.4f}·μ + {b_shared_fit:.4f}"
          f"  (zero crossing at μ = {-b_shared_fit / m_shared_fit:.2f}°)")

    pl_results: list[dict] = []
    for i, r in enumerate(sigma_fits):
        pl_results.append(dict(
            cycle=r["cycle"], hemisphere=r["hemisphere"],
            mu_peak=float(opt_pl.x[2 + 2 * i]),
            m_i=float(opt_pl.x[3 + 2 * i]),
            bin_mu=r["bin_mu"], bin_sigma=r["bin_sigma"],
        ))

    # ── 6. Cycle peak amplitudes ──────────────────────────────────────────
    # Notebook 06 Task 17 uses correctedArea > 50 MSH for the smoothed activity
    # curve — matching that threshold here keeps amplitude values comparable.
    print("Computing cycle peak amplitudes ...")
    SMOOTHING_DAYS = 365
    AMP_AREA_MIN   = 50.0 if MATCH_NOTEBOOK else 30.0
    df_amp = df[(df["CYCLE"].isin(cycles_13)) & (df["correctedArea"] > AMP_AREA_MIN)].copy()

    daily_north = df_amp[df_amp["hemisphere"] == "north"].groupby("date")["correctedArea"].sum()
    daily_south = df_amp[df_amp["hemisphere"] == "south"].groupby("date")["correctedArea"].sum()
    date_range  = pd.date_range(
        min(daily_north.index.min(), daily_south.index.min()),
        max(daily_north.index.max(), daily_south.index.max()),
        freq="D",
    )
    smooth_north = (daily_north.reindex(date_range, fill_value=0)
                    .rolling(SMOOTHING_DAYS, center=True, min_periods=SMOOTHING_DAYS // 3)
                    .mean())
    smooth_south = (daily_south.reindex(date_range, fill_value=0)
                    .rolling(SMOOTHING_DAYS, center=True, min_periods=SMOOTHING_DAYS // 3)
                    .mean())

    peak_records: list[dict] = []
    for cyc in cycles_13:
        cyc_dates = df[df["CYCLE"] == cyc]["date"]
        if len(cyc_dates) == 0:
            continue
        d_min, d_max = cyc_dates.min(), cyc_dates.max()
        for hemi, smooth in [("north", smooth_north), ("south", smooth_south)]:
            seg = smooth[(smooth.index >= d_min) & (smooth.index <= d_max)].dropna()
            if len(seg) == 0:
                continue
            peak_records.append(dict(cycle=int(cyc), hemisphere=hemi,
                                     peak_amplitude=float(seg.max())))

    peaks_df   = pd.DataFrame(peak_records)
    amp_lookup = {(row["cycle"], row["hemisphere"]): row["peak_amplitude"] / A_REF
                  for _, row in peaks_df.iterrows()}
    print(f"  Amplitudes normalised by A_REF = {A_REF:.0f} MSH  "
          f"(range {min(amp_lookup.values()):.2f}–{max(amp_lookup.values()):.2f})")

    # ── 7. Linear fits: wing parameters vs amplitude ──────────────────────
    print("Fitting wing parameters vs cycle amplitude ...")
    records_amp: list[dict] = []
    for r in pl_results:
        cyc, hemi = r["cycle"], r["hemisphere"]
        amp = amp_lookup.get((int(cyc), hemi))
        if amp is None or np.isnan(amp):
            continue
        mask   = (df["CYCLE"] == cyc) & (df["hemisphere"] == hemi) & df["tau_refined"].notna()
        df_sel = df[mask]
        if len(df_sel) == 0:
            continue
        records_amp.append(dict(
            cycle=int(cyc), hemisphere=hemi,
            amplitude=float(amp),
            mu_peak=float(r["mu_peak"]),
            m_i=float(r["m_i"]),
        ))

    df_amp_params = pd.DataFrame(records_amp)
    expected_hc   = {(int(r["cycle"]), r["hemisphere"]) for r in pl_results}
    fitted_hc     = {(row["cycle"], row["hemisphere"]) for row in records_amp}
    if missing := expected_hc - fitted_hc:
        print(f"  Dropped (no amplitude): {sorted(missing)}")
    print(f"  {len(records_amp)}/{len(pl_results)} pairs survive into optimisation.")

    # μ₀ per cycle: universal path evaluated at the earliest τ_refined observed.
    # This is the latitude above which the cycle has not yet started meaningfully.
    for rec in records_amp:
        cyc, hemi = rec["cycle"], rec["hemisphere"]
        mask    = (df["CYCLE"] == cyc) & (df["hemisphere"] == hemi) & df["tau_refined"].notna()
        df_sel  = df[mask]
        tau_start = df_sel["tau_refined"].min()
        rec["mu0"] = float(exponential_decay(tau_start, a_mu_univ, b_mu_univ))

    df_amp_params = pd.DataFrame(records_amp)

    init_fit_results: dict[str, tuple] = {}
    for col in ("mu0", "mu_peak", "m_i"):
        vals = df_amp_params[["amplitude", col]].dropna()
        x, y = vals["amplitude"].values, vals[col].values
        popt, _ = curve_fit(linear_fit, x, y, p0=[0.0, float(np.mean(y))])
        init_fit_results[col] = tuple(popt)

    a_mu0_init, b_mu0_init = init_fit_results["mu0"]

    print("  Initial amplitude-regression coefficients:")
    for col, (a, b) in init_fit_results.items():
        print(f"    {col:8s}(A) = {a:+.6f}·A + {b:.3f}")

    # ── 8. Raw observation data — no pre-computed derived quantities ──────────
    # Each entry: (amplitude, t0_refined, [(year_center, lats), ...])
    # All per-year quantities (τ, μ, σ, μ₀) are computed inside compute_nll.
    print("Extracting raw observation data ...")
    cycle_data: list[tuple] = []
    hc_index:   list[tuple] = []

    for rec in df_amp_params.to_dict("records"):
        cyc, hemi = rec["cycle"], rec["hemisphere"]
        amplitude = amp_lookup.get((int(cyc), hemi))
        if amplitude is None:
            continue
        t0_ref = t0_refined[(int(cyc), hemi)]
        df_ch  = df[(df["CYCLE"] == cyc) & (df["hemisphere"] == hemi)]
        year_blocks: list[tuple] = []
        for yr in sorted(df_ch["year"].unique()):
            year_center = yr + 0.5
            lats = df_ch.loc[df_ch["year"] == yr, "latitude"].abs().values
            if len(lats) >= 5:
                year_blocks.append((year_center, lats))
        if year_blocks:
            cycle_data.append((amplitude, t0_ref, year_blocks))
            hc_index.append((int(cyc), hemi))

    n_hc  = len(cycle_data)
    n_obs = sum(len(yb) for _, _, yb in cycle_data)
    print(f"  {n_hc} hemisphere-cycles  |  {n_obs} total yearly blocks"
          f"  (μ₀ gate is soft sigmoid, T = {MU0_SIGMOID_T}°)")

    # ── 8b. Baseline NLLs ────────────────────────────────────────────────
    print("Computing procedural baselines ...")

    # B0a: amplitude regression from Task 19 applied directly, no NLL optimisation
    nll_b0a = compute_nll(
        cycle_data,
        {"mu_peak": init_fit_results["mu_peak"], "m_i": init_fit_results["m_i"]},
        m_shared_fit, b_shared_fit,
        a_mu_univ, b_mu_univ,
        a_mu0_init, b_mu0_init,
    )
    nll_b0a_nb = compute_nll_hard_gate(
        cycle_data,
        {"mu_peak": init_fit_results["mu_peak"], "m_i": init_fit_results["m_i"]},
        m_shared_fit, b_shared_fit,
        a_mu_univ, b_mu_univ,
        a_mu0_init, b_mu0_init,
    )
    print(f"  B0a (amplitude regression, no NLL opt):  soft={nll_b0a:.5f}  hard={nll_b0a_nb:.5f}")

    # B0b: per-cycle mu_peak and m_i from the piecewise-linear joint fit (no regression)
    percycle_params = {
        (rec["cycle"], rec["hemisphere"]): (rec["mu_peak"], rec["m_i"], rec["mu0"])
        for rec in records_amp
    }
    nll_b0b = compute_nll_percycle(
        cycle_data, hc_index, percycle_params,
        m_shared_fit, b_shared_fit, a_mu_univ, b_mu_univ,
    )
    nll_b0b_nb = compute_nll_percycle_hard_gate(
        cycle_data, hc_index, percycle_params,
        m_shared_fit, b_shared_fit, a_mu_univ, b_mu_univ,
    )
    print(f"  B0b (per-cycle direct, no regression):   soft={nll_b0b:.5f}  hard={nll_b0b_nb:.5f}")

    # ── 9. Optimisation ───────────────────────────────────────────────────
    LBFGSB_OPTS = {"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8}

    bounds_8 = [
        (None, None), (None, None),   # a_mupeak, b_mupeak
        (None, None), (None, None),   # a_mi,     b_mi
        (0.0,  2.0),  (-5.0, 5.0),   # m_shared, b_shared
        (5.0, 30.0),  (1.0, 15.0),   # a_mu,     b_mu
    ]
    bounds_full = bounds_8 + [(-2.0, 2.0)] * n_hc

    # Stage 1 — 6p Nelder-Mead: amplitude regression + equatorial line, mean path fixed
    # Matches 06_2D_Wings Task 23: a_mu and b_mu are fixed at the universal path values
    # so the mu₀ hard threshold introduces no gradient discontinuity here.
    NM_OPTS   = {"maxiter": 50_000, "xatol": 1e-6, "fatol": 1e-7, "adaptive": True}
    N_STARTS  = 4

    def _obj_s1(x: np.ndarray) -> float:
        fr  = {"mu_peak": (float(x[0]), float(x[1])),
               "m_i":     (float(x[2]), float(x[3]))}
        return compute_nll(cycle_data, fr,
                           float(x[4]), float(x[5]),   # m_shared, b_shared
                           a_mu_univ, b_mu_univ,        # mean path fixed
                           a_mu0_init, b_mu0_init)

    x0_s1_base = np.array([*pack_4p(init_fit_results), m_shared_fit, b_shared_fit])
    rng_s1 = np.random.default_rng(42)
    opt_s1, nll_s1 = None, np.inf

    print(f"\nStage 1 — 6p Nelder-Mead (amplitude regression + equatorial line,"
          f" {N_STARTS} restarts) ...")
    for i in range(N_STARTS):
        x0_try = x0_s1_base if i == 0 else (
            x0_s1_base + rng_s1.standard_normal(6) * (np.abs(x0_s1_base) * 0.3 + 0.5)
        )
        opt = minimize(_obj_s1, x0_try, method="Nelder-Mead", options=NM_OPTS)
        nll = _obj_s1(opt.x)
        print(f"  restart {i}: NLL={nll:.5f}  converged={opt.success}")
        if nll < nll_s1:
            opt_s1, nll_s1 = opt, nll

    fr_s1    = {"mu_peak": (float(opt_s1.x[0]), float(opt_s1.x[1])),
                "m_i":     (float(opt_s1.x[2]), float(opt_s1.x[3]))}
    m_sh_s1  = float(opt_s1.x[4])
    b_sh_s1  = float(opt_s1.x[5])
    print(f"  best NLL = {nll_s1:.5f}")

    # Stage 2 — 8p Nelder-Mead: all global params including mean path
    # Using Nelder-Mead (not L-BFGS-B) because freeing a_mu and b_mu makes
    # the μ₀ hard threshold discontinuous w.r.t. those parameters.
    def _obj_s2(x: np.ndarray) -> float:
        fr, m_sh, b_sh, a_mu, b_mu = unpack_8p(x)
        return compute_nll(cycle_data, fr, m_sh, b_sh, a_mu, b_mu,
                           a_mu0_init, b_mu0_init)

    x0_s2_base = pack_global(fr_s1, m_sh_s1, b_sh_s1, a_mu_univ, b_mu_univ)
    rng_s2 = np.random.default_rng(7)
    opt_s2, nll_s2 = None, np.inf

    print(f"Stage 2 — 8p Nelder-Mead (all global params, {N_STARTS} restarts) ...")
    for i in range(N_STARTS):
        x0_try = x0_s2_base if i == 0 else (
            x0_s2_base + rng_s2.standard_normal(8) * (np.abs(x0_s2_base) * 0.2 + 0.3)
        )
        opt = minimize(_obj_s2, x0_try, method="Nelder-Mead", options=NM_OPTS)
        nll = _obj_s2(opt.x)
        print(f"  restart {i}: NLL={nll:.5f}  converged={opt.success}")
        if nll < nll_s2:
            opt_s2, nll_s2 = opt, nll

    fr_s2, m_sh_s2, b_sh_s2, a_mu_s2, b_mu_s2 = unpack_8p(opt_s2.x)
    print(f"  best NLL = {nll_s2:.5f}")

    # Stage 3 — (8 + N_hc)p L-BFGS-B: all global params + per-cycle timeshifts
    # Warm-started from S2; timeshifts initialised at zero.
    # a_mu and b_mu are already refined in S2, so their gradients are reliable here.
    print(f"Stage 3 — ({8 + n_hc})p L-BFGS-B"
          f" (global params + {n_hc} timeshifts) ...")

    def _obj_s3(x: np.ndarray) -> float:
        fr, m_sh, b_sh, a_mu, b_mu, dts = unpack_full(x, n_hc)
        return compute_nll(cycle_data, fr, m_sh, b_sh, a_mu, b_mu,
                           a_mu0_init, b_mu0_init, dts)

    x0_s3  = np.concatenate([opt_s2.x, np.zeros(n_hc)])
    opt_s3 = minimize(_obj_s3, x0_s3, method="L-BFGS-B",
                      bounds=bounds_full, options=LBFGSB_OPTS)
    fr_s3, m_sh_s3, b_sh_s3, a_mu_s3, b_mu_s3, dts_s3 = unpack_full(opt_s3.x, n_hc)
    nll_s3 = opt_s3.fun
    print(f"  converged={opt_s3.success}  iters={opt_s3.nit}  NLL={nll_s3:.5f}")

    # ── 10. Evaluate all stages under the hard-gate metric (notebook 06 scoring) ──
    nll_s1_nb = compute_nll_hard_gate(
        cycle_data, fr_s1, m_sh_s1, b_sh_s1, a_mu_univ, b_mu_univ,
        a_mu0_init, b_mu0_init,
    )
    nll_s2_nb = compute_nll_hard_gate(
        cycle_data, fr_s2, m_sh_s2, b_sh_s2, a_mu_s2, b_mu_s2,
        a_mu0_init, b_mu0_init,
    )
    nll_s3_nb = compute_nll_hard_gate(
        cycle_data, fr_s3, m_sh_s3, b_sh_s3, a_mu_s3, b_mu_s3,
        a_mu0_init, b_mu0_init, dts_s3,
    )

    # ── 11. Report results ────────────────────────────────────────────────
    W = 50
    print("\n" + "=" * (W + 28))
    print("  Optimisation Progression")
    print("=" * (W + 28))
    print(f"  {'Stage':<{W}s}  {'soft-gate':>10s}  {'hard-gate':>10s}")
    print(f"  {'(this script)':<{W}s}  {'(notebook 06)':>10s}")
    print(f"  {'-'*W}  {'-'*10}  {'-'*10}")
    print(f"  {'B0a: procedural — amplitude regression (Task 19)':<{W}s}"
          f"  {nll_b0a:>10.5f}  {nll_b0a_nb:>10.5f}")
    print(f"  {'B0b: procedural — per-cycle direct (no regression)':<{W}s}"
          f"  {nll_b0b:>10.5f}  {nll_b0b_nb:>10.5f}")
    print(f"  {'-'*W}  {'-'*10}  {'-'*10}")
    print(f"  {'S1: 6p NM  (amplitude regression + eq. line)':<{W}s}"
          f"  {nll_s1:>10.5f}  {nll_s1_nb:>10.5f}")
    print(f"  {'S2: 8p NM  (all global params)':<{W}s}"
          f"  {nll_s2:>10.5f}  {nll_s2_nb:>10.5f}")
    print(f"  {f'S3: {8+n_hc}p LBFGSB  (global + timeshifts)':<{W}s}"
          f"  {nll_s3:>10.5f}  {nll_s3_nb:>10.5f}")
    print()
    print(f"  S3 converged={opt_s3.success}  iters={opt_s3.nit}")
    print()
    print("  Fixed amplitude-dependent parameters (θ(A) = a·A + b,  A in MSH):")
    print(f"    μ₀(A)      = {a_mu0_init / A_REF:+.8f}·A  +  {b_mu0_init:.4f}°  ← upper lat cutoff (fixed)")
    print()
    print("  Optimised amplitude-dependent parameters:")
    print(f"    μ_peak(A)  = {fr_s3['mu_peak'][0] / A_REF:+.8f}·A  +  {fr_s3['mu_peak'][1]:.4f}°")
    print(f"    m_i(A)     = {fr_s3['m_i'][0] / A_REF:+.8f}·A  +  {fr_s3['m_i'][1]:.4f}")
    print()
    print("  Universal parameters:")
    print(f"    σ_eq(μ)    = {m_sh_s3:.4f}·μ  +  {b_sh_s3:.4f}°")
    print(f"    μ(τ)       = {a_mu_s3:.4f}° · exp(−τ / {b_mu_s3:.4f} yr)")
    print()
    print("  Per-cycle timeshift corrections Δt  [years]  (bounded ±2 yr):")
    print(f"    {'Cycle':>6s}  {'Hemi':<6s}  {'t0_refined':>12s}  {'Δt':>8s}  {'t0_final':>12s}  {'μ₀':>6s}")
    print(f"    {'-'*6}  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*12}  {'-'*6}")
    for (cyc, hemi), dt, (amp_i, _, _) in zip(hc_index, dts_s3, cycle_data):
        t0_i  = t0_refined[(cyc, hemi)]
        mu0_i = a_mu0_init * amp_i + b_mu0_init
        print(f"    {cyc:>6d}  {hemi:<6s}  {t0_i:>12.4f}  {dt:>+8.4f}  {t0_i + dt:>12.4f}  {mu0_i:>6.2f}°")
    print("=" * (W + 28))


if __name__ == "__main__":
    main()
