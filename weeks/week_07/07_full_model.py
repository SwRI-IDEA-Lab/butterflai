#!/usr/bin/env python3
"""
07_full_model.py

Direct fit of the full 10-parameter butterfly-wing model via L-BFGS-B.

The model describes the latitude distribution of sunspot emergence as a
symmetric Gaussian whose mean and spread both depend on cycle amplitude.
All ten free parameters are optimised jointly by minimising a sigmoid-
weighted negative log-likelihood (soft μ₀ threshold).

Parameter layout (10 scalars, one optimisation vector):
    [0]  a_mu0     — slope   of μ₀(A) = a_mu0·A + b_mu0
    [1]  b_mu0     — intercept
    [2]  a_mupeak  — slope   of μ_peak(A) = a_mupeak·A + b_mupeak
    [3]  b_mupeak  — intercept
    [4]  a_mi      — slope   of m_i(A) = a_mi·A + b_mi  (poleward wing slope)
    [5]  b_mi      — intercept
    [6]  m_shared  — universal equatorward line slope  σ(μ) = m_shared·μ + b_shared
    [7]  b_shared  — universal equatorward line intercept
    [8]  a_mu      — amplitude of universal mean path  μ(τ) = a_mu·exp(−τ/b_mu)
    [9]  b_mu      — e-folding time of universal mean path  [years]

Workflow
--------
1. Load sunspot-group data and compute absolute latitudes.
2. Align each hemisphere-cycle on a common time axis τ (15° crossing).
3. Fit the universal exponential mean path μ(τ) and refine τ origins.
4. Fit the per-cycle σ(μ) envelope (split-normal in latitude space).
5. Fit a universal piecewise-linear envelope shared across cycles.
6. Measure cycle peak amplitudes and correlate wing parameters with amplitude.
7. Optimise all 10 parameters jointly by minimising the soft-NLL.
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

# Sigmoid temperature [degrees] for the soft μ₀ threshold.
# Smaller T → sharper cutoff (approaches hard cutoff as T → 0).
MU0_SIGMOID_TEMP: float = 1.0

# Amplitude normalization reference [MSH].
# Dividing cycle peak amplitudes by A_REF keeps all 10 optimisation parameters
# O(1), preventing L-BFGS-B's finite-difference gradient estimation from
# drowning in floating-point noise on the tiny slope coefficients (a_mu0 etc.).
# Reported physical slopes must be divided by A_REF to recover °/MSH units.
A_REF: float = 1000.0


# ══════════════════════════════════════════════════════════════════════════
# Section 1 — Primitive model functions
# ══════════════════════════════════════════════════════════════════════════

def exponential_decay(tau: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Universal mean-latitude path as a function of cycle-aligned time.

    Parameters
    ----------
    tau : ndarray — time since cycle reference epoch  [years]
    a   : float   — initial latitude at τ = 0  [degrees]
    b   : float   — e-folding time  [years]

    Returns
    -------
    ndarray — mean sunspot emergence latitude  [degrees]
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
    Amplitude of σ(μ) described as a split-normal in latitude space.

    Used only for the per-cycle σ(μ) envelope fitting (Section 4);
    the emission model uses a symmetric Gaussian.

    Parameters
    ----------
    mu      : ndarray — mean emergence latitude  [degrees]
    A       : float   — peak spread amplitude  [degrees]
    mu_peak : float   — latitude at maximum spread  [degrees]
    sigma_L : float   — poleward half-width of the envelope  [degrees]
    sigma_R : float   — equatorward half-width of the envelope  [degrees]

    Returns
    -------
    ndarray — predicted spread σ  [degrees]
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
    Piecewise-linear model for σ(μ): universal equatorward line joined
    at μ_peak to a per-cycle poleward line, with continuity enforced.

    Equatorward branch (μ ≤ μ_peak):  σ = m_shared·μ + b_shared
    Poleward  branch  (μ > μ_peak):   σ = m_i·μ + b_i
        where b_i = σ(μ_peak) − m_i·μ_peak  (continuity).

    Parameters
    ----------
    mu       : ndarray — mean emergence latitude  [degrees]
    m_shared : float   — equatorward line slope  [degrees/degree]
    b_shared : float   — equatorward line intercept  [degrees]
    mu_peak  : float   — latitude of the σ peak for this cycle  [degrees]
    m_i      : float   — poleward line slope for this cycle  [degrees/degree]

    Returns
    -------
    ndarray — predicted spread σ (clipped to ≥ 0)  [degrees]
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

def pack_params(fit_results: dict) -> np.ndarray:
    """
    Flatten the first six amplitude-dependent parameters into a 1-D array.

    The six parameters are the (slope, intercept) pairs for μ₀(A),
    μ_peak(A), and m_i(A).

    Parameters
    ----------
    fit_results : dict — keys 'mu0', 'mu_peak', 'm_i', each a (slope, intercept) tuple.

    Returns
    -------
    ndarray, shape (6,)
    """
    return np.array([
        fit_results["mu0"][0],    fit_results["mu0"][1],
        fit_results["mu_peak"][0], fit_results["mu_peak"][1],
        fit_results["m_i"][0],    fit_results["m_i"][1],
    ])


def unpack_6p(x: np.ndarray) -> dict:
    """
    Unpack the six amplitude-dependent parameters.

    Returns
    -------
    dict with keys 'mu0', 'mu_peak', 'm_i', each a (slope, intercept) tuple.
    """
    return {
        "mu0":     (float(x[0]), float(x[1])),
        "mu_peak": (float(x[2]), float(x[3])),
        "m_i":     (float(x[4]), float(x[5])),
    }


def unpack_full_10(x: np.ndarray) -> tuple:
    """
    Unpack the full 10-parameter optimisation vector.

    Layout:
        x[0:2]   — (a_mu0,    b_mu0)    μ₀(A) coefficients
        x[2:4]   — (a_mupeak, b_mupeak) μ_peak(A) coefficients
        x[4:6]   — (a_mi,     b_mi)    wing m_i(A) coefficients
        x[6:8]   — (m_shared, b_shared) universal equatorward line
        x[8:10]  — (a_mu,     b_mu)    universal mean-path coefficients

    Returns
    -------
    fit_results : dict — amplitude-dependent parameter pairs
    m_shared    : float
    b_shared    : float
    a_mu        : float  [degrees]
    b_mu        : float  [years]
    """
    fit_results = {
        "mu0":     (float(x[0]),  float(x[1])),
        "mu_peak": (float(x[2]),  float(x[3])),
        "m_i":     (float(x[4]),  float(x[5])),
    }
    return (
        fit_results,
        float(x[6]),   # m_shared
        float(x[7]),   # b_shared
        float(x[8]),   # a_mu
        float(x[9]),   # b_mu
    )


# ══════════════════════════════════════════════════════════════════════════
# Section 3 — NLL objective
# ══════════════════════════════════════════════════════════════════════════

def compute_soft_nll(
    cycle_data: list,
    fit_results: dict,
    m_shared: float,
    b_shared: float,
    a_mu: float,
    b_mu: float,
    sigmoid_temp: float = MU0_SIGMOID_TEMP,
) -> float:
    """
    Sigmoid-weighted negative log-likelihood for the butterfly-wing model.

    Each yearly observation block is weighted by how far below μ₀ the local
    mean latitude is.  The sigmoid weight is:

        w(μ, μ₀; T) = 1 / (1 + exp((μ − μ₀) / T))

    which smoothly suppresses contributions from early-cycle years where
    μ > μ₀ (the equatorial drift has not yet begun for the active window).
    This soft boundary makes the NLL differentiable everywhere, enabling
    gradient-based optimisation.

    The emission model is a symmetric Gaussian: lat ~ N(μ, σ(μ)).

    Parameters
    ----------
    cycle_data   : list of (A, years_data) — pre-extracted data per hemisphere-cycle.
                   A = peak amplitude [MSH]; years_data = list of (τ, lats) tuples.
    fit_results  : dict — amplitude-dependent (slope, intercept) pairs for
                   'mu0', 'mu_peak', and 'm_i'.
    m_shared     : float — equatorward line slope  [degrees/degree]
    b_shared     : float — equatorward line intercept  [degrees]
    a_mu         : float — mean-path amplitude  [degrees]
    b_mu         : float — mean-path e-folding time  [years]
    sigmoid_temp : float — temperature T of the μ₀ sigmoid  [degrees]

    Returns
    -------
    float — weighted NLL normalised by total weight (nats per effective year).
            Lower is better.  Returns 1e6 if no valid observations exist.
    """
    a_mu0,    b_mu0    = fit_results["mu0"]
    a_mupeak, b_mupeak = fit_results["mu_peak"]
    a_mi,     b_mi     = fit_results["m_i"]

    total_weighted_nll = 0.0
    total_weight       = 0.0

    for amplitude, years_data in cycle_data:
        mu0_pred    = a_mu0    * amplitude + b_mu0
        mupeak_pred = a_mupeak * amplitude + b_mupeak
        mi_pred     = a_mi     * amplitude + b_mi

        for tau, latitudes in years_data:
            mu = a_mu * np.exp(-tau / b_mu)

            exponent = np.clip((mu - mu0_pred) / sigmoid_temp, -500.0, 500.0)
            weight   = 1.0 / (1.0 + np.exp(exponent))
            if weight < 1e-6:
                continue

            sigma = piecewise_linear_sigma(mu, m_shared, b_shared, mupeak_pred, mi_pred)
            if sigma <= 0.0:
                continue

            mean_logpdf = sp_norm.logpdf(latitudes, loc=mu, scale=sigma).mean()

            total_weighted_nll -= weight * mean_logpdf
            total_weight       += weight

    if total_weight < 1e-6:
        return 1e6
    return total_weighted_nll / total_weight


def compute_hard_nll(
    cycle_data: list,
    fit_results: dict,
    m_shared: float,
    b_shared: float,
    a_mu: float,
    b_mu: float,
) -> float:
    """
    Optimisation NLL: evaluate all post-15°-crossing data (τ ≥ 0).

    Uses a fixed, data-driven threshold (τ = 0, the 15°-crossing epoch) instead
    of the free μ₀ parameter.  This makes the threshold non-gameable: the
    optimiser cannot lower the NLL by adjusting μ₀ to exclude inconvenient
    blocks.  Normalises by the number of included blocks (fixed for a given
    dataset), so the metric is comparable across runs.
    """
    a_mupeak, b_mupeak = fit_results["mu_peak"]
    a_mi,     b_mi     = fit_results["m_i"]
    total, n = 0.0, 0
    for amplitude, years_data in cycle_data:
        mupeak_p = a_mupeak * amplitude + b_mupeak
        mi_p     = a_mi     * amplitude + b_mi
        for tau, lats in years_data:
            if tau < 0:          # fixed threshold: pre-crossing data excluded
                continue
            mu = a_mu * np.exp(-tau / b_mu)
            sigma = piecewise_linear_sigma(mu, m_shared, b_shared, mupeak_p, mi_p)
            if sigma <= 0.0:
                continue
            total -= sp_norm.logpdf(lats, loc=mu, scale=sigma).mean()
            n     += 1
    return total / n if n > 0 else 1e6


def compute_global_nll(
    cycle_data: list,
    fit_results: dict,
    m_shared: float,
    b_shared: float,
    a_mu: float,
    b_mu: float,
) -> float:
    """
    Scoreboard NLL: hard-threshold with normalisation by included blocks only.

    This is the metric reported in the weekly notebooks and used to compare
    models across all optimisation stages.  It is NOT used as an optimisation
    objective (use compute_hard_nll for that, which penalises data exclusion).
    """
    a_mu0,    b_mu0    = fit_results["mu0"]
    a_mupeak, b_mupeak = fit_results["mu_peak"]
    a_mi,     b_mi     = fit_results["m_i"]
    total, n = 0.0, 0
    for amplitude, years_data in cycle_data:
        mu0_p    = a_mu0    * amplitude + b_mu0
        mupeak_p = a_mupeak * amplitude + b_mupeak
        mi_p     = a_mi     * amplitude + b_mi
        for tau, lats in years_data:
            mu = a_mu * np.exp(-tau / b_mu)
            if mu > mu0_p:
                continue
            sigma = piecewise_linear_sigma(mu, m_shared, b_shared, mupeak_p, mi_p)
            if sigma <= 0.0:
                continue
            total -= sp_norm.logpdf(lats, loc=mu, scale=sigma).mean()
            n     += 1
    return total / n if n > 0 else 1e6


# ══════════════════════════════════════════════════════════════════════════
# Section 4 — Data preparation helpers
# ══════════════════════════════════════════════════════════════════════════

def find_15deg_crossing(years: np.ndarray, means: np.ndarray) -> float | None:
    """
    Linearly interpolate the decimal year at which the yearly mean
    absolute latitude crosses 15° from above.

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
    Divide τ into equal-width bins and fit a Gaussian to each bin's
    latitude distribution.

    Parameters
    ----------
    tau_values : ndarray — cycle-aligned times  [years]
    lat_values : ndarray — absolute latitudes   [degrees]
    n_bins     : int     — number of time bins

    Returns
    -------
    bin_tau   : ndarray — bin-center τ values for bins with ≥ 10 spots
    bin_mu    : ndarray — Gaussian mean (μ̂) for each qualifying bin  [degrees]
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
    Divide τ into equal-width bins and fit a Gaussian to each bin, returning
    both the mean latitude and the spread for that bin.

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
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
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

    # ── 1. Load and filter sunspot-group data ─────────────────────────────
    print("Loading data ...")
    df = pd.read_csv(DATA_PATH, parse_dates=[[0, 1, 2]], keep_date_col=False)
    df.rename(columns={"year_month_day": "date"}, inplace=True)
    df = df[df["latitude"].notna()].copy()

    df["hemisphere"]   = np.where(df["latitude"] >= 0, "north", "south")
    df["abs_latitude"] = df["latitude"].abs()
    df["year"]         = df["date"].dt.year
    df["decimal_year"] = df["date"].dt.year + df["date"].dt.dayofyear / 365.25

    # Keep only well-observed groups and cycles with reliable data
    df = df[df["correctedArea"] > 30].copy()
    cycles    = sorted(df["CYCLE"].dropna().unique())
    cycles_13 = [c for c in cycles if c >= 12]      # cycles 12+ have good coverage

    # ── 2. Align cycles: find t₀ at the 15° crossing ─────────────────────
    print("Computing cycle reference epochs (15° crossing) ...")
    t0_lookup: dict[tuple, float] = {}
    for (cyc, hemi), group in df.groupby(["CYCLE", "hemisphere"]):
        yearly_mean = group.groupby("year")["abs_latitude"].mean().sort_index()
        years, means = yearly_mean.index.values, yearly_mean.values
        t0 = find_15deg_crossing(years, means)
        if t0 is not None:
            t0_lookup[(cyc, hemi)] = t0

    df["t0"]  = df.apply(lambda r: t0_lookup.get((r["CYCLE"], r["hemisphere"]), np.nan), axis=1)
    df["tau"] = df["decimal_year"] - df["t0"]

    # ── 3. Fit universal mean path μ(τ) and refine t₀ ────────────────────
    print("Fitting universal mean path μ(τ) ...")
    N_BINS_MU = 20
    all_tau_bins, all_mu_bins = [], []
    hemicycle_bins: dict[tuple, tuple] = {}

    for cyc in cycles_13:
        for hemi in ["north", "south"]:
            if (cyc, hemi) not in t0_lookup:
                continue
            mask   = (df["CYCLE"] == cyc) & (df["hemisphere"] == hemi) & df["tau"].notna()
            df_sel = df[mask]
            if len(df_sel) < 50:
                continue
            bt, bm = bin_latitudes(
                df_sel["tau"].values,
                df_sel["abs_latitude"].values,
                N_BINS_MU,
            )
            if len(bt) < 5:
                continue
            hemicycle_bins[(cyc, hemi)] = (bt, bm)
            all_tau_bins.extend(bt)
            all_mu_bins.extend(bm)

    popt_path, _ = curve_fit(
        exponential_decay, all_tau_bins, all_mu_bins, p0=[15.0, 5.0]
    )
    a_mu_univ, b_mu_univ = popt_path
    print(f"  μ(τ) = {a_mu_univ:.2f}° · exp(−τ / {b_mu_univ:.2f} yr)")

    # Refine t₀ by minimising residuals of each cycle against the universal path
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

    # ── 4. Fit per-cycle σ(μ) envelope ───────────────────────────────────
    print("Fitting per-cycle σ(μ) envelopes ...")
    N_BINS_SIGMA = 20
    sigma_fits: list[dict] = []

    for cyc in cycles_13:
        for hemi in ["north", "south"]:
            if (cyc, hemi) not in t0_refined:
                continue
            mask   = (df["CYCLE"] == cyc) & (df["hemisphere"] == hemi) & df["tau_refined"].notna()
            df_sel = df[mask]
            if len(df_sel) < 50:
                continue
            bm_arr, bs_arr = bin_sigma(
                df_sel["tau_refined"].values,
                df_sel["abs_latitude"].values,
                N_BINS_SIGMA,
            )
            if len(bm_arr) < 5:
                continue
            try:
                p0 = [bs_arr.max(), bm_arr[np.argmax(bs_arr)], 5.0, 4.0]
                popt, _ = curve_fit(split_normal_amplitude, bm_arr, bs_arr,
                                    p0=p0, maxfev=10_000)
                A_f, mu_peak_f, sL_f, sR_f = popt
            except RuntimeError:
                continue
            # Reject physically implausible fits
            if not (0.5 < A_f < 20 and 2 < mu_peak_f < 38
                    and 0.5 < sL_f < 20 and 0.5 < sR_f < 20):
                continue
            sigma_fits.append(dict(
                cycle=cyc, hemisphere=hemi,
                A=A_f, mu_peak=mu_peak_f, sL=sL_f, sR=sR_f,
                bin_mu=bm_arr, bin_sigma=bs_arr,
            ))

    print(f"  {len(sigma_fits)} hemisphere-cycles with valid σ(μ) fits.")

    # ── 5. Fit universal piecewise-linear envelope ────────────────────────
    print("Fitting universal piecewise-linear σ(μ) envelope ...")

    # Initial guess for the equatorward (shared) line from a simple polyfit
    eq_mu    = [x for r in sigma_fits for x in r["bin_mu"][r["bin_mu"] <= r["mu_peak"]]]
    eq_sigma = [s for r in sigma_fits
                for s, m in zip(r["bin_sigma"], r["bin_mu"]) if m <= r["mu_peak"]]
    m_init, b_init = np.polyfit(eq_mu, eq_sigma, 1)
    m_i_init       = np.mean([-r["A"] / (2 * max(r["sL"], 1.0)) for r in sigma_fits])

    n_hc = len(sigma_fits)

    def piecewise_residuals(x: np.ndarray) -> float:
        """Joint squared-error over all hemisphere-cycles for the piecewise model."""
        m_sh, b_sh = x[0], x[1]
        total = 0.0
        for i, r in enumerate(sigma_fits):
            mu_peak_i = x[2 + 2 * i]
            m_i_i     = x[3 + 2 * i]
            sigma_pred = piecewise_linear_sigma(r["bin_mu"], m_sh, b_sh, mu_peak_i, m_i_i)
            total     += np.sum((r["bin_sigma"] - sigma_pred) ** 2)
        return total

    x0_pl   = np.array([m_init, b_init] + [v for r in sigma_fits
                                             for v in (r["mu_peak"], m_i_init)])
    bounds_pl = ([(0.0, 2.0), (-5.0, 5.0)]
                 + [(2.0, 38.0), (-5.0, 0.0)] * n_hc)
    opt_pl  = minimize(piecewise_residuals, x0_pl, method="L-BFGS-B", bounds=bounds_pl)

    m_shared_fit = float(opt_pl.x[0])
    b_shared_fit = float(opt_pl.x[1])
    print(f"  σ_eq(μ) = {m_shared_fit:.4f}·μ + {b_shared_fit:.4f}"
          f"  (zero crossing at μ = {-b_shared_fit / m_shared_fit:.2f}°)")

    # Collect per-cycle peak latitudes from the piecewise fit
    pl_results: list[dict] = []
    for i, r in enumerate(sigma_fits):
        pl_results.append(dict(
            cycle=r["cycle"], hemisphere=r["hemisphere"],
            mu_peak=float(opt_pl.x[2 + 2 * i]),
            m_i=float(opt_pl.x[3 + 2 * i]),
            bin_mu=r["bin_mu"], bin_sigma=r["bin_sigma"],
        ))

    # ── 6. Measure cycle peak amplitudes ─────────────────────────────────
    print("Computing cycle peak amplitudes ...")
    SMOOTHING_DAYS = 365

    df_amp = df[(df["CYCLE"] >= 12) & (df["correctedArea"] > 50)].copy()

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
            peak_records.append(dict(
                cycle=int(cyc), hemisphere=hemi,
                peak_amplitude=float(seg.max()),
            ))

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
        mask    = (df["CYCLE"] == cyc) & (df["hemisphere"] == hemi) & df["tau_refined"].notna()
        df_sel  = df[mask]
        if len(df_sel) == 0:
            continue
        tau_start = df_sel["tau_refined"].min()
        mu0_val   = float(exponential_decay(tau_start, a_mu_univ, b_mu_univ))
        records_amp.append(dict(
            cycle=int(cyc), hemisphere=hemi,
            amplitude=float(amp),
            mu0=mu0_val,
            mu_peak=float(r["mu_peak"]),
            m_i=float(r["m_i"]),
        ))

    df_amp_params = pd.DataFrame(records_amp)

    # Initial amplitude-dependent coefficients from simple linear regression
    init_fit_results: dict[str, tuple] = {}
    for col in ("mu0", "mu_peak", "m_i"):
        vals = df_amp_params[["amplitude", col]].dropna()
        x, y = vals["amplitude"].values, vals[col].values
        popt, _ = curve_fit(linear_fit, x, y, p0=[0.0, float(np.mean(y))])
        init_fit_results[col] = tuple(popt)

    print("  Initial amplitude-regression coefficients:")
    for col, (a, b) in init_fit_results.items():
        print(f"    {col:8s}(A) = {a:+.6f}·A + {b:.3f}")

    # ── 8. Build per-cycle observation cache ──────────────────────────────
    # Pre-extract latitudes grouped by (cycle, hemisphere, year) to avoid
    # re-filtering the full DataFrame inside the optimisation loop.
    print("Extracting observation cache ...")
    cycle_data: list[tuple] = []
    for rec in df_amp_params.to_dict("records"):
        cyc, hemi = rec["cycle"], rec["hemisphere"]
        amplitude = amp_lookup.get((int(cyc), hemi))
        if amplitude is None:
            continue
        df_ch = df[(df["CYCLE"] == cyc) & (df["hemisphere"] == hemi)]
        years_data: list[tuple] = []
        for yr in sorted(df_ch["year"].unique()):
            tau  = (yr + 0.5) - t0_refined[(int(cyc), hemi)]
            lats = df_ch.loc[df_ch["year"] == yr, "latitude"].abs().values
            if len(lats) >= 5:
                years_data.append((tau, lats))
        if years_data:
            cycle_data.append((amplitude, years_data))

    n_hc_total = len(cycle_data)
    n_obs_total = sum(len(yd) for _, yd in cycle_data)
    print(f"  {n_hc_total} hemisphere-cycles  |  {n_obs_total} yearly blocks")

    # ── 9. Progressive warm-start optimisation ───────────────────────────────
    # Jumping cold to 10 parameters reliably stalls in a poor local minimum.
    # Walking up the parameter tree lets each stage hand its converged solution
    # to the next, so the final 10p solve starts near the global basin.
    #
    #   S1  6p  Nelder-Mead  hard μ₀  — broad gradient-free search
    #   S2  6p  L-BFGS-B     soft μ₀  — gradient refinement
    #   S3  8p  L-BFGS-B     soft μ₀ + eq. line free
    #   S4  8p  L-BFGS-B     soft μ₀ + μ(τ) path free  (branches off S2)
    #   S5  10p L-BFGS-B     all parameters free (warm-started from S3+S4)

    LBFGSB_OPTS = {"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8}

    # ── Stage 1: 6p Nelder-Mead (hard μ₀), multi-start ───────────────────────
    N_S1_STARTS = 4
    print(f"\nStage 1 — 6p Nelder-Mead (hard μ₀), {N_S1_STARTS} restarts ...")
    def _obj_s1(x: np.ndarray) -> float:
        fr = unpack_6p(x)
        return compute_hard_nll(cycle_data, fr, m_shared_fit, b_shared_fit,
                                a_mu_univ, b_mu_univ)
    rng_s1   = np.random.default_rng(42)
    x0_base  = pack_params(init_fit_results)
    opt_s1, nll_s1 = None, np.inf
    for i in range(N_S1_STARTS):
        if i == 0:
            x0_try = x0_base.copy()
        else:
            noise  = rng_s1.standard_normal(6) * (np.abs(x0_base) * 0.3 + 0.5)
            x0_try = x0_base + noise
        opt = minimize(_obj_s1, x0_try, method="Nelder-Mead",
                       options={"maxiter": 50_000, "xatol": 1e-6,
                                "fatol": 1e-7, "adaptive": True})
        nll = _obj_s1(opt.x)
        print(f"  restart {i}: NLL={nll:.5f}  converged={opt.success}")
        if nll < nll_s1:
            opt_s1, nll_s1 = opt, nll
    fr_s1 = unpack_6p(opt_s1.x)
    gnll_s1 = compute_global_nll(cycle_data, fr_s1, m_shared_fit, b_shared_fit,
                                  a_mu_univ, b_mu_univ)
    print(f"  best NLL={nll_s1:.5f}  global-NLL={gnll_s1:.5f}")

    # ── Stage 2: 6p L-BFGS-B (soft μ₀) ──────────────────────────────────────
    print("Stage 2 — 6p L-BFGS-B (soft μ₀) ...")
    def _obj_s2(x: np.ndarray) -> float:
        fr = unpack_6p(x)
        return compute_soft_nll(cycle_data, fr, m_shared_fit, b_shared_fit,
                                a_mu_univ, b_mu_univ, MU0_SIGMOID_TEMP)
    opt_s2 = minimize(_obj_s2, pack_params(fr_s1), method="L-BFGS-B",
                      options=LBFGSB_OPTS)
    fr_s2  = unpack_6p(opt_s2.x)
    nll_s2 = opt_s2.fun
    gnll_s2 = compute_global_nll(cycle_data, fr_s2, m_shared_fit, b_shared_fit,
                                  a_mu_univ, b_mu_univ)
    print(f"  converged={opt_s2.success}  iters={opt_s2.nit}  soft-NLL={nll_s2:.5f}  global-NLL={gnll_s2:.5f}")

    # ── Stage 3: 8p L-BFGS-B (soft μ₀ + free eq. line) ──────────────────────
    print("Stage 3 — 8p L-BFGS-B (soft μ₀ + eq. line) ...")
    def _obj_s3(x: np.ndarray) -> float:
        fr = unpack_6p(x[:6])
        return compute_soft_nll(cycle_data, fr, float(x[6]), float(x[7]),
                                a_mu_univ, b_mu_univ, MU0_SIGMOID_TEMP)
    opt_s3 = minimize(
        _obj_s3,
        np.append(pack_params(fr_s2), [m_shared_fit, b_shared_fit]),
        method="L-BFGS-B",
        bounds=[(None, None)] * 6 + [(0.0, 2.0), (-5.0, 5.0)],
        options=LBFGSB_OPTS,
    )
    fr_s3        = unpack_6p(opt_s3.x[:6])
    m_sh_s3, b_sh_s3 = float(opt_s3.x[6]), float(opt_s3.x[7])
    nll_s3 = opt_s3.fun
    gnll_s3 = compute_global_nll(cycle_data, fr_s3, m_sh_s3, b_sh_s3,
                                  a_mu_univ, b_mu_univ)
    print(f"  converged={opt_s3.success}  iters={opt_s3.nit}  soft-NLL={nll_s3:.5f}  global-NLL={gnll_s3:.5f}")

    # ── Stage 4: 8p L-BFGS-B (soft μ₀ + free μ(τ) path) ─────────────────────
    # Branches off S2, not S3: the eq-line and path subspaces are explored
    # independently so their best solutions can be merged in S5.
    print("Stage 4 — 8p L-BFGS-B (soft μ₀ + μ(τ) path) ...")
    def _obj_s4(x: np.ndarray) -> float:
        fr = unpack_6p(x[:6])
        return compute_soft_nll(cycle_data, fr, m_shared_fit, b_shared_fit,
                                float(x[6]), float(x[7]), MU0_SIGMOID_TEMP)
    opt_s4 = minimize(
        _obj_s4,
        np.append(pack_params(fr_s2), [a_mu_univ, b_mu_univ]),
        method="L-BFGS-B",
        bounds=[(None, None)] * 6 + [(5.0, 30.0), (1.0, 15.0)],
        options=LBFGSB_OPTS,
    )
    fr_s4        = unpack_6p(opt_s4.x[:6])
    a_mu_s4, b_mu_s4 = float(opt_s4.x[6]), float(opt_s4.x[7])
    nll_s4 = opt_s4.fun
    gnll_s4 = compute_global_nll(cycle_data, fr_s4, m_shared_fit, b_shared_fit,
                                  a_mu_s4, b_mu_s4)
    print(f"  converged={opt_s4.success}  iters={opt_s4.nit}  soft-NLL={nll_s4:.5f}  global-NLL={gnll_s4:.5f}")

    # ── Stage 5: 10p L-BFGS-B (all parameters free), multi-start ────────────
    # Two candidate starting points:
    #   (a) Warm chain — S4's path combined with S3's eq. line
    #   (b) Direct S1 route — best Nelder-Mead basin injected straight into 10p
    print("Stage 5 — 10p L-BFGS-B (all parameters free), multi-start ...")
    def objective(x: np.ndarray) -> float:
        fr, m_sh, b_sh, a_mu, b_mu = unpack_full_10(x)
        return compute_soft_nll(cycle_data, fr, m_sh, b_sh, a_mu, b_mu,
                                MU0_SIGMOID_TEMP)
    bounds_10 = [
        (None, None), (None, None),  # a_mu0,    b_mu0
        (None, None), (None, None),  # a_mupeak, b_mupeak
        (None, None), (None, None),  # a_mi,     b_mi
        (0.0,  2.0),  (-5.0, 5.0),  # m_shared, b_shared
        (5.0, 30.0),  (1.0, 15.0),  # a_mu,     b_mu
    ]
    s5_starts = [
        ("warm chain (S3+S4)",
         np.array([*pack_params(fr_s4), m_sh_s3, b_sh_s3, a_mu_s4, b_mu_s4])),
        ("direct S1 route",
         np.array([*pack_params(fr_s1), m_shared_fit, b_shared_fit,
                   a_mu_univ, b_mu_univ])),
    ]
    result, nll_soft_final_s5 = None, np.inf
    for label, x0 in s5_starts:
        opt = minimize(objective, x0, method="L-BFGS-B", bounds=bounds_10,
                       options=LBFGSB_OPTS)
        soft_t = opt.fun
        print(f"  [{label}]  soft-NLL={soft_t:.5f}  converged={opt.success}")
        if soft_t < nll_soft_final_s5:
            result, nll_soft_final_s5 = opt, soft_t
    fr_s5, m_sh_s5, b_sh_s5, a_mu_s5, b_mu_s5 = unpack_full_10(result.x)
    gnll_s5 = compute_global_nll(cycle_data, fr_s5, m_sh_s5, b_sh_s5, a_mu_s5, b_mu_s5)

    # ── Stage 5b: 10p Nelder-Mead (global NLL), warm-started from S5 ─────────
    # S5 minimised the soft NLL; S5b refines directly against the scoreboard
    # metric (global NLL), starting from S5's result so that mu_0 is already
    # in a physically reasonable basin from the soft optimisation.
    # Nelder-Mead is used because compute_global_nll has a discontinuity at
    # μ = μ₀ that breaks gradient-based methods.
    print("Stage 5b — 10p Nelder-Mead (global NLL), warm-started from S5 ...")
    def _obj_global_10p(x: np.ndarray) -> float:
        fr, m_sh, b_sh, a_mu, b_mu = unpack_full_10(x)
        return compute_global_nll(cycle_data, fr, m_sh, b_sh, a_mu, b_mu)

    opt_s5b = minimize(
        _obj_global_10p, result.x, method="Nelder-Mead",
        bounds=bounds_10,
        options={"maxiter": 100_000, "xatol": 1e-6, "fatol": 1e-7, "adaptive": True},
    )
    fr_10, m_sh_10, b_sh_10, a_mu_10, b_mu_10 = unpack_full_10(opt_s5b.x)
    gnll_s5b = compute_global_nll(cycle_data, fr_10, m_sh_10, b_sh_10, a_mu_10, b_mu_10)
    print(f"  converged={opt_s5b.success}  iters={opt_s5b.nit}  global-NLL={gnll_s5b:.5f}")

    # ── 10. Report results ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Optimisation Progression")
    print("=" * 70)
    print(f"  {'Stage':<42s}  {'global NLL':>10s}")
    print(f"  {'-'*42}  {'-'*10}")
    print(f"  {'S1: 6p Nelder-Mead (hard μ₀)':<42s}  {gnll_s1:>10.5f}")
    print(f"  {'S2: 6p L-BFGS-B (soft μ₀)':<42s}  {gnll_s2:>10.5f}")
    print(f"  {'S3: 8p (soft μ₀ + eq. line)':<42s}  {gnll_s3:>10.5f}")
    print(f"  {'S4: 8p (soft μ₀ + path)':<42s}  {gnll_s4:>10.5f}")
    print(f"  {'─ ' * 27}")
    print(f"  {'S5: 10p L-BFGS-B (soft NLL)':<42s}  {gnll_s5:>10.5f}")
    print(f"  {'S5b: 10p Nelder-Mead (global NLL)':<42s}  {gnll_s5b:>10.5f}")
    print(f"  {'─ ' * 27}")
    print()
    print(f"  Converged (S5)  : {result.success}  iters={result.nit}")
    print(f"  Converged (S5b) : {opt_s5b.success}  iters={opt_s5b.nit}")
    print()
    print(f"  Amplitude-dependent parameters (θ(A) = a·A + b,  A in MSH):")
    print(f"    μ₀(A)      = {fr_10['mu0'][0] / A_REF:+.8f}·A  +  {fr_10['mu0'][1]:.4f}°")
    print(f"    μ_peak(A)  = {fr_10['mu_peak'][0] / A_REF:+.8f}·A  +  {fr_10['mu_peak'][1]:.4f}°")
    print(f"    m_i(A)     = {fr_10['m_i'][0] / A_REF:+.8f}·A  +  {fr_10['m_i'][1]:.4f}")
    print()
    print("  Universal parameters:")
    print(f"    σ_eq(μ)    = {m_sh_10:.4f}·μ  +  {b_sh_10:.4f}°")
    print(f"    μ(τ)       = {a_mu_10:.4f}° · exp(−τ / {b_mu_10:.4f} yr)")
    print("=" * 65)


if __name__ == "__main__":
    main()
