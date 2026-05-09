#!/usr/bin/env python3
"""
13_official_model_bootstrap.py

Produces "the official model" — the variant-A all-data fit declared as the
project's working model after the LOCO ablations (variants B, C, D, E) failed
to improve held-out NLL on this data.  Generates point-estimate parameters
with bootstrap 95 % confidence intervals.

Methodology
-----------
1. Run the full S1 → S2 → S3 pipeline on ALL hemicycles (no held-out fold).
   This produces the point estimates.
2. Bootstrap by m-out-of-n subsampling (m = 0.95 n, without replacement).
   Each draw refits the entire pipeline and records all parameters,
   including per-hemicycle effective reference epochs.
3. Reports per-parameter mean, std, median, and 2.5 / 97.5 percentile bounds.

Why hemicycle-level subsampling: the hemicycle is the unit you want to
generalise across — a future cycle the model has never seen.  Block-level
resampling would underestimate uncertainty by treating each year-block as
independent.

What the official model needs to be self-contained
--------------------------------------------------
The S1 mean-path fit refines each hemicycle's 15-degree crossing by δ_S1 to
align it with the universal exponential decay.  The S3 NLL stage adds a
further per-hemicycle correction Δt_S3.  The *effective* reference epoch
the model uses is

    t0_total = t0(15°-crossing) + δ_S1 + Δt_S3

This is the single quantity downstream code needs — once it has t0_total
for every fitted hemicycle, τ = decimal_year − t0_total feeds directly into
μ(τ), σ(μ), and the Gaussian likelihood with no further fitting required.
This script tracks all three components and reports t0_total alongside its
bootstrap distribution.

Outputs
-------
- model_specification.txt : human-readable parameter table with CIs, the
  full LOCO-justified design rationale, and the per-hemicycle t0_total
  table (the table downstream users actually consume).
- official_model.npz      : numpy archive containing the point estimates,
  bootstrap distributions, fold metadata, and the per-hemicycle t0_total
  arrays needed by butterflai_model.ButterflAIModel.
- bootstrap_diagnostics.png : per-parameter histograms showing the bootstrap
  distributions and the point estimate.

Runtime: ~30 minutes for N_BOOTSTRAP=200 on a modern laptop.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm as sp_norm
from scipy.optimize import curve_fit, minimize_scalar, minimize

# ── Paths and constants ────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
DATA_PATH  = REPO_ROOT / "data" / "composite_sunspot_groups_peak_area.csv"
OUT_DIR    = Path(__file__).resolve().parent
OUT_SPEC   = OUT_DIR / "model_specification.txt"
OUT_NPZ    = OUT_DIR / "official_model.npz"
OUT_PLOT   = OUT_DIR / "bootstrap_diagnostics.png"

A_REF = 1000.0
MU0_SIGMOID_T = 0.5
COVERAGE_MIN_FRAC = 0.95
COVERAGE_PENALTY = 1.0e4
A_MIN = 30.0   # MATCH_NOTEBOOK = False mode

N_BOOTSTRAP = 200
BOOTSTRAP_SEED = 20260505     # date-based, deterministic across reruns


# ══════════════════════════════════════════════════════════════════════════
# Section 1 — model primitives  (lifted from 11_loco_validation_powerlaw.py)
# ══════════════════════════════════════════════════════════════════════════

def exponential_decay(tau, a, b):
    return a * np.exp(-tau / b)


def split_normal_amplitude(mu, A, mu_peak, sigma_L, sigma_R):
    return np.where(
        mu >= mu_peak,
        A * np.exp(-0.5 * ((mu - mu_peak) / sigma_L) ** 2),
        A * np.exp(-0.5 * ((mu - mu_peak) / sigma_R) ** 2),
    )


def piecewise_linear_sigma(mu, m_shared, b_shared, mu_peak, m_i):
    """Variant-A σ(μ): universal equatorward arm, cycle-specific poleward."""
    sigma_at_peak = m_shared * mu_peak + b_shared
    b_poleward    = sigma_at_peak - m_i * mu_peak
    return np.clip(
        np.where(mu <= mu_peak,
                 m_shared * mu + b_shared,
                 m_i * mu + b_poleward),
        0.0, None,
    )


def linear_fit(x, a, b):
    return a * x + b


def find_15deg_crossing(years, means):
    below = means < 15.0
    if not below.any() or below.all():
        return None
    idx = int(np.argmax(below))
    if idx == 0:
        return None
    y0, mu0 = years[idx - 1], means[idx - 1]
    y1, mu1 = years[idx],     means[idx]
    return float(y0 + (15.0 - mu0) / (mu1 - mu0))


def bin_latitudes(tau_values, lat_values, n_bins):
    bins        = np.linspace(tau_values.min(), tau_values.max(), n_bins + 1)
    centers     = 0.5 * (bins[:-1] + bins[1:])
    bt, bm = [], []
    for i in range(n_bins):
        mask = (tau_values >= bins[i]) & (tau_values < bins[i + 1])
        lats = lat_values[mask]
        if len(lats) < 10:
            continue
        mu_fit, _ = sp_norm.fit(lats)
        bt.append(centers[i])
        bm.append(mu_fit)
    return np.array(bt), np.array(bm)


def bin_sigma(tau_values, lat_values, n_bins):
    bins = np.linspace(tau_values.min(), tau_values.max(), n_bins + 1)
    bm, bs = [], []
    for i in range(n_bins):
        mask = (tau_values >= bins[i]) & (tau_values < bins[i + 1])
        lats = lat_values[mask]
        if len(lats) < 10:
            continue
        mu_fit, sigma_fit = sp_norm.fit(lats)
        bm.append(mu_fit)
        bs.append(sigma_fit)
    bm_arr = np.array(bm); bs_arr = np.array(bs)
    order  = np.argsort(bm_arr)
    return bm_arr[order], bs_arr[order]


# ══════════════════════════════════════════════════════════════════════════
# Section 2 — NLL and objective construction
# ══════════════════════════════════════════════════════════════════════════

def nll_soft(cycle_data, fr, m_sh, b_sh, a_mu, b_mu, a_mu0, b_mu0, dts,
             return_details=False):
    a_mp, b_mp = fr["mu_peak"]; a_mi, b_mi = fr["m_i"]
    total = 0.0; eff = 0.0; cand = 0
    for i, (amp, t0_ref, yb) in enumerate(cycle_data):
        mu0_p = a_mu0 * amp + b_mu0
        mp_p  = a_mp  * amp + b_mp
        mi_p  = a_mi  * amp + b_mi
        dt    = float(dts[i]) if dts is not None else 0.0
        t0_e  = t0_ref + dt
        for yc, lats in yb:
            cand += 1
            tau = yc - t0_e
            mu  = a_mu * np.exp(-tau / b_mu)
            w = 1.0 / (1.0 + np.exp((mu - mu0_p) / MU0_SIGMOID_T))
            if w < 1e-4:
                continue
            sigma = piecewise_linear_sigma(mu, m_sh, b_sh, mp_p, mi_p)
            if sigma <= 0:
                continue
            ll = sp_norm.logpdf(lats, loc=mu, scale=sigma).mean()
            if not np.isfinite(ll):
                continue
            total -= w * ll
            eff   += w
    nll = total / eff if eff > 0 else 1e6
    if return_details:
        return nll, {"coverage": eff / max(cand, 1)}
    return nll


def nll_hard(cycle_data, fr, m_sh, b_sh, a_mu, b_mu, a_mu0, b_mu0, dts,
             return_details=False):
    a_mp, b_mp = fr["mu_peak"]; a_mi, b_mi = fr["m_i"]
    total = 0.0; n = 0; cand = 0
    for i, (amp, t0_ref, yb) in enumerate(cycle_data):
        mu0_p = a_mu0 * amp + b_mu0
        mp_p  = a_mp  * amp + b_mp
        mi_p  = a_mi  * amp + b_mi
        dt    = float(dts[i]) if dts is not None else 0.0
        t0_e  = t0_ref + dt
        for yc, lats in yb:
            cand += 1
            tau = yc - t0_e
            mu  = a_mu * np.exp(-tau / b_mu)
            if mu > mu0_p:
                continue
            sigma = piecewise_linear_sigma(mu, m_sh, b_sh, mp_p, mi_p)
            if sigma <= 0:
                continue
            ll = sp_norm.logpdf(lats, loc=mu, scale=sigma).mean()
            if not np.isfinite(ll):
                continue
            total -= ll
            n += 1
    nll = total / n if n > 0 else 1e6
    if return_details:
        return nll, {"coverage": n / max(cand, 1)}
    return nll


# ══════════════════════════════════════════════════════════════════════════
# Section 3 — All-data pipeline (variant A, full S1 → S2 → S3)
# ══════════════════════════════════════════════════════════════════════════

def prepare_data():
    """Load the data and compute amplitudes / 15-degree crossings.

    These quantities are computed from the full series and reused across
    bootstrap draws — they describe each hemicycle's own data and are not
    part of what's being bootstrapped.
    """
    df = pd.read_csv(DATA_PATH, parse_dates=[[0, 1, 2]], keep_date_col=False)
    df.rename(columns={"year_month_day": "date"}, inplace=True)
    df = df[df["latitude"].notna()].copy()
    df["hemisphere"]   = np.where(df["latitude"] >= 0, "north", "south")
    df["abs_latitude"] = df["latitude"].abs()
    df["year"]         = df["date"].dt.year
    df["decimal_year"] = df["date"].dt.year + df["date"].dt.dayofyear / 365.25
    df = df[df["correctedArea"] > 30].copy()

    cycles_13 = sorted([c for c in df["CYCLE"].dropna().unique() if c >= 12])

    t0_lookup = {}
    for (cyc, hemi), group in df.groupby(["CYCLE", "hemisphere"]):
        ym = group.groupby("year")["abs_latitude"].mean().sort_index()
        t0 = find_15deg_crossing(ym.index.values, ym.values)
        if t0 is not None:
            t0_lookup[(cyc, hemi)] = t0

    df_amp = df[(df["CYCLE"].isin(cycles_13)) & (df["correctedArea"] > A_MIN)].copy()
    daily_n = df_amp[df_amp["hemisphere"] == "north"].groupby("date")["correctedArea"].sum()
    daily_s = df_amp[df_amp["hemisphere"] == "south"].groupby("date")["correctedArea"].sum()
    drange  = pd.date_range(min(daily_n.index.min(), daily_s.index.min()),
                            max(daily_n.index.max(), daily_s.index.max()), freq="D")
    smooth_n = (daily_n.reindex(drange, fill_value=0)
                .rolling(365, center=True, min_periods=365 // 3).mean())
    smooth_s = (daily_s.reindex(drange, fill_value=0)
                .rolling(365, center=True, min_periods=365 // 3).mean())
    amp_lookup = {}
    for cyc in cycles_13:
        cdates = df[df["CYCLE"] == cyc]["date"]
        if len(cdates) == 0:
            continue
        d0, d1 = cdates.min(), cdates.max()
        for hemi, smooth in [("north", smooth_n), ("south", smooth_s)]:
            seg = smooth[(smooth.index >= d0) & (smooth.index <= d1)].dropna()
            if len(seg):
                amp_lookup[(int(cyc), hemi)] = float(seg.max()) / A_REF

    df["t0"]  = df.apply(lambda r: t0_lookup.get((r["CYCLE"], r["hemisphere"]), np.nan), axis=1)
    df["tau"] = df["decimal_year"] - df["t0"]
    return df, cycles_13, t0_lookup, amp_lookup


def fit_pipeline(df, cycles_13, t0_lookup, amp_lookup, training_keys):
    """Run the full S1 → S2 → S3 fit on a given set of training keys.

    Returns a dict with all fitted parameters, the cycle_data structure
    used during NLL evaluation, AND the t0_refined dict (so the caller
    can compute t0_total = t0_refined + Δt_S3 per hemicycle).  Returns
    None if the fold is infeasible (insufficient data).
    """
    # Section 3: universal mean path
    all_tau, all_mu = [], []
    hemicycle_bins  = {}
    for (cyc, hemi) in training_keys:
        if (cyc, hemi) not in t0_lookup:
            continue
        mask = (df["CYCLE"] == cyc) & (df["hemisphere"] == hemi) & df["tau"].notna()
        df_sel = df[mask]
        if len(df_sel) < 50:
            continue
        bt, bm = bin_latitudes(df_sel["tau"].values, df_sel["abs_latitude"].values, 20)
        if len(bt) < 5:
            continue
        hemicycle_bins[(cyc, hemi)] = (bt, bm)
        all_tau.extend(bt); all_mu.extend(bm)

    if len(all_tau) < 10:
        return None

    popt_path, _ = curve_fit(exponential_decay, all_tau, all_mu, p0=[15.0, 5.0])
    a_mu, b_mu = popt_path

    t0_refined = {}
    for (cyc, hemi), (bt, bm) in hemicycle_bins.items():
        def residual(dt, _bt=bt, _bm=bm):
            return np.sum((_bm - exponential_decay(_bt - dt, a_mu, b_mu)) ** 2)
        res = minimize_scalar(residual, bounds=(-4, 4), method="bounded")
        t0_refined[(cyc, hemi)] = t0_lookup[(cyc, hemi)] + res.x

    df_local = df.copy()
    df_local["t0_refined"]  = df_local.apply(
        lambda r: t0_refined.get((r["CYCLE"], r["hemisphere"]), np.nan), axis=1)
    df_local["tau_refined"] = df_local["decimal_year"] - df_local["t0_refined"]

    # Section 4: per-cycle σ(μ) split-normal fits
    sigma_fits = []
    for (cyc, hemi) in training_keys:
        if (cyc, hemi) not in t0_refined:
            continue
        mask = (df_local["CYCLE"] == cyc) & (df_local["hemisphere"] == hemi) & df_local["tau_refined"].notna()
        df_sel = df_local[mask]
        if len(df_sel) < 50:
            continue
        bm_arr, bs_arr = bin_sigma(df_sel["tau_refined"].values, df_sel["abs_latitude"].values, 20)
        if len(bm_arr) < 5:
            continue
        try:
            p0 = [bs_arr.max(), bm_arr[np.argmax(bs_arr)], 5.0, 4.0]
            _bounds = ([0.5, 2.0, 0.5, 0.5], [20.0, 38.0, 100.0, 40.0])
            popt, _ = curve_fit(split_normal_amplitude, bm_arr, bs_arr,
                                p0=p0, bounds=_bounds, maxfev=10_000)
            A_f, mp_f, sL_f, sR_f = popt
        except RuntimeError:
            continue
        if not (0.5 < A_f < 20 and 2 < mp_f < 38 and 0.5 < sL_f < 100 and 0.5 < sR_f < 40):
            continue
        sigma_fits.append(dict(cycle=cyc, hemisphere=hemi,
                               A=A_f, mu_peak=mp_f, sL=sL_f, sR=sR_f,
                               bin_mu=bm_arr, bin_sigma=bs_arr))

    if len(sigma_fits) < 5:
        return None

    # Section 5: universal piecewise-linear envelope
    eq_mu    = [x for r in sigma_fits for x in r["bin_mu"][r["bin_mu"] <= r["mu_peak"]]]
    eq_sigma = [s for r in sigma_fits for s, m in zip(r["bin_sigma"], r["bin_mu"]) if m <= r["mu_peak"]]
    m_init, b_init = np.polyfit(eq_mu, eq_sigma, 1)
    m_i_init = np.mean([-r["A"] / (2 * max(r["sL"], 1.0)) for r in sigma_fits])
    n_sf = len(sigma_fits)

    def pl_residuals(x):
        m_sh, b_sh = x[0], x[1]
        total = 0.0
        for i, r in enumerate(sigma_fits):
            sigma_pred = piecewise_linear_sigma(
                r["bin_mu"], m_sh, b_sh, x[2 + 2 * i], x[3 + 2 * i])
            total += np.sum((r["bin_sigma"] - sigma_pred) ** 2)
        return total

    x0_pl  = np.array([m_init, b_init] + [v for r in sigma_fits for v in (r["mu_peak"], m_i_init)])
    bnd_pl = ([(0.0, 2.0), (-5.0, 5.0)] + [(2.0, 38.0), (-5.0, 0.0)] * n_sf)
    opt_pl = minimize(pl_residuals, x0_pl, method="L-BFGS-B", bounds=bnd_pl)
    m_shared = float(opt_pl.x[0])
    b_shared = float(opt_pl.x[1])

    pl_results = []
    for i, r in enumerate(sigma_fits):
        pl_results.append(dict(
            cycle=r["cycle"], hemisphere=r["hemisphere"],
            mu_peak=float(opt_pl.x[2 + 2 * i]), m_i=float(opt_pl.x[3 + 2 * i]),
        ))

    # Section 7: amplitude regressions
    records_amp = []
    for r in pl_results:
        cyc, hemi = r["cycle"], r["hemisphere"]
        amp = amp_lookup.get((int(cyc), hemi))
        if amp is None or np.isnan(amp):
            continue
        records_amp.append(dict(cycle=int(cyc), hemisphere=hemi, amplitude=float(amp),
                                mu_peak=float(r["mu_peak"]), m_i=float(r["m_i"])))
    for rec in records_amp:
        cyc, hemi = rec["cycle"], rec["hemisphere"]
        mask = (df_local["CYCLE"] == cyc) & (df_local["hemisphere"] == hemi) & df_local["tau_refined"].notna()
        tau_start = df_local[mask]["tau_refined"].min()
        rec["mu0"] = float(exponential_decay(tau_start, a_mu, b_mu))

    df_amp_params = pd.DataFrame(records_amp)
    init_fit_results = {}
    for col in ("mu0", "mu_peak", "m_i"):
        vals = df_amp_params[["amplitude", col]].dropna()
        x, y = vals["amplitude"].values, vals[col].values
        popt, _ = curve_fit(linear_fit, x, y, p0=[0.0, float(np.mean(y))])
        init_fit_results[col] = tuple(popt)
    a_mu0_init, b_mu0_init = init_fit_results["mu0"]

    # Section 8: cycle_data
    cycle_data, hc_index = [], []
    for rec in df_amp_params.to_dict("records"):
        cyc, hemi = rec["cycle"], rec["hemisphere"]
        amp = amp_lookup.get((int(cyc), hemi))
        if amp is None:
            continue
        t0_ref = t0_refined[(int(cyc), hemi)]
        df_ch  = df_local[(df_local["CYCLE"] == cyc) & (df_local["hemisphere"] == hemi)]
        yb = []
        for yr in sorted(df_ch["year"].unique()):
            lats = df_ch.loc[df_ch["year"] == yr, "latitude"].abs().values
            if len(lats) >= 5:
                yb.append((yr + 0.5, lats))
        if yb:
            cycle_data.append((amp, t0_ref, yb))
            hc_index.append((int(cyc), hemi))

    n_hc = len(cycle_data)
    if n_hc < 5:
        return None

    # B0a coverage anchor
    nll_b0a, det_b0a = nll_hard(
        cycle_data,
        {"mu_peak": init_fit_results["mu_peak"],
         "m_i":     init_fit_results["m_i"]},
        m_shared, b_shared, a_mu, b_mu,
        a_mu0_init, b_mu0_init, None,
        return_details=True,
    )
    min_cov = COVERAGE_MIN_FRAC * det_b0a["coverage"]

    # Optimise: S2 (8p NM) → S3 (8 + N_hc, L-BFGS-B)
    def pack8(fr, m_sh, b_sh, a_mu_, b_mu_):
        return np.array([
            fr["mu_peak"][0], fr["mu_peak"][1],
            fr["m_i"][0],     fr["m_i"][1],
            m_sh, b_sh, a_mu_, b_mu_,
        ])

    def unpack8(x):
        return ({"mu_peak": (float(x[0]), float(x[1])),
                 "m_i":     (float(x[2]), float(x[3]))},
                float(x[4]), float(x[5]), float(x[6]), float(x[7]))

    def obj_s2(x):
        fr, m_sh, b_sh, a_mu_, b_mu_ = unpack8(x)
        nll, det = nll_soft(cycle_data, fr, m_sh, b_sh, a_mu_, b_mu_,
                            a_mu0_init, b_mu0_init, None, return_details=True)
        return nll + COVERAGE_PENALTY * max(0.0, min_cov - det["coverage"]) ** 2

    def obj_s3(x):
        fr, m_sh, b_sh, a_mu_, b_mu_ = unpack8(x[:8])
        dts = x[8:]
        nll, det = nll_soft(cycle_data, fr, m_sh, b_sh, a_mu_, b_mu_,
                            a_mu0_init, b_mu0_init, dts, return_details=True)
        return nll + COVERAGE_PENALTY * max(0.0, min_cov - det["coverage"]) ** 2

    x0 = pack8(init_fit_results, m_shared, b_shared, a_mu, b_mu)
    NM_OPTS = {"maxiter": 30_000, "xatol": 1e-5, "fatol": 1e-6, "adaptive": True}
    opt_s2 = minimize(obj_s2, x0, method="Nelder-Mead", options=NM_OPTS)

    bounds_full = [(None, None)] * 4 + [(0.0, 2.0), (-5.0, 5.0),
                                         (5.0, 30.0), (1.0, 15.0)] + [(-2.0, 2.0)] * n_hc
    x0_s3 = np.concatenate([opt_s2.x, np.zeros(n_hc)])
    opt_s3 = minimize(obj_s3, x0_s3, method="L-BFGS-B", bounds=bounds_full,
                      options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8})

    fr, m_sh, b_sh, a_mu_, b_mu_ = unpack8(opt_s3.x[:8])
    dts = opt_s3.x[8:]

    # Hard-gate scoring
    nll_hard_val, det_hard = nll_hard(
        cycle_data, fr, m_sh, b_sh, a_mu_, b_mu_,
        a_mu0_init, b_mu0_init, dts, return_details=True,
    )

    return dict(
        a_mu_peak=fr["mu_peak"][0], b_mu_peak=fr["mu_peak"][1],
        a_m_i=fr["m_i"][0],         b_m_i=fr["m_i"][1],
        m_shared=m_sh, b_shared=b_sh,
        a_mu=a_mu_, b_mu=b_mu_,
        a_mu0=a_mu0_init, b_mu0=b_mu0_init,
        delta_t0s=dts,
        t0_refined=t0_refined,        # NEW: dict {(cyc, hemi): refined epoch}
        hc_index=hc_index,
        nll_hard=nll_hard_val,
        coverage=det_hard["coverage"],
        n_hemicycles=n_hc,
    )


# ══════════════════════════════════════════════════════════════════════════
# Section 4 — Bootstrap
# ══════════════════════════════════════════════════════════════════════════

def bootstrap_fit(df, cycles_13, t0_lookup, amp_lookup, all_keys, n_draws):
    """N_BOOTSTRAP draws of resampled hemicycles.

    We use m-out-of-n subsampling without replacement (m = 0.95n) rather
    than classical bootstrap-with-replacement, because some pipeline
    components (per-cycle σ(μ) fits) are awkward to define when a cycle
    appears in the draw multiple times.  The 95 % subsample produces
    bootstrap-equivalent uncertainty estimates for the global parameters
    that are our main target.

    For per-hemicycle quantities (Δt_S3 and t0_total), each hemicycle is
    sampled in ~95 % of draws — its bootstrap distribution is built from
    those draws in which it was present.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    universe = list(all_keys)
    n_hemi   = len(universe)

    # Storage for global parameters
    keys_global = ["a_mu_peak", "b_mu_peak", "a_m_i", "b_m_i",
                   "m_shared", "b_shared", "a_mu", "b_mu",
                   "a_mu0", "b_mu0",
                   "nll_hard", "coverage"]
    samples = {k: [] for k in keys_global}
    samples["dt_summary"] = []   # (mean, std) of Δt's per draw — fold-summary
    samples["t0_total"]   = {}   # NEW: {(cyc, hemi): [t0_total values across draws]}

    n_failed = 0
    for draw in range(n_draws):
        # m-out-of-n without replacement
        n_keep = int(round(0.95 * n_hemi))
        idx = rng.choice(n_hemi, size=n_keep, replace=False)
        keys = set(universe[i] for i in idx)

        try:
            fit = fit_pipeline(df, cycles_13, t0_lookup, amp_lookup, keys)
            if fit is None:
                n_failed += 1
                continue
            for k in keys_global:
                samples[k].append(fit[k])
            dts = np.asarray(fit["delta_t0s"])
            samples["dt_summary"].append((float(np.mean(dts)), float(np.std(dts))))

            # Per-hemicycle t0_total = t0_refined (from THIS draw) + Δt_S3
            t0_ref_draw = fit["t0_refined"]
            for (cyc, hemi), dt in zip(fit["hc_index"], fit["delta_t0s"]):
                key = (int(cyc), str(hemi))
                t0_total_val = float(t0_ref_draw[(cyc, hemi)] + float(dt))
                samples["t0_total"].setdefault(key, []).append(t0_total_val)

            if (draw + 1) % 20 == 0:
                nll_so_far = np.array(samples["nll_hard"])
                print(f"  draw {draw+1:3d}/{n_draws}  "
                      f"NLL mean={nll_so_far.mean():.5f} std={nll_so_far.std():.5f}"
                      f"  failures={n_failed}")
        except Exception as e:
            n_failed += 1
            if n_failed % 5 == 0:
                print(f"  draw {draw+1}: failed ({e}); total failures={n_failed}")

    print(f"\nBootstrap completed: {n_draws - n_failed}/{n_draws} successful draws.")
    out = {k: np.array(v) for k, v in samples.items()
           if k not in ("dt_summary", "t0_total")}
    out["dt_summary"] = (np.array(samples["dt_summary"]) if samples["dt_summary"]
                        else np.zeros((0, 2)))
    out["t0_total"]   = samples["t0_total"]    # leave as dict for per-hc lookup
    return out


# ══════════════════════════════════════════════════════════════════════════
# Section 5 — Reporting
# ══════════════════════════════════════════════════════════════════════════

PARAM_DOCS = {
    "a_mu_peak":  ("μ_peak slope vs amplitude",          "° / unit-A"),
    "b_mu_peak":  ("μ_peak intercept (μ_peak at A=0)",   "°"),
    "a_m_i":      ("m_i slope vs amplitude (poleward σ slope)", "1 / unit-A"),
    "b_m_i":      ("m_i intercept (m_i at A=0)",         "—"),
    "m_shared":   ("Universal equatorward σ-line slope", "° / °"),
    "b_shared":   ("Universal equatorward σ-line intercept", "°"),
    "a_mu":       ("Mean-path amplitude μ(τ=0)",         "°"),
    "b_mu":       ("Mean-path e-folding time",           "yr"),
    "a_mu0":      ("μ_0 slope vs amplitude",             "° / unit-A"),
    "b_mu0":      ("μ_0 intercept (μ_0 at A=0)",         "°"),
    "nll_hard":   ("Hard-gate NLL (training)",           "nats / year-block"),
    "coverage":   ("Hard-gate coverage (training)",      "fraction"),
}


def summarize_t0_total(point_fit, samples_t0):
    """Build per-hemicycle t0_total table aligned with the point fit's hc_index.

    Returns a list of dicts with:
      cycle, hemisphere, t0_15deg, t0_refined, t0_total (point),
      boot_mean, boot_std, boot_lo, boot_hi, n_boot
    """
    rows = []
    for (cyc, hemi), dt_pt in zip(point_fit["hc_index"], point_fit["delta_t0s"]):
        key = (int(cyc), str(hemi))
        t0_ref_pt = point_fit["t0_refined"][(cyc, hemi)]
        t0_total_pt = t0_ref_pt + float(dt_pt)
        boot_arr = np.array(samples_t0.get(key, []))
        if boot_arr.size > 1:
            boot_mean = float(boot_arr.mean())
            boot_std  = float(boot_arr.std())
            boot_lo, boot_hi = (float(x) for x in np.percentile(boot_arr, [2.5, 97.5]))
        else:
            boot_mean = boot_std = boot_lo = boot_hi = float("nan")
        rows.append(dict(
            cycle=int(cyc), hemisphere=str(hemi),
            t0_refined=float(t0_ref_pt),
            delta_t_S3=float(dt_pt),
            t0_total=float(t0_total_pt),
            boot_mean=boot_mean, boot_std=boot_std,
            boot_lo=boot_lo, boot_hi=boot_hi,
            n_boot=int(boot_arr.size),
        ))
    return rows


def write_specification(point_fit, samples, t0_rows, t0_lookup):
    """Write a human-readable model specification with parameter CIs."""
    with open(OUT_SPEC, "w") as f:
        f.write("=" * 78 + "\n")
        f.write("ButterflAI Official Model Specification\n")
        f.write(f"Generated {datetime.now().isoformat(timespec='minutes')}\n")
        f.write("=" * 78 + "\n\n")

        f.write("MODEL CLASS\n")
        f.write("-" * 78 + "\n")
        f.write(
            "Variant A from the LOCO ablation series:\n"
            "  - Gaussian per-block log-likelihood\n"
            "  - Symmetric piecewise-linear σ(μ): universal equatorward arm,\n"
            "    cycle-specific poleward arm regressed on cycle amplitude\n"
            "  - Linear amplitude regressions:  param(A) = a · A + b\n"
            "    for μ_peak, m_i, μ_0\n"
            "  - Per-hemicycle Δt corrections, bounded ±2 yr\n"
            "  - Coverage penalty in the optimisation objective\n\n"
        )

        f.write("JUSTIFICATION FOR MODEL CLASS (LOCO results)\n")
        f.write("-" * 78 + "\n")
        f.write(
            "Three structural extensions were tested via leave-one-cycle-out\n"
            "validation against this baseline:\n"
            "  - B (Student-t likelihood + asymmetric σ): test NLL +0.007\n"
            "  - C (Student-t likelihood):                test NLL  0.000\n"
            "  - D (power-law-with-offset regressions):   test NLL +0.016\n"
            "All three failed to improve held-out NLL.  3.087 nats/year-block is\n"
            "interpreted as the generalisation floor of this model class.\n\n"
        )

        f.write("POINT ESTIMATES AND BOOTSTRAP 95 % CONFIDENCE INTERVALS\n")
        f.write("-" * 78 + "\n")
        f.write(
            f"All-data fit on N={point_fit['n_hemicycles']} hemicycles.\n"
            f"Bootstrap: {len(samples['nll_hard'])} draws of 95 % subsamples\n"
            "without replacement (m-out-of-n bootstrap, m = 0.95n).\n\n"
        )

        f.write(f"{'Parameter':<14} {'Point':>14} {'Mean':>14} {'Std':>10} "
                f"{'2.5 %':>14} {'97.5 %':>14}  Description\n")
        f.write("-" * 78 + "\n")
        for k in ["a_mu_peak", "b_mu_peak", "a_m_i", "b_m_i",
                  "m_shared", "b_shared", "a_mu", "b_mu",
                  "a_mu0", "b_mu0", "nll_hard", "coverage"]:
            p = point_fit[k]
            s = samples[k]
            mean = s.mean()
            std  = s.std()
            lo, hi = np.percentile(s, [2.5, 97.5])
            desc, units = PARAM_DOCS[k]
            f.write(f"{k:<14} {p:>14.6f} {mean:>14.6f} {std:>10.6f} "
                    f"{lo:>14.6f} {hi:>14.6f}  {desc} [{units}]\n")
        f.write("\n")

        f.write("PER-HEMICYCLE EFFECTIVE REFERENCE EPOCH  t0_total  (decimal year)\n")
        f.write("=" * 78 + "\n")
        f.write(
            "This is the canonical quantity downstream code consumes.\n"
            "  τ = decimal_year − t0_total  feeds μ(τ), σ(μ), and the Gaussian\n"
            "  likelihood with no further fitting required.\n\n"
            "Components:\n"
            "  t0_15deg    — raw 15° crossing of yearly-mean |latitude|\n"
            "  t0_refined  — t0_15deg + δ_S1 (S1 mean-path alignment)\n"
            "  Δt_S3       — S3 NLL refinement\n"
            "  t0_total    = t0_refined + Δt_S3   (THE deployment value)\n\n"
        )
        f.write(f"{'cyc':>4} {'hemi':<6} "
                f"{'t0_15deg':>10} {'t0_refined':>12} {'Δt_S3':>9} "
                f"{'t0_total':>12} {'boot_mean':>12} {'boot_std':>9} "
                f"{'2.5 %':>10} {'97.5 %':>10} {'n_boot':>7}\n")
        f.write("-" * 78 + "\n")
        for r in t0_rows:
            t0_raw = t0_lookup.get((r["cycle"], r["hemisphere"]), float("nan"))
            flag   = " (BND)" if abs(abs(r["delta_t_S3"]) - 2.0) < 0.05 else ""
            f.write(
                f"{r['cycle']:>4d} {r['hemisphere']:<6s} "
                f"{t0_raw:>10.4f} {r['t0_refined']:>12.4f} "
                f"{r['delta_t_S3']:>+9.4f} "
                f"{r['t0_total']:>12.4f} {r['boot_mean']:>12.4f} "
                f"{r['boot_std']:>9.4f} {r['boot_lo']:>10.4f} "
                f"{r['boot_hi']:>10.4f} {r['n_boot']:>7d}{flag}\n"
            )
        f.write("\n  (BND) marks Δt_S3 at the ±2 yr optimiser bound.\n\n")

        f.write("USAGE — closed form, no re-fitting required\n")
        f.write("=" * 78 + "\n")
        f.write(
            "For a hemicycle (cycle, hemisphere) with amplitude A (in MSH/1000):\n\n"
            "  t0_total   = (look up in table above)\n"
            "  τ          = decimal_year − t0_total                  [years]\n"
            "  μ(τ)       = a_mu · exp(−τ / b_mu)                    [degrees]\n"
            "  μ_0(A)     = a_mu0     · A + b_mu0\n"
            "  μ_peak(A)  = a_mu_peak · A + b_mu_peak\n"
            "  m_i(A)     = a_m_i     · A + b_m_i\n"
            "  σ(μ, A)    = piecewise_linear_sigma(\n"
            "                  μ, m_shared, b_shared, μ_peak(A), m_i(A))\n"
            "  Active?    = (μ(τ) ≤ μ_0(A))\n"
            "  p(|lat|)   = N(|lat| | μ(τ), σ(μ(τ), A))     when active\n"
            "             = 0                                otherwise\n\n"
            "For a NEW (unseen) hemicycle, t0_total is not in the table.\n"
            "Use t0_15deg from the smoothed yearly-mean |latitude| crossing\n"
            "of 15° as a proxy.  The Δt_S3 corrections are by construction\n"
            "unidentifiable for held-out cycles.\n"
        )
    print(f"Wrote {OUT_SPEC}")


def plot_bootstrap(point_fit, samples):
    """Histograms of bootstrap distributions with point estimates marked."""
    keys = ["a_mu_peak", "b_mu_peak", "a_m_i", "b_m_i",
            "m_shared", "b_shared", "a_mu", "b_mu",
            "a_mu0", "b_mu0", "nll_hard", "coverage"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.ravel()
    for ax, k in zip(axes, keys):
        s = samples[k]
        ax.hist(s, bins=30, color="tab:blue", alpha=0.7, edgecolor="white")
        ax.axvline(point_fit[k], color="tab:red", linewidth=2,
                   label=f"point: {point_fit[k]:.4f}")
        lo, hi = np.percentile(s, [2.5, 97.5])
        ax.axvline(lo, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(hi, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(f"{k}\n95 % CI: [{lo:.4f}, {hi:.4f}]", fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150)
    print(f"Wrote {OUT_PLOT}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("ButterflAI official model fit + bootstrap uncertainty")
    print("=" * 70)

    df, cycles_13, t0_lookup, amp_lookup = prepare_data()
    universe = sorted([k for k in t0_lookup
                       if k in amp_lookup and k[0] in cycles_13])
    print(f"Universe: {len(universe)} hemicycles")

    print("\n[1/2] Point-estimate fit on all data ...")
    point_fit = fit_pipeline(df, cycles_13, t0_lookup, amp_lookup, set(universe))
    if point_fit is None:
        raise RuntimeError("Full-data fit failed — cannot proceed.")
    print(f"  Hard-gate NLL = {point_fit['nll_hard']:.5f}")
    print(f"  Coverage      = {point_fit['coverage']:.3f}")
    print(f"  N hemicycles  = {point_fit['n_hemicycles']}")

    print(f"\n[2/2] Bootstrap with {N_BOOTSTRAP} draws ...")
    samples = bootstrap_fit(df, cycles_13, t0_lookup, amp_lookup,
                            universe, N_BOOTSTRAP)

    # Summarise per-hemicycle t0_total (point + bootstrap)
    t0_rows = summarize_t0_total(point_fit, samples["t0_total"])

    write_specification(point_fit, samples, t0_rows, t0_lookup)
    plot_bootstrap(point_fit, samples)

    # Build parallel arrays for the npz (clean for downstream loading)
    hc_cycles    = np.array([r["cycle"]      for r in t0_rows], dtype=int)
    hc_hemis     = np.array([r["hemisphere"] for r in t0_rows])
    t0_15deg_arr = np.array([t0_lookup[(r["cycle"], r["hemisphere"])]
                             for r in t0_rows], dtype=float)
    t0_refined_pt = np.array([r["t0_refined"]  for r in t0_rows], dtype=float)
    delta_t_S3_pt = np.array([r["delta_t_S3"]  for r in t0_rows], dtype=float)
    t0_total_pt   = np.array([r["t0_total"]    for r in t0_rows], dtype=float)
    t0_total_mean = np.array([r["boot_mean"]   for r in t0_rows], dtype=float)
    t0_total_std  = np.array([r["boot_std"]    for r in t0_rows], dtype=float)
    t0_total_lo   = np.array([r["boot_lo"]     for r in t0_rows], dtype=float)
    t0_total_hi   = np.array([r["boot_hi"]     for r in t0_rows], dtype=float)
    t0_total_n    = np.array([r["n_boot"]      for r in t0_rows], dtype=int)

    np.savez(
        OUT_NPZ,
        # ── Global parameters: point estimates ─────────────────────────
        point_a_mu_peak=point_fit["a_mu_peak"],
        point_b_mu_peak=point_fit["b_mu_peak"],
        point_a_m_i=point_fit["a_m_i"],
        point_b_m_i=point_fit["b_m_i"],
        point_m_shared=point_fit["m_shared"],
        point_b_shared=point_fit["b_shared"],
        point_a_mu=point_fit["a_mu"],
        point_b_mu=point_fit["b_mu"],
        point_a_mu0=point_fit["a_mu0"],
        point_b_mu0=point_fit["b_mu0"],
        point_nll_hard=point_fit["nll_hard"],
        point_coverage=point_fit["coverage"],
        # ── Per-hemicycle effective reference epoch ────────────────────
        hc_cycles=hc_cycles,
        hc_hemispheres=hc_hemis,
        point_t0_15deg=t0_15deg_arr,
        point_t0_refined=t0_refined_pt,
        point_delta_t_S3=delta_t_S3_pt,
        point_t0_total=t0_total_pt,
        boot_t0_total_mean=t0_total_mean,
        boot_t0_total_std=t0_total_std,
        boot_t0_total_lo=t0_total_lo,
        boot_t0_total_hi=t0_total_hi,
        boot_t0_total_n=t0_total_n,
        # ── Bootstrap distributions of global parameters ───────────────
        boot_a_mu_peak=samples["a_mu_peak"],
        boot_b_mu_peak=samples["b_mu_peak"],
        boot_a_m_i=samples["a_m_i"],
        boot_b_m_i=samples["b_m_i"],
        boot_m_shared=samples["m_shared"],
        boot_b_shared=samples["b_shared"],
        boot_a_mu=samples["a_mu"],
        boot_b_mu=samples["b_mu"],
        boot_a_mu0=samples["a_mu0"],
        boot_b_mu0=samples["b_mu0"],
        boot_nll_hard=samples["nll_hard"],
        boot_coverage=samples["coverage"],
        boot_dt_summary=samples["dt_summary"],
    )
    print(f"Wrote {OUT_NPZ}")
    print("\nDone.  Load downstream with:")
    print(f"  from butterflai_model import ButterflAIModel")
    print(f"  m = ButterflAIModel('{OUT_NPZ.name}')")
    print(f"  mu, sigma = m.gaussian(amplitude=1.5, tau=2.0)")


if __name__ == "__main__":
    main()