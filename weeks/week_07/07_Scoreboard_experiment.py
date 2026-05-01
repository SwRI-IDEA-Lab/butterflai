#!/usr/bin/env python3
"""
07_Scoreboard_experiment.py

End-to-end NLL optimisation of the solar butterfly wing model.
Sections 1–5: data prep, mean-path fit, σ(μ) model, piecewise envelope.
Tasks 17–19: cycle amplitude, peak timing, shape-amplitude relationships.
Tasks 23+: progressive parameter expansions (6p → 8p → 10p → 12p) and
the model quality scoreboard.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm as sp_norm
from scipy.optimize import curve_fit, minimize_scalar, minimize

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "composite_sunspot_groups_peak_area.csv"
MU0_TEMP  = 1.0  # sigmoid temperature T (degrees) for soft μ₀ threshold


# ── Pure functions — no side effects, safe to import ──────────────────────

def exp_decay(tau, a, b):
    return a * np.exp(-tau / b)


def split_normal_mu(mu, A, mu_peak, s_L, s_R):
    return np.where(
        mu >= mu_peak,
        A * np.exp(-0.5 * ((mu - mu_peak) / s_L) ** 2),
        A * np.exp(-0.5 * ((mu - mu_peak) / s_R) ** 2),
    )


def piecewise_linear_wing(mu, m_shared, b_shared, mu_peak, m_i):
    """Universal equatorward line + per-cycle poleward line, joined at mu_peak."""
    sigma_peak = m_shared * mu_peak + b_shared
    b_per = sigma_peak - m_i * mu_peak
    return np.clip(
        np.where(mu <= mu_peak, m_shared * mu + b_shared, m_i * mu + b_per),
        0.0, None
    )


def split_norm_logpdf(x, mu, sigma_L, sigma_R):
    """
    Log-pdf of a split-normal distribution.
    sigma_L: poleward spread (x ≥ μ); sigma_R: equatorward spread (x < μ).
    Normalisation: A = √(2/π) / (σ_L + σ_R).
    """
    log_A = 0.5 * np.log(2.0 / np.pi) - np.log(sigma_L + sigma_R)
    return np.where(
        x >= mu,
        log_A - 0.5 * ((x - mu) / sigma_L) ** 2,
        log_A - 0.5 * ((x - mu) / sigma_R) ** 2,
    )


def linear_fit(x, a, b):
    return a * x + b


# ── Parameter vector helpers ───────────────────────────────────────────────
# Layout: [a_mu0, b_mu0, a_mupeak, b_mupeak, a_mi, b_mi, ...]

def _pack(fr):
    return np.array([fr["mu0"][0], fr["mu0"][1],
                     fr["mu_peak"][0], fr["mu_peak"][1],
                     fr["m_i"][0], fr["m_i"][1]])


def _unpack(x):
    return {"mu0": (x[0], x[1]), "mu_peak": (x[2], x[3]), "m_i": (x[4], x[5])}


def _unpack_eq8(x):
    # 6p + [m_shared, b_shared]
    return _unpack(x), float(x[6]), float(x[7])


def _unpack_ext(x):
    # 6p + [a_mu, b_mu]
    return _unpack(x), float(x[6]), float(x[7])


def _unpack_10(x):
    # 6p + [a_mu, b_mu, a_mi_R, b_mi_R]
    return _unpack(x), float(x[6]), float(x[7]), (float(x[8]), float(x[9]))


def _unpack_12(x):
    # 6p + [m_shared, b_shared, a_mu, b_mu, a_mi_R, b_mi_R]
    return (_unpack(x), float(x[6]), float(x[7]),
            float(x[8]), float(x[9]), (float(x[10]), float(x[11])))


# ══════════════════════════════════════════════════════════════════════════

def main():

    # ══════════════════════════════════════════════════════════════════════
    # Section 1 — Load data
    # ══════════════════════════════════════════════════════════════════════
    df = pd.read_csv(DATA_PATH, parse_dates=[[0, 1, 2]], keep_date_col=False)
    df.rename(columns={"year_month_day": "date"}, inplace=True)
    df = df[df["latitude"].notna()].copy()

    df["hemisphere"]   = df["latitude"].apply(lambda v: "north" if v >= 0 else "south")
    df["abs_latitude"] = df["latitude"].abs()
    df["year"]         = df["date"].dt.year
    df["decimal_year"] = df["date"].dt.year + df["date"].dt.dayofyear / 365.25
    df = df[df["correctedArea"] > 30].copy()

    cycles    = sorted(df["CYCLE"].dropna().unique())
    cycles_13 = [c for c in cycles if c >= 12]

    # ══════════════════════════════════════════════════════════════════════
    # Section 2 — Standardise time: align cycles at the 15° crossing (τ)
    # ══════════════════════════════════════════════════════════════════════
    t0_lookup = {}
    for (cyc, hemi), group in df.groupby(["CYCLE", "hemisphere"]):
        yearly_mean = group.groupby("year")["abs_latitude"].mean().sort_index()
        years, means = yearly_mean.index.values, yearly_mean.values
        below = means < 15.0
        if not below.any() or below.all():
            continue
        idx1 = np.argmax(below)
        if idx1 == 0:
            continue
        y0, mu0 = years[idx1 - 1], means[idx1 - 1]
        y1, mu1 = years[idx1],     means[idx1]
        t0_lookup[(cyc, hemi)] = y0 + (15.0 - mu0) / (mu1 - mu0)

    df["t0"]  = df.apply(lambda r: t0_lookup.get((r["CYCLE"], r["hemisphere"]), np.nan), axis=1)
    df["tau"] = df["decimal_year"] - df["t0"]

    # ══════════════════════════════════════════════════════════════════════
    # Section 3 — Fit universal mean path μ(τ) and refine t₀
    # ══════════════════════════════════════════════════════════════════════
    N_BINS_13 = 20
    all_tau_bins, all_mu_bins = [], []
    hemicycle_bins_13 = {}

    for cyc in cycles_13:
        for hemi in ["north", "south"]:
            if (cyc, hemi) not in t0_lookup:
                continue
            mask   = (df["CYCLE"] == cyc) & (df["hemisphere"] == hemi) & df["tau"].notna()
            df_sel = df[mask]
            if len(df_sel) < 50:
                continue
            t_min, t_max = df_sel["tau"].min(), df_sel["tau"].max()
            bins        = np.linspace(t_min, t_max, N_BINS_13 + 1)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bt, bm = [], []
            for i in range(N_BINS_13):
                lats_bin = df_sel.loc[
                    (df_sel["tau"] >= bins[i]) & (df_sel["tau"] < bins[i + 1]),
                    "abs_latitude"].values
                if len(lats_bin) < 10:
                    continue
                mu_f, _ = sp_norm.fit(lats_bin)
                bt.append(bin_centers[i]); bm.append(mu_f)
            if len(bt) < 5:
                continue
            bt = np.array(bt); bm = np.array(bm)
            hemicycle_bins_13[(cyc, hemi)] = (bt, bm)
            all_tau_bins.extend(bt); all_mu_bins.extend(bm)

    popt_global, _ = curve_fit(exp_decay, all_tau_bins, all_mu_bins, p0=[15.0, 5.0])
    a_mu_univ, b_mu_univ = popt_global
    print(f"Universal μ(τ):  a = {a_mu_univ:.2f}°   b = {b_mu_univ:.2f} yr")

    t0_refined = {}
    for (cyc, hemi), (bt, bm) in hemicycle_bins_13.items():
        def _res(dt, _bt=bt, _bm=bm):
            return np.sum((_bm - exp_decay(_bt - dt, a_mu_univ, b_mu_univ)) ** 2)
        res = minimize_scalar(_res, bounds=(-4, 4), method="bounded")
        t0_refined[(cyc, hemi)] = t0_lookup[(cyc, hemi)] + res.x

    df["t0_refined"]  = df.apply(
        lambda r: t0_refined.get((r["CYCLE"], r["hemisphere"]), np.nan), axis=1)
    df["tau_refined"] = df["decimal_year"] - df["t0_refined"]

    # ══════════════════════════════════════════════════════════════════════
    # Section 4 — Model σ(μ): spread as a function of mean latitude
    # ══════════════════════════════════════════════════════════════════════
    N_BINS_15 = 20
    results_15 = []

    for cyc in cycles_13:
        for hemi in ["north", "south"]:
            if (cyc, hemi) not in t0_refined:
                continue
            mask   = (df["CYCLE"] == cyc) & (df["hemisphere"] == hemi) & df["tau_refined"].notna()
            df_sel = df[mask]
            if len(df_sel) < 50:
                continue
            t_min, t_max = df_sel["tau_refined"].min(), df_sel["tau_refined"].max()
            bins        = np.linspace(t_min, t_max, N_BINS_15 + 1)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bm_list, bs_list = [], []
            for i in range(N_BINS_15):
                lats_bin = df_sel.loc[
                    (df_sel["tau_refined"] >= bins[i]) & (df_sel["tau_refined"] < bins[i + 1]),
                    "abs_latitude"].values
                if len(lats_bin) < 10:
                    continue
                mu_f, sigma_f = sp_norm.fit(lats_bin)
                bm_list.append(mu_f); bs_list.append(sigma_f)
            if len(bm_list) < 5:
                continue
            bm_arr = np.array(bm_list); bs_arr = np.array(bs_list)
            sidx = np.argsort(bm_arr)
            bm_arr, bs_arr = bm_arr[sidx], bs_arr[sidx]
            try:
                p0 = [bs_arr.max(), bm_arr[np.argmax(bs_arr)], 5.0, 4.0]
                popt, _ = curve_fit(split_normal_mu, bm_arr, bs_arr, p0=p0, maxfev=10_000)
                A_f, mu_peak_f, sL_f, sR_f = popt
            except RuntimeError:
                continue
            if not (0.5 < A_f < 20 and 2 < mu_peak_f < 38 and 0.5 < sL_f < 20 and 0.5 < sR_f < 20):
                continue
            results_15.append(dict(
                cycle=cyc, hemisphere=hemi,
                A=A_f, mu_peak=mu_peak_f, sL=sL_f, sR=sR_f,
                bin_mu=bm_arr, bin_sigma=bs_arr,
            ))

    print(f"Fitted {len(results_15)} hemisphere-cycles.")

    # ══════════════════════════════════════════════════════════════════════
    # Section 5 — Universal piecewise-linear envelope
    # ══════════════════════════════════════════════════════════════════════
    eq_mu, eq_sigma = [], []
    for r in results_15:
        mask = r["bin_mu"] <= r["mu_peak"]
        eq_mu.extend(r["bin_mu"][mask]); eq_sigma.extend(r["bin_sigma"][mask])
    m_init, b_init = np.polyfit(eq_mu, eq_sigma, 1)
    m_i_init = np.mean([-r["A"] / (2 * max(r["sL"], 1.0)) for r in results_15])

    n_hc = len(results_15)

    def residuals_pl(x):
        m_sh, b_sh = x[0], x[1]
        return sum(
            np.sum((r["bin_sigma"] - piecewise_linear_wing(
                r["bin_mu"], m_sh, b_sh, x[2 + 2*i], x[3 + 2*i])) ** 2)
            for i, r in enumerate(results_15)
        )

    x0_pl = np.array([m_init, b_init] + [val for r in results_15
                                          for val in (r["mu_peak"], m_i_init)])
    bounds_pl = list(zip(
        [0.0, -5.0] + [2.0, -5.0] * n_hc,
        [2.0,  5.0] + [38.0, 0.0] * n_hc,
    ))
    opt_pl = minimize(residuals_pl, x0_pl, method="L-BFGS-B", bounds=bounds_pl)

    m_shared_fit, b_shared_fit = opt_pl.x[0], opt_pl.x[1]
    print(f"Universal line:  σ(μ) = {m_shared_fit:.4f}·μ + {b_shared_fit:.4f}")
    print(f"Zero crossing at μ = {-b_shared_fit / m_shared_fit:.2f}°")

    results_pl = []
    for i, r in enumerate(results_15):
        mu_peak_i = opt_pl.x[2 + 2*i]
        m_i       = opt_pl.x[3 + 2*i]
        results_pl.append(dict(
            cycle=r["cycle"], hemisphere=r["hemisphere"],
            m_shared=m_shared_fit, b_shared=b_shared_fit,
            mu_peak=mu_peak_i, m_i=m_i,
            bin_mu=r["bin_mu"], bin_sigma=r["bin_sigma"],
        ))

    rmse_16 = np.sqrt(np.mean([
        np.mean((r["bin_sigma"] - piecewise_linear_wing(
            r["bin_mu"], r["m_shared"], r["b_shared"], r["mu_peak"], r["m_i"])) ** 2)
        for r in results_pl
    ]))
    print(f"RMSE — piecewise-linear joint: {rmse_16:.3f}°")

    # ══════════════════════════════════════════════════════════════════════
    # Tasks 17+18 — Cycle amplitude and peak timing
    # ══════════════════════════════════════════════════════════════════════
    WIN_18  = 365
    df_amp  = df[(df["CYCLE"] >= 12) & (df["correctedArea"] > 50)].copy()

    daily_north = df_amp[df_amp["hemisphere"] == "north"].groupby("date")["correctedArea"].sum()
    daily_south = df_amp[df_amp["hemisphere"] == "south"].groupby("date")["correctedArea"].sum()
    date_range  = pd.date_range(
        min(daily_north.index.min(), daily_south.index.min()),
        max(daily_north.index.max(), daily_south.index.max()),
        freq="D",
    )
    daily_north  = daily_north.reindex(date_range, fill_value=0)
    daily_south  = daily_south.reindex(date_range, fill_value=0)
    smooth_north = daily_north.rolling(WIN_18, center=True, min_periods=WIN_18 // 3).mean()
    smooth_south = daily_south.rolling(WIN_18, center=True, min_periods=WIN_18 // 3).mean()

    peak_records = []
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
                peak_date=seg.idxmax(),
                peak_amplitude=float(seg.max()),
            ))

    peaks_df = (pd.DataFrame(peak_records)
                .sort_values(["cycle", "hemisphere"])
                .reset_index(drop=True))

    # ══════════════════════════════════════════════════════════════════════
    # Task 19 — Wing shape parameters vs cycle amplitude
    # ══════════════════════════════════════════════════════════════════════
    amp_lookup = {
        (row["cycle"], row["hemisphere"]): row["peak_amplitude"]
        for _, row in peaks_df.iterrows()
    }

    records_19 = []
    for r in results_pl:
        cyc, hemi = r["cycle"], r["hemisphere"]
        amp = amp_lookup.get((int(cyc), hemi))
        if amp is None or np.isnan(amp):
            continue
        mask = (df["CYCLE"] == cyc) & (df["hemisphere"] == hemi) & df["tau_refined"].notna()
        df_sel = df[mask]
        if len(df_sel) == 0:
            continue
        tau_start = df_sel["tau_refined"].min()
        mu0 = float(exp_decay(tau_start, a_mu_univ, b_mu_univ))
        records_19.append(dict(
            cycle=int(cyc), hemisphere=hemi,
            amplitude=float(amp), mu0=mu0,
            mu_peak=float(r["mu_peak"]),
            m_i=float(r["m_i"]),
        ))

    df19 = pd.DataFrame(records_19)

    fit_results_19 = {}
    for col in ("mu0", "mu_peak", "m_i"):
        vals = df19[["amplitude", col]].dropna()
        x, y = vals["amplitude"].values, vals[col].values
        popt, _ = curve_fit(linear_fit, x, y, p0=[0.0, float(np.mean(y))])
        fit_results_19[col] = tuple(popt)

    print("\nTask 19 linear fits:")
    for col, (a, b) in fit_results_19.items():
        print(f"  {col:8s} = {a:+.6f} · A  +  {b:.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # Scoreboard — NLL functions (closures over fitted constants)
    # ══════════════════════════════════════════════════════════════════════
    cycle_data_global = []
    for rec in df19.to_dict("records"):
        cyc, hemi = rec["cycle"], rec["hemisphere"]
        A = amp_lookup.get((int(cyc), hemi))
        if A is None:
            continue
        df_ch = df[(df["CYCLE"] == cyc) & (df["hemisphere"] == hemi)]
        years_data = []
        for yr in sorted(df_ch["year"].unique()):
            tau  = (yr + 0.5) - t0_refined[(int(cyc), hemi)]
            mu   = exp_decay(tau, a_mu_univ, b_mu_univ)
            lats = df_ch.loc[df_ch["year"] == yr, "latitude"].abs().values
            if len(lats) >= 5:
                years_data.append((tau, mu, lats))
        if years_data:
            cycle_data_global.append((A, years_data))

    def compute_global_nll(fit_results, m_sh=None, b_sh=None, a_mu=None, b_mu=None,
                           mi_R_coeffs=None):
        """
        Mean per-year-normalised NLL across all hemisphere-cycles (hard μ₀ cutoff).

        Parameters
        ----------
        fit_results : dict — keys 'mu0', 'mu_peak', 'm_i', each a (slope, intercept) tuple.
        m_sh, b_sh : float, optional — override universal equatorial line.
        a_mu, b_mu : float, optional — override universal mean-path μ(τ) = a·exp(−τ/b).
        mi_R_coeffs : (float, float) or None — (slope, intercept) for equatorward m_i_R(A).

        Returns
        -------
        float — lower is better (nats/year).
        """
        if m_sh is None: m_sh = m_shared_fit
        if b_sh is None: b_sh = b_shared_fit
        if a_mu is None: a_mu = a_mu_univ
        if b_mu is None: b_mu = b_mu_univ
        a_mu0,    b_mu0    = fit_results["mu0"]
        a_mupeak, b_mupeak = fit_results["mu_peak"]
        a_mi,     b_mi     = fit_results["m_i"]
        use_split = mi_R_coeffs is not None
        if use_split:
            a_mi_R, b_mi_R = mi_R_coeffs
        total, n_terms = 0.0, 0
        for A, years_data in cycle_data_global:
            mu0_p    = a_mu0 * A + b_mu0
            mupeak_p = a_mupeak * A + b_mupeak
            mi_p     = a_mi * A + b_mi
            mi_R_p   = (a_mi_R * A + b_mi_R) if use_split else mi_p
            for tau, _mu_cached, lats in years_data:
                mu = a_mu * np.exp(-tau / b_mu)
                if mu > mu0_p:
                    continue
                sigma_L = piecewise_linear_wing(mu, m_sh, b_sh, mupeak_p, mi_p)
                sigma_R = piecewise_linear_wing(mu, m_sh, b_sh, mupeak_p, mi_R_p)
                if sigma_L <= 0 or sigma_R <= 0:
                    continue
                if use_split:
                    total -= split_norm_logpdf(lats, mu, sigma_L, sigma_R).mean()
                else:
                    total -= sp_norm.logpdf(lats, loc=mu, scale=sigma_L).mean()
                n_terms += 1
        return total / n_terms if n_terms > 0 else 1e6

    def compute_global_nll_soft(fit_results, T=MU0_TEMP, m_sh=None, b_sh=None,
                                a_mu=None, b_mu=None, mi_R_coeffs=None):
        """
        Sigmoid-weighted NLL (differentiable μ₀ threshold, for L-BFGS-B optimisation).

        Weight: w(μ, μ₀) = 1/(1 + exp((μ − μ₀)/T)).  Normalised by Σw.
        Same signature as compute_global_nll; returns nats/year (weighted).
        """
        if m_sh is None: m_sh = m_shared_fit
        if b_sh is None: b_sh = b_shared_fit
        if a_mu is None: a_mu = a_mu_univ
        if b_mu is None: b_mu = b_mu_univ
        a_mu0,    b_mu0    = fit_results["mu0"]
        a_mupeak, b_mupeak = fit_results["mu_peak"]
        a_mi,     b_mi     = fit_results["m_i"]
        use_split = mi_R_coeffs is not None
        if use_split:
            a_mi_R, b_mi_R = mi_R_coeffs
        total_w, sum_w = 0.0, 0.0
        for A, years_data in cycle_data_global:
            mu0_p    = a_mu0 * A + b_mu0
            mupeak_p = a_mupeak * A + b_mupeak
            mi_p     = a_mi * A + b_mi
            mi_R_p   = (a_mi_R * A + b_mi_R) if use_split else mi_p
            for tau, _mu_cached, lats in years_data:
                mu = a_mu * np.exp(-tau / b_mu)
                w = 1.0 / (1.0 + np.exp(np.clip((mu - mu0_p) / T, -500, 500)))
                if w < 1e-6:
                    continue
                sigma_L = piecewise_linear_wing(mu, m_sh, b_sh, mupeak_p, mi_p)
                sigma_R = piecewise_linear_wing(mu, m_sh, b_sh, mupeak_p, mi_R_p)
                if sigma_L <= 0 or sigma_R <= 0:
                    continue
                if use_split:
                    total_w -= w * split_norm_logpdf(lats, mu, sigma_L, sigma_R).mean()
                else:
                    total_w -= w * sp_norm.logpdf(lats, loc=mu, scale=sigma_L).mean()
                sum_w += w
        return total_w / sum_w if sum_w > 1e-6 else 1e6

    nll_baseline = compute_global_nll(fit_results_19)
    n_hc_global  = len(cycle_data_global)
    n_terms_all  = sum(len(yd) for _, yd in cycle_data_global)
    print(f"\nDataset: {n_hc_global} hemisphere-cycles  |  {n_terms_all} valid year-cycle terms")
    print(f"Baseline NLL (Task 19 two-stage fits): {nll_baseline:.5f} nats/year")

    # ══════════════════════════════════════════════════════════════════════
    # Task 23 — End-to-end NLL optimisation (Nelder-Mead, 6p, hard cutoff)
    # ══════════════════════════════════════════════════════════════════════
    print("\nTask 23: Nelder-Mead 6p optimisation ...")
    opt_23 = minimize(
        lambda x: compute_global_nll(_unpack(x)),
        _pack(fit_results_19),
        method="Nelder-Mead",
        options={"maxiter": 50_000, "xatol": 1e-6, "fatol": 1e-7, "adaptive": True},
    )
    fit_results_23 = _unpack(opt_23.x)
    nll_23_hard    = compute_global_nll(fit_results_23)
    print(f"  converged={opt_23.success}  iters={opt_23.nit}  NLL={nll_23_hard:.5f}")

    # ══════════════════════════════════════════════════════════════════════
    # Soft μ₀ threshold — L-BFGS-B (6p)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\nSoft μ₀ L-BFGS-B 6p (T={MU0_TEMP}°) ...")

    def _nll_soft_6p(x):
        return compute_global_nll_soft(_unpack(x), T=MU0_TEMP)

    opt_soft = minimize(
        _nll_soft_6p,
        _pack(fit_results_23),
        method="L-BFGS-B",
        options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8},
    )
    fit_results_soft = _unpack(opt_soft.x)
    nll_soft_obj  = compute_global_nll_soft(fit_results_soft)
    nll_soft_hard = compute_global_nll(fit_results_soft)
    print(f"  converged={opt_soft.success}  iters={opt_soft.nit}  NLL(hard)={nll_soft_hard:.5f}")

    # ══════════════════════════════════════════════════════════════════════
    # Soft μ₀ + equatorial line — L-BFGS-B (8p eq)
    # [a_mu0, b_mu0, a_mupeak, b_mupeak, a_mi, b_mi, m_shared, b_shared]
    # ══════════════════════════════════════════════════════════════════════
    print(f"\nSoft μ₀ + eq. line L-BFGS-B 8p ...")

    def _nll_soft_eq8(x):
        fr, m_sh, b_sh = _unpack_eq8(x)
        return compute_global_nll_soft(fr, T=MU0_TEMP, m_sh=m_sh, b_sh=b_sh)

    x0_eq8 = np.append(_pack(fit_results_soft), [m_shared_fit, b_shared_fit])
    opt_eq8 = minimize(
        _nll_soft_eq8,
        x0_eq8,
        method="L-BFGS-B",
        bounds=[(None, None)] * 6 + [(0.0, 2.0), (-5.0, 5.0)],
        options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8},
    )
    fit_results_eq8, m_shared_eq8, b_shared_eq8 = _unpack_eq8(opt_eq8.x)
    nll_eq8_obj  = compute_global_nll_soft(fit_results_eq8, m_sh=m_shared_eq8, b_sh=b_shared_eq8)
    nll_eq8_hard = compute_global_nll(fit_results_eq8, m_sh=m_shared_eq8, b_sh=b_shared_eq8)
    print(f"  converged={opt_eq8.success}  iters={opt_eq8.nit}  NLL(hard)={nll_eq8_hard:.5f}")
    print(f"  m_shared={m_shared_eq8:.4f}  b_shared={b_shared_eq8:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Soft μ₀ + universal path — L-BFGS-B (8p path)
    # [a_mu0, b_mu0, a_mupeak, b_mupeak, a_mi, b_mi, a_mu, b_mu]
    # ══════════════════════════════════════════════════════════════════════
    print(f"\nSoft μ₀ + universal path L-BFGS-B 8p ...")

    def _nll_soft_ext(x):
        fr, a_mu, b_mu = _unpack_ext(x)
        return compute_global_nll_soft(fr, T=MU0_TEMP, a_mu=a_mu, b_mu=b_mu)

    x0_ext = np.append(_pack(fit_results_soft), [a_mu_univ, b_mu_univ])
    opt_ext = minimize(
        _nll_soft_ext,
        x0_ext,
        method="L-BFGS-B",
        bounds=[(None, None)] * 6 + [(5.0, 30.0), (1.0, 15.0)],
        options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8},
    )
    fit_results_ext, a_mu_opt, b_mu_opt = _unpack_ext(opt_ext.x)
    nll_ext_obj  = compute_global_nll_soft(fit_results_ext, a_mu=a_mu_opt, b_mu=b_mu_opt)
    nll_ext_hard = compute_global_nll(fit_results_ext, a_mu=a_mu_opt, b_mu=b_mu_opt)
    print(f"  converged={opt_ext.success}  iters={opt_ext.nit}  NLL(hard)={nll_ext_hard:.5f}")
    print(f"  a_mu={a_mu_opt:.4f}°  b_mu={b_mu_opt:.4f} yr")

    # ══════════════════════════════════════════════════════════════════════
    # Split-Gaussian + soft μ₀ + universal path — L-BFGS-B (10p)
    # [a_mu0, b_mu0, a_mupeak, b_mupeak, a_mi, b_mi, a_mu, b_mu, a_mi_R, b_mi_R]
    # σ_L (poleward, x≥μ) ← m_i;  σ_R (equatorward, x<μ) ← m_i_R
    # ══════════════════════════════════════════════════════════════════════
    print(f"\nSplit-Gaussian + path L-BFGS-B 10p ...")

    def _nll_split_10(x):
        fr, a_mu, b_mu, mi_R = _unpack_10(x)
        return compute_global_nll_soft(fr, T=MU0_TEMP, a_mu=a_mu, b_mu=b_mu, mi_R_coeffs=mi_R)

    a_mi_ext, b_mi_ext = fit_results_ext["m_i"]
    x0_10 = np.append(_pack(fit_results_ext), [a_mu_opt, b_mu_opt, a_mi_ext, b_mi_ext])
    opt_10 = minimize(
        _nll_split_10,
        x0_10,
        method="L-BFGS-B",
        bounds=[(None, None)] * 6 + [(5.0, 30.0), (1.0, 15.0)] + [(None, None)] * 2,
        options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8},
    )
    fit_results_10, a_mu_10, b_mu_10, mi_R_10 = _unpack_10(opt_10.x)
    nll_10_obj  = compute_global_nll_soft(fit_results_10, a_mu=a_mu_10, b_mu=b_mu_10,
                                          mi_R_coeffs=mi_R_10)
    nll_10_hard = compute_global_nll(fit_results_10, a_mu=a_mu_10, b_mu=b_mu_10,
                                     mi_R_coeffs=mi_R_10)
    print(f"  converged={opt_10.success}  iters={opt_10.nit}  NLL(hard)={nll_10_hard:.5f}")
    print(f"  a_mu={a_mu_10:.4f}°  b_mu={b_mu_10:.4f} yr")

    # ══════════════════════════════════════════════════════════════════════
    # Full model — all free parameters — L-BFGS-B (12p)
    # [a_mu0, b_mu0, a_mupeak, b_mupeak, a_mi, b_mi,
    #  m_shared, b_shared, a_mu, b_mu, a_mi_R, b_mi_R]
    # ══════════════════════════════════════════════════════════════════════
    print(f"\nFull model L-BFGS-B 12p ...")

    def _nll_full_12(x):
        fr, m_sh, b_sh, a_mu, b_mu, mi_R = _unpack_12(x)
        return compute_global_nll_soft(fr, T=MU0_TEMP, m_sh=m_sh, b_sh=b_sh,
                                       a_mu=a_mu, b_mu=b_mu, mi_R_coeffs=mi_R)

    a_mi_R_10, b_mi_R_10 = mi_R_10
    x0_12 = np.array([
        *_pack(fit_results_10),
        m_shared_eq8, b_shared_eq8,
        a_mu_10, b_mu_10,
        a_mi_R_10, b_mi_R_10,
    ])
    bounds_12 = ([(None, None)] * 6
                 + [(0.0, 2.0), (-5.0, 5.0)]
                 + [(5.0, 30.0), (1.0, 15.0)]
                 + [(None, None)] * 2)
    opt_12 = minimize(
        _nll_full_12,
        x0_12,
        method="L-BFGS-B",
        bounds=bounds_12,
        options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8},
    )
    fit_results_12, m_shared_12, b_shared_12, a_mu_12, b_mu_12, mi_R_12 = _unpack_12(opt_12.x)
    nll_12_obj  = compute_global_nll_soft(fit_results_12, m_sh=m_shared_12, b_sh=b_shared_12,
                                          a_mu=a_mu_12, b_mu=b_mu_12, mi_R_coeffs=mi_R_12)
    nll_12_hard = compute_global_nll(fit_results_12, m_sh=m_shared_12, b_sh=b_shared_12,
                                     a_mu=a_mu_12, b_mu=b_mu_12, mi_R_coeffs=mi_R_12)
    print(f"  converged={opt_12.success}  iters={opt_12.nit}  NLL(hard)={nll_12_hard:.5f}")

    # ══════════════════════════════════════════════════════════════════════
    # Model Quality Scoreboard
    # ══════════════════════════════════════════════════════════════════════
    def _pct(nll):
        return (nll_baseline - nll) / abs(nll_baseline) * 100

    print("\n" + "=" * 70)
    print("  Model Quality Scoreboard")
    print("=" * 70)
    print(f"  {'Model':<44s}  {'hard NLL':>10s}  {'vs baseline':>12s}")
    print(f"  {'-'*44}  {'-'*10}  {'-'*12}")
    print(f"  {'Baseline (Task 19, two-stage)':<44s}  {nll_baseline:>10.5f}  {'—':>12s}")
    print(f"  {'Task 23  (hard cutoff, Nelder-Mead, 6p)':<44s}  {nll_23_hard:>10.5f}  {_pct(nll_23_hard):>+11.2f}%")
    print(f"  {f'Soft μ₀ T={MU0_TEMP}° (L-BFGS-B, 6p)':<44s}  {nll_soft_hard:>10.5f}  {_pct(nll_soft_hard):>+11.2f}%")
    print(f"  {'Soft μ₀ + eq. line (L-BFGS-B, 8p eq)':<44s}  {nll_eq8_hard:>10.5f}  {_pct(nll_eq8_hard):>+11.2f}%")
    print(f"  {'Soft μ₀ + path (L-BFGS-B, 8p)':<44s}  {nll_ext_hard:>10.5f}  {_pct(nll_ext_hard):>+11.2f}%")
    print(f"  {'Split-Gaussian + path (L-BFGS-B, 10p)':<44s}  {nll_10_hard:>10.5f}  {_pct(nll_10_hard):>+11.2f}%")
    print(f"  {'─ ' * 33}")
    print(f"  {'Full model (all params, L-BFGS-B, 12p)':<44s}  {nll_12_hard:>10.5f}  {_pct(nll_12_hard):>+11.2f}%")
    print(f"  {'─ ' * 33}")
    print()
    print(f"  {'Model':<44s}  {'soft NLL':>10s}  {'note':>18s}")
    print(f"  {'-'*44}  {'-'*10}  {'-'*18}")
    print(f"  {'Soft μ₀  (6p)':<44s}  {nll_soft_obj:>10.5f}  {'← 6p objective':>18s}")
    print(f"  {'Soft+eq. line (8p eq)':<44s}  {nll_eq8_obj:>10.5f}  {'← 8p(eq) objective':>18s}")
    print(f"  {'Soft+path (8p)':<44s}  {nll_ext_obj:>10.5f}  {'← 8p objective':>18s}")
    print(f"  {'Split+path (10p)':<44s}  {nll_10_obj:>10.5f}  {'← 10p objective':>18s}")
    print(f"  {'─ ' * 33}")
    print(f"  {'Full model (12p)':<44s}  {nll_12_obj:>10.5f}  {'← 12p objective':>18s}")
    print(f"  {'─ ' * 33}")
    print("=" * 70)


if __name__ == "__main__":
    main()
