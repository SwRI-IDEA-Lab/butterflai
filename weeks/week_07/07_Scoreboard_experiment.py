#!/usr/bin/env python3
"""
96_2D_Wings_Metrics.py

Reference implementation covering Sections 1–5 and Tasks 17–20 of
06_2D_Wings_metrics.ipynb, plus the scoreboard metric setup (compute_global_nll)
and a full implementation of Task 23 (end-to-end NLL optimisation of the
amplitude–shape relationships).

Non-implemented tasks (21, 22, 24, 25) are intentionally omitted.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import norm as sp_norm
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit, minimize_scalar, minimize

# ── Repo-relative data path ────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "composite_sunspot_groups_peak_area.csv"


# ══════════════════════════════════════════════════════════════════════════
# Section 1 — Load data
# ══════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH, parse_dates=[[0, 1, 2]], keep_date_col=False)
df.rename(columns={"year_month_day": "date"}, inplace=True)
df = df[df["latitude"].notna()].copy()

df["hemisphere"]   = df["latitude"].apply(lambda v: "north" if v >= 0 else "south")
df["abs_latitude"] = df["latitude"].abs()
df["year"]         = df["date"].dt.year
df["decimal_year"] = df["date"].dt.year + df["date"].dt.dayofyear / 365.25

df = df[df["correctedArea"] > 30].copy()

cycles    = sorted(df["CYCLE"].dropna().unique())
n_cycles  = len(cycles)
cmap_full = cm.get_cmap("tab20", n_cycles)
cycle_colors = {cyc: cmap_full(i) for i, cyc in enumerate(cycles)}

fig, ax = plt.subplots(figsize=(14, 3))
for cyc in cycles:
    df_cyc = df[df["CYCLE"] == cyc]
    ax.scatter(df_cyc["date"], df_cyc["latitude"], s=3,
               c=[cycle_colors[cyc]], alpha=0.5, edgecolors="none")
ax.set_xlabel("Date")
ax.set_ylabel("Latitude (degrees)")
ax.set_ylim(-45, 45)
ax.axhline(0, color="k", linewidth=0.5, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════
# Section 2 — Standardise time: align cycles at the 15° crossing (τ)
# ══════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════
# Section 3 — Fit universal mean path μ(τ) and refine t₀
# ══════════════════════════════════════════════════════════════════════════
def exp_decay(tau, a, b):
    return a * np.exp(-tau / b)


N_BINS_13  = 20
cycles_13  = [c for c in sorted(df["CYCLE"].dropna().unique()) if c >= 12]
cmap_13    = cm.get_cmap("tab20", len(cycles_13))
cyc_idx_13 = {c: i for i, c in enumerate(cycles_13)}

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

df["t0_refined"]  = df.apply(lambda r: t0_refined.get((r["CYCLE"], r["hemisphere"]), np.nan), axis=1)
df["tau_refined"] = df["decimal_year"] - df["t0_refined"]

TAU_GRID_13 = np.linspace(-8, 8, 300)

fig1, ax1 = plt.subplots(figsize=(12, 5))
for (cyc, hemi) in hemicycle_bins_13:
    mask = (df["CYCLE"] == cyc) & (df["hemisphere"] == hemi) & df["tau_refined"].notna()
    grp  = df[mask]
    ax1.scatter(grp["tau_refined"], grp["abs_latitude"],
                s=10, color=cmap_13(cyc_idx_13[cyc]), alpha=0.25, edgecolors="none")
ax1.plot(TAU_GRID_13, exp_decay(TAU_GRID_13, a_mu_univ, b_mu_univ),
         color="black", linewidth=2.5,
         label=f"Universal μ(τ)  a={a_mu_univ:.1f}°  b={b_mu_univ:.1f} yr")
ax1.axvline(0, color="black", linewidth=1.5, linestyle="--", label="τ = 0")
ax1.axhline(15, color="gray", linewidth=1, linestyle=":", alpha=0.6)
ax1.set_xlabel("τ (years relative to refined t₀)")
ax1.set_ylabel("|Latitude| (degrees)")
ax1.set_title("All hemisphere-cycles aligned to the universal mean path")
ax1.set_ylim(0, 45); ax1.set_xlim(-8, 8)
ax1.legend(loc="upper right")
sm1 = plt.cm.ScalarMappable(cmap="tab20",
                              norm=plt.Normalize(vmin=min(cycles_13), vmax=max(cycles_13)))
sm1.set_array([])
fig1.colorbar(sm1, ax=ax1, pad=0.02, label="Cycle number")
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════
# Section 4 — Model σ(μ): spread as a function of mean latitude
# ══════════════════════════════════════════════════════════════════════════
def split_normal_mu(mu, A, mu_peak, s_L, s_R):
    return np.where(
        mu >= mu_peak,
        A * np.exp(-0.5 * ((mu - mu_peak) / s_L) ** 2),
        A * np.exp(-0.5 * ((mu - mu_peak) / s_R) ** 2),
    )


N_BINS_15 = 20
MU_GRID   = np.linspace(2, 42, 300)
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
            sigma_curve=split_normal_mu(MU_GRID, A_f, mu_peak_f, sL_f, sR_f),
            bin_mu=bm_arr, bin_sigma=bs_arr,
        ))

print(f"Fitted {len(results_15)} hemisphere-cycles.")


# ══════════════════════════════════════════════════════════════════════════
# Section 5 — Universal piecewise-linear envelope
# ══════════════════════════════════════════════════════════════════════════
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

    The distribution is Gaussian on each side of the mode μ but with
    independent widths:
        f(x | μ, σ_L, σ_R) ∝  exp(−(x−μ)²/(2σ_L²))  for x ≥ μ  (poleward)
                               exp(−(x−μ)²/(2σ_R²))  for x < μ  (equatorward)

    Normalisation constant: A = √(2/π) / (σ_L + σ_R).

    Parameters
    ----------
    x : array-like   — observed absolute latitudes
    mu : float       — predicted mean latitude (mode of the distribution)
    sigma_L : float  — poleward spread  (x ≥ μ side, high-latitude tail)
    sigma_R : float  — equatorward spread (x < μ side, low-latitude tail)
    """
    log_A = 0.5 * np.log(2.0 / np.pi) - np.log(sigma_L + sigma_R)
    return np.where(
        x >= mu,
        log_A - 0.5 * ((x - mu) / sigma_L) ** 2,
        log_A - 0.5 * ((x - mu) / sigma_R) ** 2,
    )


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


x0 = np.array([m_init, b_init] + [val for r in results_15
                                   for val in (r["mu_peak"], m_i_init)])
bounds = list(zip(
    [0.0, -5.0] + [2.0, -5.0] * n_hc,
    [2.0,  5.0] + [38.0, 0.0] * n_hc,
))
opt = minimize(residuals_pl, x0, method="L-BFGS-B", bounds=bounds)

m_shared_fit, b_shared_fit = opt.x[0], opt.x[1]
print(f"Universal line:  σ(μ) = {m_shared_fit:.4f}·μ + {b_shared_fit:.4f}")
print(f"Zero crossing at μ = {-b_shared_fit / m_shared_fit:.2f}°")

MU_GRID_PL = np.linspace(2, 42, 300)
results_pl = []
for i, r in enumerate(results_15):
    mu_peak_i    = opt.x[2 + 2*i]
    m_i          = opt.x[3 + 2*i]
    sigma_peak_i = m_shared_fit * mu_peak_i + b_shared_fit
    results_pl.append(dict(
        cycle=r["cycle"], hemisphere=r["hemisphere"],
        m_shared=m_shared_fit, b_shared=b_shared_fit,
        mu_peak=mu_peak_i, m_i=m_i, sigma_peak=sigma_peak_i,
        sigma_curve=piecewise_linear_wing(
            MU_GRID_PL, m_shared_fit, b_shared_fit, mu_peak_i, m_i),
        bin_mu=r["bin_mu"], bin_sigma=r["bin_sigma"],
    ))


def rmse_pl(res_list):
    return np.sqrt(np.mean([
        np.mean((r["bin_sigma"] - piecewise_linear_wing(
            r["bin_mu"], r["m_shared"], r["b_shared"], r["mu_peak"], r["m_i"])) ** 2)
        for r in res_list
    ]))


rmse_16 = rmse_pl(results_pl)
print(f"RMSE — piecewise-linear joint: {rmse_16:.3f}°")

fig, ax = plt.subplots(figsize=(12, 5))
for r in results_pl:
    c = cmap_13(cyc_idx_13.get(r["cycle"], 0))
    ax.scatter(r["bin_mu"], r["bin_sigma"], s=20, color=c, alpha=0.8, edgecolors="none", zorder=1)
    ax.plot(MU_GRID_PL, r["sigma_curve"], color=c, linewidth=1.2, alpha=0.7, zorder=2)
mu_eq = np.linspace(0, max(r["mu_peak"] for r in results_pl) + 2, 200)
ax.plot(mu_eq, np.clip(m_shared_fit * mu_eq + b_shared_fit, 0, None),
        color="tab:red", linewidth=2.5, linestyle="--",
        label=f"Universal equatorward line  σ = {m_shared_fit:.3f}·μ + {b_shared_fit:.3f}",
        zorder=5)
ax.invert_xaxis()
ax.set_xlim(42, 2)
ax.set_ylim(0, 12)
ax.set_xlabel("|μ| (mean emergence latitude, °)")
ax.set_ylabel("σ (degrees)")
ax.set_title(
    f"Universal piecewise-linear envelope  "
    f"σ = {m_shared_fit:.3f}·μ + {b_shared_fit:.3f}  |  RMSE = {rmse_16:.3f}°"
)
ax.legend(loc="upper left", fontsize=9)
sm = plt.cm.ScalarMappable(cmap="tab20",
                            norm=plt.Normalize(vmin=min(cycles_13), vmax=max(cycles_13)))
sm.set_array([])
fig.colorbar(sm, ax=ax, pad=0.02, label="Cycle number")
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════
# Task 17 — Solar Cycle Amplitude from Total Sunspot Area
# ══════════════════════════════════════════════════════════════════════════
df_amp = df[(df["CYCLE"] >= 12) & (df["correctedArea"] > 50)].copy()

daily_north = df_amp[df_amp["hemisphere"] == "north"].groupby("date")["correctedArea"].sum()
daily_south = df_amp[df_amp["hemisphere"] == "south"].groupby("date")["correctedArea"].sum()

date_range_17 = pd.date_range(
    min(daily_north.index.min(), daily_south.index.min()),
    max(daily_north.index.max(), daily_south.index.max()),
    freq="D",
)
daily_north = daily_north.reindex(date_range_17, fill_value=0)
daily_south = daily_south.reindex(date_range_17, fill_value=0)

windows_17  = {"3 mo": 91, "6 mo": 182, "12 mo": 365, "24 mo": 730}
line_styles = [("tab:blue", 1.0), ("tab:orange", 1.2), ("tab:green", 1.6), ("tab:red", 2.0)]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for ax, hemi_label, daily in zip(axes, ["North", "South"], [daily_north, daily_south]):
    for (label, win), (col, lw) in zip(windows_17.items(), line_styles):
        smoothed = daily.rolling(win, center=True, min_periods=win // 3).mean()
        ax.plot(smoothed.index, smoothed.values, label=label, color=col, linewidth=lw, alpha=0.85)
    ax.set_ylabel("Total corrected area (MSH)")
    ax.set_title(f"{hemi_label} hemisphere — smoothed total sunspot area "
                 "(cycles ≥ 12, correctedArea > 50 MSH)")
    ax.legend(loc="upper right", title="Smoothing window")
axes[1].set_xlabel("Date")
plt.tight_layout()
plt.show()

print("Best window: 12 months — resolves cycle rise/peak/decay without over-smoothing.")


# ══════════════════════════════════════════════════════════════════════════
# Task 18 — Peak Amplitude and Timing of Each Hemispheric Cycle
# ══════════════════════════════════════════════════════════════════════════
WIN_18 = 365

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
print(peaks_df.to_string(index=False))

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for ax, hemi, smooth, color in zip(
        axes,
        ["north", "south"],
        [smooth_north, smooth_south],
        ["steelblue", "tomato"]):
    ax.plot(smooth.index, smooth.values, color=color, linewidth=1.2, alpha=0.8)
    for _, row in peaks_df[peaks_df["hemisphere"] == hemi].iterrows():
        ax.scatter(row["peak_date"], row["peak_amplitude"], s=70, color="black", zorder=5)
        ax.annotate(str(int(row["cycle"])),
                    (row["peak_date"], row["peak_amplitude"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_ylabel("Total corrected area (MSH)")
    ax.set_title(f"{hemi.capitalize()} hemisphere — 12-month smoothed area with detected peaks")
axes[1].set_xlabel("Date")
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════
# Task 19 — Relating Cycle Amplitude to Wing Shape Parameters
# ══════════════════════════════════════════════════════════════════════════
def linear_fit(x, a, b):
    return a * x + b


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
print(df19.to_string(index=False))

param_info = [
    ("mu0",     r"$\mu_0$ — earliest mean latitude (°)",       "tab:purple"),
    ("mu_peak", r"$\mu_{\rm peak}$ — detachment latitude (°)", "tab:orange"),
    ("m_i",     r"$m_i$ — poleward slope",                     "tab:green"),
]

fit_results_19 = {}
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (col, ylabel, color) in zip(axes, param_info):
    vals = df19[["amplitude", col]].dropna()
    x, y = vals["amplitude"].values, vals[col].values
    popt, _ = curve_fit(linear_fit, x, y, p0=[0.0, float(np.mean(y))])
    a_fit, b_fit = popt
    fit_results_19[col] = (a_fit, b_fit)

    amp_grid_19 = np.linspace(x.min() * 0.9, x.max() * 1.05, 200)
    ax.scatter(x, y, color=color, s=40, alpha=0.80, edgecolors="none", zorder=3)
    ax.plot(amp_grid_19, linear_fit(amp_grid_19, a_fit, b_fit),
            color="black", linewidth=2,
            label=f"y = {a_fit:.5f}·A + {b_fit:.2f}")
    for _, row in df19.iterrows():
        if pd.notna(row["amplitude"]) and pd.notna(row[col]):
            ax.annotate(f"{int(row['cycle'])}{row['hemisphere'][0].upper()}",
                        (row["amplitude"], row[col]),
                        fontsize=5.5, alpha=0.55, xytext=(2, 2),
                        textcoords="offset points")
    ax.set_xlabel("Peak amplitude (MSH)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{col} vs amplitude")
    ax.legend(fontsize=8)

plt.suptitle("Task 19 — Wing shape parameters vs cycle peak amplitude", fontsize=12, y=1.01)
plt.tight_layout()
plt.show()

print("\nFitted linear relationships:")
for col, (a, b) in fit_results_19.items():
    print(f"  {col:8s} = {a:+.6f} · A  +  {b:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# Task 20 — Putting It All Together: Synthesizing a Wing from Amplitude
# ══════════════════════════════════════════════════════════════════════════
cycle_number = 16
hemisphere   = "south"

mask = (df["CYCLE"] == cycle_number) & (df["hemisphere"] == hemisphere)
df_cyc_hemi  = df[mask].copy()
years_in_cycle = sorted(df_cyc_hemi["year"].unique())
n_years = len(years_in_cycle)
cmap     = plt.get_cmap("viridis", n_years)
lat_grid = np.linspace(0, 45, 300) if hemisphere == "north" else np.linspace(-45, 0, 300)
kde_width_days = 250

# ── KDE butterfly diagram (mirrors Task 8 from earlier weeks) ─────────────
fig, ax = plt.subplots(figsize=(12, 5))
scatter_color = "tab:red" if hemisphere == "north" else "tab:blue"
ax.scatter(df_cyc_hemi["date"], df_cyc_hemi["latitude"],
           s=2, color=scatter_color, alpha=0.25, zorder=1)
for i, yr in enumerate(years_in_cycle):
    yr_lats = df_cyc_hemi.loc[df_cyc_hemi["year"] == yr, "latitude"].values
    if len(yr_lats) < 5:
        continue
    kde = gaussian_kde(yr_lats, bw_method=0.3)
    kde_vals = kde(lat_grid)
    kde_scaled = kde_vals / kde_vals.max() * kde_width_days
    center  = pd.Timestamp(f"{int(yr)}-07-01")
    x_curve = [center + pd.Timedelta(days=float(v)) for v in kde_scaled]
    color   = cmap(i)
    ax.plot(x_curve, lat_grid, color=color, linewidth=1.8, alpha=0.9, zorder=3)
    ax.fill_betweenx(lat_grid, [center] * len(lat_grid), x_curve,
                     color=color, alpha=0.20, zorder=2)
    median_lat = np.median(yr_lats)
    kde_at_med = float(kde(np.array([median_lat]))[0]) / kde_vals.max() * kde_width_days
    ax.plot([center, center + pd.Timedelta(days=kde_at_med)],
            [median_lat, median_lat], color=color, linewidth=2, linestyle="--", zorder=4)
sm = plt.cm.ScalarMappable(cmap="viridis",
                            norm=plt.Normalize(vmin=years_in_cycle[0], vmax=years_in_cycle[-1]))
sm.set_array([])
fig.colorbar(sm, ax=ax, pad=0.02).set_label("Year")
ax.set_title(f"Cycle {cycle_number} ({hemisphere}) — KDE butterfly diagram")
ax.set_xlabel("Date"); ax.set_ylabel("Latitude (degrees)")
ax.set_ylim((0, 45) if hemisphere == "north" else (-45, 0))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ── Synthesize wing from amplitude ────────────────────────────────────────
amp_20      = amp_lookup.get((cycle_number, hemisphere))
mu0_pred    = linear_fit(amp_20, *fit_results_19["mu0"])
mupeak_pred = linear_fit(amp_20, *fit_results_19["mu_peak"])
mi_pred     = linear_fit(amp_20, *fit_results_19["m_i"])
print(f"Cycle {cycle_number} ({hemisphere}): peak amplitude = {amp_20:.0f} MSH")
print(f"  Predicted μ₀     = {mu0_pred:.2f}°")
print(f"  Predicted μ_peak = {mupeak_pred:.2f}°")
print(f"  Predicted m_i    = {mi_pred:.4f}")

t0_ref_20  = t0_refined[(cycle_number, hemisphere)]
year_ann   = np.array(years_in_cycle)
tau_ann    = (year_ann + 0.5) - t0_ref_20
mu_ann     = exp_decay(tau_ann, a_mu_univ, b_mu_univ)
sigma_ann  = np.array([
    piecewise_linear_wing(mu, m_shared_fit, b_shared_fit, mupeak_pred, mi_pred)
    for mu in mu_ann
])

sign     = +1 if hemisphere == "north" else -1
cmap_syn = plt.get_cmap("plasma", len(year_ann))

fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(df_cyc_hemi["date"], df_cyc_hemi["latitude"],
           s=2, color=scatter_color, alpha=0.15, zorder=1)

for i, yr in enumerate(years_in_cycle):
    yr_lats = df_cyc_hemi.loc[df_cyc_hemi["year"] == yr, "latitude"].values
    if len(yr_lats) < 5:
        continue
    kde_obj  = gaussian_kde(yr_lats, bw_method=0.3)
    kde_vals = kde_obj(lat_grid)
    kde_sc   = kde_vals / kde_vals.max() * kde_width_days
    center   = pd.Timestamp(f"{int(yr)}-07-01")
    x_curve  = [center + pd.Timedelta(days=float(v)) for v in kde_sc]
    col_r    = cmap(i)
    ax.plot(x_curve, lat_grid, color=col_r, linewidth=1.5, alpha=0.65, zorder=3)
    ax.fill_betweenx(lat_grid, [center] * len(lat_grid), x_curve,
                     color=col_r, alpha=0.12, zorder=2)

for j, (yr, mu, sigma) in enumerate(zip(year_ann, mu_ann, sigma_ann)):
    if sigma <= 0:
        continue
    center_lat = sign * mu
    gauss_vals = sp_norm.pdf(lat_grid, loc=center_lat, scale=sigma)
    if gauss_vals.max() == 0:
        continue
    gauss_sc = gauss_vals / gauss_vals.max() * kde_width_days
    center   = pd.Timestamp(f"{int(yr)}-07-01")
    x_curve  = [center + pd.Timedelta(days=float(v)) for v in gauss_sc]
    col_s    = cmap_syn(j)
    ax.plot(x_curve, lat_grid, color=col_s, linewidth=2.2, linestyle="--",
            alpha=0.95, zorder=5)
    ax.fill_betweenx(lat_grid, [center] * len(lat_grid), x_curve,
                     color=col_s, alpha=0.10, zorder=4)

sm_r = plt.cm.ScalarMappable(cmap="viridis",
    norm=plt.Normalize(vmin=years_in_cycle[0], vmax=years_in_cycle[-1]))
sm_r.set_array([])
fig.colorbar(sm_r, ax=ax, pad=0.02).set_label("Year (real KDE)")
leg_elements = [
    Line2D([0], [0], color="gray",       linewidth=1.5, alpha=0.7,   label="Real KDE (viridis)"),
    Line2D([0], [0], color="black",      linewidth=2.2, linestyle="--", label="Synthetic Gaussian (plasma)"),
    Line2D([0], [0], color=scatter_color, marker="o",  markersize=4,
           linestyle="None", alpha=0.4,  label="Observed sunspot groups"),
]
ax.legend(handles=leg_elements, loc="upper right", fontsize=9)
ax.set_title(
    f"Cycle {cycle_number} ({hemisphere}) — Real KDE vs Synthetic Gaussian\n"
    f"Amplitude = {amp_20:.0f} MSH  →  μ₀ = {mu0_pred:.1f}°   "
    f"μ_peak = {mupeak_pred:.1f}°   m_i = {mi_pred:.3f}"
)
ax.set_xlabel("Date"); ax.set_ylabel("Latitude (°)")
ax.set_ylim((0, 45) if hemisphere == "north" else (-45, 0))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════
# Scoreboard setup — compute_global_nll
# ══════════════════════════════════════════════════════════════════════════

# Build (cycle, amplitude, years_data) cache reused in Tasks 23–24.
# Each years_data entry: (tau, mu, abs_latitudes).
df19_records_global = []
cycle_data_global   = []

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
        df19_records_global.append(rec)
        cycle_data_global.append((A, years_data))


def compute_global_nll(fit_results, m_sh=None, b_sh=None, a_mu=None, b_mu=None,
                       mi_R_coeffs=None):
    """
    Mean per-year-normalized NLL across all hemisphere-cycles.

    Parameters
    ----------
    fit_results : dict with keys 'mu0', 'mu_peak', 'm_i'
        Each value is a (slope, intercept) tuple.
    m_sh, b_sh : float, optional
        Override the universal equatorial line parameters.
    a_mu, b_mu : float, optional
        Universal mean-path parameters μ(τ) = a·exp(−τ/b).
        Default: module-level a_mu_univ, b_mu_univ.
    mi_R_coeffs : (float, float) or None
        (slope, intercept) for the equatorward-spread amplitude relationship
        m_i_R(A) = slope·A + intercept.  When None, falls back to the
        symmetric Gaussian using m_i for both sides.

    Returns
    -------
    float — lower is better (nats/year).
    """
    if m_sh  is None: m_sh  = m_shared_fit
    if b_sh  is None: b_sh  = b_shared_fit
    if a_mu  is None: a_mu  = a_mu_univ
    if b_mu  is None: b_mu  = b_mu_univ
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


MU0_TEMP = 1.0  # sigmoid temperature T (degrees) for the soft μ₀ threshold


def compute_global_nll_soft(fit_results, T=MU0_TEMP, m_sh=None, b_sh=None,
                            a_mu=None, b_mu=None, mi_R_coeffs=None):
    """
    Sigmoid-weighted version of compute_global_nll.

    Replaces the hard μ > μ₀ cutoff with a continuous weight:
        w(μ, μ₀) = 1 / (1 + exp((μ − μ₀) / T))
    so every year contributes, but years well above μ₀ (pre-cycle) are
    down-weighted to near zero.  The normalisation uses the sum of weights
    rather than a count of included terms, recovering the hard-threshold
    behaviour as T → 0.

    This formulation is differentiable in all parameters (including a_mu,
    b_mu, and the split-Gaussian m_i_R), enabling L-BFGS-B optimisation.

    Parameters
    ----------
    fit_results : dict with keys 'mu0', 'mu_peak', 'm_i'
        Each value is a (slope, intercept) tuple.
    T : float
        Sigmoid temperature in degrees (default: MU0_TEMP = 1°).
    m_sh, b_sh : float, optional
        Override the universal equatorial line parameters.
    a_mu, b_mu : float, optional
        Universal mean-path parameters μ(τ) = a·exp(−τ/b).
        Default: module-level a_mu_univ, b_mu_univ.
    mi_R_coeffs : (float, float) or None
        (slope, intercept) for the equatorward-spread amplitude relationship
        m_i_R(A) = slope·A + intercept.  When None, falls back to the
        symmetric Gaussian using m_i for both sides.

    Returns
    -------
    float — lower is better (nats/year, weighted).
    """
    if m_sh  is None: m_sh  = m_shared_fit
    if b_sh  is None: b_sh  = b_shared_fit
    if a_mu  is None: a_mu  = a_mu_univ
    if b_mu  is None: b_mu  = b_mu_univ
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


# ══════════════════════════════════════════════════════════════════════════
# Task 23 — End-to-End NLL Optimisation of Amplitude Relationships
# ══════════════════════════════════════════════════════════════════════════
# Optimise all six linear coefficients simultaneously by minimising
# compute_global_nll.  The hard μ₀ threshold makes the objective
# discontinuous in (a_mu0, b_mu0), so we use gradient-free Nelder-Mead.

print("\nTask 23: End-to-end NLL optimisation (Nelder-Mead) ...")

# Pack/unpack helpers — parameter vector layout:
#   [a_mu0, b_mu0, a_mupeak, b_mupeak, a_mi, b_mi]
def _pack(fr):
    return np.array([
        fr["mu0"][0],    fr["mu0"][1],
        fr["mu_peak"][0], fr["mu_peak"][1],
        fr["m_i"][0],    fr["m_i"][1],
    ])


def _unpack(x):
    return {
        "mu0":    (x[0], x[1]),
        "mu_peak": (x[2], x[3]),
        "m_i":    (x[4], x[5]),
    }


def _nll_vec(x):
    return compute_global_nll(_unpack(x))


x0_23 = _pack(fit_results_19)

opt_23 = minimize(
    _nll_vec,
    x0_23,
    method="Nelder-Mead",
    options={"maxiter": 50_000, "xatol": 1e-6, "fatol": 1e-7, "adaptive": True},
)

fit_results_23 = _unpack(opt_23.x)
nll_23         = compute_global_nll(fit_results_23)

print(f"Nelder-Mead converged: {opt_23.success}  ({opt_23.message})")
print(f"  Iterations: {opt_23.nit}   Function evaluations: {opt_23.nfev}")

print("\nOptimised coefficients (Task 23) vs Task 19 two-stage fits:")
print(f"  {'param':8s}  {'slope (19)':>12s}  {'slope (23)':>12s}  "
      f"{'intercept (19)':>15s}  {'intercept (23)':>15s}")
for key in ("mu0", "mu_peak", "m_i"):
    a19, b19 = fit_results_19[key]
    a23, b23 = fit_results_23[key]
    print(f"  {key:8s}  {a19:>12.6f}  {a23:>12.6f}  {b19:>15.4f}  {b23:>15.4f}")

# ── Scatter plots: Task 19 vs Task 23 regression lines ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (col, ylabel, color) in zip(axes, param_info):
    vals = df19[["amplitude", col]].dropna()
    x, y = vals["amplitude"].values, vals[col].values
    amp_g = np.linspace(x.min() * 0.9, x.max() * 1.05, 200)

    ax.scatter(x, y, color=color, s=40, alpha=0.80, edgecolors="none", zorder=3)

    a19, b19 = fit_results_19[col]
    ax.plot(amp_g, linear_fit(amp_g, a19, b19),
            color="black", linewidth=2, linestyle="--",
            label=f"Task 19  y = {a19:.5f}·A + {b19:.2f}")

    a23, b23 = fit_results_23[col]
    ax.plot(amp_g, linear_fit(amp_g, a23, b23),
            color="tab:red", linewidth=2, linestyle="-",
            label=f"Task 23  y = {a23:.5f}·A + {b23:.2f}")

    for _, row in df19.iterrows():
        if pd.notna(row["amplitude"]) and pd.notna(row[col]):
            ax.annotate(f"{int(row['cycle'])}{row['hemisphere'][0].upper()}",
                        (row["amplitude"], row[col]),
                        fontsize=5.5, alpha=0.55, xytext=(2, 2),
                        textcoords="offset points")

    ax.set_xlabel("Peak amplitude (MSH)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{col} vs amplitude")
    ax.legend(fontsize=7)

plt.suptitle("Task 23 — End-to-end NLL optimisation vs Task 19 two-stage fits",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════
# Soft μ₀ Threshold — L-BFGS-B Optimisation
# ══════════════════════════════════════════════════════════════════════════
# With the sigmoid weighting the objective is differentiable everywhere,
# so L-BFGS-B can exploit gradient information.  We initialise from the
# Task 23 Nelder-Mead solution and let L-BFGS-B refine further.

print(f"\nSoft μ₀ + L-BFGS-B optimisation (T = {MU0_TEMP}°) ...")


def _nll_soft_vec(x):
    return compute_global_nll_soft(_unpack(x), T=MU0_TEMP)


x0_soft = _pack(fit_results_23)   # warm-start from Task 23

opt_soft = minimize(
    _nll_soft_vec,
    x0_soft,
    method="L-BFGS-B",
    options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8},
)

fit_results_soft = _unpack(opt_soft.x)
nll_soft_obj  = compute_global_nll_soft(fit_results_soft)   # soft metric (what was optimised)
nll_soft_hard = compute_global_nll(fit_results_soft)        # hard metric (scoreboard currency)

print(f"L-BFGS-B converged: {opt_soft.success}  ({opt_soft.message})")
print(f"  Iterations: {opt_soft.nit}   Function evaluations: {opt_soft.nfev}")

print("\nOptimised coefficients — all three strategies:")
print(f"  {'param':8s}  {'slope 19':>10s}  {'slope 23':>10s}  {'slope soft':>10s}  "
      f"  {'int 19':>8s}  {'int 23':>8s}  {'int soft':>8s}")
for key in ("mu0", "mu_peak", "m_i"):
    a19, b19   = fit_results_19[key]
    a23, b23   = fit_results_23[key]
    asf, bsf   = fit_results_soft[key]
    print(f"  {key:8s}  {a19:>10.6f}  {a23:>10.6f}  {asf:>10.6f}  "
          f"  {b19:>8.3f}  {b23:>8.3f}  {bsf:>8.3f}")

# ── Scatter plots: all three regression lines ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (col, ylabel, color) in zip(axes, param_info):
    vals = df19[["amplitude", col]].dropna()
    x, y = vals["amplitude"].values, vals[col].values
    amp_g = np.linspace(x.min() * 0.9, x.max() * 1.05, 200)

    ax.scatter(x, y, color=color, s=40, alpha=0.80, edgecolors="none", zorder=3)

    for (fr, lbl, ls, lc) in [
        (fit_results_19,   "Task 19",       "--", "black"),
        (fit_results_23,   "Task 23 (NM)",  "-",  "tab:red"),
        (fit_results_soft, f"Soft T={MU0_TEMP}° (LBFGSB)", "-.", "tab:blue"),
    ]:
        a, b = fr[col]
        ax.plot(amp_g, linear_fit(amp_g, a, b), color=lc, linewidth=2,
                linestyle=ls, label=f"{lbl}  {a:.5f}·A + {b:.2f}")

    for _, row in df19.iterrows():
        if pd.notna(row["amplitude"]) and pd.notna(row[col]):
            ax.annotate(f"{int(row['cycle'])}{row['hemisphere'][0].upper()}",
                        (row["amplitude"], row[col]),
                        fontsize=5.5, alpha=0.55, xytext=(2, 2),
                        textcoords="offset points")

    ax.set_xlabel("Peak amplitude (MSH)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{col} vs amplitude")
    ax.legend(fontsize=6)

plt.suptitle(
    f"Soft μ₀ threshold (T = {MU0_TEMP}°) + L-BFGS-B vs prior strategies",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════
# Soft μ₀ + Equatorial Line (m_shared, b_shared) — L-BFGS-B, 8 parameters
# ══════════════════════════════════════════════════════════════════════════
# Extend the 6-param soft model by also optimising the two universal
# equatorial-line parameters m_shared and b_shared.  In Section 5 these
# were fixed by a two-stage LSQ fit on binned (μ, σ) points; freeing them
# lets the global NLL adjust the baseline σ(μ) jointly with all amplitude
# relationships rather than treating it as a pre-fitted constant.
#
# Parameter vector layout (8 total):
#   [a_mu0, b_mu0, a_mupeak, b_mupeak, a_mi, b_mi, m_shared, b_shared]
#    0      1      2         3         4     5     6          7

print(f"\nSoft μ₀ + equatorial line (8-param eq) L-BFGS-B (T = {MU0_TEMP}°) ...")


def _pack_eq8(fr, m_sh, b_sh):
    return np.array([
        fr["mu0"][0],     fr["mu0"][1],
        fr["mu_peak"][0], fr["mu_peak"][1],
        fr["m_i"][0],     fr["m_i"][1],
        m_sh, b_sh,
    ])


def _unpack_eq8(x):
    fr = {
        "mu0":     (x[0], x[1]),
        "mu_peak": (x[2], x[3]),
        "m_i":     (x[4], x[5]),
    }
    return fr, float(x[6]), float(x[7])


def _nll_soft_eq8_vec(x):
    fr, m_sh, b_sh = _unpack_eq8(x)
    return compute_global_nll_soft(fr, T=MU0_TEMP, m_sh=m_sh, b_sh=b_sh)


x0_eq8     = _pack_eq8(fit_results_soft, m_shared_fit, b_shared_fit)
# m_shared > 0: spread must increase with mean latitude; b_shared weakly bounded
bounds_eq8 = [(None, None)] * 6 + [(0.0, 2.0), (-5.0, 5.0)]

opt_eq8 = minimize(
    _nll_soft_eq8_vec,
    x0_eq8,
    method="L-BFGS-B",
    bounds=bounds_eq8,
    options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8},
)

fit_results_eq8, m_shared_eq8, b_shared_eq8 = _unpack_eq8(opt_eq8.x)

nll_eq8_obj  = compute_global_nll_soft(fit_results_eq8, m_sh=m_shared_eq8, b_sh=b_shared_eq8)
nll_eq8_hard = compute_global_nll(fit_results_eq8, m_sh=m_shared_eq8, b_sh=b_shared_eq8)

print(f"L-BFGS-B (eq8) converged: {opt_eq8.success}  ({opt_eq8.message})")
print(f"  Iterations: {opt_eq8.nit}   Function evaluations: {opt_eq8.nfev}")
print(f"\n  Optimised equatorial line:  m_shared = {m_shared_eq8:.4f}  (was {m_shared_fit:.4f})")
print(f"                              b_shared = {b_shared_eq8:.4f}  (was {b_shared_fit:.4f})")
print(f"  Zero crossing at μ = {-b_shared_eq8 / m_shared_eq8:.2f}°"
      f"  (was {-b_shared_fit / m_shared_fit:.2f}°)")

print("\nAmplitude-relationship coefficients  [eq8 vs prior strategies]:")
print(f"  {'param':8s}  {'slope 19':>10s}  {'slope 23':>10s}  "
      f"{'slope soft':>10s}  {'slope eq8':>10s}  "
      f"  {'int 19':>7s}  {'int 23':>7s}  {'int soft':>8s}  {'int eq8':>8s}")
for key in ("mu0", "mu_peak", "m_i"):
    a19, b19 = fit_results_19[key]
    a23, b23 = fit_results_23[key]
    asf, bsf = fit_results_soft[key]
    aeq, beq = fit_results_eq8[key]
    print(f"  {key:8s}  {a19:>10.6f}  {a23:>10.6f}  {asf:>10.6f}  {aeq:>10.6f}  "
          f"  {b19:>7.3f}  {b23:>7.3f}  {bsf:>8.3f}  {beq:>8.3f}")

# ── Scatter plots: amplitude relationships ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (col, ylabel, color) in zip(axes, param_info):
    vals = df19[["amplitude", col]].dropna()
    x, y = vals["amplitude"].values, vals[col].values
    amp_g = np.linspace(x.min() * 0.9, x.max() * 1.05, 200)

    ax.scatter(x, y, color=color, s=40, alpha=0.80, edgecolors="none", zorder=3)

    for (fr, lbl, ls, lc) in [
        (fit_results_19,   "Task 19",       "--", "black"),
        (fit_results_23,   "Task 23 (NM)",  "-",  "tab:red"),
        (fit_results_soft, "Soft 6p",       "-.", "tab:blue"),
        (fit_results_eq8,  "Soft 8p (eq)",  ":",  "tab:green"),
    ]:
        a, b = fr[col]
        ax.plot(amp_g, linear_fit(amp_g, a, b), color=lc, linewidth=2,
                linestyle=ls, label=f"{lbl}  {a:.5f}·A + {b:.2f}")

    for _, row in df19.iterrows():
        if pd.notna(row["amplitude"]) and pd.notna(row[col]):
            ax.annotate(f"{int(row['cycle'])}{row['hemisphere'][0].upper()}",
                        (row["amplitude"], row[col]),
                        fontsize=5.5, alpha=0.55, xytext=(2, 2),
                        textcoords="offset points")

    ax.set_xlabel("Peak amplitude (MSH)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{col} vs amplitude")
    ax.legend(fontsize=6)

plt.suptitle(
    f"8p (eq. line): Soft μ₀ + m_shared/b_shared  "
    f"(m={m_shared_eq8:.4f}, b={b_shared_eq8:.4f})",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════
# Soft μ₀ + Universal Path (a_mu, b_mu) — L-BFGS-B, 8 parameters
# ══════════════════════════════════════════════════════════════════════════
# Extend the 6-parameter soft model by also optimising the two universal
# mean-path coefficients a_mu_univ and b_mu_univ.  These remain shared
# across all hemisphere-cycles; each cycle still has its own pre-computed
# time delay (t0_refined) built into the cached τ values.
#
# Parameter vector layout (8 total):
#   [a_mu0, b_mu0, a_mupeak, b_mupeak, a_mi, b_mi, a_mu, b_mu]
#    0      1      2         3         4     5     6     7

print(f"\nSoft μ₀ + universal path (8-param) L-BFGS-B optimisation (T = {MU0_TEMP}°) ...")


def _pack_ext(fr, a_mu, b_mu):
    return np.array([
        fr["mu0"][0],     fr["mu0"][1],
        fr["mu_peak"][0], fr["mu_peak"][1],
        fr["m_i"][0],     fr["m_i"][1],
        a_mu, b_mu,
    ])


def _unpack_ext(x):
    fr = {
        "mu0":    (x[0], x[1]),
        "mu_peak": (x[2], x[3]),
        "m_i":    (x[4], x[5]),
    }
    return fr, float(x[6]), float(x[7])


def _nll_soft_ext_vec(x):
    fr, a_mu, b_mu = _unpack_ext(x)
    return compute_global_nll_soft(fr, T=MU0_TEMP, a_mu=a_mu, b_mu=b_mu)


# Warm-start from the 6-param soft result; a_mu/b_mu start at Section 3 values
x0_ext = _pack_ext(fit_results_soft, a_mu_univ, b_mu_univ)

# Bounds: a_mu ∈ [5, 30] degrees, b_mu ∈ [1, 15] years; rest unconstrained
bounds_ext = [(None, None)] * 6 + [(5.0, 30.0), (1.0, 15.0)]

opt_ext = minimize(
    _nll_soft_ext_vec,
    x0_ext,
    method="L-BFGS-B",
    bounds=bounds_ext,
    options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8},
)

fit_results_ext, a_mu_opt, b_mu_opt = _unpack_ext(opt_ext.x)

# Evaluate on both the soft metric and the hard-threshold metric
nll_ext_obj  = compute_global_nll_soft(fit_results_ext, a_mu=a_mu_opt, b_mu=b_mu_opt)
nll_ext_hard = compute_global_nll(fit_results_ext, a_mu=a_mu_opt, b_mu=b_mu_opt)

print(f"L-BFGS-B (8-param) converged: {opt_ext.success}  ({opt_ext.message})")
print(f"  Iterations: {opt_ext.nit}   Function evaluations: {opt_ext.nfev}")
print(f"\n  Optimised universal path:  a_mu = {a_mu_opt:.4f}°  (was {a_mu_univ:.4f}°)")
print(f"                             b_mu = {b_mu_opt:.4f} yr  (was {b_mu_univ:.4f} yr)")

print("\nOptimised amplitude-relationship coefficients — all strategies:")
print(f"  {'param':8s}  {'slope 19':>10s}  {'slope 23':>10s}  "
      f"{'slope soft6':>11s}  {'slope ext8':>10s}  "
      f"  {'int 19':>7s}  {'int 23':>7s}  {'int soft6':>9s}  {'int ext8':>8s}")
for key in ("mu0", "mu_peak", "m_i"):
    a19, b19 = fit_results_19[key]
    a23, b23 = fit_results_23[key]
    asf, bsf = fit_results_soft[key]
    aex, bex = fit_results_ext[key]
    print(f"  {key:8s}  {a19:>10.6f}  {a23:>10.6f}  {asf:>11.6f}  {aex:>10.6f}  "
          f"  {b19:>7.3f}  {b23:>7.3f}  {bsf:>9.3f}  {bex:>8.3f}")

# ── Scatter plots: all four regression lines ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (col, ylabel, color) in zip(axes, param_info):
    vals = df19[["amplitude", col]].dropna()
    x, y = vals["amplitude"].values, vals[col].values
    amp_g = np.linspace(x.min() * 0.9, x.max() * 1.05, 200)

    ax.scatter(x, y, color=color, s=40, alpha=0.80, edgecolors="none", zorder=3)

    for (fr, lbl, ls, lc) in [
        (fit_results_19,  "Task 19",            "--", "black"),
        (fit_results_23,  "Task 23 (NM)",       "-",  "tab:red"),
        (fit_results_soft, f"Soft 6p",          "-.", "tab:blue"),
        (fit_results_ext,  f"Soft+path 8p",     ":",  "tab:green"),
    ]:
        a, b = fr[col]
        ax.plot(amp_g, linear_fit(amp_g, a, b), color=lc, linewidth=2,
                linestyle=ls, label=f"{lbl}  {a:.5f}·A + {b:.2f}")

    for _, row in df19.iterrows():
        if pd.notna(row["amplitude"]) and pd.notna(row[col]):
            ax.annotate(f"{int(row['cycle'])}{row['hemisphere'][0].upper()}",
                        (row["amplitude"], row[col]),
                        fontsize=5.5, alpha=0.55, xytext=(2, 2),
                        textcoords="offset points")

    ax.set_xlabel("Peak amplitude (MSH)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{col} vs amplitude")
    ax.legend(fontsize=6)

plt.suptitle(
    f"8-param: Soft μ₀ + universal path  "
    f"(a_mu={a_mu_opt:.3f}°, b_mu={b_mu_opt:.3f} yr)",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════
# Split-Gaussian + Soft μ₀ + Universal Path — L-BFGS-B, 10 parameters
# ══════════════════════════════════════════════════════════════════════════
# Extend the 8p model by replacing the symmetric per-year Gaussian with a
# split-normal that has independent poleward (σ_L) and equatorward (σ_R)
# widths.  σ_L is still controlled by m_i; the new m_i_R controls σ_R.
# Both follow the same piecewise-linear σ(μ) form, so each gets its own
# amplitude relationship: m_i_R(A) = a_mi_R · A + b_mi_R.
#
# Parameter vector layout (10 total):
#   [a_mu0, b_mu0, a_mupeak, b_mupeak, a_mi, b_mi, a_mu, b_mu, a_mi_R, b_mi_R]
#    0      1      2         3         4     5     6     7     8       9
#
# σ_L = piecewise_linear_wing(μ, m_sh, b_sh, μ_peak, m_i)    ← poleward  (x ≥ μ)
# σ_R = piecewise_linear_wing(μ, m_sh, b_sh, μ_peak, m_i_R)  ← equatorward (x < μ)
# log p(x) = log(√(2/π)/(σ_L+σ_R)) − ½((x−μ)/σ_{L or R})²

print(f"\nSplit-Gaussian + Soft μ₀ + universal path (10-param) L-BFGS-B (T = {MU0_TEMP}°) ...")


def _pack_10(fr, a_mu, b_mu, mi_R_coeffs):
    return np.array([
        fr["mu0"][0],     fr["mu0"][1],
        fr["mu_peak"][0], fr["mu_peak"][1],
        fr["m_i"][0],     fr["m_i"][1],
        a_mu, b_mu,
        mi_R_coeffs[0],   mi_R_coeffs[1],
    ])


def _unpack_10(x):
    fr = {
        "mu0":    (x[0], x[1]),
        "mu_peak": (x[2], x[3]),
        "m_i":    (x[4], x[5]),
    }
    return fr, float(x[6]), float(x[7]), (float(x[8]), float(x[9]))


def _nll_split_vec(x):
    fr, a_mu, b_mu, mi_R_coeffs = _unpack_10(x)
    return compute_global_nll_soft(fr, T=MU0_TEMP, a_mu=a_mu, b_mu=b_mu,
                                   mi_R_coeffs=mi_R_coeffs)


# Warm-start: 8p result + m_i_R initialised symmetric (= m_i from 8p fit)
a_mi_ext, b_mi_ext = fit_results_ext["m_i"]
x0_10 = _pack_10(fit_results_ext, a_mu_opt, b_mu_opt,
                 (a_mi_ext, b_mi_ext))

bounds_10 = [(None, None)] * 6 + [(5.0, 30.0), (1.0, 15.0)] + [(None, None)] * 2

opt_10 = minimize(
    _nll_split_vec,
    x0_10,
    method="L-BFGS-B",
    bounds=bounds_10,
    options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8},
)

fit_results_10, a_mu_10, b_mu_10, mi_R_10 = _unpack_10(opt_10.x)

nll_10_obj  = compute_global_nll_soft(fit_results_10, a_mu=a_mu_10, b_mu=b_mu_10,
                                      mi_R_coeffs=mi_R_10)
nll_10_hard = compute_global_nll(fit_results_10, a_mu=a_mu_10, b_mu=b_mu_10,
                                 mi_R_coeffs=mi_R_10)

print(f"L-BFGS-B (10-param) converged: {opt_10.success}  ({opt_10.message})")
print(f"  Iterations: {opt_10.nit}   Function evaluations: {opt_10.nfev}")
print(f"\n  Universal path:  a_mu = {a_mu_10:.4f}°  b_mu = {b_mu_10:.4f} yr")

a_mi_R_10, b_mi_R_10 = mi_R_10
print(f"\n  Amplitude relationships:")
print(f"  {'param':8s}  {'slope 8p':>10s}  {'slope 10p':>10s}    {'int 8p':>8s}  {'int 10p':>8s}")
for key in ("mu0", "mu_peak", "m_i"):
    a8, b8   = fit_results_ext[key]
    a10, b10 = fit_results_10[key]
    print(f"  {key:8s}  {a8:>10.6f}  {a10:>10.6f}    {b8:>8.3f}  {b10:>8.3f}")
print(f"  {'m_i_R':8s}  {a_mi_ext:>10.6f}  {a_mi_R_10:>10.6f}    "
      f"{b_mi_ext:>8.3f}  {b_mi_R_10:>8.3f}  ← equatorward spread")

# ── Scatter plots: m_i vs m_i_R amplitude relationships ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (col, ylabel, color) in zip(axes, param_info):
    vals = df19[["amplitude", col]].dropna()
    x, y = vals["amplitude"].values, vals[col].values
    amp_g = np.linspace(x.min() * 0.9, x.max() * 1.05, 200)

    ax.scatter(x, y, color=color, s=40, alpha=0.80, edgecolors="none", zorder=3)

    a8, b8   = fit_results_ext[col]
    a10, b10 = fit_results_10[col]
    ax.plot(amp_g, linear_fit(amp_g, a8,  b8),  color="tab:green", linewidth=2,
            linestyle=":", label=f"8p  {a8:.5f}·A + {b8:.2f}")
    ax.plot(amp_g, linear_fit(amp_g, a10, b10), color="tab:purple", linewidth=2,
            linestyle="-", label=f"10p {a10:.5f}·A + {b10:.2f}")
    if col == "m_i":
        ax.plot(amp_g, linear_fit(amp_g, a_mi_R_10, b_mi_R_10),
                color="tab:orange", linewidth=2, linestyle="--",
                label=f"10p m_i_R {a_mi_R_10:.5f}·A + {b_mi_R_10:.2f}")

    for _, row in df19.iterrows():
        if pd.notna(row["amplitude"]) and pd.notna(row[col]):
            ax.annotate(f"{int(row['cycle'])}{row['hemisphere'][0].upper()}",
                        (row["amplitude"], row[col]),
                        fontsize=5.5, alpha=0.55, xytext=(2, 2),
                        textcoords="offset points")

    ax.set_xlabel("Peak amplitude (MSH)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{col} vs amplitude")
    ax.legend(fontsize=6)

plt.suptitle(
    f"10-param: Split-Gaussian + Soft μ₀ + universal path  "
    f"(a_mu={a_mu_10:.3f}°, b_mu={b_mu_10:.3f} yr)\n"
    f"m_i (poleward σ_L) vs m_i_R (equatorward σ_R) — rightmost panel shows both",
    fontsize=11, y=1.02,
)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════
# Full Model — All Free Parameters — L-BFGS-B, 12 parameters
# ══════════════════════════════════════════════════════════════════════════
# Combine every parameter touched by any prior experiment into a single
# joint optimisation:
#   • 6 amplitude-relationship coefficients       (Task 23 / soft 6p)
#   • m_shared, b_shared  (universal eq. line)    (8p eq)
#   • a_mu, b_mu          (universal mean path)   (8p path)
#   • a_mi_R, b_mi_R      (split-Gaussian σ_R)    (10p)
#
# Warm-started from the 10p result for the bulk + m_shared_eq8/b_shared_eq8
# for the equatorial line (the two separately optimised 8p variants meet here).
#
# Parameter vector layout (12 total):
#   [a_mu0, b_mu0, a_mupeak, b_mupeak, a_mi, b_mi,
#    m_shared, b_shared, a_mu, b_mu, a_mi_R, b_mi_R]
#    0       1      2       3      4    5
#    6        7       8     9     10      11

print(f"\nFull model (12-param) L-BFGS-B (T = {MU0_TEMP}°) ...")


def _pack_12(fr, m_sh, b_sh, a_mu, b_mu, mi_R_coeffs):
    return np.array([
        fr["mu0"][0],     fr["mu0"][1],
        fr["mu_peak"][0], fr["mu_peak"][1],
        fr["m_i"][0],     fr["m_i"][1],
        m_sh, b_sh,
        a_mu, b_mu,
        mi_R_coeffs[0],   mi_R_coeffs[1],
    ])


def _unpack_12(x):
    fr = {
        "mu0":     (x[0], x[1]),
        "mu_peak": (x[2], x[3]),
        "m_i":     (x[4], x[5]),
    }
    return (fr, float(x[6]), float(x[7]),
            float(x[8]), float(x[9]),
            (float(x[10]), float(x[11])))


def _nll_full_12_vec(x):
    fr, m_sh, b_sh, a_mu, b_mu, mi_R_coeffs = _unpack_12(x)
    return compute_global_nll_soft(fr, T=MU0_TEMP, m_sh=m_sh, b_sh=b_sh,
                                   a_mu=a_mu, b_mu=b_mu, mi_R_coeffs=mi_R_coeffs)


x0_12 = _pack_12(fit_results_10, m_shared_eq8, b_shared_eq8,
                 a_mu_10, b_mu_10, mi_R_10)

bounds_12 = (
    [(None, None)] * 6            # amplitude relationship coefficients
    + [(0.0, 2.0), (-5.0, 5.0)]   # m_shared ∈ [0,2], b_shared ∈ [-5,5]
    + [(5.0, 30.0), (1.0, 15.0)]  # a_mu (°), b_mu (yr)
    + [(None, None)] * 2          # a_mi_R, b_mi_R
)

opt_12 = minimize(
    _nll_full_12_vec,
    x0_12,
    method="L-BFGS-B",
    bounds=bounds_12,
    options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8},
)

fit_results_12, m_shared_12, b_shared_12, a_mu_12, b_mu_12, mi_R_12 = _unpack_12(opt_12.x)
a_mi_R_12, b_mi_R_12 = mi_R_12

nll_12_obj  = compute_global_nll_soft(fit_results_12, m_sh=m_shared_12, b_sh=b_shared_12,
                                      a_mu=a_mu_12, b_mu=b_mu_12, mi_R_coeffs=mi_R_12)
nll_12_hard = compute_global_nll(fit_results_12, m_sh=m_shared_12, b_sh=b_shared_12,
                                 a_mu=a_mu_12, b_mu=b_mu_12, mi_R_coeffs=mi_R_12)

print(f"L-BFGS-B (12p) converged: {opt_12.success}  ({opt_12.message})")
print(f"  Iterations: {opt_12.nit}   Function evaluations: {opt_12.nfev}")
print(f"\n  Universal path:    a_mu = {a_mu_12:.4f}°  b_mu = {b_mu_12:.4f} yr"
      f"  (10p: {a_mu_10:.4f}°, {b_mu_10:.4f} yr)")
print(f"  Equatorial line:   m_shared = {m_shared_12:.4f}  b_shared = {b_shared_12:.4f}"
      f"  (eq8: {m_shared_eq8:.4f}, {b_shared_eq8:.4f})")
print(f"  Zero crossing at μ = {-b_shared_12 / m_shared_12:.2f}°")

a_mi_R_10, b_mi_R_10 = mi_R_10
print(f"\n  {'param':8s}  {'slope 10p':>10s}  {'slope 12p':>10s}    {'int 10p':>8s}  {'int 12p':>8s}")
for key in ("mu0", "mu_peak", "m_i"):
    a10, b10 = fit_results_10[key]
    a12, b12 = fit_results_12[key]
    print(f"  {key:8s}  {a10:>10.6f}  {a12:>10.6f}    {b10:>8.3f}  {b12:>8.3f}")
print(f"  {'m_i_R':8s}  {a_mi_R_10:>10.6f}  {a_mi_R_12:>10.6f}    {b_mi_R_10:>8.3f}  {b_mi_R_12:>8.3f}"
      "  ← equatorward spread")

# ── Scatter plots: all strategies ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (col, ylabel, color) in zip(axes, param_info):
    vals = df19[["amplitude", col]].dropna()
    x, y = vals["amplitude"].values, vals[col].values
    amp_g = np.linspace(x.min() * 0.9, x.max() * 1.05, 200)

    ax.scatter(x, y, color=color, s=40, alpha=0.80, edgecolors="none", zorder=3)

    for (fr, lbl, ls, lc, lw) in [
        (fit_results_19,   "Task 19",        "--",         "black",      1.5),
        (fit_results_23,   "Task 23 (NM)",   "-",          "tab:red",    1.5),
        (fit_results_soft, "Soft 6p",        "-.",         "tab:blue",   1.5),
        (fit_results_eq8,  "Soft 8p (eq)",   ":",          "tab:green",  1.5),
        (fit_results_12,   "Full 12p",       (0, (5, 1)),  "tab:purple", 2.5),
    ]:
        a, b = fr[col]
        ax.plot(amp_g, linear_fit(amp_g, a, b), color=lc, linewidth=lw,
                linestyle=ls, label=f"{lbl}  {a:.5f}·A + {b:.2f}", zorder=4 if lw > 2 else 3)

    for _, row in df19.iterrows():
        if pd.notna(row["amplitude"]) and pd.notna(row[col]):
            ax.annotate(f"{int(row['cycle'])}{row['hemisphere'][0].upper()}",
                        (row["amplitude"], row[col]),
                        fontsize=5.5, alpha=0.55, xytext=(2, 2),
                        textcoords="offset points")

    ax.set_xlabel("Peak amplitude (MSH)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{col} vs amplitude")
    ax.legend(fontsize=6)

plt.suptitle(
    f"Full 12p: all free parameters  "
    f"(a_mu={a_mu_12:.3f}°, b_mu={b_mu_12:.3f} yr, "
    f"m_sh={m_shared_12:.4f}, b_sh={b_shared_12:.4f})",
    fontsize=11, y=1.02,
)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════
# Model Quality Scoreboard
# ══════════════════════════════════════════════════════════════════════════
#
# NLL values differ along two independent axes:
#
# METRIC AXIS — how years near the cycle boundary are handled
# ─────────────────────────────────────────────────────────────────────────
#   hard  (compute_global_nll)
#         Binary cutoff: years with μ > μ₀ are dropped entirely; the rest
#         contribute equally.  This is the SCOREBOARD CURRENCY — all model
#         rows are evaluated with it so they are directly comparable.
#
#   soft  (compute_global_nll_soft)
#         Sigmoid weight w = 1/(1+exp((μ−μ₀)/T)), T=1°.  Every year
#         contributes, but pre-cycle years (μ >> μ₀) are down-weighted to
#         ≈ 0.  Normalised by Σw instead of a count.  This is what L-BFGS-B
#         actually minimised, so soft models score best on this metric.
#
# MODEL AXIS — parameters optimised end-to-end
# ─────────────────────────────────────────────────────────────────────────
#   6p   6 amplitude-relationship coefficients; path fixed.
#   8p   same 6 + a_mu/b_mu of the universal mean path.
#   10p  same 8 + split-Gaussian: m_i_R(A) for equatorward spread σ_R.
#        σ_L (poleward, x≥μ) ← m_i;  σ_R (equatorward, x<μ) ← m_i_R.
#        log p = log(√(2/π)/(σ_L+σ_R)) − ½((x−μ)/σ_{L or R})²
#
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────
#             hard metric (scoreboard)   soft metric (objective)
#   6p        nll_soft_hard              nll_soft_obj  ← minimised
#   8p        nll_ext_hard               nll_ext_obj   ← minimised
#   10p       nll_10_hard                nll_10_obj    ← minimised
#
# Interpretation:
#   nll_Xp_hard < nll_{X-2}p_hard → extra parameters genuinely help
#   nll_Xp_hard ≈ nll_{X-2}p_hard → diminishing returns; simpler is better
#   nll_soft_obj < nll_soft_hard   → expected: sigmoid normalisation assigns
#                                    lower cost than the hard count
# ══════════════════════════════════════════════════════════════════════════

nll_23_hard = compute_global_nll(fit_results_23)

pct_23   = (nll_baseline - nll_23_hard)   / abs(nll_baseline) * 100
pct_soft = (nll_baseline - nll_soft_hard) / abs(nll_baseline) * 100
pct_eq8  = (nll_baseline - nll_eq8_hard)  / abs(nll_baseline) * 100
pct_ext  = (nll_baseline - nll_ext_hard)  / abs(nll_baseline) * 100
pct_10   = (nll_baseline - nll_10_hard)   / abs(nll_baseline) * 100
pct_12   = (nll_baseline - nll_12_hard)   / abs(nll_baseline) * 100

print("\n" + "=" * 70)
print("  Model Quality Scoreboard")
print("=" * 70)
print("  METRIC — hard: binary μ>μ₀ cutoff, equal-weight mean NLL")
print(f"           soft: sigmoid w=1/(1+exp((μ−μ₀)/{MU0_TEMP}°)), Σw-normalised")
print()
print("  MODEL  — 6p:       6 amplitude coefficients; equatorial line and path fixed")
print("           8p (eq): + m_shared/b_shared (universal equatorial line)")
print("           8p:      + a_mu/b_mu (universal mean path)")
print("           10p:     + m_i_R (split-Gaussian equatorward spread)")
print("           12p:     all of the above jointly")
print()
print(f"  {'Model':<44s}  {'hard NLL':>10s}  {'vs baseline':>12s}")
print(f"  {'-'*44}  {'-'*10}  {'-'*12}")
print(f"  {'Baseline (Task 19, two-stage)':<44s}  {nll_baseline:>10.5f}  {'—':>12s}")
print(f"  {'Task 23  (hard cutoff, Nelder-Mead, 6p)':<44s}  {nll_23_hard:>10.5f}  {pct_23:>+11.2f}%")
print(f"  {f'Soft μ₀ T={MU0_TEMP}° (L-BFGS-B, 6p)':<44s}  {nll_soft_hard:>10.5f}  {pct_soft:>+11.2f}%")
print(f"  {'Soft μ₀ + eq. line (L-BFGS-B, 8p eq)':<44s}  {nll_eq8_hard:>10.5f}  {pct_eq8:>+11.2f}%")
print(f"  {'Soft μ₀ + path (L-BFGS-B, 8p)':<44s}  {nll_ext_hard:>10.5f}  {pct_ext:>+11.2f}%")
print(f"  {'Split-Gaussian + path (L-BFGS-B, 10p)':<44s}  {nll_10_hard:>10.5f}  {pct_10:>+11.2f}%")
print(f"  {'─ ' * 33}")
print(f"  {'Full model (all params, L-BFGS-B, 12p)':<44s}  {nll_12_hard:>10.5f}  {pct_12:>+11.2f}%")
print(f"  {'─ ' * 33}")
print()
print(f"  {'Model':<44s}  {'soft NLL':>10s}  {'note':>14s}")
print(f"  {'-'*44}  {'-'*10}  {'-'*14}")
print(f"  {'Soft μ₀  (6p) — what L-BFGS-B minimised':<44s}  {nll_soft_obj:>10.5f}  {'← 6p objective':>18s}")
print(f"  {'Soft+eq. line (8p eq) — minimised':<44s}  {nll_eq8_obj:>10.5f}  {'← 8p(eq) objective':>18s}")
print(f"  {'Soft+path (8p) — what L-BFGS-B minimised':<44s}  {nll_ext_obj:>10.5f}  {'← 8p objective':>18s}")
print(f"  {'Split+path (10p) — what L-BFGS-B minimised':<44s}  {nll_10_obj:>10.5f}  {'← 10p objective':>18s}")
print(f"  {'─ ' * 33}")
print(f"  {'Full model (12p) — what L-BFGS-B minimised':<44s}  {nll_12_obj:>10.5f}  {'← 12p objective':>18s}")
print(f"  {'─ ' * 33}")
print("=" * 70)
