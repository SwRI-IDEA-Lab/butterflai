#!/usr/bin/env python3
"""
fit_sarge_sigma_width.py

Fits the scale parameter k (replacing 400 in the SARGE paper) in the
relationship between instantaneous total sunspot area and emergence-latitude
spread:

    σ_λ(A) = 1.5° + 3.8° × [1 − exp(−A_Total / k)]

The constants 1.5°, 3.8°, and the mean-path parameters are fixed to
SARGE/paper values and are NOT optimization targets.  Only k is fit.

Data:   composite_sunspot_groups_peak_area.csv
Filter: correctedArea > 30 MSH  (matches the base threshold in
        06_2D_Wings_metrics.ipynb, cell 3)
Cycles: ≥ 12 (same as the notebook)

The loss is the mean per-year-normalized NLL across every valid
(cycle, hemisphere, year) bin, matching the scoreboard metric used in
Tasks 21–24 of 06_2D_Wings_metrics.ipynb.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm as sp_norm
from scipy.optimize import curve_fit, minimize_scalar

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
DATA_PATH  = REPO_ROOT / "data" / "composite_sunspot_groups_peak_area.csv"

# ── SARGE constants (paper-prescribed, NOT optimised) ─────────────────────
SARGE_SIGMA_MIN = 1.5   # degrees — floor spread
SARGE_SIGMA_AMP = 3.8   # degrees — maximum additional spread

# ── Data-pipeline constants (mirror 06_2D_Wings_metrics.ipynb) ─────────────
AREA_THRESHOLD  = 30    # MSH — correctedArea filter (cell 3)
MIN_CYCLE       = 12    # first cycle with reliable composite data
N_BINS          = 20    # tau-bins for μ(τ) fitting
MIN_LATS_BIN    = 10    # minimum latitudes per tau-bin to include in fit
MIN_LATS_YEAR   = 5     # minimum latitudes per year to enter NLL cache
SMOOTH_WIN      = 365   # rolling window (days) for total-area smoothing


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load and pre-process data
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH, parse_dates=[[0, 1, 2]], keep_date_col=False)
df.rename(columns={"year_month_day": "date"}, inplace=True)
df = df[df["latitude"].notna()].copy()

df["hemisphere"]   = df["latitude"].apply(lambda v: "north" if v >= 0 else "south")
df["abs_latitude"] = df["latitude"].abs()
df["year"]         = df["date"].dt.year
df["decimal_year"] = df["date"].dt.year + df["date"].dt.dayofyear / 365.25

df = df[df["correctedArea"] > AREA_THRESHOLD].copy()
print(f"Rows after correctedArea > {AREA_THRESHOLD} filter: {len(df):,}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Standardise time: align cycles at the 15° crossing (τ)
#     (mirrors cell 5 of 06_2D_Wings_metrics.ipynb)
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Fit universal mean path μ(τ) and refine t₀ per hemisphere-cycle
#     (mirrors cells 7–8 of 06_2D_Wings_metrics.ipynb)
# ══════════════════════════════════════════════════════════════════════════════
def exp_decay(tau, a, b):
    return a * np.exp(-tau / b)


cycles_13 = [c for c in sorted(df["CYCLE"].dropna().unique()) if c >= MIN_CYCLE]

all_tau_bins, all_mu_bins = [], []
hemicycle_bins = {}

for cyc in cycles_13:
    for hemi in ["north", "south"]:
        if (cyc, hemi) not in t0_lookup:
            continue
        mask   = (df["CYCLE"] == cyc) & (df["hemisphere"] == hemi) & df["tau"].notna()
        df_sel = df[mask]
        if len(df_sel) < 50:
            continue
        t_min, t_max = df_sel["tau"].min(), df_sel["tau"].max()
        bins         = np.linspace(t_min, t_max, N_BINS + 1)
        bin_centers  = 0.5 * (bins[:-1] + bins[1:])
        bt, bm = [], []
        for i in range(N_BINS):
            lats_bin = df_sel.loc[
                (df_sel["tau"] >= bins[i]) & (df_sel["tau"] < bins[i + 1]),
                "abs_latitude"].values
            if len(lats_bin) < MIN_LATS_BIN:
                continue
            mu_f, _ = sp_norm.fit(lats_bin)
            bt.append(bin_centers[i])
            bm.append(mu_f)
        if len(bt) < 5:
            continue
        bt, bm = np.array(bt), np.array(bm)
        hemicycle_bins[(cyc, hemi)] = (bt, bm)
        all_tau_bins.extend(bt)
        all_mu_bins.extend(bm)

popt, _ = curve_fit(exp_decay, all_tau_bins, all_mu_bins, p0=[15.0, 5.0])
a_mu_univ, b_mu_univ = popt
print(f"Universal μ(τ):  a = {a_mu_univ:.4f}°   b = {b_mu_univ:.4f} yr")

# Refine t₀ per hemisphere-cycle
t0_refined = {}
for (cyc, hemi), (bt, bm) in hemicycle_bins.items():
    def _res(dt, _bt=bt, _bm=bm):
        return np.sum((_bm - exp_decay(_bt - dt, a_mu_univ, b_mu_univ)) ** 2)
    res = minimize_scalar(_res, bounds=(-4, 4), method="bounded")
    t0_refined[(cyc, hemi)] = t0_lookup[(cyc, hemi)] + res.x

df["t0_refined"]  = df.apply(
    lambda r: t0_refined.get((r["CYCLE"], r["hemisphere"]), np.nan), axis=1)
df["tau_refined"] = df["decimal_year"] - df["t0_refined"]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Global total sunspot area (both hemispheres), smoothed
#
#     SARGE derives σ_λ from the global total area A_Total, not a per-hemisphere
#     area — it draws each AR's latitude from N(μ_λ, σ_λ) where σ_λ comes from
#     the whole-Sun activity level.  We therefore sum across hemispheres before
#     smoothing.  (mirrors Task 17 of 06_2D_Wings_metrics.ipynb)
# ══════════════════════════════════════════════════════════════════════════════
daily_north = df[df["hemisphere"] == "north"].groupby("date")["correctedArea"].sum()
daily_south = df[df["hemisphere"] == "south"].groupby("date")["correctedArea"].sum()

date_range = pd.date_range(
    min(daily_north.index.min(), daily_south.index.min()),
    max(daily_north.index.max(), daily_south.index.max()),
    freq="D",
)
daily_north = daily_north.reindex(date_range, fill_value=0)
daily_south = daily_south.reindex(date_range, fill_value=0)
daily_total = daily_north + daily_south   # global total

smooth_total = daily_total.rolling(SMOOTH_WIN, center=True, min_periods=SMOOTH_WIN // 3).mean()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Build per-(cycle, hemisphere, year) data cache
#     Each entry: observed absolute latitudes, predicted μ, global A_Total
# ══════════════════════════════════════════════════════════════════════════════
cycle_year_data = []   # list of (lats, mu, A_total)

for cyc in cycles_13:
    for hemi in ["north", "south"]:
        if (cyc, hemi) not in t0_refined:
            continue
        mask  = (df["CYCLE"] == cyc) & (df["hemisphere"] == hemi) & df["tau_refined"].notna()
        df_ch = df[mask]
        for yr in sorted(df_ch["year"].unique()):
            lats = df_ch.loc[df_ch["year"] == yr, "abs_latitude"].values
            if len(lats) < MIN_LATS_YEAR:
                continue
            tau_mid = (yr + 0.5) - t0_refined[(cyc, hemi)]
            mu      = float(exp_decay(tau_mid, a_mu_univ, b_mu_univ))
            # Global smoothed total area at mid-year (July 2 ≈ day 183)
            mid_date = pd.Timestamp(year=yr, month=7, day=2)
            idx      = smooth_total.index.get_indexer([mid_date], method="nearest")[0]
            A_total  = float(smooth_total.iloc[idx])
            if np.isnan(A_total) or A_total <= 0:
                continue
            cycle_year_data.append((lats, mu, A_total))

n_entries = len(cycle_year_data)
print(f"NLL cache: {n_entries} valid (cycle, hemisphere, year) entries")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SARGE σ model and NLL objective
# ══════════════════════════════════════════════════════════════════════════════
def sarge_sigma(A_total: float, k: float) -> float:
    """
    SARGE emergence-latitude spread.

    Parameters
    ----------
    A_total : total sunspot area in MSH (= µHem) at time of emergence
    k       : scale parameter to be fit (replaces 400 in the paper)

    Returns
    -------
    σ_λ in degrees
    """
    return SARGE_SIGMA_MIN + SARGE_SIGMA_AMP * (1.0 - np.exp(-A_total / k))


def nll_objective(k: float) -> float:
    """
    Mean per-year-normalised NLL across all cache entries.

    Uses the same convention as compute_global_nll in 06_2D_Wings_metrics.ipynb:
    for each year-bin, compute −mean(log p(lats | μ, σ_SARGE)) and average
    across all bins.
    """
    if k <= 0:
        return 1e10
    total = 0.0
    for lats, mu, A_total in cycle_year_data:
        sigma = sarge_sigma(A_total, k)
        total -= sp_norm.logpdf(lats, loc=mu, scale=sigma).mean()
    return total / n_entries


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Optimise k
#
#     The SARGE paper derives k = 400 from SSN-based area estimates
#     (A = 10 × SSN_v2, giving global totals in the hundreds-to-thousands µHem).
#     Our composite catalog directly measures area, so the smoothed global total
#     spans a much smaller range (median ≈ 100 MSH, max ≈ 500 MSH).  k must
#     therefore be calibrated to this measurement scale — the optimum is
#     expected to lie well below 400.
# ══════════════════════════════════════════════════════════════════════════════
A_vals = [A for _, _, A in cycle_year_data]
print(f"Global area range in cache — "
      f"min: {min(A_vals):.1f}  median: {np.median(A_vals):.1f}  "
      f"mean: {np.mean(A_vals):.1f}  max: {max(A_vals):.1f}  MSH")

result = minimize_scalar(
    nll_objective,
    bounds=(0.5, 2000.0),
    method="bounded",
    options={"xatol": 0.1},
)
k_fit    = result.x
nll_fit  = result.fun
nll_400  = nll_objective(400.0)

print(f"\n{'─'*55}")
print(f"  Fitted k         : {k_fit:>10.1f}  MSH")
print(f"  NLL at k = {k_fit:<6.1f}  : {nll_fit:>10.5f}  nats/year")
print(f"  NLL at k = 400   : {nll_400:>10.5f}  nats/year")
print(f"  ΔNLL             : {nll_fit - nll_400:>+10.5f}  nats/year")
print(f"{'─'*55}")
A_median = float(np.median(A_vals))
A_max    = float(np.max(A_vals))
print(f"\n  σ_λ(A=0)         = {sarge_sigma(0,        k_fit):.2f}°  (no activity)")
print(f"  σ_λ(A={A_median:.0f})    = {sarge_sigma(A_median, k_fit):.2f}°  (median activity)")
print(f"  σ_λ(A={A_max:.0f})   = {sarge_sigma(A_max,   k_fit):.2f}°  (peak activity)")
print(f"  σ_λ(A→∞)         = {SARGE_SIGMA_MIN + SARGE_SIGMA_AMP:.2f}°  (saturation)")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Diagnostic plots
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Left: NLL vs k (log scale to reveal shallow minimum)
k_grid   = np.logspace(np.log10(0.5), np.log10(2000), 300)
nll_grid = [nll_objective(k) for k in k_grid]

axes[0].plot(k_grid, nll_grid, color="steelblue", linewidth=2)
axes[0].axvline(k_fit, color="tab:red",    linewidth=2,   linestyle="--",
                label=f"k_fit = {k_fit:.1f}  (NLL = {nll_fit:.4f})")
axes[0].axvline(400,   color="tab:orange", linewidth=1.5, linestyle=":",
                label=f"k = 400 paper  (NLL = {nll_400:.4f})")
axes[0].axhline(nll_fit, color="tab:red",    linewidth=1, linestyle="--", alpha=0.4)
axes[0].axhline(nll_400, color="tab:orange", linewidth=1, linestyle=":",  alpha=0.4)
axes[0].set_xscale("log")
axes[0].set_xlabel("k  (MSH, log scale)")
axes[0].set_ylabel("Mean NLL  (nats / year)")
axes[0].set_title("NLL objective vs. scale parameter k")
axes[0].legend(fontsize=8)

# Middle: σ_λ(A) curves over the data range
A_grid = np.linspace(0, max(A_vals) * 1.1, 400)
axes[1].plot(A_grid, sarge_sigma(A_grid, k_fit),
             color="tab:red",    linewidth=2,
             label=f"k = {k_fit:.1f}  (fitted to data)")
axes[1].plot(A_grid, sarge_sigma(A_grid, 400.0),
             color="tab:orange", linewidth=2, linestyle="--",
             label="k = 400  (SARGE paper)")
axes[1].axvline(np.median(A_vals), color="gray", linewidth=1, linestyle=":",
                label=f"median A = {np.median(A_vals):.0f}")
axes[1].set_xlabel("A_Total  (MSH, global total)")
axes[1].set_ylabel("σ_λ  (degrees)")
axes[1].set_title("σ_λ(A) over the observed area range")
axes[1].legend(fontsize=8)

# Right: histogram of A_Total values in cache
axes[2].hist(A_vals, bins=30, color="steelblue", alpha=0.75, edgecolor="white")
axes[2].axvline(np.median(A_vals), color="gray",       linestyle="--", label=f"median = {np.median(A_vals):.0f}")
axes[2].axvline(400,               color="tab:orange",  linestyle=":",  label="k = 400 (paper)")
axes[2].axvline(k_fit,             color="tab:red",     linestyle="--", label=f"k_fit = {k_fit:.1f}")
axes[2].set_xlabel("A_Total  (MSH, global total)")
axes[2].set_ylabel("Count  (year-bins)")
axes[2].set_title("Distribution of A_Total in NLL cache")
axes[2].legend(fontsize=8)

plt.suptitle(
    f"SARGE σ_λ fit — butterfly data (cycles ≥ {MIN_CYCLE}, correctedArea > {AREA_THRESHOLD} MSH)\n"
    f"σ_λ = {SARGE_SIGMA_MIN}° + {SARGE_SIGMA_AMP}° × [1 − exp(−A_Total / k)]"
    f"     k_fit = {k_fit:.1f}  (paper: 400)"
    f"     NLL: {nll_fit:.4f} (fitted)  vs  {nll_400:.4f} (paper)  "
    f"[ΔNLL = {nll_fit - nll_400:+.4f} nats/yr]",
    fontsize=10,
)
plt.tight_layout()
out_path = Path(__file__).parent / "sarge_sigma_fit.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nDiagnostic plot saved → {out_path}")
plt.show()
