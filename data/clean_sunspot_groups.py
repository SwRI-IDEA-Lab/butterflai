"""
clean_sunspot_groups.py

Reduce composite_sunspot_groups_daily_measurements_10_23.csv so that each
sunspot group (uniqueID) appears exactly once, at the observation with its
maximum corrected area.  Rows with no uniqueID represent days with no observed
spots and are kept as-is.

Outputs
-------
composite_sunspot_groups_peak_area.csv   — cleaned dataset
butterfly_input_vs_output.png            — two-panel butterfly diagram comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent
RAW_PATH  = DATA_DIR / "composite_sunspot_groups_daily_measurements_10_23.csv"
OUT_CSV   = DATA_DIR / "composite_sunspot_groups_peak_area.csv"
OUT_FIG   = DATA_DIR / "butterfly_input_vs_output.png"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading data …")
df = pd.read_csv(RAW_PATH)
print(f"  Raw shape: {df.shape}")

# ---------------------------------------------------------------------------
# Build a decimal-year time column (used for plotting)
# ---------------------------------------------------------------------------
# Use day-of-year / 365.25 to avoid leap-year edge cases.
day_of_year = (
    pd.to_datetime(
        df[["year", "month", "day"]]
        .rename(columns={"year": "year", "month": "month", "day": "day"})
    ).dt.day_of_year
)
df["time"] = df["year"] + (day_of_year - 1) / 365.25

# ---------------------------------------------------------------------------
# Split: rows with a uniqueID vs. spotless days (no uniqueID)
# ---------------------------------------------------------------------------
has_id = df["uniqueID"].notna()
df_spotless = df[~has_id].copy()   # days without observed spots — keep all
df_groups   = df[ has_id].copy()   # individual group observations — deduplicate

print(f"  Spotless / no-ID rows (kept as-is): {len(df_spotless)}")
print(f"  Group observation rows: {len(df_groups)}")

# ---------------------------------------------------------------------------
# Reduce: keep one row per uniqueID — the observation at peak corrected area
# ---------------------------------------------------------------------------
# Sort chronologically so idxmax returns the first occurrence of the maximum
# when there are ties.
df_groups_sorted = df_groups.sort_values("time")
idx_peak = df_groups_sorted.groupby("uniqueID")["correctedArea"].idxmax()
df_peak = df_groups_sorted.loc[idx_peak]

print(f"  Unique groups after deduplication: {len(df_peak)}")

# Recombine and sort by time, preserving original row order within a day.
df_clean = (
    pd.concat([df_spotless, df_peak], ignore_index=True)
    .sort_values("time")
    .reset_index(drop=True)
)

print(f"  Final dataset shape: {df_clean.shape}")
print(f"  Time range: {df_clean['time'].min():.1f} – {df_clean['time'].max():.1f}")

with_lat = df_clean["latitude"].notna()
print(f"  Latitude range (group rows): {df_clean.loc[with_lat,'latitude'].min():.1f}° – "
      f"{df_clean.loc[with_lat,'latitude'].max():.1f}°")
with_area = df_clean["correctedArea"].notna()
print(f"  Corrected area range: {df_clean.loc[with_area,'correctedArea'].min():.0f} – "
      f"{df_clean.loc[with_area,'correctedArea'].max():.0f} μHem")

# ---------------------------------------------------------------------------
# Save cleaned CSV
# ---------------------------------------------------------------------------
df_clean.to_csv(OUT_CSV, index=False)
print(f"\nSaved cleaned CSV → {OUT_CSV}")

# ---------------------------------------------------------------------------
# Figure: two-panel butterfly diagram
# ---------------------------------------------------------------------------
def scatter_butterfly(ax, time, lat, area, title, *, alpha=0.15, s_scale=0.04):
    """Scatter plot coloured by hemisphere, sized by corrected area."""
    # Clip area for marker sizing — very large groups would overwhelm the plot
    s = np.clip(area, 1, 5000) * s_scale + 0.5

    north = lat >= 0
    south = lat <  0

    ax.scatter(time[north], lat[north], s=s[north],
               color="tab:red",  alpha=alpha, linewidths=0, rasterized=True)
    ax.scatter(time[south], lat[south], s=s[south],
               color="tab:blue", alpha=alpha, linewidths=0, rasterized=True)

    ax.axhline(0, color="0.5", lw=0.5, ls="--")
    ax.set_ylabel("Latitude (°)", fontsize=10)
    ax.set_ylim(-45, 45)
    ax.set_yticks([-40, -20, 0, 20, 40])
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=9)


print("Generating butterfly diagram …")

fig, axes = plt.subplots(
    2, 1,
    figsize=(14, 7),
    sharex=True,
    gridspec_kw={"hspace": 0.08},
)

# Scatter plots only use rows that have both latitude and corrected area.
raw_plot  = df_groups_sorted.dropna(subset=["latitude", "correctedArea"])
clean_plot = df_peak.dropna(subset=["latitude", "correctedArea"])

# --- Top panel: all daily group observations (before deduplication) ---
scatter_butterfly(
    axes[0],
    raw_plot["time"].values,
    raw_plot["latitude"].values,
    raw_plot["correctedArea"].values,
    f"Input — all daily group observations (N = {len(raw_plot):,})",
)

# --- Bottom panel: one row per group at peak corrected area ---
scatter_butterfly(
    axes[1],
    clean_plot["time"].values,
    clean_plot["latitude"].values,
    clean_plot["correctedArea"].values,
    f"Output — one observation per group at peak corrected area "
    f"(N = {len(clean_plot):,})",
    alpha=0.25,
)

axes[1].set_xlabel("Year", fontsize=10)

# Shared legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red",
           markersize=6, label="Northern hemisphere"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
           markersize=6, label="Southern hemisphere"),
]
axes[0].legend(handles=legend_elements, loc="upper left", fontsize=8,
               framealpha=0.7)

fig.suptitle(
    "Solar Butterfly Diagram — Composite Sunspot Group Catalog (Cycles 7–25)",
    fontsize=12, fontweight="bold", y=0.98,
)

fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure    → {OUT_FIG}")
print("Done.")
