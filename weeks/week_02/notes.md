# Week 02 — Session Notes (March 27, 2026)

## Notebook: `02_distributions.ipynb`

---

## Data

- **File:** `data/composite_sunspot_groups_daily_measurements_10_23.csv`
- **Content:** Daily sunspot group measurements, 1825–2023
- **Key derived columns:**
  - `df["date"]` — parsed from the first three columns (year/month/day), using a version-invariant construction (see Shea's note in cell 4)
  - `df["hemisphere"]` — `"north"` if `latitude >= 0`, else `"south"`
  - `df["year"]` — `df["date"].dt.year`

---

## pandas `read_csv` versioning fix (Shea's note)

`keep_date_col=False` is now the default and the keyword is no longer valid. The original `parse_dates` call used a list-of-lists which is also deprecated. Fixed by constructing the date string manually:

```python
df["date"] = pd.to_datetime(
    df.iloc[:, 0].astype(str) + "-" +
    df.iloc[:, 1].astype(str).str.zfill(2) + "-" +
    df.iloc[:, 2].astype(str).str.zfill(2)
)
df.drop(columns=df.columns[:3], inplace=True)
```

---

## Task 6 — Empirical histogram (1987, north)

- Year: 1987, Hemisphere: north
- 15-bin density-normalised histogram
- Vertical lines: Q1, Q3, median (black solid), mean (blue dash-dot), mode (green dotted)
- IQR shaded in orange

**Shea's note:** The 1987 northern hemisphere data contains a mix of two solar cycles — Cycle 22 (~83%, 200/240) and Cycle 21 (~9%, 21/240). There is a striking outlier at ~57° which is a real Cycle 22 emergence from August 1987 — not a data error. The cycle overlap makes the Gaussian wider and pulls the mean equatorward.

---

## Task 7 — Gaussian fitting (three plots)

Uses `scipy.stats.norm.fit(data)` → `(mu, sigma)` (MLE).

All three plots include:
- Histogram (density-normalised)
- Gaussian fit curve (black solid)
- Vertical lines: **μ** (solid) and **μ ± σ** (dashed)

| Plot | Data | Color |
|------|------|-------|
| 1 | 1987 north, all cycles — stacked bars by cycle (C22=blue, C21=red) | dual legend |
| 2 | Full Cycle 22, north | orange |
| 3 | 1987 north, Cycle 22 only | blue |

**Summary (approximate):**
- All cycles 1987 north: μ ≈ 14–15°, σ wider due to cycle mix
- Full Cycle 22 north: σ largest (spans whole cycle's latitude range)
- 1987 C22 only: σ narrower, μ slightly higher than mixed sample

---

## Task 8 — KDE evolution (Spörer's Law)

Uses `scipy.stats.gaussian_kde(bw_method=0.35)`.

- Cycle 22, northern hemisphere
- One KDE curve per year, colored by a **discrete `rainbow_r` colormap** (red = early years → purple = late years)
- Continuous colorbar with year labels
- Equatorward drift of the KDE peak over time demonstrates **Spörer's Law**

---

## Key library notes

| Function | Purpose |
|----------|---------|
| `scipy.stats.norm.fit(data)` | MLE Gaussian fit → `(mu, sigma)` |
| `scipy.stats.gaussian_kde(bw_method=0.35)` | KDE estimation |
| `plt.cm.get_cmap("rainbow_r", N)` | Discrete reversed rainbow colormap |
| `matplotlib.colors.BoundaryNorm` | Forces colorbar into hard discrete bands |
| `matplotlib.patches.Patch` | Custom legend elements |

---

## Notebook recovery note

During this session, Tasks 6 and 7 cells were accidentally cleared. They were reconstructed from conversation memory. If cells are lost again, the code above is the authoritative source. VS Code local history is at `~/.config/Code/User/History/-11d54fe5/` but may not have saved content.
