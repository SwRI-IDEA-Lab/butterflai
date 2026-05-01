"""
mat_to_csv.py — Regenerate composite_sunspot_groups_daily_measurements_10_23.csv
               with correct, globally unique uniqueID values.

Background
----------
The existing composite CSV contains authoritative correctedArea values from
AMJ's calibration pipeline but has uniqueID corrupted by limited float
precision in the original MATLAB export (≤ 3 significant figures, which
collapses distinct 6-digit IDs like 187234, 187235 → 187000).

Strategy
--------
For each row that has both a (corrupted) uniqueID and a position (latitude,
longitude), match it back to the original instrument file by
    (year, month, day, round(lat, 1°), round(lon, 1°))
and recover the source group number.  A fresh, globally unique string ID
is then assigned that encodes survey origin, and year for KMAS (whose group
numbers reset annually).

New ID conventions
------------------
  GPR_<n>           Royal Greenwich Observatory, group #n
                    (cumulative across years 1874-1976; globally unique)
  DPD_<n>           Debrecen Photoheliographic Data, group #n
                    (cumulative across years 1974-2016; globally unique)
  KMAS_<yyyy>_<n>   Kislovodsk Mountain Station, group #n of year yyyy
                    (IDs reset each year; year qualifier makes them unique)
  SC_<id>           Schwabe catalog group (id may be decimal, e.g. SC_984.1)
  SP_<n>            Spoerer catalog group #n

Rows without a uniqueID (spotless days / no-observation records) keep NaN.

Usage
-----
    python mat_to_csv.py
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent
RAW_CSV  = DATA_DIR / "composite_sunspot_groups_daily_measurements_10_23.csv"
OUT_CSV  = RAW_CSV   # overwrite in-place (authoritative source)


# ── helpers ──────────────────────────────────────────────────────────────────

def _fmt_sc_id(val: float) -> str:
    """Format a Schwabe/Spoerer uniqueID float as a minimal string."""
    if val == int(val):
        return str(int(val))
    # Remove trailing zeros: 984.10 → "984.1"
    return f"{val:.10g}"


def _round_key(lat: float, lon: float) -> tuple[float, float]:
    # Cast to Python float before rounding: numpy float64 and Python float have
    # different __round__ implementations that can produce different results for
    # values like 265.15 (numpy: 265.2, Python: 265.1). Using float() ensures
    # the same rounding rule applies when building the lookup and querying it.
    return round(float(lat), 1), round(float(lon), 1)


# ── parse GPR / DPD instrument files ────────────────────────────────────────
# Both use the same text format:
#   g YYYY MM DD HH MM SS  GROUP_ID  ... lat lon ...
# where col 7 = group_id (may have alpha suffix like "278b"),
#       col 12 = heliographic latitude, col 13 = Carrington longitude.

def _parse_gpr_dpd(directory: Path, glob: str) -> dict[tuple, str]:
    """Return lookup (year, month, day, lat_r, lon_r) → group_id_str."""
    lookup: dict[tuple, str] = {}
    collisions: list[tuple] = []

    for fpath in sorted(directory.glob(glob)):
        for line in open(fpath, errors="replace"):
            if not line.startswith("g "):
                continue
            cols = line.split()
            if len(cols) < 14:
                continue
            try:
                yr, mo, dy = int(cols[1]), int(cols[2]), int(cols[3])
                gid_str = re.sub(r"[^0-9]", "", cols[7])
                if not gid_str or gid_str == "0":
                    continue
                lat = float(cols[12])
                lon = float(cols[13])
            except (ValueError, IndexError):
                continue

            key = (yr, mo, dy) + _round_key(lat, lon)
            if key in lookup and lookup[key] != gid_str:
                collisions.append(key)
            lookup[key] = gid_str

    if collisions:
        warnings.warn(
            f"{directory.name}: {len(set(collisions))} date+position keys mapped "
            "to more than one group — keeping last assignment."
        )
    return lookup


# ── parse KMAS instrument files ──────────────────────────────────────────────
# Format (two variants depending on whether lat/lon were measured):
#   with position:    YYYYMMDD.frac  group_id  lat  lon  foreshor  ...
#   without position: YYYYMMDD.frac  group_id  foreshor  ...
#
# Distinguish: if |col[2]| > 1 it's latitude; if 0 ≤ col[2] ≤ 1 it's
# foreshortening.  (Sunspot latitudes satisfy |lat| ≥ 5° by Spörer's law,
# so the ±1° cutoff is safe in practice.)

def _parse_kmas(directory: Path, glob: str) -> dict[tuple, str]:
    """Return lookup (year, month, day, lat_r, lon_r) → 'KMAS_{year}_{gid}'."""
    lookup: dict[tuple, str] = {}
    collisions: list[tuple] = []

    for fpath in sorted(directory.glob(glob)):
        for line in open(fpath, encoding="latin-1", errors="replace"):
            cols = line.split()
            if len(cols) < 7:
                continue
            try:
                dt  = float(cols[0])
                gid = int(cols[1])
                if gid == 0:
                    continue
                v2  = float(cols[2])
            except (ValueError, IndexError):
                continue

            # Determine whether lat/lon are present
            has_position = abs(v2) > 1.0 or cols[2].startswith("+")
            if not has_position:
                continue

            try:
                lat = v2
                lon = float(cols[3])
            except (ValueError, IndexError):
                continue

            date_int = int(dt)
            yr  = date_int // 10000
            mo  = (date_int // 100) % 100
            dy  = date_int % 100
            uid = f"KMAS_{yr}_{gid}"
            key = (yr, mo, dy) + _round_key(lat, lon)

            if key in lookup and lookup[key] != uid:
                collisions.append(key)
            lookup[key] = uid

    if collisions:
        warnings.warn(
            f"KMAS: {len(set(collisions))} date+position keys mapped "
            "to more than one group — keeping last assignment."
        )
    return lookup


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── load composite CSV ────────────────────────────────────────────────
    print(f"Loading {RAW_CSV.name} …")
    df = pd.read_csv(RAW_CSV)
    print(f"  {len(df):,} rows")

    # Guard: if uniqueID is already string-typed the CSV has been processed.
    # Restore the original with:
    #   git show 54a68d4:data/composite_sunspot_groups_daily_measurements_10_23.csv \
    #       > data/composite_sunspot_groups_daily_measurements_10_23.csv
    if df["uniqueID"].dtype == object:
        raise SystemExit(
            "ERROR: uniqueID column is already string-typed — the CSV has been "
            "processed before.\nRestore the original from git before re-running:\n"
            "  git show 54a68d4:data/composite_sunspot_groups_daily_measurements_10_23.csv"
            " > data/composite_sunspot_groups_daily_measurements_10_23.csv"
        )

    # ── build source lookups ──────────────────────────────────────────────
    print("Parsing GPR files …")
    gpr_lookup = _parse_gpr_dpd(DATA_DIR / "GPR", "GPR*.txt")
    print(f"  {len(gpr_lookup):,} (date, position) keys")

    print("Parsing DPD files …")
    dpd_lookup = _parse_gpr_dpd(DATA_DIR / "DPD", "DPD*.txt")
    print(f"  {len(dpd_lookup):,} (date, position) keys")

    print("Parsing KMAS files …")
    kmas_lookup = _parse_kmas(DATA_DIR / "KMAS", "k*.dat")
    print(f"  {len(kmas_lookup):,} (date, position) keys")

    # ── assign new uniqueIDs ──────────────────────────────────────────────
    new_ids:    list[object] = [np.nan] * len(df)
    unmatched:  dict[int, int] = {}   # survey → count of unmatched rows

    for idx, row in df.iterrows():
        survey = row["survey"]
        uid    = row["uniqueID"]
        lat    = row["latitude"]
        lon    = row["longitude"]

        # rows without a uniqueID are spotless / no-observation days
        if pd.isna(uid):
            continue

        yr = int(row["year"])
        mo = int(row["month"])
        dy = int(row["day"])

        # ── Schwabe and Spoerer: no instrument files, preserve as string ──
        if survey == 1:
            new_ids[idx] = f"SC_{_fmt_sc_id(uid)}"
            continue
        if survey == 2:
            new_ids[idx] = f"SP_{int(uid)}"
            continue

        # ── instrument surveys: match by (date, lat, lon) ─────────────────
        if pd.isna(lat) or pd.isna(lon):
            # No position data — cannot match; keep corrupted ID as fallback
            new_ids[idx] = f"UNMATCHED_{survey}_{int(uid)}"
            unmatched[int(survey)] = unmatched.get(int(survey), 0) + 1
            continue

        key = (yr, mo, dy) + _round_key(lat, lon)

        if survey == 1004:                     # RGO / GPR
            gid = gpr_lookup.get(key)
            if gid is not None:
                new_ids[idx] = f"GPR_{gid}"
            else:
                new_ids[idx] = f"GPR_UNMATCHED_{yr}{mo:02d}{dy:02d}"
                unmatched[1004] = unmatched.get(1004, 0) + 1

        elif survey == 1003:                   # Debrecen / DPD
            gid = dpd_lookup.get(key)
            if gid is not None:
                new_ids[idx] = f"DPD_{gid}"
            else:
                new_ids[idx] = f"DPD_UNMATCHED_{yr}{mo:02d}{dy:02d}"
                unmatched[1003] = unmatched.get(1003, 0) + 1

        elif survey == 1002:                   # KMAS
            uid_str = kmas_lookup.get(key)
            if uid_str is not None:
                new_ids[idx] = uid_str
            else:
                # Mark as needing grouping; use row index as placeholder
                new_ids[idx] = None   # resolved in the grouping pass below
                unmatched[1002] = unmatched.get(1002, 0) + 1

        else:
            new_ids[idx] = f"SURVEY{int(survey)}_UNMATCHED_{int(uid)}"
            unmatched[int(survey)] = unmatched.get(int(survey), 0) + 1

    df["uniqueID"] = new_ids

    # ── second pass: group unmatched KMAS rows by temporal proximity + lat ──
    # Source files lacked lat/lon for these observations (likely from a different
    # version of the KMAS catalog used when building the composite).  Group
    # consecutive observations at the same rounded (lat, lon) within a 30-day
    # window — they belong to the same sunspot group — then assign a unique
    # KMAS_{year}_UN{n} label per cluster.
    kmas_none = df[(df["survey"] == 1002) & df["uniqueID"].isna() & df["latitude"].notna()].copy()

    if len(kmas_none) > 0:
        kmas_none = kmas_none.sort_values(["year", "month", "day"])
        kmas_none["_date"] = pd.to_datetime(kmas_none[["year", "month", "day"]])
        kmas_none["_lat_r"] = kmas_none["latitude"].apply(lambda x: round(float(x), 1))
        kmas_none["_lon_r"] = kmas_none["longitude"].apply(lambda x: round(float(x), 1))

        # Greedy grouping: active = {(lat_r, lon_r): (last_date, cluster_id)}
        active: dict[tuple, tuple] = {}
        cluster_counter: dict[int, int] = {}   # year → next cluster number
        cluster_ids: dict[int, str] = {}        # row idx → cluster string ID
        WINDOW_DAYS = 30

        for ridx, rrow in kmas_none.iterrows():
            yr_r = int(rrow["year"])
            lat_r = rrow["_lat_r"]
            lon_r = rrow["_lon_r"]
            date_r = rrow["_date"]
            pos_key = (lat_r, lon_r)

            if pos_key in active:
                last_date, cid = active[pos_key]
                if (date_r - last_date).days <= WINDOW_DAYS:
                    active[pos_key] = (date_r, cid)
                    cluster_ids[ridx] = cid
                    continue
            # New cluster
            n = cluster_counter.get(yr_r, 0) + 1
            cluster_counter[yr_r] = n
            cid = f"KMAS_{yr_r}_UN{n}"
            active[pos_key] = (date_r, cid)
            cluster_ids[ridx] = cid

        for ridx, cid in cluster_ids.items():
            df.at[ridx, "uniqueID"] = cid

        print(f"\nUnmatched KMAS rows regrouped into "
              f"{len(set(cluster_ids.values()))} synthetic clusters.")

    # ── diagnostics ───────────────────────────────────────────────────────
    remaining_none = (df["survey"] == 1002) & df["uniqueID"].isna() & df["latitude"].notna()
    if remaining_none.any():
        print(f"WARNING — {remaining_none.sum()} KMAS rows still unresolved.")

    if unmatched:
        print("\nSummary of rows not matched to source instrument files:")
        for s, n in sorted(unmatched.items()):
            label = {1002: "KMAS (regrouped above)"}.get(s, "")
            print(f"  survey {s}: {n} rows  {label}")
    else:
        print("\nAll rows matched successfully.")

    print("\nNew uniqueID value counts (unique IDs per survey):")
    for survey in [1, 2, 1002, 1003, 1004]:
        mask = df["survey"] == survey
        n_ids = df.loc[mask, "uniqueID"].dropna().nunique()
        print(f"  survey {survey}: {n_ids} unique IDs")

    # ── write output ──────────────────────────────────────────────────────
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df):,} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
