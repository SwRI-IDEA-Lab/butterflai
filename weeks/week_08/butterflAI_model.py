"""
butterflai_model.py — self-contained ButterflAI official model.

Loads the fitted parameters from official_model.npz and provides everything
needed to evaluate the model at arbitrary (amplitude, τ) — without re-fitting.

API
---
    from butterflai_model import ButterflAIModel

    m = ButterflAIModel()                       # loads default npz
    mu, sigma = m.gaussian(A=1.5, tau=2.0)      # at standard year τ=2
    p   = m.density(A=1.5, tau=2.0, abs_lat=18) # p(|lat|=18° | A, τ)
    tau = m.to_tau(cycle=23, hemisphere='north', decimal_year=2001.5)
    mu0 = m.mu_0(A=1.5)                         # cap latitude (active gate)
    on  = m.is_active(A=1.5, tau=8.0)           # past μ₀ cutoff?

τ ("standard year") is centred on the hemicycle's effective reference epoch:

    τ = decimal_year − t0_total(cycle, hemisphere)

At τ = 0, μ(τ) = a_mu ≈ 15° — the model's reference latitude.

For a hemicycle outside the fitted set (a future cycle), t0_total is
unidentifiable from the fit alone, since the per-hemicycle Δt_S3
corrections are by construction degenerate with held-out data.  The
recommended proxy is the 15° crossing of the smoothed yearly-mean
|latitude|; pass it explicitly via `register_t0(cycle, hemi, t0_value)`.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.stats import norm as sp_norm

DEFAULT_NPZ = Path(__file__).resolve().parent / "official_model.npz"


# ──────────────────────────────────────────────────────────────────────────
# Shape primitives (duplicated here so the module is import-self-contained)
# ──────────────────────────────────────────────────────────────────────────

def piecewise_linear_sigma(mu, m_shared, b_shared, mu_peak, m_i):
    """Variant-A σ(μ): universal equatorward arm, cycle-amplitude-dependent
    poleward arm.  Shape continuous at μ_peak by construction.
    """
    sigma_at_peak = m_shared * mu_peak + b_shared
    b_poleward    = sigma_at_peak - m_i * mu_peak
    return np.clip(
        np.where(mu <= mu_peak,
                 m_shared * mu + b_shared,
                 m_i * mu + b_poleward),
        0.0, None,
    )


# ──────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class _GlobalParams:
    """Point estimates of the 10 global parameters."""
    # μ(τ) mean path
    a_mu: float
    b_mu: float
    # Amplitude regressions:  param(A) = a · A + b
    a_mu0: float;     b_mu0: float
    a_mu_peak: float; b_mu_peak: float
    a_m_i: float;     b_m_i: float
    # Universal equatorward σ-line
    m_shared: float;  b_shared: float
    # Bookkeeping
    nll_hard: float = float("nan")
    coverage: float = float("nan")


class ButterflAIModel:
    """Closed-form butterfly-diagram density model with bootstrap CI access."""

    def __init__(self, npz_path: Path | str = DEFAULT_NPZ):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"{npz_path} not found.  Run "
                                    "13_official_model_bootstrap.py first.")
        self._npz = np.load(npz_path, allow_pickle=True)
        d = self._npz

        self.params = _GlobalParams(
            a_mu      = float(d["point_a_mu"]),
            b_mu      = float(d["point_b_mu"]),
            a_mu0     = float(d["point_a_mu0"]),
            b_mu0     = float(d["point_b_mu0"]),
            a_mu_peak = float(d["point_a_mu_peak"]),
            b_mu_peak = float(d["point_b_mu_peak"]),
            a_m_i     = float(d["point_a_m_i"]),
            b_m_i     = float(d["point_b_m_i"]),
            m_shared  = float(d["point_m_shared"]),
            b_shared  = float(d["point_b_shared"]),
            nll_hard  = float(d["point_nll_hard"]),
            coverage  = float(d["point_coverage"]),
        )

        # Per-hemicycle effective reference epoch
        cycles = d["hc_cycles"].astype(int)
        hemis  = d["hc_hemispheres"]
        t0     = d["point_t0_total"].astype(float)
        # str() coerces numpy bytes/strings to native strings
        self._t0_total = {(int(c), str(h)): float(v)
                          for c, h, v in zip(cycles, hemis, t0)}

        # Optional: bootstrap CIs on t0_total
        self._t0_ci = {(int(c), str(h)): (float(lo), float(hi))
                       for c, h, lo, hi in zip(cycles, hemis,
                                               d["boot_t0_total_lo"],
                                               d["boot_t0_total_hi"])}

    # ── Amplitude regressions ──────────────────────────────────────────
    def mu_peak(self, A):
        A = np.asarray(A, dtype=float)
        return self.params.a_mu_peak * A + self.params.b_mu_peak

    def m_i(self, A):
        A = np.asarray(A, dtype=float)
        return self.params.a_m_i * A + self.params.b_m_i

    def mu_0(self, A):
        A = np.asarray(A, dtype=float)
        return self.params.a_mu0 * A + self.params.b_mu0

    # ── Mean path and spread ───────────────────────────────────────────
    def mu(self, tau):
        tau = np.asarray(tau, dtype=float)
        return self.params.a_mu * np.exp(-tau / self.params.b_mu)

    def sigma(self, mu_value, A):
        mu_value = np.asarray(mu_value, dtype=float)
        return piecewise_linear_sigma(
            mu_value,
            self.params.m_shared, self.params.b_shared,
            self.mu_peak(A), self.m_i(A),
        )

    # ── Convenience ────────────────────────────────────────────────────
    def gaussian(self, A, tau):
        """Returns (μ, σ) of the |latitude| Gaussian at standard year τ.

        Both inputs broadcast.  The hard gate at μ₀(A) is NOT applied here
        — caller decides whether to gate.  Use `is_active` or `density`
        for gated output.
        """
        mu_t = self.mu(tau)
        sig_t = self.sigma(mu_t, A)
        return mu_t, sig_t

    def density(self, A, tau, abs_lat):
        """p(|latitude| | A, τ).  Zero past μ₀(A) (hard gate).

        All three inputs broadcast.  Use np.abs(latitude) for raw signed
        latitudes — the model is symmetric.
        """
        mu_t  = self.mu(tau)
        sig_t = self.sigma(mu_t, A)
        cap   = self.mu_0(A)
        pdf   = sp_norm.pdf(np.asarray(abs_lat, dtype=float),
                            loc=mu_t, scale=sig_t)
        return np.where(mu_t <= cap, pdf, 0.0)

    def is_active(self, A, tau):
        """Boolean: is the hemicycle still emerging at standard year τ?"""
        return self.mu(tau) <= self.mu_0(A)

    # ── Reference-epoch lookup ─────────────────────────────────────────
    def lookup_t0(self, cycle, hemisphere):
        key = (int(cycle), str(hemisphere))
        if key not in self._t0_total:
            raise KeyError(
                f"No fitted t0 for {key}.  Either this hemicycle was not in "
                f"the training set, or the hemisphere string is wrong "
                f"(use 'north'/'south').  For an unseen hemicycle, supply "
                f"your own t0 estimate via `register_t0`."
            )
        return self._t0_total[key]

    def lookup_t0_ci(self, cycle, hemisphere):
        """Bootstrap 95 % CI on t0_total, or (nan, nan) if too few samples."""
        return self._t0_ci[(int(cycle), str(hemisphere))]

    def register_t0(self, cycle, hemisphere, t0_total):
        """Add a t0_total for an unseen hemicycle (e.g., a future cycle).

        For new data, supply t0 from the 15° crossing of the smoothed
        yearly-mean |latitude| — this is the nearest equivalent of the
        in-sample t0_15deg.  The S1+S3 corrections cannot be reconstructed
        without held-out data and add ~0.1–0.5 yr of irreducible error
        for unseen hemicycles.
        """
        self._t0_total[(int(cycle), str(hemisphere))] = float(t0_total)

    def to_tau(self, cycle, hemisphere, decimal_year):
        return np.asarray(decimal_year, dtype=float) - self.lookup_t0(cycle, hemisphere)

    def known_hemicycles(self) -> list[tuple[int, str]]:
        return sorted(self._t0_total.keys())

    # ── Bootstrap access (for downstream uncertainty propagation) ─────
    def bootstrap_global_params(self) -> dict[str, np.ndarray]:
        """Return the bootstrap distributions for the 10 global parameters."""
        return {
            "a_mu":      self._npz["boot_a_mu"],
            "b_mu":      self._npz["boot_b_mu"],
            "a_mu0":     self._npz["boot_a_mu0"],
            "b_mu0":     self._npz["boot_b_mu0"],
            "a_mu_peak": self._npz["boot_a_mu_peak"],
            "b_mu_peak": self._npz["boot_b_mu_peak"],
            "a_m_i":     self._npz["boot_a_m_i"],
            "b_m_i":     self._npz["boot_b_m_i"],
            "m_shared":  self._npz["boot_m_shared"],
            "b_shared":  self._npz["boot_b_shared"],
        }

    # ── Pretty print ───────────────────────────────────────────────────
    def __repr__(self):
        p = self.params
        return (f"<ButterflAIModel  N_hemicycles={len(self._t0_total)}  "
                f"NLL={p.nll_hard:.4f}  coverage={p.coverage:.3f}>")


# ──────────────────────────────────────────────────────────────────────────
# Demo / smoke test
# ──────────────────────────────────────────────────────────────────────────

def _demo():
    """Tiny self-test that runs without the data file — uses synthetic params."""
    import tempfile, os
    # Build a minimal fake npz so the demo runs without a real fit
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        path = tmp.name
    try:
        np.savez(
            path,
            point_a_mu_peak=0.5, point_b_mu_peak=18.0,
            point_a_m_i=-0.1,   point_b_m_i=-0.4,
            point_m_shared=0.18, point_b_shared=1.5,
            point_a_mu=15.0,    point_b_mu=4.5,
            point_a_mu0=2.0,    point_b_mu0=28.0,
            point_nll_hard=3.087, point_coverage=0.95,
            hc_cycles=np.array([23, 23], dtype=int),
            hc_hemispheres=np.array(["north", "south"]),
            point_t0_15deg=np.array([1997.5, 1997.7]),
            point_t0_refined=np.array([1997.6, 1997.8]),
            point_delta_t_S3=np.array([0.05, -0.03]),
            point_t0_total=np.array([1997.65, 1997.77]),
            boot_t0_total_mean=np.array([1997.65, 1997.77]),
            boot_t0_total_std=np.array([0.05, 0.06]),
            boot_t0_total_lo=np.array([1997.55, 1997.65]),
            boot_t0_total_hi=np.array([1997.75, 1997.89]),
            boot_t0_total_n=np.array([190, 190], dtype=int),
            boot_a_mu_peak=np.array([0.5]),
            boot_b_mu_peak=np.array([18.0]),
            boot_a_m_i=np.array([-0.1]),
            boot_b_m_i=np.array([-0.4]),
            boot_m_shared=np.array([0.18]),
            boot_b_shared=np.array([1.5]),
            boot_a_mu=np.array([15.0]),
            boot_b_mu=np.array([4.5]),
            boot_a_mu0=np.array([2.0]),
            boot_b_mu0=np.array([28.0]),
            boot_nll_hard=np.array([3.087]),
            boot_coverage=np.array([0.95]),
            boot_dt_summary=np.zeros((1, 2)),
        )
        m = ButterflAIModel(path)
        print(m)

        A = 1.5
        taus = np.linspace(-1, 8, 6)
        for tau in taus:
            mu, sig = m.gaussian(A, tau)
            active = m.is_active(A, tau)
            print(f"  τ={tau:+5.2f}  μ={mu:6.2f}°  σ={sig:5.2f}°  "
                  f"active={bool(active)}")

        print(f"\n  μ_0(A=1.5)   = {m.mu_0(1.5):.2f}°")
        print(f"  density(A=1.5, τ=2, |lat|=18°) = {m.density(1.5, 2.0, 18.0):.4f}")
        print(f"  to_tau(cyc=23, hemi='north', year=2000) = "
              f"{m.to_tau(23, 'north', 2000.0):.3f}")
        print(f"  t0 95% CI for cycle 23 north = {m.lookup_t0_ci(23, 'north')}")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    _demo()