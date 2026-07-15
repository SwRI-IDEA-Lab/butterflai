"""Distributional goodness-of-fit and calibration metrics for butterfly
diagrams.

This module is shared infrastructure: it is imported by both diffusion model
families (the residual model in ``conditioned_infrastructure`` and the
logit/softmax model in ``empirical_infrastructure``) and by the week-11
evaluate notebooks. It deliberately has **no model imports** — every function
takes plain NumPy arrays — so it stays on the ``weeks/ -> infrastructure/``
side of the dependency graph and can be unit-tested in isolation.

Vocabulary
----------
Each butterfly *window* is summarized by a per-latitude **density** over a
fixed grid of ``K`` bins (default ``K = 15``, ``bin_width = 3`` degrees, so the
grid spans |latitude| 0-45 deg). A density ``p`` of shape ``(K,)`` carries
units of 1/deg; ``p * bin_width`` is the probability mass per bin and sums to 1
for a proper distribution.

The diffusion model is a *generative ensemble*: conditioned on a window's cycle
features it produces many candidate densities. The right yardsticks are
therefore distributional (does the cloud of samples match the data?) and
calibration-based (is the ensemble spread honest?), not pointwise likelihood.

Conventions
-----------
- ``*_dens`` arguments are densities (1/deg) of shape ``(K,)`` or ``(N, K)``.
- ``bin_centers`` is ``(K,)`` in degrees; ``bin_width`` in degrees.
- Returned distances in physical EMD units are **degrees of latitude**.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

__all__ = [
    "normalize_density",
    "emd_density",
    "energy_distance_to_point",
    "energy_distance_samples",
    "mmd_rbf",
    "density_moments",
    "moment_errors",
    "crps_ensemble",
    "crps_ensemble_batch",
    "rank_of",
    "rank_histogram",
    "bin_rank_histogram",
    "distributional_report",
]


# ---------------------------------------------------------------------------
# Density bookkeeping
# ---------------------------------------------------------------------------

def normalize_density(dens, bin_width: float = 3.0, eps: float = 1e-12):
    """Clip a (possibly model-emitted) density to a proper probability density.

    Residual-model densities are ``classical + residual`` and may carry small
    negative excursions or fail to integrate to exactly 1. For the
    distribution-shape metrics below we first project onto the probability
    simplex: clip negatives to zero, then renormalize so ``sum(p)*bin_width ==
    1``.

    Parameters
    ----------
    dens : ndarray, shape (..., K)
        Density in 1/deg. The last axis is the latitude bin.
    bin_width : float
        Bin width in degrees.
    eps : float
        Floor on the total mass before dividing, to avoid 0/0.

    Returns
    -------
    ndarray, shape (..., K)
        Non-negative density that integrates to 1 over latitude.
    """
    p = np.clip(np.asarray(dens, dtype=np.float64), 0.0, None)
    mass = p * bin_width
    total = mass.sum(axis=-1, keepdims=True)
    total = np.clip(total, eps, None)
    return (mass / total) / bin_width


# ---------------------------------------------------------------------------
# Optimal-transport / shape distances
# ---------------------------------------------------------------------------

def emd_density(p_dens, q_dens, bin_centers, bin_width: float = 3.0) -> float:
    """1-D Wasserstein-1 (earth-mover) distance between two densities.

    Both densities live on the same ``bin_centers`` grid, so this is the
    classic 1-D optimal transport cost, returned in **degrees of latitude** —
    directly interpretable as "how far, on average, mass must move to turn the
    model density into the empirical one." Sensitive to the equatorward-drift
    shape in a way pointwise NLL is not.

    Parameters
    ----------
    p_dens, q_dens : ndarray, shape (K,)
        Densities (1/deg). Internally clipped/renormalized to proper masses.
    bin_centers : ndarray, shape (K,)
        Bin-center latitudes in degrees.
    bin_width : float
        Bin width in degrees (used only for normalization).

    Returns
    -------
    float
        W1 distance in degrees.
    """
    p = normalize_density(p_dens, bin_width) * bin_width  # mass
    q = normalize_density(q_dens, bin_width) * bin_width
    return float(wasserstein_distance(
        np.asarray(bin_centers, dtype=np.float64),
        np.asarray(bin_centers, dtype=np.float64),
        u_weights=p, v_weights=q,
    ))


def _pairwise_euclidean(A, B):
    """Euclidean distance matrix between rows of A (M,K) and B (N,K)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    # (M,1,K) - (1,N,K) -> (M,N,K)
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.einsum("mnk,mnk->mn", diff, diff))


def energy_distance_to_point(samples_MK, target_K) -> float:
    """Energy distance between an ensemble and a single target density.

    The empirical butterfly density for a window is a *single* observed vector,
    so the model ensemble is compared against a degenerate (point-mass) target
    distribution. The general energy distance

        E = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||

    then collapses (the Y-Y' term vanishes) to

        E = 2 mean_i ||x_i - target|| - mean_{i,j} ||x_i - x_j||.

    A model that sits near the target *and* keeps healthy internal spread is
    rewarded; a collapsed ensemble pinned away from the target is penalized.

    Parameters
    ----------
    samples_MK : ndarray, shape (M, K)
        Ensemble of model densities for one window.
    target_K : ndarray, shape (K,)
        The empirical density for that window.

    Returns
    -------
    float
        Energy distance (>= 0), in the L2 metric on density vectors.
    """
    X = np.asarray(samples_MK, dtype=np.float64)
    y = np.asarray(target_K, dtype=np.float64)[None, :]
    cross = _pairwise_euclidean(X, y).mean()          # mean_i ||x_i - target||
    within = _pairwise_euclidean(X, X).mean()         # mean_{i,j} ||x_i - x_j||
    return float(2.0 * cross - within)


def energy_distance_samples(A, B) -> float:
    """Two-sample energy distance between ensembles ``A`` (M,K) and ``B`` (N,K).

    ``E = 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||``. Symmetric, non-negative, and
    zero iff the two ensembles share a distribution. Captures full multivariate
    shape — including inter-bin correlations that a 2nd-moment covariance
    distance misses.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    cross = _pairwise_euclidean(A, B).mean()
    aa = _pairwise_euclidean(A, A).mean()
    bb = _pairwise_euclidean(B, B).mean()
    return float(2.0 * cross - aa - bb)


def mmd_rbf(A, B, bandwidth: float = None) -> float:
    """Biased squared-MMD between ensembles with an RBF kernel.

    Alternative to :func:`energy_distance_samples`. ``bandwidth`` defaults to
    the median pairwise L2 distance over the pooled sample (median heuristic).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if bandwidth is None:
        pooled = np.vstack([A, B])
        d = _pairwise_euclidean(pooled, pooled)
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        bandwidth = float(med) if med > 0 else 1.0
    gamma = 1.0 / (2.0 * bandwidth ** 2)

    def k(X, Y):
        return np.exp(-gamma * _pairwise_euclidean(X, Y) ** 2)

    return float(k(A, A).mean() + k(B, B).mean() - 2.0 * k(A, B).mean())


# ---------------------------------------------------------------------------
# Physical moments (Spoerer drift, spread, skew)
# ---------------------------------------------------------------------------

def density_moments(p_dens, bin_centers, bin_width: float = 3.0):
    """Mean latitude, spread, and skew of one window's density.

    ``mu`` (the first moment) is the mean emergence |latitude| — i.e. the
    Spoerer-law drift point for that window. ``sigma`` is the spread and
    ``skew`` the dimensionless third standardized moment.

    Returns
    -------
    (mu, sigma, skew) : tuple of float
        ``mu`` and ``sigma`` in degrees; ``skew`` dimensionless. ``skew`` is 0
        when ``sigma`` is degenerate.
    """
    p = normalize_density(p_dens, bin_width)
    w = p * bin_width                      # mass per bin (sums to 1)
    c = np.asarray(bin_centers, dtype=np.float64)
    mu = float((w * c).sum())
    var = float((w * (c - mu) ** 2).sum())
    sigma = float(np.sqrt(max(var, 0.0)))
    if sigma <= 1e-8:
        return mu, sigma, 0.0
    skew = float((w * ((c - mu) / sigma) ** 3).sum())
    return mu, sigma, skew


def moment_errors(model_dens, emp_dens, bin_centers, bin_width: float = 3.0,
                  tau=None) -> dict:
    """Per-window moment errors between model and empirical densities.

    Parameters
    ----------
    model_dens, emp_dens : ndarray, shape (N, K)
        Per-window densities (e.g. the ensemble-mean model density vs the
        empirical density), aligned row-for-row.
    bin_centers : ndarray, shape (K,)
    bin_width : float
    tau : ndarray, shape (N,), optional
        Cycle phase per window; if given, the per-window moment arrays are
        returned too so the drift law ``mu(tau)`` can be plotted.

    Returns
    -------
    dict
        ``mu_mae``, ``sigma_mae``, ``skew_mae`` (means of |model - emp|), plus
        ``mu_model``/``mu_emp``/``sigma_model``/``sigma_emp`` arrays and (if
        provided) ``tau`` for plotting.
    """
    model_dens = np.atleast_2d(np.asarray(model_dens, dtype=np.float64))
    emp_dens = np.atleast_2d(np.asarray(emp_dens, dtype=np.float64))
    n = model_dens.shape[0]

    mm = np.empty(n); ms = np.empty(n); mk = np.empty(n)
    em = np.empty(n); es = np.empty(n); ek = np.empty(n)
    for i in range(n):
        mm[i], ms[i], mk[i] = density_moments(model_dens[i], bin_centers, bin_width)
        em[i], es[i], ek[i] = density_moments(emp_dens[i], bin_centers, bin_width)

    out = {
        "mu_mae":    float(np.mean(np.abs(mm - em))),
        "sigma_mae": float(np.mean(np.abs(ms - es))),
        "skew_mae":  float(np.mean(np.abs(mk - ek))),
        "mu_model": mm, "mu_emp": em,
        "sigma_model": ms, "sigma_emp": es,
    }
    if tau is not None:
        out["tau"] = np.asarray(tau, dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Calibration: CRPS and rank histograms
# ---------------------------------------------------------------------------

def crps_ensemble(obs: float, ensemble) -> float:
    """Continuous Ranked Probability Score of a scalar against an ensemble.

    Closed-form ensemble estimator (Hersbach 2000):

        CRPS = mean_i |m_i - obs| - 1/2 * mean_{i,j} |m_i - m_j|.

    A *proper* scoring rule: it rewards an ensemble whose members straddle the
    observation (calibration) while staying tight (sharpness), and — unlike NLL
    — it is not destabilized by flooring or a single under-weighted bin. Lower
    is better; 0 for a perfect deterministic forecast.

    Parameters
    ----------
    obs : float
        Observed scalar summary (e.g. a window's mean latitude).
    ensemble : ndarray, shape (M,)
        Ensemble of forecast summaries for that window.

    Returns
    -------
    float
    """
    m = np.asarray(ensemble, dtype=np.float64).ravel()
    term1 = np.abs(m - float(obs)).mean()
    term2 = np.abs(m[:, None] - m[None, :]).mean()
    return float(term1 - 0.5 * term2)


def crps_ensemble_batch(obs_N, ensemble_NM) -> float:
    """Mean CRPS over ``N`` windows. ``ensemble_NM`` is (N, M)."""
    obs_N = np.asarray(obs_N, dtype=np.float64).ravel()
    ens = np.asarray(ensemble_NM, dtype=np.float64)
    return float(np.mean([crps_ensemble(obs_N[i], ens[i]) for i in range(len(obs_N))]))


def rank_of(obs: float, ensemble, rng=None) -> int:
    """Rank of an observation within an ensemble, in ``[0, M]``.

    Counts ensemble members strictly below ``obs``; ties are broken at random
    so a calibrated ensemble yields a uniform rank distribution.
    """
    m = np.asarray(ensemble, dtype=np.float64).ravel()
    below = int(np.sum(m < obs))
    ties = int(np.sum(m == obs))
    if ties:
        rng = np.random.default_rng() if rng is None else rng
        below += int(rng.integers(0, ties + 1))
    return below


def rank_histogram(obs_N, ensemble_NM, rng=None) -> np.ndarray:
    """Rank (Talagrand) histogram of observations within their ensembles.

    A flat histogram indicates a calibrated ensemble; a U shape means the
    ensemble is under-dispersed (over-confident / too sharp); a dome means it
    is over-dispersed. Returns counts over ``M + 1`` rank bins.
    """
    ens = np.asarray(ensemble_NM, dtype=np.float64)
    M = ens.shape[1]
    rng = np.random.default_rng(0) if rng is None else rng
    counts = np.zeros(M + 1, dtype=np.int64)
    obs_N = np.asarray(obs_N, dtype=np.float64).ravel()
    for i in range(len(obs_N)):
        counts[rank_of(obs_N[i], ens[i], rng=rng)] += 1
    return counts


def bin_rank_histogram(emp_dens_NK, ensemble_NMK, rng=None) -> np.ndarray:
    """Per-bin rank histogram of empirical density values within the ensemble.

    For each window and latitude bin, ranks the empirical density value among
    the ``M`` generated density values for that bin, then pools the ranks over
    all windows and bins. Probes calibration at the level of individual bins
    rather than a scalar summary. Returns counts over ``M + 1`` rank bins.
    """
    emp = np.asarray(emp_dens_NK, dtype=np.float64)
    ens = np.asarray(ensemble_NMK, dtype=np.float64)   # (N, M, K)
    N, M, K = ens.shape
    rng = np.random.default_rng(0) if rng is None else rng
    counts = np.zeros(M + 1, dtype=np.int64)
    for i in range(N):
        for b in range(K):
            counts[rank_of(emp[i, b], ens[i, :, b], rng=rng)] += 1
    return counts


# ---------------------------------------------------------------------------
# High-level report (one call from the training callback / notebook)
# ---------------------------------------------------------------------------

def distributional_report(ensembles_NMK, emp_dens_NK, bin_centers,
                          bin_width: float = 3.0, tau=None,
                          rng=None) -> dict:
    """Aggregate the full metric suite over a set of windows.

    Parameters
    ----------
    ensembles_NMK : ndarray, shape (N, M, K)
        For each of ``N`` windows, ``M`` model-sampled densities.
    emp_dens_NK : ndarray, shape (N, K)
        The empirical density for each window.
    bin_centers : ndarray, shape (K,)
    bin_width : float
    tau : ndarray, shape (N,), optional
        Cycle phase per window (for the drift arrays).
    rng : np.random.Generator, optional

    Returns
    -------
    dict
        Scalar metrics (``emd``, ``energy``, ``mu_mae``, ``sigma_mae``,
        ``skew_mae``, ``crps_mu``, ``crps_sigma``), the ``rank_hist_mu`` array,
        and the moment arrays for plotting.
    """
    ens = np.asarray(ensembles_NMK, dtype=np.float64)
    emp = np.asarray(emp_dens_NK, dtype=np.float64)
    N, M, K = ens.shape
    rng = np.random.default_rng(0) if rng is None else rng

    mean_dens = ens.mean(axis=1)                       # (N, K) ensemble mean

    emd_vals = np.array([
        emd_density(mean_dens[i], emp[i], bin_centers, bin_width)
        for i in range(N)
    ])
    energy_vals = np.array([
        energy_distance_to_point(ens[i], emp[i]) for i in range(N)
    ])

    # Scalar summaries for CRPS / rank histograms: per-window mean latitude
    # (mu) and spread (sigma).
    mu_obs = np.empty(N); sig_obs = np.empty(N)
    mu_ens = np.empty((N, M)); sig_ens = np.empty((N, M))
    for i in range(N):
        mu_obs[i], sig_obs[i], _ = density_moments(emp[i], bin_centers, bin_width)
        for j in range(M):
            mu_ens[i, j], sig_ens[i, j], _ = density_moments(
                ens[i, j], bin_centers, bin_width)

    mom = moment_errors(mean_dens, emp, bin_centers, bin_width, tau=tau)

    return {
        "emd":        float(emd_vals.mean()),
        "energy":     float(energy_vals.mean()),
        "mu_mae":     mom["mu_mae"],
        "sigma_mae":  mom["sigma_mae"],
        "skew_mae":   mom["skew_mae"],
        "crps_mu":    crps_ensemble_batch(mu_obs, mu_ens),
        "crps_sigma": crps_ensemble_batch(sig_obs, sig_ens),
        "rank_hist_mu": rank_histogram(mu_obs, mu_ens, rng=rng),
        # arrays for plotting
        "mu_model": mom["mu_model"], "mu_emp": mom["mu_emp"],
        "sigma_model": mom["sigma_model"], "sigma_emp": mom["sigma_emp"],
        **({"tau": mom["tau"]} if tau is not None else {}),
    }


# ---------------------------------------------------------------------------
# Sanity checks (run as a script: ``python distribution_metrics.py``)
# ---------------------------------------------------------------------------

def _self_test():
    rng = np.random.default_rng(0)
    K = 15
    bin_width = 3.0
    bin_centers = (np.arange(K) + 0.5) * bin_width

    def gauss_density(mu, sigma):
        p = np.exp(-0.5 * ((bin_centers - mu) / sigma) ** 2)
        return normalize_density(p, bin_width)

    p = gauss_density(20.0, 6.0)
    q = gauss_density(12.0, 6.0)

    # EMD: identical -> 0; shifted by 8 deg -> ~8 deg.
    assert emd_density(p, p, bin_centers, bin_width) < 1e-9
    emd_pq = emd_density(p, q, bin_centers, bin_width)
    assert 6.0 < emd_pq < 10.0, emd_pq

    # Energy distance: symmetric, non-negative, ~0 for identical ensembles.
    A = np.stack([gauss_density(20 + rng.normal(0, 0.5), 6.0) for _ in range(40)])
    B = np.stack([gauss_density(20 + rng.normal(0, 0.5), 6.0) for _ in range(40)])
    assert energy_distance_samples(A, A) < 1e-9
    assert energy_distance_samples(A, B) >= -1e-9
    assert abs(energy_distance_samples(A, B) - energy_distance_samples(B, A)) < 1e-9

    # Energy distance to a point >= 0.
    assert energy_distance_to_point(A, p) >= -1e-9

    # CRPS: perfect deterministic ensemble -> 0; spread-out -> positive.
    assert crps_ensemble(5.0, np.full(20, 5.0)) < 1e-9
    assert crps_ensemble(5.0, rng.normal(5.0, 2.0, size=200)) > 0.0

    # Rank histogram of a calibrated ensemble is roughly flat.
    N, M = 4000, 19
    obs = rng.normal(0, 1, size=N)
    ens = rng.normal(0, 1, size=(N, M))
    hist = rank_histogram(obs, ens, rng=rng)
    assert hist.shape == (M + 1,)
    expected = N / (M + 1)
    assert np.all(np.abs(hist - expected) < 0.35 * expected), hist

    # Moments recover a known Gaussian center.
    mu, sigma, skew = density_moments(gauss_density(18.0, 5.0), bin_centers, bin_width)
    assert abs(mu - 18.0) < 1.0, mu
    assert abs(skew) < 0.5, skew

    # End-to-end report shape.
    ens_NMK = np.stack([
        np.stack([gauss_density(15 + rng.normal(0, 1), 6.0) for _ in range(16)])
        for _ in range(8)
    ])
    emp_NK = np.stack([gauss_density(15.0, 6.0) for _ in range(8)])
    rep = distributional_report(ens_NMK, emp_NK, bin_centers, bin_width,
                                tau=np.linspace(0, 1, 8))
    for key in ("emd", "energy", "mu_mae", "sigma_mae", "crps_mu"):
        assert np.isfinite(rep[key]), (key, rep[key])
    assert rep["rank_hist_mu"].sum() == 8

    print("distribution_metrics self-test passed.")


if __name__ == "__main__":
    _self_test()
