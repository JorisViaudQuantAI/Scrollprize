# ---------------------------------------------------------------------------------------------------
# methods_ledger/code/linefold_stat.py
# Provided for REVIEW of our arithmetic, not as a packaged tool. No support, no tests, no stable interface.
# Sanitised copy of a private lane script: local paths, job names and every location of unread material
# were removed; the numerical part is unchanged.
# Ledger entry: 05_label_free_and_physics.md L5
# ---------------------------------------------------------------------------------------------------
"""Frozen line-pitch folding statistic (definition in 05_label_free_and_physics.md L5). numpy/scipy only."""
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d

PMIN, PMAX = 3.0, 9.0
DETREND_MM = 8.0


def fill_invalid(p, ok):
    p = np.asarray(p, np.float64).copy()
    ok = np.asarray(ok, bool) & np.isfinite(p)
    if ok.sum() < 3:
        return None
    idx = np.arange(len(p))
    p[~ok] = np.interp(idx[~ok], idx[ok], p[ok])
    return p


def prep(p, ok, px_mm):
    """detrend (Gaussian sigma 8 mm) and standardise a profile; returns r (len n) or None"""
    p = fill_invalid(p, ok)
    if p is None:
        return None
    r = p - gaussian_filter1d(p, DETREND_MM / px_mm, mode='reflect')
    ok = np.asarray(ok, bool)
    sd = r[ok].std()
    if not np.isfinite(sd) or sd <= 0:
        return None
    r = (r - r[ok].mean()) / sd
    r[~ok] = 0.0
    return r


def freq_grid(H_mm, pmin=PMIN, pmax=PMAX, over=0.05):
    df = over / H_mm
    f = np.arange(1.0 / pmax, 1.0 / pmin + df, df)
    return 1.0 / f


def z2(r, px_mm, P, nh=2, ok=None):
    """Buccheri Z^2_nh at pitches P (mm). r standardised profile."""
    y = np.arange(len(r)) * px_mm
    if ok is not None:
        r = r[ok]; y = y[ok]
    n = len(r)
    out = np.zeros(len(P))
    ph = 2 * np.pi * y[None, :] / np.asarray(P)[:, None]
    for k in range(1, nh + 1):
        c = (r[None] * np.cos(k * ph)).sum(1); s = (r[None] * np.sin(k * ph)).sum(1)
        out += c * c + s * s
    return 2.0 / n * out


def phase1(r, px_mm, P, ok=None):
    """phase (cycles) of the first harmonic maximum: line centres at y = P*(phase + k)"""
    y = np.arange(len(r)) * px_mm
    if ok is not None:
        r = r[ok]; y = y[ok]
    ph = 2 * np.pi * y / P
    c = (r * np.cos(ph)).sum(); s = (r * np.sin(ph)).sum()
    return (np.arctan2(s, c) / (2 * np.pi)) % 1.0


def fold(r, px_mm, P, nb=20, ok=None, phase0=0.0):
    y = np.arange(len(r)) * px_mm
    if ok is not None:
        r = r[ok]; y = y[ok]
    b = (((y / P - phase0) % 1.0) * nb).astype(int)
    m = np.array([r[b == i].mean() if (b == i).any() else np.nan for i in range(nb)])
    e = np.array([r[b == i].std() / np.sqrt(max((b == i).sum(), 1)) for i in range(nb)])
    return m, e


def fit_null(vals):
    """scaled chi2 (c, nu) by moments"""
    vals = np.asarray(vals, np.float64)
    m, v = vals.mean(), vals.var()
    c = v / (2 * m); nu = 2 * m * m / v
    return c, nu


def p_local(Z, c, nu):
    return stats.chi2.sf(np.asarray(Z) / c, nu)


def n_trials(H_mm, nh=2):
    return max(1.0, nh * H_mm * (1.0 / PMIN - 1.0 / PMAX))


def z_of_p(p):
    p = np.clip(p, 1e-300, 1.0)
    return stats.norm.isf(p)


def z_global(Z, c, nu, H_mm):
    p = p_local(Z, c, nu)
    Nt = n_trials(H_mm)
    pg = -np.expm1(Nt * np.log1p(-np.clip(p, 0, 1 - 1e-16)))
    pg = np.where(p * Nt < 1e-6, p * Nt, pg)
    return z_of_p(pg)
