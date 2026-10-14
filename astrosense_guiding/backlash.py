"""Declination backlash estimation.

The Dec drive is modelled as a *play* (dead-band) operator acting on the
cumulative commanded pulse time ``u`` (ms)::

    y = play_b(u)            # actual mechanical position, ms-equivalent
    dec_drift[k] = offset_seg + a1*t + a2*t^2 + g * y[k] + noise

For every candidate width ``b`` the remaining parameters are linear and are
solved by least squares.  The profile RSS(b) gives the best estimate and a
1-sigma interval.  Backlash is only observable when the Dec guide pulses
reverse direction, so the number of reversals is reported as well.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parser import GuideSession
from .reconstruct import OpenLoop, segment_matrix


def play_operator(u: np.ndarray, b: float, first_dir: float = 1.0) -> np.ndarray:
    """Symmetric play operator of total width *b*."""
    y = np.empty_like(u, dtype=float)
    h = b / 2.0
    yk = u[0] - first_dir * h
    for k, uk in enumerate(u):
        if yk < uk - h:
            yk = uk - h
        elif yk > uk + h:
            yk = uk + h
        y[k] = yk
    return y


@dataclass
class BacklashResult:
    ok: bool
    message: str
    backlash_ms: float | None = None
    backlash_ms_lo: float | None = None
    backlash_ms_hi: float | None = None
    backlash_arcsec: float | None = None
    gain_ratio: float | None = None  # fitted / calibrated Dec rate
    n_reversals: int = 0
    phd2_comp_ms: float | None = None
    grid: np.ndarray | None = None
    rss: np.ndarray | None = None
    t: np.ndarray | None = None
    model: np.ndarray | None = None
    observed: np.ndarray | None = None
    residual: np.ndarray | None = None
    confidence: str = "low"


def count_reversals(signed_ms: np.ndarray, min_ms: float = 1.0) -> int:
    s = np.sign(signed_ms[np.abs(signed_ms) >= min_ms])
    return int(np.sum(s[1:] != s[:-1])) if s.size > 1 else 0


def estimate_backlash(s: GuideSession, ol: OpenLoop, max_ms: float | None = None,
                      step_ms: float | None = None) -> BacklashResult:
    pl = ol.dec_plant
    signed_all = pl.signed_ms(s.dec_dur, s.dec_dir)
    nrev = count_reversals(signed_all)
    comp = s.params.get("backlash_pulse_ms")
    if ol.t.size < 60:
        return BacklashResult(False, "有効フレームが少なすぎます", n_reversals=nrev, phd2_comp_ms=comp)
    if nrev < 2:
        return BacklashResult(False, "Dec ガイドの方向反転がほぼ無く、バックラッシュは観測できません"
                              "（一方向ガイド・Dec ガイド無効の可能性）", n_reversals=nrev, phd2_comp_ms=comp)

    nz = signed_all[np.abs(signed_all) > 0]
    first_dir = float(np.sign(nz[0])) if nz.size else 1.0
    # observed Dec position (raw, arcsec) and command at the valid frames
    u_all = np.concatenate([[0.0], np.cumsum(signed_all)[:-1]])
    idx = ol.idx
    t = ol.t - ol.t[0]
    tn = t / max(t[-1], 1.0)
    yobs = ol.dec_raw_as  # closed-loop raw = drift + mechanical correction
    S = segment_matrix(ol.segment, ol.n_segments)
    base = np.hstack([S, tn[:, None], tn[:, None] ** 2])

    if max_ms is None:
        span = np.ptp(u_all) if u_all.size else 0
        max_ms = float(min(5000.0, max(200.0, 0.8 * span)))
    if step_ms is None:
        step_ms = max(5.0, max_ms / 200)
    def evaluate(g):
        rs = np.empty(g.size)
        gn = np.empty(g.size)
        for i, b in enumerate(g):
            y_mech = play_operator(u_all, b, first_dir)[idx]
            X = np.hstack([base, y_mech[:, None]])
            coef, *_ = np.linalg.lstsq(X, yobs, rcond=None)
            r = yobs - X @ coef
            rs[i] = r @ r
            gn[i] = coef[-1]
        return rs, gn

    grid = np.arange(0.0, max_ms + step_ms, step_ms)
    rss, gains = evaluate(grid)
    j0 = int(np.argmin(rss))
    at_edge = j0 == grid.size - 1
    # local refinement (step/10) around the coarse optimum
    fine = np.arange(max(0.0, grid[j0] - 2 * step_ms), grid[j0] + 2 * step_ms + 1e-9, step_ms / 10)
    rf, gf = evaluate(fine)
    grid = np.concatenate([grid, fine])
    rss = np.concatenate([rss, rf])
    gains = np.concatenate([gains, gf])
    o = np.argsort(grid, kind="stable")
    grid, rss, gains = grid[o], rss[o], gains[o]
    j = int(np.argmin(rss))
    b = float(grid[j])
    n, p = yobs.size, base.shape[1] + 2
    sigma2 = rss[j] / max(n - p, 1)
    inside = grid[rss - rss[j] <= sigma2]
    lo, hi = float(inside.min()), float(inside.max())

    y_mech = play_operator(u_all, b, first_dir)[idx]
    X = np.hstack([base, y_mech[:, None]])
    coef, *_ = np.linalg.lstsq(X, yobs, rcond=None)
    model = X @ coef
    cal_gain = pl.rate_px_s / 1000.0 * ol.pixel_scale * np.sign(gains[j] or 1)  # arcsec per ms
    g = float(coef[-1])
    gain_ratio = g / cal_gain if cal_gain else None
    b_as = abs(b * g)

    rel_width = (hi - lo) / max(b, step_ms)
    if nrev >= 10 and rel_width < 0.5 and not at_edge:
        conf = "high"
    elif nrev >= 4 and not at_edge:
        conf = "medium"
    else:
        conf = "low"
    msg = "OK"
    if at_edge:
        msg = "探索範囲の上限に達しました。実際のバックラッシュはさらに大きい可能性があります"
    elif gain_ratio is not None and not (0.5 < gain_ratio < 2.0):
        msg = f"フィットした Dec ゲインがキャリブレーション値の {gain_ratio:.2f} 倍です。再キャリブレーション推奨"
    return BacklashResult(True, msg, b, lo, hi, b_as, gain_ratio, nrev, comp, grid, rss,
                          ol.t, model, yobs, yobs - model, conf)
