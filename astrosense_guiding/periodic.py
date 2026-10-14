"""Periodic-error (PE) estimation on the reconstructed RA drift."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .reconstruct import OpenLoop, segment_matrix, stitch

SIDEREAL_DAY = 86164.0905  # s

# Worm-wheel tooth counts for common mounts (period = sidereal day / teeth)
MOUNT_WORMS: list[tuple[tuple[str, ...], int, str]] = [
    (("HEQ5",), 135, "Sky-Watcher HEQ5"),
    (("EQ6", "NEQ6", "AZ-EQ6", "AZEQ6", "EQ6-R", "EQ8"), 180, "Sky-Watcher EQ6 family"),
    (("EQ5", "AZ-EQ5", "AZEQ5", "EXOS2", "EXOS-2", "SXD", "SXP", "SX2", "SXD2"), 144, "EQ5 / Vixen SX class"),
    (("EQ3",), 130, "Sky-Watcher EQ3"),
    (("CGEM", "AVX", "CGX", "CEM60", "CEM70", "CEM120"), 180, "Celestron / iOptron (180T)"),
    (("GEM45", "CEM40", "CEM26", "GEM28"), 144, "iOptron (144T)"),
    (("G11", "GM8", "TITAN"), 360, "Losmandy (360T)"),
    (("MACH1", "MACH2", "AP1100", "AP900"), 225, "Astro-Physics (225T)"),
]


def worm_candidates(mount: str | None, profile: str | None = None) -> list[tuple[float, str]]:
    text = " ".join(x for x in (mount, profile) if x).upper().replace(" ", "")
    out: list[tuple[float, str]] = []
    for keys, teeth, label in MOUNT_WORMS:
        for k in keys:
            if k.replace(" ", "") in text:
                out.append((SIDEREAL_DAY / teeth, f"{label} ({teeth}T)"))
                break
    # "HEQ5/6" style strings match both; keep unique periods
    uniq: dict[float, str] = {}
    for p, lab in out:
        uniq.setdefault(round(p, 1), lab)
    return [(p, lab) for p, lab in uniq.items()]


def ls_amplitude(t: np.ndarray, y: np.ndarray, freqs: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Least-squares sinusoid amplitude at each frequency (Hz), floating mean.

    Equivalent to a generalised Lomb–Scargle periodogram but returns the
    fitted semi-amplitude directly, which is what a PE analysis needs.
    """
    t = np.asarray(t, float) - t[0]
    y = np.asarray(y, float) - np.mean(y)
    amps = np.empty(freqs.size)
    for i in range(0, freqs.size, chunk):
        w = 2 * np.pi * freqs[i:i + chunk, None]
        c = np.cos(w * t)
        s = np.sin(w * t)
        c -= c.mean(axis=1, keepdims=True)
        s -= s.mean(axis=1, keepdims=True)
        cc = (c * c).sum(1)
        ss = (s * s).sum(1)
        cs = (c * s).sum(1)
        yc = (c * y).sum(1)
        ys = (s * y).sum(1)
        det = cc * ss - cs * cs
        det = np.where(np.abs(det) < 1e-12, np.nan, det)
        a = (yc * ss - ys * cs) / det
        b = (ys * cc - yc * cs) / det
        amps[i:i + chunk] = np.hypot(a, b)
    return amps


@dataclass
class Peak:
    period: float
    amplitude: float  # semi-amplitude, arcsec (sky)
    snr: float  # amplitude / median spectrum amplitude
    label: str = ""


@dataclass
class PEResult:
    ok: bool
    message: str
    worm_period: float | None = None
    worm_source: str = ""
    n_harmonics: int = 0
    harmonics: list[dict] = field(default_factory=list)  # n, amplitude, phase_rad
    pp_sky: float | None = None  # peak-to-peak, arcsec on sky
    pp_axis: float | None = None  # peak-to-peak referred to RA axis (/cos dec)
    cycles_covered: float = 0.0
    drift_rate_as_per_min: float | None = None
    explained_fraction: float | None = None
    residual_rms: float | None = None
    peaks: list[Peak] = field(default_factory=list)
    # arrays for plotting / downstream use
    freqs: np.ndarray | None = None
    spectrum: np.ndarray | None = None
    t: np.ndarray | None = None
    drift_stitched: np.ndarray | None = None  # offsets & trend removed
    pe_model: np.ndarray | None = None  # PE curve at t
    residual: np.ndarray | None = None  # drift - full model
    t0_epoch_offset: float = 0.0
    coef_trend: tuple[float, float] = (0.0, 0.0)


def harmonic_design(t: np.ndarray, period: float, nh: int) -> np.ndarray:
    cols = []
    for n in range(1, nh + 1):
        w = 2 * np.pi * n / period
        cols += [np.sin(w * t), np.cos(w * t)]
    return np.column_stack(cols) if cols else np.zeros((t.size, 0))


def fit_model(t, y, segment, n_seg, period, nh, poly=2):
    """Least squares: per-segment offsets + polynomial trend + PE harmonics."""
    tc = (t - t[0])
    tn = tc / max(tc[-1], 1.0)
    P = np.column_stack([tn ** k for k in range(1, poly + 1)]) if poly else np.zeros((t.size, 0))
    S = segment_matrix(segment, n_seg)
    H = harmonic_design(tc, period, nh) if period else np.zeros((t.size, 0))
    X = np.hstack([S, P, H])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    ns, npoly = S.shape[1], P.shape[1]
    off = S @ coef[:ns]
    trend = P @ coef[ns:ns + npoly]
    pe = H @ coef[ns + npoly:]
    return coef[ns + npoly:], off, trend, pe, coef[ns:ns + npoly]


def _harmonic_score(freqs, amps, period, nmax=4):
    score = 0.0
    for n in range(1, nmax + 1):
        f = n / period
        if f > freqs[-1] or f < freqs[0]:
            continue
        j = np.argmin(np.abs(freqs - f))
        lo, hi = max(j - 2, 0), min(j + 3, freqs.size)
        score += np.nanmax(amps[lo:hi]) ** 2
    return score


def estimate_pe(ol: OpenLoop, mount: str | None = None, profile: str | None = None,
                worm_period: float | None = None, max_harmonics: int = 6,
                dec_deg: float | None = None) -> PEResult:
    t = ol.t
    y = ol.ra_drift_as
    if t.size < 60:
        return PEResult(False, "有効フレームが少なすぎます (<60)")
    T = t[-1] - t[0]
    cad = float(np.median(np.diff(t)))

    # 1) stitched + detrended series for the periodogram
    ys = stitch(t, y, ol.segment)
    tn = t - t[0]
    ys_d = ys - np.polyval(np.polyfit(tn, ys, 2), tn)

    fmin = 1.0 / max(T / 1.2, 4 * cad)
    fmax = 1.0 / (2.5 * cad)
    df = 1.0 / (T * 8)
    freqs = np.arange(fmin, fmax, df)
    if freqs.size < 10:
        return PEResult(False, "観測時間が短すぎて周波数解析できません")
    amps = ls_amplitude(t, ys_d, freqs)
    med = float(np.nanmedian(amps))

    # 2) choose the worm period
    source = ""
    if worm_period:
        source = "ユーザー指定"
    else:
        cands = worm_candidates(mount, profile)
        cands = [(p, lab) for p, lab in cands if p < T * 1.2]
        if cands:
            scores = [_harmonic_score(freqs, amps, p) for p, _ in cands]
            best = int(np.argmax(scores))
            worm_period, source = cands[best][0], "マウント名から推定: " + cands[best][1]
        else:
            long = freqs < 1 / 60.0
            if long.any():
                j = int(np.nanargmax(np.where(long, amps, -np.inf)))
                worm_period, source = 1 / freqs[j], "スペクトル最大ピーク"
    if not worm_period:
        return PEResult(False, "ウォーム周期を決定できません", freqs=freqs, spectrum=amps)

    # 3) refine period when >= 2 cycles are available
    cycles = T / worm_period
    if cycles >= 2.0 and "ユーザー" not in source:
        grid = worm_period * np.linspace(0.97, 1.03, 61)
        rss = []
        for P in grid:
            _, off, tr, pe, _ = fit_model(t, y, ol.segment, ol.n_segments, P, 3)
            rss.append(np.sum((y - off - tr - pe) ** 2))
        Pbest = grid[int(np.argmin(rss))]
        if abs(Pbest - worm_period) / worm_period > 0.002:
            source += f" → 周期を {Pbest:.1f} s に微調整"
        worm_period = float(Pbest)

    nh = max(1, min(max_harmonics, int(worm_period / (4 * cad))))
    hc, off, trend, pe, ptr = fit_model(t, y, ol.segment, ol.n_segments, worm_period, nh)
    resid = y - off - trend - pe
    harmonics = []
    for n in range(1, nh + 1):
        a, b = hc[2 * (n - 1)], hc[2 * (n - 1) + 1]
        harmonics.append({"n": n, "period": worm_period / n,
                          "amplitude": float(math.hypot(a, b)),
                          "phase_rad": float(math.atan2(b, a))})
    ph = np.linspace(0, worm_period, 2000)
    curve = harmonic_design(ph, worm_period, nh) @ hc
    pp_sky = float(curve.max() - curve.min())
    cosd = math.cos(math.radians(dec_deg)) if dec_deg is not None and abs(dec_deg) < 85 else None
    pp_axis = pp_sky / cosd if cosd else None
    var_no_pe = np.var(y - off - trend)
    explained = float(1 - np.var(resid) / var_no_pe) if var_no_pe > 0 else None
    drift_rate = float(ptr[0] / max(T, 1) * 60) if ptr.size else None

    # 4) significant peaks left in the residual (after removing PE + trend),
    #    which avoids spectral leakage/side-lobes of the strong worm signal
    peaks: list[Peak] = []
    from scipy.signal import find_peaks
    amps_r = ls_amplitude(t, resid, freqs)
    med_r = float(np.nanmedian(amps_r))
    idx, _ = find_peaks(amps_r, height=5 * med_r, distance=4)
    for j in sorted(idx, key=lambda j: -amps_r[j])[:10]:
        P = 1 / freqs[j]
        ratio = worm_period / P
        n = round(ratio)
        label = f"ウォーム第{n}高調波(残)" if n >= 1 and abs(ratio - n) < 0.03 * max(n, 1) else "非ウォーム成分"
        peaks.append(Peak(float(P), float(amps_r[j]), float(amps_r[j] / med_r), label))

    msg = "OK"
    if cycles < 1.0:
        msg = f"ログ長がウォーム周期の {cycles:.2f} 周期分しかありません。振幅は参考値です"
    elif cycles < 2.0:
        msg = f"ウォーム {cycles:.1f} 周期分。2周期以上あると精度が上がります"

    return PEResult(
        ok=True, message=msg, worm_period=float(worm_period), worm_source=source,
        n_harmonics=nh, harmonics=harmonics, pp_sky=pp_sky, pp_axis=pp_axis,
        cycles_covered=float(cycles), drift_rate_as_per_min=drift_rate,
        explained_fraction=explained, residual_rms=float(np.std(resid)), peaks=peaks,
        freqs=freqs, spectrum=amps, t=t, drift_stitched=y - off, pe_model=pe, residual=resid,
        coef_trend=tuple(ptr[:2]) if ptr.size >= 2 else (0.0, 0.0),
    )
