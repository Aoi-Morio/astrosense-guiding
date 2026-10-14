"""Vibration, wind gusts and over-correction diagnostics.

Guide exposures (typically 1–5 s) integrate any mechanical vibration faster
than the frame rate, so true structural ringing (Hz-range) cannot be resolved
directly; it is aliased or shows up as extra white scatter.  What *can* be
measured from a guide log is:

* periodic components between ~2.5 frames and a few minutes that are not worm
  harmonics (motor / transfer gears, belt, fan, ...)
* transient bursts (wind gusts, cable snags, touching the mount)
* oscillation caused by over-correction (negative lag-1 autocorrelation and
  alternating guide pulses)
* the white-noise floor (seeing + centroid noise)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .parser import GuideSession
from .periodic import ls_amplitude
from .reconstruct import OpenLoop


@dataclass
class Burst:
    t_start: float
    t_end: float
    peak_arcsec: float
    axes: str
    star_mass_ratio: float | None  # median star mass during burst / overall median
    likely: str


@dataclass
class VibrationResult:
    nyquist_hz: float
    noise_ra: float
    noise_dec: float
    lag1_ra: float
    lag1_dec: float
    closed_lag1_ra: float
    closed_lag1_dec: float
    alt_frac_ra: float  # fraction of consecutive pulses that switch direction
    alt_frac_dec: float
    oscillation_ra: bool
    oscillation_dec: bool
    peaks: list[dict] = field(default_factory=list)
    bursts: list[Burst] = field(default_factory=list)
    burst_time_fraction: float = 0.0
    freqs: np.ndarray | None = None
    spec_ra: np.ndarray | None = None
    spec_dec: np.ndarray | None = None
    t: np.ndarray | None = None
    res_ra: np.ndarray | None = None
    res_dec: np.ndarray | None = None


def _robust_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _lag1(x: np.ndarray) -> float:
    x = x[np.isfinite(x)] - np.nanmean(x)
    if x.size < 3 or np.dot(x, x) == 0:
        return float("nan")
    return float(np.dot(x[1:], x[:-1]) / np.dot(x, x))


def _alt_frac(dur: np.ndarray, dirs: np.ndarray) -> float:
    d = dirs[dur > 0]
    if d.size < 3:
        return float("nan")
    return float(np.mean(d[1:] != d[:-1]))


def _highpass(t: np.ndarray, y: np.ndarray, window_s: float) -> np.ndarray:
    """Remove slow content with a running median (robust against bursts)."""
    out = np.empty_like(y)
    h = window_s / 2
    lo = 0
    hi = 0
    n = y.size
    for i in range(n):
        while t[lo] < t[i] - h:
            lo += 1
        while hi < n and t[hi] <= t[i] + h:
            hi += 1
        out[i] = y[i] - np.median(y[lo:hi])
    return out


def analyze_vibration(s: GuideSession, ol: OpenLoop, res_ra: np.ndarray | None = None,
                      res_dec: np.ndarray | None = None, worm_period: float | None = None,
                      z_thresh: float = 4.5) -> VibrationResult:
    t = ol.t
    cad = s.cadence
    nyq = 1.0 / (2 * cad) if cad and np.isfinite(cad) else float("nan")
    # residuals after PE/backlash models, high-passed to remove what is left of slow drift
    rr = res_ra if res_ra is not None else ol.ra_drift_as
    rd = res_dec if res_dec is not None else ol.dec_drift_as
    win = max(30 * cad, 60.0)
    hr = _highpass(t, rr, win)
    hd = _highpass(t, rd, win)

    noise_ra, noise_dec = _robust_std(hr), _robust_std(hd)

    # --- periodic components faster than ~2 min that are not worm harmonics
    peaks: list[dict] = []
    freqs = spec_ra = spec_dec = None
    if t.size > 100:
        fmin = 1.0 / min(240.0, (t[-1] - t[0]) / 3)
        fmax = 1.0 / (2.5 * cad)
        if fmax > fmin:
            freqs = np.linspace(fmin, fmax, 1500)
            spec_ra = ls_amplitude(t, hr, freqs)
            spec_dec = ls_amplitude(t, hd, freqs)
            from scipy.signal import find_peaks
            for axis, sp in (("RA", spec_ra), ("Dec", spec_dec)):
                med = np.nanmedian(sp)
                idx, _ = find_peaks(sp, height=4.0 * med, distance=5)
                for j in sorted(idx, key=lambda j: -sp[j])[:5]:
                    P = 1 / freqs[j]
                    harm = None
                    if worm_period:
                        r = worm_period / P
                        if abs(r - round(r)) < 0.03 * round(r) and round(r) >= 1:
                            harm = int(round(r))
                    peaks.append({"axis": axis, "period": float(P), "amplitude": float(sp[j]),
                                  "snr": float(sp[j] / med), "worm_harmonic": harm})

    # --- bursts: robust z-score of the 2D residual
    rad = np.hypot(hr - np.median(hr), hd - np.median(hd))
    sig = np.hypot(noise_ra, noise_dec)
    z = rad / sig if sig > 0 else np.zeros_like(rad)
    flag = z > z_thresh
    # dilate to merge flags separated by short quiet gaps (one gust = one event)
    fl = flag.copy()
    for sh in range(1, 4):
        fl[sh:] |= flag[:-sh]
        fl[:-sh] |= flag[sh:]
    bursts: list[Burst] = []
    mass = s.star_mass[ol.idx]
    mmed = np.nanmedian(mass) if np.isfinite(mass).any() else np.nan
    i = 0
    while i < fl.size:
        if fl[i]:
            j = i
            while j + 1 < fl.size and fl[j + 1]:
                j += 1
            if flag[i:j + 1].sum() >= 1:
                seg = slice(i, j + 1)
                pra = np.max(np.abs(hr[seg]))
                pdc = np.max(np.abs(hd[seg]))
                axes = "RA+Dec" if min(pra, pdc) > 0.5 * max(pra, pdc) else ("RA" if pra > pdc else "Dec")
                mr = float(np.nanmedian(mass[seg]) / mmed) if np.isfinite(mmed) and mmed > 0 else None
                if mr is not None and mr < 0.6:
                    likely = "雲・透明度低下（星の光量低下を伴う）"
                elif axes == "RA+Dec":
                    likely = "風・接触などの機械的外乱"
                else:
                    likely = f"{axes} 軸の機械的外乱（ギア噛み込み・ケーブル等）"
                bursts.append(Burst(float(t[i]), float(t[j]), float(rad[seg].max()), axes, mr, likely))
            i = j + 1
        else:
            i += 1
    btime = sum(max(b.t_end - b.t_start, cad) for b in bursts)
    frac = btime / max(t[-1] - t[0], 1.0)

    # --- over-correction / oscillation on the closed-loop error
    v = s.valid
    cl_ra, cl_dec = _lag1(s.ra_raw[v]), _lag1(s.dec_raw[v])
    af_ra, af_dec = _alt_frac(s.ra_dur, s.ra_dir), _alt_frac(s.dec_dur, s.dec_dir)
    osc_ra = bool(cl_ra < -0.25 and af_ra > 0.55)
    osc_dec = bool(cl_dec < -0.25 and af_dec > 0.55)

    return VibrationResult(
        nyquist_hz=nyq, noise_ra=noise_ra, noise_dec=noise_dec,
        lag1_ra=_lag1(hr), lag1_dec=_lag1(hd), closed_lag1_ra=cl_ra, closed_lag1_dec=cl_dec,
        alt_frac_ra=af_ra, alt_frac_dec=af_dec, oscillation_ra=osc_ra, oscillation_dec=osc_dec,
        peaks=peaks, bursts=bursts, burst_time_fraction=float(frac),
        freqs=freqs, spec_ra=spec_ra, spec_dec=spec_dec, t=t, res_ra=hr, res_dec=hd,
    )
