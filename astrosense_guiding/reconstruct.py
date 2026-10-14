"""Reconstruct the *uncorrected* (open-loop) mount tracking error.

PHD2 only logs the closed-loop error (star offset after previous corrections)
and the guide pulses it sent.  Because the effect of every pulse is known from
calibration (px per second of pulse), the open-loop drift can be rebuilt as::

    drift[k] = raw[k] - sum_{j<k} effect(pulse[j])

A pulse sent after frame *j* is assumed to be complete before frame *j+1*.
The direction semantics (does a West pulse decrease or increase the RA
distance?) are inferred from the log itself: PHD2 always chooses the pulse
direction that reduces the measured guide distance.

Lock-position changes (dither) make the raw distance jump, so the series is
split into *segments* whose constant offsets are estimated later.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .parser import GuideSession

SIDEREAL_RATE = 15.041  # arcsec / s


@dataclass
class AxisPlant:
    """Linear pulse -> displacement model for one axis."""

    name: str
    rate_px_s: float  # displacement magnitude per second of pulse
    rate_source: str  # "calibration", "guide_speed", "default"
    dir_sign: dict  # direction letter -> +1/-1 effect on the raw distance

    def signed_effect_px(self, dur_ms: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        sgn = np.array([self.dir_sign.get(str(d), 0.0) for d in dirs], dtype=float)
        return sgn * dur_ms / 1000.0 * self.rate_px_s

    def signed_ms(self, dur_ms: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        sgn = np.array([self.dir_sign.get(str(d), 0.0) for d in dirs], dtype=float)
        return sgn * dur_ms


@dataclass
class OpenLoop:
    t: np.ndarray  # valid frame times [s]
    idx: np.ndarray  # indices into the session arrays
    segment: np.ndarray  # segment id per valid frame
    ra_raw_as: np.ndarray  # closed-loop error [arcsec]
    dec_raw_as: np.ndarray
    ra_drift_as: np.ndarray  # reconstructed open-loop drift, per-segment offsets NOT removed
    dec_drift_as: np.ndarray
    ra_cmd_ms: np.ndarray  # cumulative signed pulse (ms) applied before each valid frame
    dec_cmd_ms: np.ndarray
    ra_plant: AxisPlant
    dec_plant: AxisPlant
    pixel_scale: float
    n_segments: int


def _infer_dir_sign(guide: np.ndarray, raw: np.ndarray, dur: np.ndarray, dirs: np.ndarray,
                    letters: tuple[str, str], default: tuple[float, float]) -> dict:
    """Effect sign of each direction letter on the raw distance."""
    ref = np.where(np.isfinite(guide), guide, raw)
    out = {}
    for letter, dflt in zip(letters, default):
        m = (dirs == letter) & (dur > 0) & np.isfinite(ref)
        if m.sum() >= 3:
            med = np.median(np.sign(ref[m]))
            out[letter] = -float(np.sign(med)) if med != 0 else dflt
        else:
            out[letter] = dflt
    a, b = letters
    if out[a] == out[b]:  # inconsistent – fall back to the PHD2 convention
        out = {a: default[0], b: default[1]}
    return out


def build_plants(s: GuideSession) -> tuple[AxisPlant, AxisPlant]:
    p = s.params
    ps = s.pixel_scale or 1.0
    cosd = math.cos(math.radians(p.get("dec_deg", 0.0) or 0.0))

    def rate(cal_key, speed_key, use_cos):
        if p.get(cal_key):
            return abs(p[cal_key]), "calibration"
        if p.get(speed_key):
            return p[speed_key] * (cosd if use_cos else 1.0) / ps, "guide_speed"
        return 0.5 * SIDEREAL_RATE * (cosd if use_cos else 1.0) / ps, "default"

    r_ra, src_ra = rate("x_rate_px_s", "ra_guide_speed", True)
    r_dec, src_dec = rate("y_rate_px_s", "dec_guide_speed", False)
    ra_sign = _infer_dir_sign(s.ra_guide, s.ra_raw, s.ra_dur, s.ra_dir, ("E", "W"), (1.0, -1.0))
    dec_sign = _infer_dir_sign(s.dec_guide, s.dec_raw, s.dec_dur, s.dec_dir, ("N", "S"), (1.0, -1.0))
    return (AxisPlant("RA", r_ra, src_ra, ra_sign), AxisPlant("Dec", r_dec, src_dec, dec_sign))


def segment_ids(s: GuideSession, gap_factor: float = 10.0) -> np.ndarray:
    """Segment id for every frame: new segment at dither events and long gaps."""
    n = s.n_frames
    seg = np.zeros(n, dtype=int)
    breaks = np.zeros(n, dtype=bool)
    for ev in s.events:
        if ev.kind == "dither" and 0 <= ev.frame_index + 1 < n:
            breaks[ev.frame_index + 1] = True
    if n > 2:
        dt = np.diff(s.t)
        cad = np.median(dt)
        breaks[1:] |= dt > gap_factor * cad
    seg = np.cumsum(breaks)
    return seg


def reconstruct(s: GuideSession) -> OpenLoop:
    ra_pl, dec_pl = build_plants(s)
    ps = s.pixel_scale or 1.0
    v = s.valid
    # effect of pulse at frame j appears from frame j+1 on
    ra_eff = ra_pl.signed_effect_px(s.ra_dur, s.ra_dir)
    dec_eff = dec_pl.signed_effect_px(s.dec_dur, s.dec_dir)
    ra_cum = np.concatenate([[0.0], np.cumsum(ra_eff)[:-1]])
    dec_cum = np.concatenate([[0.0], np.cumsum(dec_eff)[:-1]])
    ra_ms = np.concatenate([[0.0], np.cumsum(ra_pl.signed_ms(s.ra_dur, s.ra_dir))[:-1]])
    dec_ms = np.concatenate([[0.0], np.cumsum(dec_pl.signed_ms(s.dec_dur, s.dec_dir))[:-1]])
    seg_all = segment_ids(s)
    idx = np.flatnonzero(v)
    seg = seg_all[idx]
    # renumber to consecutive ids
    _, seg = np.unique(seg, return_inverse=True)
    return OpenLoop(
        t=s.t[idx], idx=idx, segment=seg,
        ra_raw_as=s.ra_raw[idx] * ps, dec_raw_as=s.dec_raw[idx] * ps,
        ra_drift_as=(s.ra_raw[idx] - ra_cum[idx]) * ps,
        dec_drift_as=(s.dec_raw[idx] - dec_cum[idx]) * ps,
        ra_cmd_ms=ra_ms[idx], dec_cmd_ms=dec_ms[idx],
        ra_plant=ra_pl, dec_plant=dec_pl, pixel_scale=ps,
        n_segments=int(seg.max() + 1) if seg.size else 0,
    )


def segment_matrix(segment: np.ndarray, n_segments: int) -> np.ndarray:
    """One-hot design columns for per-segment offsets."""
    M = np.zeros((segment.size, max(n_segments, 1)))
    M[np.arange(segment.size), segment] = 1.0
    return M


def stitch(t: np.ndarray, y: np.ndarray, segment: np.ndarray, n_edge: int = 8) -> np.ndarray:
    """Remove offset jumps between segments by local linear extrapolation."""
    out = y.astype(float).copy()
    for sid in range(1, int(segment.max()) + 1 if segment.size else 0):
        prev = np.flatnonzero(segment == sid - 1)
        cur = np.flatnonzero(segment == sid)
        if prev.size == 0 or cur.size == 0:
            continue
        a = prev[-n_edge:]
        b = cur[:n_edge]
        tb = t[b[0]]
        # level of previous segment extrapolated to tb, vs level of current segment at tb
        if a.size >= 3:
            pa = np.polyfit(t[a] - tb, out[a], 1)
            lev_a = pa[1]
        else:
            lev_a = out[a].mean()
        if b.size >= 3:
            pb = np.polyfit(t[b] - tb, out[b], 1)
            lev_b = pb[1]
        else:
            lev_b = out[b].mean()
        out[cur] += lev_a - lev_b
        # propagate to later segments too: they will be handled in later iterations
        later = np.flatnonzero(segment > sid)
        out[later] += lev_a - lev_b
    return out
