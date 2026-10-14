"""Foundation for predictive guiding.

This module does **not** talk to a mount.  It provides:

* :class:`PECModel` – a harmonic periodic-error model tied to absolute time,
  which can predict the drift (and its rate) at any future instant;
* :func:`replay` – offline closed-loop replay of the reconstructed drift with
  a PHD2-like feedback controller, optionally augmented with model
  feed-forward, to quantify how much prediction would help;
* :func:`forecast_commands` – the feed-forward pulse schedule that a future
  predictive controller would send (ms + direction per guide step).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import numpy as np

from .periodic import PEResult, fit_model, harmonic_design
from .reconstruct import AxisPlant, OpenLoop


def _parse_start(start: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(start.strip(), fmt)
        except ValueError:
            pass
    return None


@dataclass
class PECModel:
    period: float  # s
    coefs: list[float]  # sin/cos pairs, arcsec (sky)
    drift_rate: float  # linear drift, arcsec / s
    epoch: str  # ISO time corresponding to t = 0
    t0: float = 0.0  # session-relative time that maps to epoch

    @property
    def n_harmonics(self) -> int:
        return len(self.coefs) // 2

    def _tau(self, t_rel: np.ndarray) -> np.ndarray:
        return np.asarray(t_rel, float) - self.t0

    def predict(self, t_rel) -> np.ndarray:
        """Predicted drift (PE + linear drift) at session-relative times [arcsec]."""
        tau = self._tau(np.atleast_1d(t_rel))
        H = harmonic_design(tau, self.period, self.n_harmonics)
        return H @ np.asarray(self.coefs) + self.drift_rate * tau

    def pe_only(self, t_rel) -> np.ndarray:
        tau = self._tau(np.atleast_1d(t_rel))
        return harmonic_design(tau, self.period, self.n_harmonics) @ np.asarray(self.coefs)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_fit(cls, ol: OpenLoop, period: float, nh: int, epoch: str,
                 train: slice | None = None) -> "PECModel":
        t, y, seg = ol.t, ol.ra_drift_as, ol.segment
        if train is not None:
            t, y, seg = t[train], y[train], seg[train]
        _, seg = np.unique(seg, return_inverse=True)
        hc, _, _, _, ptr = fit_model(t, y, seg, int(seg.max()) + 1, period, nh, poly=1)
        T = max(t[-1] - t[0], 1.0)
        # fit_model uses tau = t - t[0]
        return cls(period=float(period), coefs=[float(c) for c in hc],
                   drift_rate=float(ptr[0] / T) if ptr.size else 0.0, epoch=epoch, t0=float(t[0]))


@dataclass
class ControllerConfig:
    aggression: float = 0.7
    min_move_as: float = 0.3
    hysteresis: float = 0.1
    max_pulse_ms: float = 2500.0


@dataclass
class ReplayResult:
    name: str
    rms: float
    pp: float
    error: np.ndarray


def replay(t: np.ndarray, drift: np.ndarray, rate_as_per_ms: float, cfg: ControllerConfig,
           model: PECModel | None = None, name: str = "") -> ReplayResult:
    """Replay a drift series through a hysteresis feedback controller.

    ``drift`` is the open-loop error (arcsec) including seeing noise.  The
    measured error at frame k is ``drift[k] + corr[k]``; after each frame the
    controller applies a correction that takes effect before frame k+1.  When
    ``model`` is given, the predicted drift change over the next interval is
    cancelled in advance (feed-forward).
    """
    n = t.size
    corr = 0.0
    prev = 0.0
    err = np.empty(n)
    maxc = rate_as_per_ms * cfg.max_pulse_ms
    pred = model.predict(t) if model is not None else None
    for k in range(n):
        e = drift[k] + corr
        err[k] = e
        if k == n - 1:
            break
        # feedback (PHD2 hysteresis algorithm)
        g = (1 - cfg.hysteresis) * e + cfg.hysteresis * prev
        prev = g
        fb = cfg.aggression * g if abs(g) >= cfg.min_move_as else 0.0
        ff = (pred[k + 1] - pred[k]) if pred is not None else 0.0
        u = float(np.clip(-fb - ff, -maxc, maxc))
        corr += u
    return ReplayResult(name, float(np.sqrt(np.mean(err ** 2))), float(np.ptp(err)), err)


def evaluate_pec(ol: OpenLoop, pe: PEResult, epoch: str, cfg: ControllerConfig | None = None) -> dict:
    """Compare feedback-only vs feedback+PEC feed-forward on held-out data."""
    if not pe.ok or not pe.worm_period:
        return {"ok": False, "message": "PE モデルが無いため評価できません"}
    cfg = cfg or ControllerConfig()
    t = ol.t
    T = t[-1] - t[0]
    holdout = T / pe.worm_period >= 2.0
    if holdout:
        h = int(np.searchsorted(t, t[0] + T / 2))
        model = PECModel.from_fit(ol, pe.worm_period, pe.n_harmonics, epoch, slice(0, h))
        test = slice(h, None)
    else:
        model = PECModel.from_fit(ol, pe.worm_period, pe.n_harmonics, epoch)
        test = slice(0, None)
    drift = pe.drift_stitched  # offsets removed, trend kept
    rate = ol.ra_plant.rate_px_s / 1000.0 * ol.pixel_scale
    tt, dd = t[test], drift[test] - drift[test][0]
    base = replay(tt, dd, rate, cfg, None, "フィードバックのみ (PHD2相当)")
    pec = replay(tt, dd, rate, cfg, model, "フィードバック + PEC予測")
    imp = 1 - pec.rms / base.rms if base.rms > 0 else 0.0
    return {
        "ok": True,
        "holdout": holdout,
        "message": ("後半データで検証（前半で学習）" if holdout
                    else "ウォーム2周期未満のため学習データ内での評価（楽観的）"),
        "baseline_rms": base.rms, "pec_rms": pec.rms, "improvement": float(imp),
        "t": tt, "baseline_error": base.error, "pec_error": pec.error, "model": model,
    }


def forecast_commands(model: PECModel, plant: AxisPlant, pixel_scale: float, t_start: float,
                      horizon_s: float, cadence_s: float) -> list[dict]:
    """Feed-forward pulse schedule cancelling the predicted drift.

    Returns one entry per guide step with absolute time, pulse length (ms) and
    direction letter.  Intended for a future controller – nothing is sent.
    """
    ts = np.arange(t_start, t_start + horizon_s + 1e-9, cadence_s)
    pred = model.predict(ts)
    rate = plant.rate_px_s / 1000.0 * pixel_scale  # arcsec per ms
    epoch = _parse_start(model.epoch)
    inv = {v: k for k, v in plant.dir_sign.items()}  # effect sign -> letter
    out = []
    for k in range(ts.size - 1):
        delta = pred[k + 1] - pred[k]  # predicted drift increase
        need = -delta  # correction to apply
        ms = abs(need) / rate if rate > 0 else 0.0
        letter = inv.get(float(np.sign(need)), "") if ms > 0 else ""
        when = (epoch + timedelta(seconds=float(ts[k]))).isoformat(timespec="seconds") if epoch else None
        out.append({"t_rel": float(ts[k]), "time": when, "pulse_ms": round(float(ms), 1),
                    "direction": letter, "predicted_drift_as": float(pred[k])})
    return out
