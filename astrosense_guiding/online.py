"""Experimental causal harmonic RLS; accepts reconstructed open-loop RA only.

Differences remove segment offsets. Differenced seeing noise is correlated,
so P is an algorithmic information matrix, not a calibrated uncertainty.
No hardware commands are issued. Period is fixed, not learned.
"""
from __future__ import annotations

import numpy as np

from .periodic import harmonic_design
from .predict import PECModel


class OnlinePEC:
    def __init__(self, period: float, n_harmonics: int = 3, *,
                 forgetting: float = 0.999, prior_variance: float = 1000.0,
                 max_gap_s: float = 30.0, t0: float = 0.0):
        if not np.isfinite(period) or period <= 0:
            raise ValueError("period must be finite and positive")
        if not isinstance(n_harmonics, int) or n_harmonics < 1:
            raise ValueError("n_harmonics must be a positive integer")
        if not 0 < forgetting <= 1:
            raise ValueError("forgetting must be in (0, 1]")
        if not np.isfinite(prior_variance) or prior_variance <= 0:
            raise ValueError("prior_variance must be finite and positive")
        if not np.isfinite(max_gap_s) or max_gap_s <= 0 or not np.isfinite(t0):
            raise ValueError("invalid time configuration")
        self.period, self.n_harmonics = float(period), n_harmonics
        self.forgetting, self.max_gap_s, self.t0 = forgetting, max_gap_s, t0
        self.coef = np.zeros(2 * n_harmonics + 1)
        self.P = np.eye(self.coef.size) * prior_variance
        self.n_updates = 0
        self._previous = None

    def _features(self, t):
        tau = np.atleast_1d(np.asarray(t, dtype=float)) - self.t0
        if not np.all(np.isfinite(tau)):
            raise ValueError("times must be finite")
        return np.column_stack((harmonic_design(tau, self.period, self.n_harmonics),
                                tau / self.period))

    def update(self, t: float, drift_as: float, *, segment: int = 0,
               valid: bool = True) -> bool:
        """Update after observing a frame; False means no parameter update.

        Caller must mark dither/settling/outlier frames invalid and change
        segment at lock-position changes. Invalid frames break continuity.
        Duplicate or reversed timestamps raise without modifying state.
        """
        if not valid or not np.isfinite(t) or not np.isfinite(drift_as):
            self._previous = None
            return False
        previous = self._previous
        if previous is not None and t <= previous[0]:
            raise ValueError("timestamps must increase")
        current = (float(t), float(drift_as), segment)
        if previous is None or segment != previous[2] or t - previous[0] > self.max_gap_s:
            self._previous = current
            return False
        x = self._features(t)[0] - self._features(previous[0])[0]
        target = drift_as - previous[1]
        px = self.P @ x
        gain = px / (self.forgetting + x @ px)
        self.coef += gain * (target - x @ self.coef)
        self.P = (self.P - np.outer(gain, px)) / self.forgetting
        self.P = (self.P + self.P.T) / 2
        self._previous = current
        self.n_updates += 1
        return True

    def predict(self, t):
        """Relative PE + drift; arbitrary segment offsets are excluded."""
        return self._features(t) @ self.coef

    def snapshot(self, epoch: str = "") -> PECModel:
        """Independent frozen model compatible with existing replay/forecast."""
        return PECModel(self.period, self.coef[:-1].tolist(),
                        float(self.coef[-1] / self.period), epoch, self.t0)
