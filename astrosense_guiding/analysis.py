"""High-level analysis pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backlash import BacklashResult, estimate_backlash
from .parser import GuideLog, GuideSession, parse_file
from .periodic import PEResult, estimate_pe
from .predict import evaluate_pec, forecast_commands
from .reconstruct import OpenLoop, reconstruct
from .vibration import VibrationResult, analyze_vibration


@dataclass
class Analysis:
    session: GuideSession
    openloop: OpenLoop
    pe: PEResult
    backlash: BacklashResult
    vibration: VibrationResult
    pec_eval: dict
    forecast: list[dict]


def analyze_session(s: GuideSession, worm_period: float | None = None,
                    forecast_horizon_s: float = 120.0) -> Analysis:
    ol = reconstruct(s)
    pe = estimate_pe(ol, s.params.get("mount"), s.params.get("profile"), worm_period,
                     dec_deg=s.dec_deg)
    bl = estimate_backlash(s, ol)
    vib = analyze_vibration(s, ol, pe.residual if pe.ok else None,
                            bl.residual if bl.ok else None, pe.worm_period)
    ev = evaluate_pec(ol, pe, s.start)
    fc: list[dict] = []
    if ev.get("ok"):
        fc = forecast_commands(ev["model"], ol.ra_plant, ol.pixel_scale,
                               float(ol.t[-1]), forecast_horizon_s, s.cadence)
    return Analysis(s, ol, pe, bl, vib, ev, fc)


def load(path: str) -> GuideLog:
    return parse_file(path)


def usable_sessions(log: GuideLog, min_frames: int = 60) -> list[GuideSession]:
    return [s for s in log.sessions if int(s.valid.sum()) >= min_frames]


def to_jsonable(a: Analysis) -> dict:
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items() if not isinstance(v, np.ndarray) and k != "model"}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x

    pe, bl, vib = a.pe, a.backlash, a.vibration
    out = {
        "session": {"index": a.session.index, "start": a.session.start,
                    "frames": a.session.n_frames, "valid": int(a.session.valid.sum()),
                    "duration_s": a.session.duration, "cadence_s": a.session.cadence,
                    "params": a.session.params},
        "plant": {"ra_rate_px_s": a.openloop.ra_plant.rate_px_s, "ra_rate_source": a.openloop.ra_plant.rate_source,
                  "ra_dir_sign": a.openloop.ra_plant.dir_sign,
                  "dec_rate_px_s": a.openloop.dec_plant.rate_px_s, "dec_rate_source": a.openloop.dec_plant.rate_source,
                  "dec_dir_sign": a.openloop.dec_plant.dir_sign, "segments": a.openloop.n_segments},
        "periodic_error": {"ok": pe.ok, "message": pe.message, "worm_period_s": pe.worm_period,
                           "worm_source": pe.worm_source, "cycles": pe.cycles_covered,
                           "pp_sky_arcsec": pe.pp_sky, "pp_axis_arcsec": pe.pp_axis,
                           "harmonics": pe.harmonics, "drift_as_per_min": pe.drift_rate_as_per_min,
                           "explained_fraction": pe.explained_fraction, "residual_rms": pe.residual_rms,
                           "peaks": [p.__dict__ for p in pe.peaks]},
        "backlash": {"ok": bl.ok, "message": bl.message, "ms": bl.backlash_ms,
                     "ms_1sigma": [bl.backlash_ms_lo, bl.backlash_ms_hi], "arcsec": bl.backlash_arcsec,
                     "gain_ratio": bl.gain_ratio, "reversals": bl.n_reversals,
                     "confidence": bl.confidence, "phd2_comp_ms": bl.phd2_comp_ms},
        "vibration": {"nyquist_hz": vib.nyquist_hz, "noise_ra": vib.noise_ra, "noise_dec": vib.noise_dec,
                      "lag1_ra": vib.lag1_ra, "lag1_dec": vib.lag1_dec,
                      "closed_lag1_ra": vib.closed_lag1_ra, "closed_lag1_dec": vib.closed_lag1_dec,
                      "alt_frac_ra": vib.alt_frac_ra, "alt_frac_dec": vib.alt_frac_dec,
                      "oscillation_ra": vib.oscillation_ra, "oscillation_dec": vib.oscillation_dec,
                      "peaks": vib.peaks, "bursts": [b.__dict__ for b in vib.bursts],
                      "burst_time_fraction": vib.burst_time_fraction},
        "pec_evaluation": a.pec_eval,
        "pec_model": a.pec_eval["model"].to_dict() if a.pec_eval.get("model") else None,
        "feedforward_forecast": a.forecast,
    }
    return clean(out)
