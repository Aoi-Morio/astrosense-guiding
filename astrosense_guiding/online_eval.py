"""Causal, ideal-plant offline evaluation of harmonic online learning.

Controllers receive only the current closed-loop measurement. The simulator
alone knows noise-free drift. Pulse effects are exact and complete by the next
frame; cadence, calibration, and worm period are known. No hardware I/O.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .online import OnlinePEC
from .predict import ControllerConfig, PECModel
from .periodic import harmonic_design


class CausalController:
    """Offline reference controller with a scheduled, constant guide cadence.

    Call step once per frame, including invalid ones. correction_as represents
    completed commands before this frame; the returned signed pulse completes
    before the next frame. Positive pulse means positive displacement.
    """

    def __init__(self, mode: str, period: float, cadence_s: float, warmup_s: float,
                 rate_as_per_ms: float, *, n_harmonics: int = 4,
                 forgetting: float = 0.995, cfg: ControllerConfig | None = None):
        if mode not in {"feedback", "fixed", "online"}:
            raise ValueError("unknown controller mode")
        for name, value in (("cadence_s", cadence_s), ("warmup_s", warmup_s),
                            ("rate_as_per_ms", rate_as_per_ms)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.cfg = cfg or ControllerConfig()
        if (not 0 <= self.cfg.aggression <= 1 or not 0 <= self.cfg.hysteresis < 1
                or not np.isfinite(self.cfg.min_move_as) or self.cfg.min_move_as < 0
                or not np.isfinite(self.cfg.max_pulse_ms) or self.cfg.max_pulse_ms <= 0):
            raise ValueError("invalid controller configuration")
        self.mode, self.period, self.cadence_s = mode, period, cadence_s
        self.warmup_s, self.rate = warmup_s, rate_as_per_ms
        self.learner = OnlinePEC(period, n_harmonics, forgetting=forgetting)
        self.n_harmonics = n_harmonics
        self.fixed_model: PECModel | None = None
        self.correction_as = 0.0
        self.prediction_as = 0.0
        self.saturated = False
        self.ready = False
        self._previous_error = 0.0
        self._start = None
        self._last_time = None
        self._train_t: list[float] = []
        self._train_y: list[float] = []
        self._fixed_fit_attempted = False

    def step(self, t: float, measured_error_as: float, *, valid: bool = True) -> float:
        """Consume one observation and return a bounded signed pulse in ms."""
        if not np.isfinite(t):
            raise ValueError("time must be finite")
        if self._last_time is not None and not np.isclose(
                t - self._last_time, self.cadence_s, rtol=1e-7, atol=1e-8):
            raise ValueError("step must follow the configured cadence")
        self._last_time = t
        if self._start is None:
            self._start = t
            self.learner.t0 = t
        self.prediction_as, self.saturated = 0.0, False
        if not valid or not np.isfinite(measured_error_as):
            self.learner.update(t, np.nan, valid=False)
            self._previous_error = 0.0
            return 0.0

        # Reconstruct from this controller's own completed pulse history.
        observation = measured_error_as - self.correction_as
        elapsed = t - self._start
        if self.mode == "online":
            self.learner.update(t, observation)
            self.ready = elapsed >= self.warmup_s and self.learner.n_updates >= 2 * self.n_harmonics + 1
            if self.ready:
                pred = self.learner.predict([t, t + self.cadence_s])
                self.prediction_as = float(pred[1] - pred[0])
        elif self.mode == "fixed":
            if elapsed < self.warmup_s:
                self._train_t.append(t)
                self._train_y.append(observation)
            elif not self._fixed_fit_attempted:
                # Fit exactly once, from observations strictly before warmup ends.
                self._fixed_fit_attempted = True
                tau = np.asarray(self._train_t) - self._start
                design = np.column_stack((harmonic_design(tau, self.period, self.n_harmonics),
                                          tau / self.period, np.ones(tau.size)))
                coef, _, rank, _ = np.linalg.lstsq(design, self._train_y, rcond=None)
                if rank == design.shape[1]:
                    self.fixed_model = PECModel(self.period, coef[:-2].tolist(),
                                                float(coef[-2] / self.period), "", self._start)
                # Insufficient prefix data leaves this controller in feedback mode.
            self.ready = self.fixed_model is not None
            if self.ready:
                pred = self.fixed_model.predict([t, t + self.cadence_s])
                self.prediction_as = float(pred[1] - pred[0])

        cfg = self.cfg
        g = (1 - cfg.hysteresis) * measured_error_as + cfg.hysteresis * self._previous_error
        self._previous_error = g
        feedback = cfg.aggression * g if abs(g) >= cfg.min_move_as else 0.0
        requested_ms = (-feedback - self.prediction_as) / self.rate
        # Keep the complete-before-next-frame assumption true even at saturation.
        limit = min(cfg.max_pulse_ms, self.cadence_s * 1000)
        pulse_ms = float(np.clip(requested_ms, -limit, limit))
        self.saturated = abs(requested_ms) > limit
        self.correction_as += pulse_ms * self.rate
        return pulse_ms


@dataclass
class OnlineTrace:
    t: np.ndarray
    tracking_error: np.ndarray
    measured_error: np.ndarray
    pulse_ms: np.ndarray
    predicted_delta: np.ndarray
    saturated: np.ndarray
    ready: np.ndarray
    valid: np.ndarray


def simulate(drift_as, noise_as, *, mode: str, period: float, cadence_s: float = 2.15,
             warmup_s: float, rate_as_per_ms: float = 0.007,
             forgetting: float = 0.995, n_harmonics: int = 4,
             valid=None, cfg: ControllerConfig | None = None) -> OnlineTrace:
    """Replay identical exogenous drift/noise through an independent controller.

    True tracking RMS includes lost-star frames. Seeing is additive measurement
    noise and does not itself move the simulated mount. Validity is an input
    available at observation time, not inferred from future samples.
    """
    drift, noise = np.asarray(drift_as, float), np.asarray(noise_as, float)
    if drift.ndim != 1 or drift.size < 2 or noise.shape != drift.shape:
        raise ValueError("drift and noise must be matching 1-D arrays, length >= 2")
    if not np.all(np.isfinite(drift)) or not np.all(np.isfinite(noise)):
        raise ValueError("drift and noise must be finite")
    valid = np.ones(drift.size, bool) if valid is None else np.asarray(valid, bool)
    if valid.shape != drift.shape:
        raise ValueError("valid must match drift")
    controller = CausalController(mode, period, cadence_s, warmup_s, rate_as_per_ms,
                                  forgetting=forgetting, n_harmonics=n_harmonics, cfg=cfg)
    t = np.arange(drift.size) * cadence_s
    error, measured, pulse, pred = [np.empty(drift.size) for _ in range(4)]
    saturated, ready = [np.zeros(drift.size, bool) for _ in range(2)]
    for k, tk in enumerate(t):
        error[k] = drift[k] + controller.correction_as
        measured[k] = error[k] + noise[k] if valid[k] else np.nan
        pulse[k] = controller.step(float(tk), float(measured[k]), valid=bool(valid[k]))
        pred[k], saturated[k], ready[k] = controller.prediction_as, controller.saturated, controller.ready
    # No command is scheduled after the final observation.
    pulse[-1], pred[-1], saturated[-1] = 0.0, 0.0, False
    return OnlineTrace(t, error, measured, pulse, pred, saturated, ready, valid.copy())


@dataclass
class BenchmarkConfig:
    period_s: float = 638.3
    cadence_s: float = 2.15
    duration_cycles: float = 12.0
    warmup_cycles: float = 3.0
    late_start_cycles: float = 8.0
    forgetting: float = 0.995
    n_harmonics: int = 4
    rate_as_per_ms: float = 0.007


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def run_benchmark(seeds=range(1, 6), config: BenchmarkConfig | None = None):
    """Paired-seed, prespecified scenarios; no per-case parameter tuning."""
    cfg = config or BenchmarkConfig()
    if not (0 < cfg.warmup_cycles < cfg.late_start_cycles < cfg.duration_cycles):
        raise ValueError("require 0 < warmup < late_start < duration cycles")
    if (not np.isfinite(cfg.period_s) or cfg.period_s <= 0
            or not np.isfinite(cfg.cadence_s) or cfg.cadence_s <= 0
            or not np.isfinite(cfg.duration_cycles)):
        raise ValueError("invalid benchmark time configuration")
    seeds = list(seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    t = np.arange(0, cfg.duration_cycles * cfg.period_s, cfg.cadence_s)
    cycles = t / cfg.period_s
    # Fundamental plus second/fourth harmonics, in sky arcseconds.
    pe = 7.5 * np.sin(2 * np.pi * cycles + .4) + 2.35 * np.sin(4 * np.pi * cycles + 1.3) + .94 * np.sin(8 * np.pi * cycles - .6)
    scenarios = [
        ("stationary_low", .2, False, 0.0),
        ("stationary_typical", .6, False, 0.0),
        ("stationary_high", 1.2, False, 0.0),
        ("amplitude_ramp", .6, True, 0.0),
        ("amplitude_ramp_drops", .6, True, .05),
    ]
    rows, examples = [], {}
    for label, seeing, ramp, drop_probability in scenarios:
        scale = 1 + np.clip((cycles - cfg.warmup_cycles) / 2, 0, 1) if ramp else np.ones(t.size)
        drift = scale * pe + .003 * t
        drift -= drift[0]
        for seed in seeds:
            # Identical noise stream between scenarios; masks use a separate RNG.
            noise = np.random.default_rng(seed).normal(0, seeing, t.size)
            valid = np.random.default_rng(seed + 100000).random(t.size) >= drop_probability
            for mode in ("feedback", "fixed", "online"):
                trace = simulate(drift, noise, mode=mode, period=cfg.period_s,
                                 cadence_s=cfg.cadence_s, warmup_s=cfg.warmup_cycles * cfg.period_s,
                                 rate_as_per_ms=cfg.rate_as_per_ms, forgetting=cfg.forgetting,
                                 n_harmonics=cfg.n_harmonics, valid=valid)
                masks = {"post_warmup": cycles >= cfg.warmup_cycles,
                         "late": cycles >= cfg.late_start_cycles}
                for window, mask in masks.items():
                    observed = mask & valid
                    # Forecast error is scored only for issued intervals, never
                    # used as an input to the controller.
                    forecast_mask = mask[:-1] & valid[:-1]
                    delta_error = trace.predicted_delta[:-1] - np.diff(drift)
                    rows.append({"scenario": label, "seed": int(seed), "mode": mode,
                                 "window": window, "tracking_rms_as": _rms(trace.tracking_error[mask]),
                                 "measured_rms_as": _rms(trace.measured_error[observed]) if observed.any() else None,
                                 "delta_rmse_as": _rms(delta_error[forecast_mask]) if forecast_mask.any() else None,
                                 "saturated_frames": int(trace.saturated[mask].sum()),
                                 "ready_frames": int(trace.ready[observed].sum()),
                                 "frames": int(mask.sum()), "valid_frames": int(observed.sum()),
                                 "max_abs_pulse_ms": float(np.max(np.abs(trace.pulse_ms[mask])))})
                if seed == seeds[0] and label in {"stationary_typical", "amplitude_ramp"}:
                    examples[(label, mode)] = trace
    summary = []
    for label, seeing, ramp, drops in scenarios:
        for window in ("post_warmup", "late"):
            modes = {}
            for mode in ("feedback", "fixed", "online"):
                selected = [r for r in rows if (r["scenario"], r["window"], r["mode"]) == (label, window, mode)]
                values = [r["tracking_rms_as"] for r in selected]
                measured_values = [r["measured_rms_as"] for r in selected if r["measured_rms_as"] is not None]
                modes[mode] = {"mean_tracking_rms_as": float(np.mean(values)),
                               "min_tracking_rms_as": float(np.min(values)),
                               "max_tracking_rms_as": float(np.max(values)),
                               "mean_measured_rms_as": float(np.mean(measured_values)) if measured_values else None}
            online = modes["online"]["mean_tracking_rms_as"]
            summary.append({"scenario": label, "seeing_as": seeing, "amplitude_ramp": ramp,
                            "drop_probability": drops, "window": window, "modes": modes,
                            "online_improvement_vs_feedback": 1 - online / modes["feedback"]["mean_tracking_rms_as"],
                            "online_improvement_vs_fixed": 1 - online / modes["fixed"]["mean_tracking_rms_as"]})
    return {"config": asdict(cfg), "controller_config": asdict(ControllerConfig()),
            "numpy_version": np.__version__, "seeds": seeds,
            "assumptions": ["Known fixed period, cadence, and exact pulse calibration",
                            "Instantaneous additive seeing measurement noise; no exposure integration",
                            "Fixed PEC fits only the pre-warmup prefix; online updates only on current/past frames",
                            "Amplitude doubles smoothly during the two cycles after warmup",
                            "No gusts, dithers, actuator delay, calibration error, or changing worm period",
                            "True tracking RMS includes dropped frames; measured RMS excludes them",
                            "No parameter tuning or model selection on benchmark results"],
            "summary": summary, "runs": rows}, examples


def benchmark_markdown(result: dict) -> str:
    cfg = result["config"]
    lines = ["# AstroSense Guiding オンライン予測評価", "",
             f"乱数seed: {result['seeds']}。ウォーム周期 {cfg['period_s']} s、"
             f"全{cfg['duration_cycles']:g}周期、学習待機{cfg['warmup_cycles']:g}周期。",
             f"RLS忘却係数 {cfg['forgetting']}、{cfg['n_harmonics']}高調波。",
             f"観測間隔 {cfg['cadence_s']} s、補正レート {cfg['rate_as_per_ms']} 秒角/ms。",
             f"FB設定: {result['controller_config']}。NumPy {result['numpy_version']}。",
             "", "指標は真の追尾誤差RMS（秒角）。seeingを含む観測RMSと各実行値はJSON参照。",
             "平均は同じseedを使った実行間平均。負の改善率は悪化を表す。", "",
             "| 条件 | 評価区間 | FB | 固定PEC | RLS | RLS対FB | RLS対固定PEC |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in result["summary"]:
        modes = row["modes"]
        interval = "学習待機後すべて" if row['window'] == 'post_warmup' else f"{cfg['late_start_cycles']:g}周期以降"
        lines.append(f"| {row['scenario']} | {interval} | "
                     f"{modes['feedback']['mean_tracking_rms_as']:.4f} | "
                     f"{modes['fixed']['mean_tracking_rms_as']:.4f} | "
                     f"{modes['online']['mean_tracking_rms_as']:.4f} | "
                     f"{row['online_improvement_vs_feedback']:+.1%} | "
                     f"{row['online_improvement_vs_fixed']:+.1%} |")
    lines += ["", "## 評価条件", "", *[f"- {a}" for a in result["assumptions"]], "",
              "seeing条件は0.2/0.6/1.2秒角の独立な観測ノイズ。振幅変化条件では、",
              "学習待機終了後の2周期でPEの全高調波の振幅が1倍から2倍へ線形に変わる。",
              "欠測条件は5%のランダムな星喪失で、そのフレームでは学習と補正を止める。",
              f"固定PECは最初の{cfg['warmup_cycles']:g}周期の観測を一括で最小二乗フィットし、その後は固定する。",
              "RLSは逐次更新し、全方式とも同じフィードバック・パルス上限を使用する。", "",
              "真の追尾誤差はシミュレータ側だけが採点に使う。制御器は閉ループ観測と自身の",
              "補正履歴から学習する。未来の観測値や真値をモデル選択・指令生成に使わない。", "",
              "合成データ・理想的な赤道儀モデルでの比較であり、実機性能は未検証。",
              "既存ログ解析の0.83→0.80秒角は観測誤差の別評価で、この表とは直接比較しない。", ""]
    return "\n".join(lines)


def plot_benchmark(result: dict, examples: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for j, scenario in enumerate(("stationary_typical", "amplitude_ramp")):
        for mode in ("feedback", "fixed", "online"):
            trace = examples[(scenario, mode)]
            window = max(1, int(.5 * result['config']['period_s'] / result['config']['cadence_s']))
            rolling_rms = np.sqrt(np.convolve(trace.tracking_error ** 2,
                                             np.ones(window) / window, mode='valid'))
            axes[0, j].plot(trace.t[window - 1:] / result['config']['period_s'], rolling_rms,
                            label=mode, alpha=.85, linewidth=1)
        axes[0, j].set(title=f"{scenario} (seed {result['seeds'][0]})",
                       xlabel="Worm cycles", ylabel="Trailing half-cycle tracking RMS [arcsec]")
        axes[0, j].axvline(result['config']['warmup_cycles'], color='k', linestyle='--', linewidth=.8)
        axes[0, j].legend()
    labels = [r['scenario'] for r in result['summary'] if r['window'] == 'late']
    for j, window in enumerate(("post_warmup", "late")):
        selected = [r for r in result['summary'] if r['window'] == window]
        for i, mode in enumerate(("feedback", "fixed", "online")):
            values = [r['modes'][mode]['mean_tracking_rms_as'] for r in selected]
            low = [v - r['modes'][mode]['min_tracking_rms_as'] for v, r in zip(values, selected)]
            high = [r['modes'][mode]['max_tracking_rms_as'] - v for v, r in zip(values, selected)]
            axes[1, j].bar(np.arange(len(labels)) + (i - 1) * .25, values, width=.25,
                           label=mode, yerr=[low, high], capsize=2)
        axes[1, j].set(title=f"{window}: mean and seed range", ylabel="True tracking RMS [arcsec]")
        axes[1, j].set_xticks(range(len(labels)), labels, rotation=22, ha='right', fontsize=8)
        axes[1, j].legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)
