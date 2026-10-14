import numpy as np
import pytest

from astrosense_guiding.online_eval import CausalController, simulate
from astrosense_guiding.predict import ControllerConfig


@pytest.mark.parametrize('mode', ['feedback', 'fixed', 'online'])
def test_future_samples_cannot_change_past_commands(mode):
    t = np.arange(700.)
    drift = 4 * np.sin(2 * np.pi * t / 100)
    noise = np.random.default_rng(3).normal(0, .4, t.size)
    kwargs = dict(mode=mode, period=100, cadence_s=1, warmup_s=200, n_harmonics=1)
    first = simulate(drift, noise, **kwargs)
    changed_drift, changed_noise = drift.copy(), noise.copy()
    changed_drift[350:] += 100
    changed_noise[350:] *= 100
    second = simulate(changed_drift, changed_noise, **kwargs)
    # Includes both warmup and active prediction frames. The pulse at frame
    # 349 may use the scheduled time of frame 350, but never its measurement.
    np.testing.assert_array_equal(first.pulse_ms[:350], second.pulse_ms[:350])
    np.testing.assert_array_equal(first.predicted_delta[:350], second.predicted_delta[:350])


@pytest.mark.parametrize('mode', ['feedback', 'fixed', 'online'])
def test_cold_start_and_invalid_frames_use_no_feedforward(mode):
    t = np.arange(600.)
    valid = np.ones(600, bool)
    valid[300:320] = False
    trace = simulate(4 * np.sin(2 * np.pi * t / 100), np.zeros(600),
                     mode=mode, period=100, cadence_s=1, warmup_s=200,
                     n_harmonics=1, valid=valid)
    assert not trace.ready[:200].any()
    assert not trace.predicted_delta[:200].any()
    assert not trace.pulse_ms[~valid].any()
    assert np.isfinite(trace.tracking_error).all()
    assert np.isnan(trace.measured_error[~valid]).all()


def test_reconstructs_own_pulses_and_predicts_noise_free_drift():
    t = np.arange(1000.)
    drift = 4 * np.sin(2 * np.pi * t / 100) + .003 * t
    kwargs = dict(period=100, cadence_s=1, warmup_s=300, n_harmonics=1)
    feedback = simulate(drift, np.zeros(t.size), mode='feedback', **kwargs)
    fixed = simulate(drift, np.zeros(t.size), mode='fixed', **kwargs)
    online = simulate(drift, np.zeros(t.size), mode='online', **kwargs)
    assert np.std(fixed.tracking_error[500:]) < 1e-10
    assert np.std(online.tracking_error[500:]) < .001
    assert np.std(feedback.tracking_error[500:]) > .1
    np.testing.assert_allclose(online.tracking_error[1:],
                               drift[1:] + np.cumsum(online.pulse_ms[:-1]) * .007)


def test_limits_signed_pulses_to_config_and_cadence():
    cfg = ControllerConfig(max_pulse_ms=100)
    controller = CausalController('feedback', 100, 2, 200, .007, cfg=cfg)
    assert controller.step(0, 1e6) == -100
    assert controller.saturated
    assert controller.step(2, -1e6) == 100
    controller = CausalController('feedback', 100, .1, 200, .007)
    assert controller.step(0, 1e6) == -100


def test_fixed_fit_uses_only_prefix_and_remains_frozen():
    controller = CausalController('fixed', 100, 1, 300, .007, n_harmonics=1)
    for t in range(301):
        measured = 4 * np.sin(2 * np.pi * t / 100) + controller.correction_as
        controller.step(t, measured)
    assert controller.ready
    before = controller.fixed_model.to_dict()
    for t in range(301, 500):
        controller.step(t, 1000 + controller.correction_as)
    assert controller.fixed_model.to_dict() == before


def test_all_lost_prefix_falls_back_to_feedback():
    controller = CausalController('fixed', 100, 1, 200, .007, n_harmonics=1)
    for t in range(201):
        controller.step(t, np.nan, valid=False)
    assert np.isfinite(controller.step(201, 1.0))
    assert not controller.ready


@pytest.mark.parametrize('drift,noise', [([], []), ([0, 1], [0]), ([0, np.nan], [0, 0])])
def test_rejects_invalid_simulation_inputs(drift, noise):
    with pytest.raises(ValueError):
        simulate(drift, noise, mode='online', period=100, warmup_s=200)


def test_online_adapts_to_amplitude_change_with_noisy_observations():
    t = np.arange(2000.)
    scale = 1 + np.clip((t - 300) / 200, 0, 1)
    drift = scale * 4 * np.sin(2 * np.pi * t / 100)
    noise = np.random.default_rng(22).normal(0, .2, t.size)
    kwargs = dict(period=100, cadence_s=1, warmup_s=300, n_harmonics=1)
    fixed = simulate(drift, noise, mode='fixed', **kwargs)
    online = simulate(drift, noise, mode='online', **kwargs)
    rms = lambda x: np.sqrt(np.mean(x[-500:] ** 2))
    assert rms(online.tracking_error) < .6 * rms(fixed.tracking_error)
    np.testing.assert_allclose(online.measured_error - online.tracking_error, noise, atol=1e-12)


def test_benchmark_contains_paired_runs_and_serializable_metrics():
    import json
    from astrosense_guiding.online_eval import BenchmarkConfig, benchmark_markdown, run_benchmark

    result, examples = run_benchmark([7], BenchmarkConfig(period_s=80, cadence_s=2))
    assert len(result['runs']) == 5 * 3 * 2
    assert len(result['summary']) == 5 * 2
    assert len(examples) == 2 * 3
    assert all(row['max_abs_pulse_ms'] <= 2000 for row in result['runs'])
    assert all(row['frames'] >= row['valid_frames'] for row in result['runs'])
    json.dumps(result, allow_nan=False)
    assert 'RLS' in benchmark_markdown(result)
