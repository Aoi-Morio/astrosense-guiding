import numpy as np
import pytest

from astrosense_guiding.backlash import play_operator, count_reversals
from astrosense_guiding.parser import parse_text
from astrosense_guiding.reconstruct import reconstruct, stitch
from astrosense_guiding.predict import PECModel, ControllerConfig, replay


def test_parser_aliases_drop_dither_and_pulse_timing():
    log = parse_text('''PHD2 version 2.5
Calibration Begins at 2026-01-01 00:00:00
STAR LOST
Guiding Begins at 2026-01-01 00:01:00
Pixel scale = 2.0 arc-sec/px
xRate = 1.0, yRate = 1.0
Frame,Time,Mount,RADistanceRaw,DECDistanceRaw,RADuration,RADirection,DECDuration,DECDirection
1,0,Mount,0,0,1000,E,0,
2,2,DROP,,,500,E,0,
INFO: DITHER by 2.0, -1.0
3,4,Mount,3.5,0,0,,0,
''')
    assert log.n_calibrations == log.n_calibration_star_lost == 1
    session = log.longest()
    assert session.valid.tolist() == [True, False, True]
    assert session.events[0].frame_index == 1
    ol = reconstruct(session)
    np.testing.assert_allclose(ol.ra_cmd_ms, [0, 1500])
    np.testing.assert_allclose(ol.ra_drift_as, [0, 4])
    assert ol.segment.tolist() == [0, 1]


def test_calibration_only():
    log = parse_text('Calibration Begins at now\nSTAR LOST\n')
    assert log.longest() is None
    assert log.n_calibration_star_lost == 1


def test_stitch_preserves_slope_across_multiple_offsets():
    t = np.arange(30.)
    segment = np.repeat([0, 1, 2], 10)
    truth = 0.3 * t + 2
    y = truth + np.choose(segment, [0, 20, -7])
    np.testing.assert_allclose(stitch(t, y, segment), truth, atol=1e-12)


def test_play_reversal_requires_full_backlash_width():
    u = np.array([0, 100, 200, 150, 100, 50, 0])
    np.testing.assert_allclose(play_operator(u, 100), [-50, 50, 150, 150, 150, 100, 50])
    np.testing.assert_allclose(play_operator(u, 0), u)
    assert count_reversals(np.array([100, 0, 0.5, -100, 0, 100])) == 2


def test_exact_feedforward_cancels_noise_free_drift():
    t = np.arange(0, 300, 2.)
    model = PECModel(100, [4, 0], 0.001, '')
    drift = model.predict(t)
    cfg = ControllerConfig(min_move_as=0)
    baseline = replay(t, drift, 0.01, cfg)
    result = replay(t, drift, 0.01, cfg, model)
    assert baseline.rms > 0.1
    assert result.rms < 1e-12
