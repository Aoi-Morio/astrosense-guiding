import numpy as np
import pytest
from astrosense_guiding.online import OnlinePEC


def test_causal_fit_predicts_unseen_cycle_with_segment_offsets():
    model = OnlinePEC(100, 1, forgetting=1, prior_variance=1e6)
    truth = lambda t: 4 * np.sin(2 * np.pi * t / 100) + 2 * np.cos(2 * np.pi * t / 100) + .003 * t
    for t in np.arange(0, 500, 2.):
        segment = int(t >= 250)
        model.update(t, truth(t) + 50 * segment, segment=segment)
    future = np.arange(500, 600, 2.)
    np.testing.assert_allclose(np.diff(model.predict(future)), np.diff(truth(future)), atol=1e-5)
    snapshot = model.snapshot()
    np.testing.assert_allclose(snapshot.predict(future), model.predict(future))
    before = snapshot.coefs.copy()
    model.update(500, truth(500) + 51, segment=1)
    assert snapshot.coefs == before


def test_invalid_gap_and_dither_do_not_train_on_jumps():
    model = OnlinePEC(100, 1, max_gap_s=10)
    assert not model.update(0, 0)
    assert model.update(2, 1)
    before = model.coef.copy()
    assert not model.update(4, 1000, segment=1)
    assert not model.update(6, np.nan)
    assert not model.update(8, 3000, segment=1)
    assert not model.update(30, 4000, segment=1)
    np.testing.assert_array_equal(model.coef, before)
    with pytest.raises(ValueError):
        model.update(30, 5000, segment=1)


@pytest.mark.parametrize('kwargs', [dict(period=0), dict(period=np.nan),
    dict(period=100, forgetting=0), dict(period=100, forgetting=1.1),
    dict(period=100, n_harmonics=0), dict(period=100, max_gap_s=-1)])
def test_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        OnlinePEC(**kwargs)
