import json

from astrosense_guiding.analysis import analyze_session, to_jsonable
from astrosense_guiding.parser import parse_text
from astrosense_guiding.synth import SynthConfig, generate


def test_synthetic_pipeline_recovers_known_truth():
    cfg = SynthConfig(dither_every_s=600)
    session = parse_text(generate(cfg)).longest()
    result = analyze_session(session)
    assert result.pe.ok and result.backlash.ok
    assert abs(result.pe.worm_period - cfg.worm_period_s) < 0.1
    assert abs(result.pe.pp_sky - cfg.truth()['pe_pp_sky']) < 0.5
    assert abs(result.backlash.backlash_ms - cfg.dec_backlash_ms) < 50
    assert result.pec_eval['holdout']
    assert result.pec_eval['pec_rms'] < result.pec_eval['baseline_rms']
    assert result.forecast
    json.dumps(to_jsonable(result), allow_nan=False)
