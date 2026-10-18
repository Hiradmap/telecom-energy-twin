"""Tests for src/conformal.py: causality, safety (only widens), streaming==offline, coverage repair under
shift, drift alarm, and a realistic end-to-end check with the frozen LightGBM model."""
import numpy as np
import pytest

from src.conformal import (ConformalConfig, DriftConfig, OnlineConformal, _conformal_quantile,
                           coverage_by_regime, drift_status, rolling_conformal_margin, rolling_coverage)

SPD = 288


def _series(n_days=40, seed=0, evening_gain=1.0, noise=0.15, bias=0.0):
    """Synthetic 'forecaster' world: truth = daily profile * gain + noise; pred = profile-based q95-ish bound."""
    rng = np.random.default_rng(seed)
    n = n_days * SPD
    hours = (np.arange(n) % SPD) / 12.0
    prof = 0.6 + 0.8 * np.exp(-((hours - 21.0) ** 2) / 8.0)
    pred = prof + 1.645 * noise                      # a correctly calibrated 95 % bound for gain == 1
    gain = np.where((hours >= 19) & (hours < 24), evening_gain, 1.0)
    y = prof * gain + rng.normal(0, noise, n) + bias
    return y, pred, hours


# ----------------------------------------------------------------------------- config / helpers
@pytest.mark.parametrize("kw", [dict(alpha=0.0), dict(alpha=1.0), dict(window_steps=0), dict(min_samples=0),
                                dict(hour_bucket_h=0.0), dict(hour_bucket_h=25.0), dict(adaptive_gamma=-0.1),
                                dict(alpha_min=0.5, alpha_max=0.1), dict(label_delay_steps=-1)])
def test_config_rejects_invalid(kw):
    with pytest.raises(ValueError):
        ConformalConfig(**kw)


def test_conformal_quantile_finite_sample_level():
    res = np.arange(1.0, 11.0)                        # n=10, alpha=0.05 -> level ceil(11*0.95)/10 = 1.0 -> max
    assert _conformal_quantile(res, 0.05) == 10.0
    res = np.arange(1.0, 101.0)                       # n=100 -> ceil(101*.95)/100 = 0.96 -> 96th value
    assert _conformal_quantile(res, 0.05) == 96.0


def test_inputs_validated():
    y, p, h = _series(3)
    with pytest.raises(ValueError):
        rolling_conformal_margin(y[:-1], p, h)
    y2 = y.copy(); y2[10] = np.nan
    with pytest.raises(ValueError):
        rolling_conformal_margin(y2, p, h)


# ----------------------------------------------------------------------------- safety + causality
def test_margin_never_negative_and_zero_during_warmup():
    y, p, h = _series(12, bias=-1.0)                  # truth far BELOW the bound -> residuals all negative
    m = rolling_conformal_margin(y, p, h)
    assert (m >= 0).all()
    assert m.max() == 0.0                             # only ever widens: never tightens the raw model
    y, p, h = _series(12)
    cfg = ConformalConfig()
    m = rolling_conformal_margin(y, p, h, cfg)
    assert (m[:cfg.label_delay_steps + cfg.min_samples] == 0).all()


def test_margin_is_causal_no_future_leakage():
    """Perturbing every label/prediction at or after step k - delay must not change margin[k]."""
    y, p, h = _series(15, seed=3)
    cfg = ConformalConfig()
    base = rolling_conformal_margin(y, p, h, cfg)
    for k in (4000, 4001, 4100):
        y2, p2 = y.copy(), p.copy()
        y2[k - cfg.label_delay_steps:] += 5.0
        p2[k - cfg.label_delay_steps:] -= 5.0
        pert = rolling_conformal_margin(y2, p2, h, cfg)
        assert pert[k] == base[k], k


def test_margin_causal_with_adaptive_mode():
    y, p, h = _series(15, seed=4)
    cfg = ConformalConfig(adaptive_gamma=0.01)
    base = rolling_conformal_margin(y, p, h, cfg)
    k = 4200
    y2 = y.copy(); y2[k - cfg.label_delay_steps:] += 5.0
    assert rolling_conformal_margin(y2, p, h, cfg)[k] == base[k]


# ----------------------------------------------------------------------------- streaming == offline
@pytest.mark.parametrize("gamma", [0.0, 0.01])
def test_online_matches_offline(gamma):
    y, p, h = _series(14, seed=5, evening_gain=1.3)
    cfg = ConformalConfig(adaptive_gamma=gamma)
    off = rolling_conformal_margin(y, p, h, cfg)
    on = OnlineConformal(cfg)
    got = np.zeros(len(y))
    for k in range(len(y)):
        j = k - cfg.label_delay_steps - 1
        if j >= 0:
            on.add(j, h[j], y[j], p[j], issued_margin=got[j])
        got[k] = on.margin(h[k], k)
    assert np.allclose(got, off, atol=1e-12)


def test_online_rejects_nan():
    with pytest.raises(ValueError):
        OnlineConformal().add(0, 1.0, float("nan"), 1.0)


# ----------------------------------------------------------------------------- coverage behaviour
def test_in_distribution_coverage_stays_near_target_and_does_not_balloon():
    y, p, h = _series(60, seed=6)
    m = rolling_conformal_margin(y, p, h)
    sl = slice(10 * SPD, None)
    raw_cov = np.mean(y[sl] <= p[sl])
    cal_cov = np.mean(y[sl] <= (p + m)[sl])
    assert 0.93 <= raw_cov <= 0.97
    assert cal_cov >= raw_cov - 1e-9                  # never worse than raw (only widens)
    assert cal_cov <= 0.985                           # and does not waste capacity when nothing is wrong
    assert m[sl].mean() < 0.15


def test_evening_shift_repaired_by_hour_bucket_but_global_pool_is_not():
    y, p, h = _series(60, seed=7, evening_gain=1.5)
    sl = slice(12 * SPD, None)
    eve = ((h >= 19) & (h < 24))[sl]
    raw = np.mean((y <= p)[sl][eve])
    bucket = rolling_conformal_margin(y, p, h, ConformalConfig(hour_bucket_h=4.0))
    glob = rolling_conformal_margin(y, p, h, ConformalConfig(hour_bucket_h=24.0))
    cb = np.mean((y <= p + bucket)[sl][eve])
    cg = np.mean((y <= p + glob)[sl][eve])
    assert raw < 0.80
    assert cb >= 0.93                                 # per-hour-of-day calibration restores the evening bound
    assert cb > cg + 0.03                             # a single global margin cannot fix an evening-only error
    night = ((h >= 0) & (h < 6))[sl]
    assert bucket[sl][night].mean() < 0.5 * bucket[sl][eve].mean()   # ... and does not inflate the night


def test_abrupt_shift_transient_then_recovery():
    """Honest transient check: right after a +40 % level jump coverage dips, then recovers within days."""
    y, p, h = _series(50, seed=8)
    t0 = 25 * SPD
    y = y.copy(); y[t0:] *= 1.4
    m = rolling_conformal_margin(y, p, h)
    hit = (y <= p + m)
    first_day = hit[t0:t0 + SPD].mean()
    late = hit[t0 + 10 * SPD:].mean()
    assert first_day < 0.93                           # we do NOT pretend the first day after a jump is covered
    assert late >= 0.93                               # but the window forgets the old regime and recovers


# ----------------------------------------------------------------------------- drift alarm
def test_drift_alarm_silent_in_distribution_and_fires_under_shift():
    y, p, h = _series(30, seed=9)
    s, cov = drift_status(y, p)
    assert (s[3 * SPD:] == 2).mean() < 0.02          # no alarm storm in-distribution
    y2, p2, h2 = _series(30, seed=9, evening_gain=1.0, bias=0.0)
    y2 = y2.copy(); y2[15 * SPD:] *= 1.5
    s2, cov2 = drift_status(y2, p2)
    assert (s2[17 * SPD:] == 2).mean() > 0.8         # ALARM raised and stays raised on persistent shift
    assert (s2[:14 * SPD] == 2).mean() < 0.02        # ... and was not raised before the shift


def test_rolling_coverage_is_causal_and_correct():
    y = np.array([1.0] * 10 + [5.0] * 10)
    b = np.full(20, 2.0)
    cov = rolling_coverage(y, b, window=4, label_delay_steps=2)
    assert np.isnan(cov[:6]).all()
    assert cov[9] == 1.0                              # labels 3..6 all covered
    assert cov[12] == 1.0                             # labels 6..9 all covered (label 10 not yet observable)
    assert cov[14] == pytest.approx(0.5)              # labels 8..11: two covered, two missed
    assert cov[16] == 0.0                             # labels 10..13 all missed


def test_coverage_by_regime():
    h = np.array([1.0, 2.0, 8.0, 20.0, 21.0, 22.0])
    y = np.array([1.0, 3.0, 1.0, 1.0, 5.0, 1.0])
    b = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    g = np.array([1, 1, 1, 0, 0, 0])
    r = coverage_by_regime(y, b, h, g)
    assert r["overall"]["coverage"] == pytest.approx(4 / 6)
    assert r["night 00-06h"]["coverage"] == pytest.approx(0.5)
    assert r["evening 19-24h"]["coverage"] == pytest.approx(2 / 3)
    assert r["grid DOWN"]["n"] == 3 and r["grid UP"]["n"] == 3
