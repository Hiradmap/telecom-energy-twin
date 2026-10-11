"""Property + regression tests. Run:  .venv/bin/python -m pytest tests -q"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from tests.reference import twin_ref as T
from tests.arbiter_invariants import violations


# ----------------------------------------------------------------------------- arbiter properties
arbiter_inputs = st.builds(
    T.ArbiterInput,
    grid_up=st.booleans(),
    current=st.integers(0, 4),
    proposal=st.integers(0, 4),
    bound=st.floats(0.0, 3.5),
    measured=st.floats(0.0, 3.5),
    telemetry_age_min=st.floats(0.0, 60.0),
    steps_since_sleep=st.integers(0, 6),
    sleeps_left_today=st.integers(0, 4),
    lease_valid=st.booleans(),
    backhaul_down=st.booleans(),
    v_batt=st.floats(40.0, 55.0),
    v_backstop=st.just(45.0),
    eff_soc=st.floats(0.0, 1.0),
    t_batt=st.floats(20.0, 60.0),
)


@settings(max_examples=2000, deadline=None)
@given(arbiter_inputs)
def test_arbiter_invariants_hold_for_all_states(x):
    fin, why = T.arbitrate(x)
    assert violations(x, fin) == [], (x, fin, why)


def test_regression_forecast_miss_cannot_sleep_into_measured_overload():
    """Found by stress test: bound=0.29 but measured=2.05 used to sleep 4 -> 2 (1 carrier for 2.05 CU)."""
    x = T.ArbiterInput(grid_up=True, current=4, proposal=2, bound=0.29, measured=2.05, telemetry_age_min=0,
                       steps_since_sleep=5, sleeps_left_today=3, lease_valid=True, backhaul_down=False,
                       v_batt=54, v_backstop=45, eff_soc=1.0, t_batt=25)
    fin, _ = T.arbitrate(x)
    assert fin == 4


def test_invariant_checker_can_actually_fail():
    """Guards against the old vacuous test: a deliberately broken decision must be flagged."""
    x = T.ArbiterInput(grid_up=True, current=4, proposal=2, bound=0.2, measured=2.5, telemetry_age_min=0,
                       steps_since_sleep=5, sleeps_left_today=3, lease_valid=True, backhaul_down=False,
                       v_batt=54, v_backstop=45, eff_soc=1.0, t_batt=25)
    assert "sleep-into-measured-overload" in violations(x, 2)
    assert "anchor-floor-violated" in violations(x, 1)
    assert "battery-critical-not-limited" in violations(
        T.ArbiterInput(False, 4, 4, 1, 1, 0, 9, 4, True, False, 44.0, 45.0, 0.9, 25.0), 4)


# ----------------------------------------------------------------------------- battery
@pytest.mark.parametrize("chem", ["VRLA", "LFP"])
@pytest.mark.parametrize("soh", [1.0, 0.6, 0.25])
def test_battery_cannot_deliver_more_than_it_holds(chem, soh):
    """Regression: SOC clamped at 0 but voltage stayed above BLVD -> 46.8 kWh from a 4.2 kWh bank."""
    b = T.Battery(chem, soh=soh, soc=1.0)
    P = T.site_power_kw(1, 0, 0, "outage")          # lightest load (SURVIVAL)
    delivered_kwh, t, dead = 0.0, 0, False
    while t < 120 * 3600:
        V = b.discharge(P, 300)
        if V < b.c["v_blvd"]:
            dead = True
            break
        delivered_kwh += P * 300 / 3600
        t += 300
    nameplate_kwh = b.C * 48.0 / 1000.0
    assert dead, "battery never tripped BLVD (immortal battery)"
    # Peukert + polarisation make usable energy <= nameplate; allow small slack for step granularity.
    assert delivered_kwh <= nameplate_kwh * 1.15, (delivered_kwh, nameplate_kwh)


def test_battery_soc_is_monotone_on_discharge():
    b = T.Battery("VRLA", soh=0.6, soc=1.0)
    last = b.soc
    for _ in range(100):
        b.discharge(1.5, 300)
        assert b.soc <= last + 1e-12
        last = b.soc


# ----------------------------------------------------------------------------- planner energy
@pytest.mark.parametrize("soh", [1.0, 0.6, 0.4, 0.25])
def test_planner_energy_uses_aged_capacity(soh):
    b = T.Battery("VRLA", soh=soh, soc=1.0)
    e = T.usable_energy_wh(b, 0.15)
    assert e == pytest.approx(0.85 * 200.0 * soh * 48.0)
    # A worn battery must offer proportionally less energy than a fresh one.
    assert e <= T.usable_energy_wh(T.Battery("VRLA", soh=1.0, soc=1.0), 0.15) * soh + 1e-6


def test_planner_never_picks_tier_it_cannot_afford():
    P = [T.site_power_kw(t, 0 if t < 2 else t - 1, 0, "outage") for t in range(5)]
    for soh in (1.0, 0.6, 0.4, 0.25):
        b = T.Battery("VRLA", soh=soh, soc=0.9)
        for q50, q90 in [(1.0, 1.1), (2.0, 2.2), (3.0, 3.3), (4.0, 4.4)]:
            tier = T.planner_tier(T.usable_energy_wh(b, 0.35), T.usable_energy_wh(b, 0.15), q50, q90, P, 0.1)
            if tier > 1:
                need = P[1] * 1000 * q90 * 1.1 + (P[tier] - P[1]) * 1000 * q50
                assert need <= T.usable_energy_wh(b, 0.15) + 1e-6


# ----------------------------------------------------------------------------- forecaster split
def test_split_has_no_time_overlap_and_covers_every_site():
    from src.forecaster import split_by_site
    ts = pd.date_range("2026-01-01", periods=1000, freq="5min", tz="Asia/Tehran")
    df = pd.concat([pd.DataFrame({"timestamp": ts, "site_id": s}) for s in ("A", "B", "C")], ignore_index=True)
    tr, te = split_by_site(df)
    assert tr.timestamp.max() < te.timestamp.min()
    assert set(tr.site_id) == set(te.site_id) == {"A", "B", "C"}
    # purge gap: no training row's 60-min target window reaches into the test period
    assert te.timestamp.min() - tr.timestamp.max() >= pd.Timedelta(minutes=60)
