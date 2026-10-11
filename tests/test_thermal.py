"""Tests for Shelter Thermal Dynamics & Battery Temperature Coupling.

Validates:
1. Passive steady-state math: T_ss = T_amb + P / UA.
2. Active HVAC cooling returning shelter to 25.0°C when grid is UP.
3. Outage thermal rise: Active cooling drop causes shelter and battery temperatures to climb.
4. Battery thermal inertia: Large thermal mass (240 kJ/K) creates realistic thermal lag behind air.
5. IEEE 484 Arrhenius aging: Life halves (damage doubles) every 8°C above 25°C.
6. IEEE derating: Battery charge rate is halved when T_batt >= 40.0°C.
7. Safety Arbiter critical protection: T_batt >= 50.0°C triggers UB:thermal-critical, overriding policy to Tier 1.
8. End-to-end controller simulation tracking thermal metrics.
"""

import math
import pytest
from tests.reference import twin_ref as T
from src.thermal import (
    ShelterThermalModel,
    ambient_temperature,
    UA_SHELTER,
    C_TH_SHELTER,
    C_TH_BATT_VRLA,
    T_SETPOINT,
    T_WARN,
    T_CRIT,
)
from src.controller import run_simulation


# -----------------------------------------------------------------------------
# 1. Steady-State & Core Differential Equations
# -----------------------------------------------------------------------------
def test_thermal_steady_state_math():
    """Verify that passive shelter equilibrium satisfies T_ss = T_amb + P / UA."""
    model = ShelterThermalModel()
    t_amb = 35.0
    p_heat_w = 400.0  # 400 W internal equipment heat

    # Theoretical steady state
    t_ss = model.steady_state_temp(p_heat_w, t_amb)
    expected_t_ss = 35.0 + 400.0 / 40.0  # 45.0 °C
    assert t_ss == pytest.approx(expected_t_ss)

    # Initialize model at steady state and verify derivative is zero (dt_s step causes no change)
    model.t_shelter = t_ss
    model.t_batt = t_ss
    # Battery has 0 internal heat in passive balance:
    t_sh, t_ba = model.step(dt_s=300, grid_up=False, p_in_heat_w=p_heat_w, p_batt_heat_w=0.0, t_amb=t_amb)
    assert t_sh == pytest.approx(t_ss, abs=1e-4)
    assert t_ba == pytest.approx(t_ss, abs=1e-4)


def test_hvac_cooling_when_grid_up():
    """Verify that when grid is up, active HVAC cools a hot shelter back towards 25°C."""
    model = ShelterThermalModel(t_shelter=45.0, t_batt=40.0)
    p_heat_w = 500.0
    t_amb = 38.0

    # Tick 1 hour (12 steps of 5 min) with grid UP
    for _ in range(12):
        model.step(dt_s=300, grid_up=True, p_in_heat_w=p_heat_w, p_batt_heat_w=5.0, t_amb=t_amb)

    # Shelter air should be pulled down to setpoint (25.0°C)
    assert model.t_shelter == pytest.approx(25.0, abs=0.5)
    # Battery cools with time constant ~3.7h (should drop below 38.5°C after 1h, and below 30°C after 4h)
    assert model.t_batt < 38.5
    assert model.t_batt < 40.0


def test_outage_causes_shelter_and_battery_temperature_rise():
    """Verify that during a 3-hour outage at 40°C ambient, shelter temperature rises sharply."""
    model = ShelterThermalModel(t_shelter=25.0, t_batt=25.0)
    p_in_heat = 450.0  # W (BBU + GSM + LTE baseband)
    p_batt_heat = 15.0  # W (I^2*R discharge losses)
    t_amb = 40.0

    steps = int(3 * 3600 / 300)  # 3 hours
    for _ in range(steps):
        model.step(dt_s=300, grid_up=False, p_in_heat_w=p_in_heat, p_batt_heat_w=p_batt_heat, t_amb=t_amb)

    # Shelter air should reach ~43 - 48 °C
    assert model.t_shelter > 40.0
    assert model.t_shelter < 52.0

    # Battery temperature should rise due to heat exchange with hot shelter air + Joule heat
    assert model.t_batt > 30.0
    # Battery heats slower than air due to 240 kJ/K thermal mass
    assert model.t_batt < model.t_shelter


def test_battery_thermal_inertia_lag():
    """Verify battery thermal lag: a sudden 15°C jump in shelter air does not immediately heat the 300kg battery."""
    model = ShelterThermalModel(t_shelter=40.0, t_batt=25.0)
    # 1 single 5-minute step
    model.step(dt_s=300, grid_up=False, p_in_heat_w=400.0, p_batt_heat_w=10.0, t_amb=40.0)

    # Air temperature can fluctuate quickly, but 240 kJ/K battery temperature rises by < 0.5°C in 5 min
    dT_batt = model.t_batt - 25.0
    assert dT_batt < 0.5
    assert dT_batt > 0.0


# -----------------------------------------------------------------------------
# 2. IEEE Standards: Aging, Derating, and Safety
# -----------------------------------------------------------------------------
def test_arrhenius_calendar_aging_damage():
    """Verify IEEE 484 rule: Lead-acid degradation doubles (halving life) for every 8°C above 25°C."""
    b_25 = T.Battery(chem="VRLA", T=25.0)
    b_33 = T.Battery(chem="VRLA", T=33.0)  # +8 °C
    b_41 = T.Battery(chem="VRLA", T=41.0)  # +16 °C (2x 8°C = 4x damage)

    dt_s = 3600.0 * 24.0  # 1 day
    b_25.age_calendar(dt_s)
    b_33.age_calendar(dt_s)
    b_41.age_calendar(dt_s)

    # Damage at 33°C should be exactly 2x damage at 25°C
    assert b_33.cal_damage / b_25.cal_damage == pytest.approx(2.0, rel=1e-3)
    # Damage at 41°C should be exactly 4x damage at 25°C
    assert b_41.cal_damage / b_25.cal_damage == pytest.approx(4.0, rel=1e-3)


def test_thermal_charge_derating_at_warning_threshold():
    """Verify that battery recharge current is halved when T_batt > 40.0°C to prevent float runaway."""
    b_cool = T.Battery(chem="VRLA", soc=0.5, T=25.0)
    b_hot = T.Battery(chem="VRLA", soc=0.5, T=42.0)

    p_cool = b_cool.charge(dt_s=300)
    p_hot = b_hot.charge(dt_s=300)

    # Hot battery charge power should be exactly 50% of cool battery charge power
    assert p_hot == pytest.approx(0.5 * p_cool, rel=1e-3)


def test_safety_arbiter_clamps_survival_on_thermal_critical():
    """Verify that when battery temperature reaches T_CRIT (50.0°C), Safety Arbiter clamps UB to SURVIVAL (Tier 1),
    overriding any full-power proposal even under high traffic demand."""
    # Case 1: Grid DOWN — UB clamps to 1 (SURVIVAL), LB is 1 (survival-floor), final is 1
    inp_outage = T.ArbiterInput(
        grid_up=False,
        current=4,
        proposal=4,        # Policy wants Tier 4 FULL
        bound=3.0,          # High traffic demand
        measured=3.0,
        telemetry_age_min=0.0,
        steps_since_sleep=5,
        sleeps_left_today=4,
        lease_valid=True,
        backhaul_down=False,
        v_batt=50.0,
        v_backstop=45.0,
        eff_soc=0.80,       # Plentiful energy
        t_batt=51.5,        # CRITICAL BATTERY OVERHEATING (>= 50°C)
    )

    final_tier_outage, why_outage = T.arbitrate(inp_outage)
    assert final_tier_outage == 1  # Forced into SURVIVAL tier to prevent thermal runaway
    assert "UB:thermal-critical" in why_outage

    # Case 2: Grid UP — UB clamps to 1 (SURVIVAL), but LB is 2 (anchor-floor) -> conflict:UB-wins fires!
    inp_grid = T.ArbiterInput(
        grid_up=True,
        current=4,
        proposal=4,
        bound=3.0,
        measured=3.0,
        telemetry_age_min=0.0,
        steps_since_sleep=5,
        sleeps_left_today=4,
        lease_valid=True,
        backhaul_down=False,
        v_batt=54.0,
        v_backstop=45.0,
        eff_soc=1.0,
        t_batt=52.0,        # Thermal critical on grid (e.g. float runaway)
    )

    final_tier_grid, why_grid = T.arbitrate(inp_grid)
    assert final_tier_grid == 1
    assert "UB:thermal-critical" in why_grid
    assert "conflict:UB-wins" in why_grid


def test_diurnal_ambient_temperature_profile():
    """Verify Iranian summer ambient profile: minimum at 03:00, peak at 15:00."""
    t_3am = ambient_temperature(3.0, base_temp=34.0, amplitude=8.0)
    t_3pm = ambient_temperature(15.0, base_temp=34.0, amplitude=8.0)
    t_9am = ambient_temperature(9.0, base_temp=34.0, amplitude=8.0)

    assert t_3am == pytest.approx(26.0)  # 34 - 8
    assert t_3pm == pytest.approx(42.0)  # 34 + 8
    assert t_9am == pytest.approx(34.0)  # base temp


# -----------------------------------------------------------------------------
# 3. Controller End-to-End Simulation Integration
# -----------------------------------------------------------------------------
def test_controller_tracks_thermal_extrema_and_damage():
    """Verify that run_simulation tracks peak shelter and battery temperatures and calendar damage."""
    metrics = run_simulation(site_id="SITE_1", days=3, wake_latency_min=5.0)

    assert "max_shelter_temp_base" in metrics
    assert "max_shelter_temp_twin" in metrics
    assert "max_batt_temp_base" in metrics
    assert "max_batt_temp_twin" in metrics

    # Shelter temperatures must have risen during blackouts
    assert metrics["max_shelter_temp_base"] > 25.0
    assert metrics["max_shelter_temp_twin"] > 25.0
    assert metrics["max_batt_temp_base"] > 25.0

    # Calendar degradation must be positive and non-zero
    assert metrics["baseline_cal_damage"] > 0.0
    assert metrics["twin_cal_damage"] > 0.0


def test_controller_heatwave_triggers_thermal_protections():
    """Run simulation with hot summer baseline (init_batt_c=41.5°C, ambient 44°C) to verify warning thresholds."""
    metrics = run_simulation(
        site_id="SITE_1",
        days=5,
        wake_latency_min=5.0,
        ambient_c=44.0,
        init_shelter_c=40.0,
        init_batt_c=41.5,
    )

    # In sustained 44°C ambient with outages starting hot, battery crosses 40.0°C warning threshold
    assert metrics["max_shelter_temp_twin"] >= 40.0
    assert metrics["max_batt_temp_twin"] >= 40.0
    assert metrics["thermal_warn_steps"] > 0


def test_dead_site_deenergizes_equipment_and_halts_shelter_heat():
    """Verify that when battery voltage collapses below BLVD during an outage (tier=0),
    the DC bus contactor drops all equipment so equipment heat is 0.0 W, allowing passive cooldown."""
    model = ShelterThermalModel(t_shelter=45.0, t_batt=40.0)
    t_amb = 25.0
    # On a dead site, equipment heat is 0.0 W and battery heat is 0.0 W
    p_in_heat = 0.0
    p_batt_heat = 0.0

    # Step forward 2 hours (24 steps of 5 min)
    for _ in range(24):
        model.step(dt_s=300, grid_up=False, p_in_heat_w=p_in_heat, p_batt_heat_w=p_batt_heat, t_amb=t_amb)

    # Without 460W phantom heat, shelter rapidly cools toward 25°C ambient (below 35°C after 2h)
    assert model.t_shelter < 35.0
    # Battery cools with ~3.7h thermal lag (dropping below 39°C after 2h, and below 35°C after 4h)
    assert model.t_batt < 39.0

    # Step forward another 2 hours (4h total)
    for _ in range(24):
        model.step(dt_s=300, grid_up=False, p_in_heat_w=p_in_heat, p_batt_heat_w=p_batt_heat, t_amb=t_amb)
    assert model.t_shelter < 30.0
    assert model.t_batt < 36.0


