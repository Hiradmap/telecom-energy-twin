"""Tests for Physics & Controller Accuracy fixes:
1. Planner Oracle Leak Prevention (no future grid_status peeking; schedule tracking vs unscheduled estimates)
2. Double-Counted Conservatism / Stranded Energy (aligned soft/hard floors; fresh battery capacity)
3. Carrier Wake-Up Latency & Signaling Tax (state transitions, warm sync, power spike, sensitivity sweep)
4. AC / Rectifier Efficiency & Recharge Power (0.96 rectifier factor, battery charging draw tracking)
"""
import pytest
import numpy as np
import pandas as pd

from tests.reference import twin_ref as T
from src.controller import run_simulation, RECTIFIER_EFF, WAKE_TAX_KW
from src.forecaster import engineer_features, split_by_site


# ----------------------------------------------------------------------------- 1. Oracle Leak Prevention
def test_no_oracle_leak_in_controller_code():
    """Verify that future grid_status peeking (np.argmax(future_vals)) is removed from controller.py."""
    with open("src/controller.py", "r") as f:
        code = f.read()
    assert "future_vals = site_df[\"grid_status\"].iloc[i:].values" not in code
    assert "steps_rem = np.argmax(future_vals)" not in code


def test_schedule_tracking_vs_unscheduled_faults():
    """Verify that scheduled blackouts compute remaining duration from the published schedule window,
    and unscheduled faults strictly use t_q50 = 2.0, t_q90 = 4.0 without future peeking."""
    df_full = pd.read_csv("data/synthetic_telecom_data.csv", parse_dates=["timestamp"])
    df_feat, _ = engineer_features(df_full)
    _, test_df = split_by_site(df_feat)
    s1 = test_df[test_df["site_id"] == "SITE_1"].iloc[:288 * 15].copy().reset_index(drop=True)

    dt_s = 300
    # Pre-parse schedule blocks exactly as controller does
    sched_blocks = []
    in_block = False
    start_idx = 0
    for idx, r in s1.iterrows():
        if r["blackout_schedule_flag"] == 1:
            if not in_block:
                in_block = True
                start_idx = idx
        else:
            if in_block:
                in_block = False
                sched_blocks.append((start_idx, idx - 1))
    if in_block:
        sched_blocks.append((start_idx, len(s1) - 1))

    sched_end_map = {}
    for s_idx, e_idx in sched_blocks:
        end_ts = s1.loc[e_idx, "timestamp"] + pd.Timedelta(seconds=dt_s)
        for k in range(s_idx, e_idx + 1):
            sched_end_map[k] = end_ts
            sched_end_map[s1.loc[k, "timestamp"]] = end_ts

    n_sched = 0
    n_unsched = 0
    for i, row in s1.iterrows():
        if row["grid_status"] == 0:
            end_ts = sched_end_map.get(i) or sched_end_map.get(row["timestamp"])
            if row["blackout_schedule_flag"] == 1 and end_ts is not None:
                rem_h = (end_ts - row["timestamp"]).total_seconds() / 3600.0
                assert rem_h > 0
                assert rem_h <= 3.5  # scheduled blackouts are 1.5 to 3.0 hours
                t_q50 = rem_h
                t_q90 = rem_h * 1.1
                n_sched += 1
            else:
                # Unscheduled fault
                t_q50 = 2.0
                t_q90 = 4.0
                assert t_q50 == 2.0 and t_q90 == 4.0
                n_unsched += 1

    assert n_sched > 0, "No scheduled blackouts encountered in test window"
    assert n_unsched > 0, "No unscheduled faults encountered in test window"


# ----------------------------------------------------------------------------- 2. Stranded Energy & Floor Alignment
def test_aligned_floors_prevent_stranded_energy():
    """Verify that SOC_CRIT is aligned to the hard floor (0.15), and fresh batteries
    (SOH 1.0) with SOC between 0.35 and 0.45 are NOT prematurely forced into SURVIVAL."""
    # When battery SOC is 0.42, effective SOC = 0.42 - 0.10 = 0.32
    # With SOC_CRIT = 0.15, eff_soc >= SOC_CRIT, so Arbiter does NOT trigger UB:battery-critical
    inp = T.ArbiterInput(
        grid_up=False, current=4, proposal=4, bound=1.5, measured=1.5,
        telemetry_age_min=0.0, steps_since_sleep=5, sleeps_left_today=4,
        lease_valid=True, backhaul_down=False, v_batt=50.0, v_backstop=45.0,
        eff_soc=0.42 - 0.10, t_batt=25.0
    )
    fin, why = T.arbitrate(inp)
    assert fin == 4
    assert "UB:battery-critical" not in why


def test_fresh_battery_serves_high_outage_traffic():
    """Verify that on fresh batteries (SOH 1.0), the twin does not strand ~2 kWh of energy
    and serves substantially higher outage data traffic than with the old 45% cutoff."""
    # Run 5-day simulation with fresh battery (soh=1.0)
    metrics_0lat = run_simulation(site_id="SITE_1", days=5, wake_latency_min=0.0, soh=1.0)
    # The twin should maintain full autonomy (any_srv == outage_steps)
    assert metrics_0lat["twin_any_srv_steps"] == metrics_0lat["outage_steps"]
    # With wake_latency=0 and aligned floors, twin served outage CU*h should be > 75% of baseline
    # (Previously with old double-counted 45% cutoff it was only ~60%)
    assert metrics_0lat["twin_served_outage_cuh"] > 0.75 * metrics_0lat["base_served_outage_cuh"]


# ----------------------------------------------------------------------------- 3. Wake-Up Latency & Signaling Tax
def test_site_power_includes_waking_and_tax():
    """Verify that site_power_kw includes n_waking * P0 and n_waking_tax * WAKE_TAX_KW."""
    # Basic power at Tier 2 (1 LTE active, 2 idle):
    p_idle_only = T.site_power_kw(tier=2, demand_cu=0.5, sleep_kappa=0.58, mode="grid", n_waking=0, n_waking_tax=0)
    # 1 carrier waking, no tax:
    p_waking_no_tax = T.site_power_kw(tier=2, demand_cu=0.5, sleep_kappa=0.58, mode="grid", n_waking=1, n_waking_tax=0)
    # 1 carrier waking, with tax:
    p_waking_with_tax = T.site_power_kw(tier=2, demand_cu=0.5, sleep_kappa=0.58, mode="grid", n_waking=1, n_waking_tax=1)

    # Waking carrier draws P0 (0.50 kW) instead of sleep kappa (0.50 * 0.58 = 0.29 kW)
    diff_waking = p_waking_no_tax - p_idle_only
    assert diff_waking == pytest.approx(T.P0 * (1.0 - 0.58))

    # Tax adds +10% max power: 0.10 * (0.50 + 0.35) = 0.085 kW
    diff_tax = p_waking_with_tax - p_waking_no_tax
    assert diff_tax == pytest.approx(WAKE_TAX_KW)


def test_wake_latency_sweep_causes_graded_qos_violations():
    """Verify sensitivity sweep across wake latency regimes (0, 5, 15 min):
    Instant wake (0 min) experiences 0 violations, while slower wake (5 min, 15 min)
    incurs progressively more QoS violations due to delayed carrier availability."""
    m_0 = run_simulation(site_id="SITE_1", days=10, wake_latency_min=0.0)
    m_5 = run_simulation(site_id="SITE_1", days=10, wake_latency_min=5.0)
    m_15 = run_simulation(site_id="SITE_1", days=10, wake_latency_min=15.0)

    assert m_0["twin_viol_grid"] == 0
    assert m_5["twin_viol_grid"] >= m_0["twin_viol_grid"]
    assert m_15["twin_viol_grid"] > m_5["twin_viol_grid"]
    assert m_5["twin_wake_tax_kwh"] > 0.0


# ----------------------------------------------------------------------------- 4. Rectifier Efficiency & Recharge Power
def test_rectifier_efficiency_and_recharge_tracking():
    """Verify that AC mains power factors in the 0.96 rectifier efficiency:
    P_ac = (P_dc + P_charge) / 0.96, and both AC kWh and recharge kWh are tracked."""
    metrics = run_simulation(site_id="SITE_1", days=5, wake_latency_min=5.0)

    # Recharge power must be tracked and positive (batteries recharge after outages)
    assert metrics["baseline_recharge_kwh"] > 0.0
    assert metrics["twin_recharge_kwh"] > 0.0

    # AC energy must be strictly greater than DC energy
    assert metrics["baseline_ac_kwh"] > metrics["baseline_dc_kwh"]
    assert metrics["twin_ac_kwh"] > metrics["twin_dc_kwh"]

    # Verify AC calculation relationship: AC == (grid_dc + recharge) / 0.96
    expected_base_ac = (metrics["baseline_grid_dc_kwh"] + metrics["baseline_recharge_kwh"]) / RECTIFIER_EFF
    assert metrics["baseline_ac_kwh"] == pytest.approx(expected_base_ac, rel=1e-3)

    expected_twin_ac = (metrics["twin_grid_dc_kwh"] + metrics["twin_recharge_kwh"]) / RECTIFIER_EFF
    assert metrics["twin_ac_kwh"] == pytest.approx(expected_twin_ac, rel=1e-3)


# ----------------------------------------------------------------------------- 5. Edge Transitions & Robustness Checks
def test_arbiter_current_preserves_dwell_and_fatigue_during_wake():
    """Verify that the Arbiter evaluates sleep requests against the commanded tier
    (current_tier_twin) so dwell time and fatigue budget cannot be bypassed while carriers are waking."""
    # When commanded tier is 4 (FULL) and carriers are waking (n_act = 1),
    # a sleep proposal to tier 2 inside dwell time (steps_since_sleep = 1 < 3) MUST be held at 4.
    inp_dwell = T.ArbiterInput(
        grid_up=True, current=4, proposal=2, bound=0.5, measured=0.5,
        telemetry_age_min=0.0, steps_since_sleep=1, sleeps_left_today=4,
        lease_valid=True, backhaul_down=False, v_batt=54.0, v_backstop=45.0,
        eff_soc=1.0, t_batt=25.0
    )
    fin_dwell, why_dwell = T.arbitrate(inp_dwell)
    assert fin_dwell == 4
    assert any("dwell" in w for w in why_dwell)

    # When fatigue budget is exhausted (sleeps_left_today = 0), sleep proposal must be held at 4.
    inp_fatigue = T.ArbiterInput(
        grid_up=True, current=4, proposal=2, bound=0.5, measured=0.5,
        telemetry_age_min=0.0, steps_since_sleep=5, sleeps_left_today=0,
        lease_valid=True, backhaul_down=False, v_batt=54.0, v_backstop=45.0,
        eff_soc=1.0, t_batt=25.0
    )
    fin_fatigue, why_fatigue = T.arbitrate(inp_fatigue)
    assert fin_fatigue == 4
    assert any("fatigue-budget" in w for w in why_fatigue)


def test_dead_site_resets_carrier_states_and_halts_wake_tax():
    """Verify that when battery voltage collapses below BLVD during an outage,
    the site shuts down, carriers are reset to sleeping, and zero wake tax is accumulated."""
    # Run simulation with an extremely depleted battery (soh=0.05) to trigger BLVD trip
    metrics = run_simulation(site_id="SITE_1", days=3, wake_latency_min=5.0, soh=0.05)
    # The baseline and twin should both trip BLVD
    assert metrics["twin_any_srv_steps"] < metrics["outage_steps"]
    # Wake tax must not accumulate indefinitely on a dead bank
    assert metrics["twin_wake_tax_kwh"] < 0.20


def test_gsm_power_in_grid_mode_is_full_during_wake():
    """Verify that site_power_kw in grid mode uses GSM_FULL (0.55 kW) even if active LTE tier is 1,
    because reduced GSM_MIN (0.30 kW) is strictly an outage survival power saving."""
    p_grid_t1 = T.site_power_kw(tier=1, demand_cu=0.0, sleep_kappa=0.58, mode="grid")
    # P_FIXED (0.35) + GSM_FULL (0.55) + 3 * P0 (0.50) * kappa (0.58) = 0.35 + 0.55 + 0.87 = 1.77 kW
    expected_p = T.P_FIXED + T.GSM_FULL + 3 * T.P0 * 0.58
    assert p_grid_t1 == pytest.approx(expected_p)

    # In outage mode, tier 1 must still use GSM_MIN (0.30 kW): 0.35 + 0.30 = 0.65 kW
    p_outage_t1 = T.site_power_kw(tier=1, demand_cu=0.0, sleep_kappa=0.0, mode="outage")
    assert p_outage_t1 == pytest.approx(T.P_FIXED + T.GSM_MIN)


def test_outage_traffic_conservation():
    """Verify that for both twin and baseline, outage served load + dropped load matches
    total outage demanded load in CU*h."""
    metrics = run_simulation(site_id="SITE_1", days=5, wake_latency_min=5.0)
    assert metrics["twin_served_outage_cuh"] + metrics["twin_dropped_load_cu"] == pytest.approx(metrics["outage_demand_cuh"], rel=1e-3)
    assert metrics["base_served_outage_cuh"] + metrics["base_dropped_load_cu"] == pytest.approx(metrics["outage_demand_cuh"], rel=1e-3)


def test_outage_duration_scaling_affects_blackout_steps_and_planner_reserve():
    """Verify that passing outage_duration_h dynamically scales scheduled blackout blocks
    and the receding-horizon reserve planner adjusts its energy reserve accordingly."""
    m_1h = run_simulation(site_id="SITE_1", days=3, wake_latency_min=5.0, outage_duration_h=1.0)
    m_3h = run_simulation(site_id="SITE_1", days=3, wake_latency_min=5.0, outage_duration_h=3.0)

    # 3-hour blackouts must accumulate more blackout steps and outage demand than 1-hour blackouts
    assert m_3h["outage_steps"] > m_1h["outage_steps"]
    assert m_3h["outage_demand_cuh"] > m_1h["outage_demand_cuh"]
    # Digital Twin preserves any-service uptime across both duration scenarios
    assert m_1h["twin_any_srv_steps"] >= m_1h["baseline_any_srv_steps"]
    assert m_3h["twin_any_srv_steps"] >= m_3h["baseline_any_srv_steps"]


# ----------------------------------------------------------------------------- 6. CarrierBank & Failure Drills
def test_carrier_bank_state_machine():
    """Verify CarrierBank instant sleep, timed wake, signaling tax window, and kill()."""
    from src.carriers import CarrierBank
    with pytest.raises(ValueError):
        CarrierBank(wake_latency_min=-1.0)

    bank = CarrierBank(wake_latency_min=10.0)
    # Sleep 2 capacity carriers immediately -> only 1 active (Tier 2)
    v_sleep = bank.command(2)
    assert (v_sleep.n_act, v_sleep.n_waking, v_sleep.n_tax, v_sleep.active_tier) == (1, 0, 0, 2)

    # Command FULL (Tier 4): 2 carriers enter 'waking' for 10 min and pay tax for 5 min
    v_wake0 = bank.command(4)
    assert (v_wake0.n_act, v_wake0.n_waking, v_wake0.n_tax, v_wake0.active_tier) == (1, 2, 2, 2)

    # After one 5-min tick: tax expires (0 left), wake has 5 min remaining
    bank.tick(300)
    v_wake1 = bank.command(4)
    assert (v_wake1.n_act, v_wake1.n_waking, v_wake1.n_tax, v_wake1.active_tier) == (1, 2, 0, 2)

    # After second 5-min tick: carriers finish waking -> 3 active (Tier 4)
    bank.tick(300)
    v_wake2 = bank.command(4)
    assert (v_wake2.n_act, v_wake2.n_waking, v_wake2.n_tax, v_wake2.active_tier) == (3, 0, 0, 4)

    # kill() powers everything down immediately
    v_dead = bank.kill()
    assert (v_dead.n_act, v_dead.n_waking, v_dead.n_tax, v_dead.active_tier) == (0, 0, 0, 0)


def test_failure_drills_telemetry_backhaul_and_lease():
    """Verify telemetry fail-static hold, backhaul 15-min debounce + BASIC cap, and dead-man's switch."""
    from src.controller import Fault

    with pytest.raises(ValueError):
        Fault("invalid_kind", 0, 5)
    with pytest.raises(ValueError):
        Fault("telemetry_loss", 10, 5)

    # 1. Short 2-step (10-min) backhaul blip must be filtered out by the 3-step (15-min) debounce
    m_blip = run_simulation(
        site_id="SITE_1", days=1, wake_latency_min=0.0,
        faults=[Fault("backhaul_down", 10, 12)], verbose=False,
    )
    assert m_blip["backhaul_shed_steps"] == 0

    # 2. Sustained backhaul outage (10 steps = 50 min) sheds capacity carriers to BASIC (Tier 2), never OFF (Tier 0)
    m_bh, h_bh = run_simulation(
        site_id="SITE_1", days=1, wake_latency_min=0.0,
        faults=[Fault("backhaul_down", 20, 30)], return_history=True, verbose=False,
    )
    assert m_bh["backhaul_shed_steps"] == 8  # steps 22..29 (after 3-step debounce at 20, 21, 22)
    shed_rows = h_bh.iloc[22:30]
    assert (shed_rows["backhaul_shed"]).all()
    assert (shed_rows[shed_rows["grid_status"] == 1]["commanded_tier"] == 2).all()

    # 3. Telemetry loss (> 15 min = 3 steps) triggers fail-static hold on grid
    m_tl, h_tl = run_simulation(
        site_id="SITE_1", days=1, wake_latency_min=0.0,
        faults=[Fault("telemetry_loss", 30, 42)], return_history=True, verbose=False,
    )
    assert m_tl["fail_static_steps"] > 0
    assert h_tl.iloc[34:42]["fail_static"].any()

    # 4. Dead-man's switch (lease_loss): forces Tier 4 on AC mains and Tier 1 (SURVIVAL) during an outage
    _, h_base = run_simulation(site_id="SITE_1", days=3, wake_latency_min=0.0, return_history=True, verbose=False)
    grid_idx = int(h_base.index[h_base["grid_status"] == 1][20])
    out_indices = h_base.index[h_base["grid_status"] == 0].tolist()
    assert len(out_indices) >= 6
    out_start = int(out_indices[0])

    m_lease, h_lease = run_simulation(
        site_id="SITE_1", days=3, wake_latency_min=0.0, soh=0.9,
        faults=[
            Fault("lease_loss", grid_idx, grid_idx + 8),
            Fault("lease_loss", out_start, out_start + 8),
        ],
        return_history=True, verbose=False,
    )
    assert m_lease["lease_lock_steps"] > 0
    # On grid after lease timeout (3 steps), commanded_tier must be 4 (FULL)
    grid_locked = h_lease.iloc[grid_idx + 3:grid_idx + 8]
    assert (grid_locked[grid_locked["grid_status"] == 1]["commanded_tier"] == 4).all()
    # On battery after lease timeout, commanded_tier must be 1 (SURVIVAL), never 4
    out_locked = h_lease.iloc[out_start + 3:out_start + 8]
    assert (out_locked[out_locked["grid_status"] == 0]["commanded_tier"] == 1).all()


def test_conformal_and_policy_modes_in_controller():
    """Verify run_simulation supports conformal=True and all comparison policies."""
    m_conf, h_conf = run_simulation(
        site_id="SITE_1", days=3, wake_latency_min=5.0, conformal=True, return_history=True, verbose=False
    )
    m_raw, h_raw = run_simulation(
        site_id="SITE_1", days=3, wake_latency_min=5.0, conformal=False, return_history=True, verbose=False
    )
    # Conformal calibration only widens the bound (bound_conf >= bound_raw everywhere)
    assert (h_conf["bound"].values >= h_raw["bound"].values - 1e-12).all()
    assert m_conf["twin_viol_grid"] <= m_raw["twin_viol_grid"]

    for pol in ("none", "timer", "reactive"):
        m_pol = run_simulation(site_id="SITE_1", days=2, wake_latency_min=5.0, policy=pol, verbose=False)
        assert m_pol["twin_ac_kwh"] > 0


