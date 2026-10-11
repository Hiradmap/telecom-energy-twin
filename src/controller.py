import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
import warnings

# Suppress lightgbm feature name warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure imports work from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.forecaster import engineer_features, split_by_site
from src.grid_client import BargheManClient
from src.thermal import ShelterThermalModel, ambient_temperature, T_WARN, T_CRIT
from tests.reference import twin_ref

BURST = 1.25   # sub-5-minute burst factor used in violation checks (Parameter Freeze)
RECTIFIER_EFF = getattr(twin_ref, "RECTIFIER_EFF", 0.96)
WAKE_TAX_KW = getattr(twin_ref, "WAKE_TAX_KW", 0.085)


def run_simulation(
    site_id="SITE_1",
    days=30,
    wake_latency_min=5.0,
    soh=None,
    grid_client=None,
    ambient_c=None,
    init_shelter_c=25.0,
    init_batt_c=25.0,
    outage_duration_h=None,
    return_history=False,
):
    print(f"--- Starting MVP Simulation Loop for {site_id} (up to {days} days, OUT-OF-SAMPLE window) ---")

    # 1. Load Data & Engineer Features
    data_path = "data/synthetic_telecom_data.csv"
    if not os.path.exists(data_path):
        print("Error: Synthetic data not found.")
        return None

    df_full = pd.read_csv(data_path, parse_dates=["timestamp"])
    print("Pre-computing historical features for Forecaster...")
    df_feat, feature_cols = engineer_features(df_full)

    # Evaluate ONLY on the held-out test period (the forecaster never trained on it)
    _, test_df = split_by_site(df_feat)
    steps = int(days * 288)
    site_df = test_df[test_df["site_id"] == site_id].iloc[:steps].copy().reset_index(drop=True)
    days = len(site_df) / 288.0

    # If custom outage_duration_h is requested, adjust scheduled blackout blocks in site_df
    if outage_duration_h is not None:
        target_steps = max(1, int(round(float(outage_duration_h) * 12)))
        is_sched = site_df["blackout_schedule_flag"].values
        starts = np.where((is_sched == 1) & (np.pad(is_sched[:-1], (1, 0), constant_values=0) == 0))[0]

        for s_idx in starts:
            orig_end = s_idx
            while orig_end < len(site_df) and site_df.at[orig_end, "blackout_schedule_flag"] == 1:
                orig_end += 1

            new_end = min(s_idx + target_steps, len(site_df))
            site_df.loc[s_idx:new_end - 1, "grid_status"] = 0
            site_df.loc[s_idx:new_end - 1, "blackout_schedule_flag"] = 1
            site_df.loc[s_idx:new_end - 1, "unscheduled_fault_flag"] = 0
            if new_end < orig_end:
                site_df.loc[new_end:orig_end - 1, "grid_status"] = 1
                site_df.loc[new_end:orig_end - 1, "blackout_schedule_flag"] = 0

        site_df["traffic_state_label"] = np.where(site_df["grid_status"] == 0, 5, 0)

    # 2. Load Forecaster Model
    model_path = "models/lgbm_q95_forecaster.txt"
    if not os.path.exists(model_path):
        print("Error: Forecaster model not found. Run src/forecaster.py first.")
        return None
    model = lgb.Booster(model_file=model_path)
    # Batch inference (one call instead of one per 5-min step)
    predicted_peaks = model.predict(site_df[feature_cols])
    print(f"Test window: {site_df['timestamp'].iloc[0]} -> {site_df['timestamp'].iloc[-1]} ({days:.1f} days)")

    # 3. Initialize Physical Twins & Thermal Models
    initial_soh = float(soh) if soh is not None else site_df["battery_soh"].iloc[0]
    batt_base = twin_ref.Battery(chem="VRLA", soh=initial_soh, Ah_nom=200.0, soc=1.0, T=float(init_batt_c))
    batt_twin = twin_ref.Battery(chem="VRLA", soh=initial_soh, Ah_nom=200.0, soc=1.0, T=float(init_batt_c))

    thermal_base = ShelterThermalModel(t_shelter=float(init_shelter_c), t_batt=float(init_batt_c))
    thermal_twin = ShelterThermalModel(t_shelter=float(init_shelter_c), t_batt=float(init_batt_c))

    dt_s = 300  # 5 minute steps
    P_tiers = [twin_ref.site_power_kw(t, 0, 0, "outage") for t in range(5)]

    # Published blackout schedule lookup via Bargh-e Man Adapter
    if grid_client is None:
        grid_client = BargheManClient(mode="mock")
    sched_end_map = grid_client.build_schedule_map(site_df, site_id=site_id, dt_s=dt_s)

    # 4. Metrics Tracking
    metrics = {
        "baseline_ac_kwh": 0.0,
        "twin_ac_kwh": 0.0,
        "baseline_grid_dc_kwh": 0.0,
        "twin_grid_dc_kwh": 0.0,
        "baseline_outage_dc_kwh": 0.0,
        "twin_outage_dc_kwh": 0.0,
        "baseline_dc_kwh": 0.0,
        "twin_dc_kwh": 0.0,
        "baseline_recharge_kwh": 0.0,
        "twin_recharge_kwh": 0.0,
        "twin_wake_tax_kwh": 0.0,
        "baseline_any_srv_steps": 0,
        "twin_any_srv_steps": 0,
        "baseline_full_srv_steps": 0,
        "twin_full_srv_steps": 0,
        "outage_steps": 0,
        "twin_dropped_load_cu": 0.0,
        "base_dropped_load_cu": 0.0,
        "twin_served_outage_cuh": 0.0,
        "base_served_outage_cuh": 0.0,
        "outage_demand_cuh": 0.0,
        "twin_viol_grid": 0,
        "base_viol_grid": 0,
        "shield_interventions": 0,
        "shield_ub_conflicts": 0,
        "max_shelter_temp_base": 25.0,
        "max_shelter_temp_twin": 25.0,
        "max_batt_temp_base": 25.0,
        "max_batt_temp_twin": 25.0,
        "thermal_warn_steps": 0,
        "thermal_crit_interventions": 0,
        "baseline_cal_damage": 0.0,
        "twin_cal_damage": 0.0,
    }

    current_tier_twin = 4
    steps_since_sleep = 0
    sleeps_left_today = 4
    site_dead_base = False
    site_dead_twin = False
    v_batt_base = 54.0
    v_batt_twin = 54.0

    # Carrier state tracking: 'active', 'waking', 'sleeping'
    carrier_states = ["active", "active", "active"]
    carrier_wake_rem = [0.0, 0.0, 0.0]
    carrier_tax_rem = [0.0, 0.0, 0.0]

    history_rows = []

    print("Running Event Loop (Ticking physics engine step-by-step)...")

    for i, row in site_df.iterrows():
        grid_up = bool(row["grid_status"] == 1)
        actual_demand = row["organic_load_cu"]

        # Calculate ambient temperature (user override or Iranian diurnal profile)
        t_amb = float(ambient_c) if ambient_c is not None else ambient_temperature(row["hour"])

        if not grid_up:
            metrics["outage_steps"] += 1
            metrics["outage_demand_cuh"] += actual_demand * dt_s / 3600.0

        # Reset hardware fatigue cap at midnight
        if row["hour"] < 0.1 and i > 0 and site_df.iloc[i-1]["hour"] > 23.0:
            sleeps_left_today = 4

        # ==========================================
        # BASELINE LOGIC (Run FULL until BLVD cuts)
        # ==========================================
        if grid_up:
            site_dead_base = False
            v_batt_base = 54.0
            p_charge_b = batt_base.charge(dt_s=dt_s) if batt_base.soc < 1.0 else 0.0
            tier_base = 4
            p_base_dc = twin_ref.site_power_kw(4, actual_demand, 0.58, "grid")
            p_base_ac = (p_base_dc + p_charge_b) / RECTIFIER_EFF
            metrics["baseline_ac_kwh"] += (p_base_ac * dt_s) / 3600.0
            metrics["baseline_recharge_kwh"] += (p_charge_b * dt_s) / 3600.0
            metrics["baseline_grid_dc_kwh"] += (p_base_dc * dt_s) / 3600.0
            metrics["baseline_dc_kwh"] += (p_base_dc * dt_s) / 3600.0
            if actual_demand * BURST > twin_ref.TIER_N_LTE[4]:
                metrics["base_viol_grid"] += 1
        else:
            if site_dead_base or v_batt_base < batt_base.c["v_blvd"]:
                site_dead_base = True
                tier_base = 0
                p_base_dc = 0.0
            else:
                tier_base = 4
                p_base_dc = twin_ref.site_power_kw(4, actual_demand, 0.0, "outage")
                v_batt_base = batt_base.discharge(p_base_dc, dt_s)
                metrics["baseline_outage_dc_kwh"] += (p_base_dc * dt_s) / 3600.0
                metrics["baseline_dc_kwh"] += (p_base_dc * dt_s) / 3600.0

        if not grid_up and tier_base > 0:
            metrics["baseline_any_srv_steps"] += 1
        if not grid_up and tier_base == 4:
            metrics["baseline_full_srv_steps"] += 1
        if not grid_up:
            served_b = min(actual_demand, twin_ref.TIER_N_LTE[tier_base]) if tier_base > 0 else 0.0
            metrics["base_served_outage_cuh"] += served_b * dt_s / 3600.0
            metrics["base_dropped_load_cu"] += (actual_demand - served_b) * dt_s / 3600.0

        # Baseline Thermal Coupling
        if grid_up:
            p_base_equip = 350.0 + 110.0 + 20.0 * twin_ref.TIER_N_LTE[tier_base]
            p_base_rec_loss = p_base_ac * (1.0 - RECTIFIER_EFF) * 1000.0
            p_in_heat_b = p_base_equip + p_base_rec_loss
            i_chg_b = p_charge_b * 1000.0 / 54.0
            p_batt_heat_b = (i_chg_b ** 2) * batt_base.R + (0.05 * i_chg_b * 54.0 if batt_base.soc > 0.85 else 0.0)
        else:
            if site_dead_base or tier_base == 0:
                p_base_equip = 0.0
                p_in_heat_b = 0.0
                p_batt_heat_b = 0.0
            else:
                p_base_equip = 350.0 + (60.0 if tier_base == 1 else 110.0) + 20.0 * twin_ref.TIER_N_LTE[tier_base]
                p_in_heat_b = p_base_equip
                i_dis_b = p_base_dc * 1000.0 / max(v_batt_base, 30.0)
                p_batt_heat_b = (i_dis_b ** 2) * batt_base.R

        thermal_base.step(dt_s, grid_up, p_in_heat_b, p_batt_heat_b, t_amb, c_batt=240_000.0)
        batt_base.T = thermal_base.t_batt
        batt_base.age_calendar(dt_s)

        # ==========================================
        # TWIN LOGIC (Forecaster -> Planner -> Shield)
        # ==========================================

        # 1. AI Forecast (pre-computed in batch above)
        predicted_peak = predicted_peaks[i]

        # 2. Propose Action
        if grid_up:
            # Energy Saver Mode
            if predicted_peak <= 1.0:
                proposal = 2
            elif predicted_peak <= 2.0:
                proposal = 3
            else:
                proposal = 4

            # Emergency hardware override (bypass AI if actual surge happens now)
            if actual_demand > twin_ref.TIER_N_LTE[proposal]:
                proposal = 4
        else:
            # Outage Survival Mode (Receding-Horizon Planner)
            end_ts = sched_end_map.get(i) or sched_end_map.get(row["timestamp"])
            if row["blackout_schedule_flag"] == 1 and end_ts is not None:
                # Read published Bargh-e Man schedule window without peeking future grid_status
                remaining_h = max(dt_s / 3600.0, (end_ts - row["timestamp"]).total_seconds() / 3600.0)
                t_q50 = remaining_h
                t_q90 = remaining_h * 1.1  # 10% safety margin
            else:
                # Unscheduled fault
                t_q50 = 2.0
                t_q90 = 4.0

            # Available energy budget bounds (aged capacity; see twin_ref.usable_energy_wh).
            E_soft = twin_ref.usable_energy_wh(batt_twin, 0.35)
            E_hard = twin_ref.usable_energy_wh(batt_twin, 0.15)

            proposal = twin_ref.planner_tier(E_soft, E_hard, t_q50, t_q90, P_tiers, 0.1)

        # 3. Arbitrate (Safety Shield / Priority Lattice with Thermal Coupling)
        arb_inp = twin_ref.ArbiterInput(
            grid_up=grid_up, current=current_tier_twin, proposal=proposal, bound=predicted_peak,
            measured=actual_demand, telemetry_age_min=0.0, steps_since_sleep=steps_since_sleep,
            sleeps_left_today=sleeps_left_today, lease_valid=True, backhaul_down=False,
            v_batt=v_batt_twin, v_backstop=45.0, eff_soc=batt_twin.soc - 0.10, t_batt=batt_twin.T
        )

        final_tier, why = twin_ref.arbitrate(arb_inp)
        if final_tier != proposal:
            metrics["shield_interventions"] += 1
        if "conflict:UB-wins" in why:
            metrics["shield_ub_conflicts"] += 1
        if "UB:thermal-critical" in why:
            metrics["thermal_crit_interventions"] += 1

        # Update hardware fatigue tracking
        if final_tier != current_tier_twin:
            steps_since_sleep = 0
            if final_tier < current_tier_twin:
                sleeps_left_today -= 1
        else:
            steps_since_sleep += 1
        current_tier_twin = final_tier

        # Carrier state management (Wake-Up Latency & Signaling Tax)
        if not grid_up and (site_dead_twin or v_batt_twin < batt_twin.c["v_blvd"]):
            site_dead_twin = True
            carrier_states = ["sleeping", "sleeping", "sleeping"]
            carrier_wake_rem = [0.0, 0.0, 0.0]
            carrier_tax_rem = [0.0, 0.0, 0.0]
            n_act = 0
            n_waking = 0
            n_tax = 0
            active_tier_step = 0
        else:
            target_n = twin_ref.TIER_N_LTE[final_tier]

            for c_idx in range(3):
                if c_idx < target_n:
                    if carrier_states[c_idx] == "sleeping":
                        if wake_latency_min <= 0:
                            carrier_states[c_idx] = "active"
                            carrier_wake_rem[c_idx] = 0.0
                        else:
                            carrier_states[c_idx] = "waking"
                            carrier_wake_rem[c_idx] = float(wake_latency_min)
                        carrier_tax_rem[c_idx] = 5.0  # 5 minutes of +10% signaling tax
                else:
                    # Sleep/power down is instantaneous
                    carrier_states[c_idx] = "sleeping"
                    carrier_wake_rem[c_idx] = 0.0
                    carrier_tax_rem[c_idx] = 0.0

            n_act = sum(1 for s in carrier_states if s == "active")
            n_waking = sum(1 for s in carrier_states if s == "waking")
            n_tax = sum(1 for t in carrier_tax_rem if t > 0.0)
            metrics["twin_wake_tax_kwh"] += (n_tax * WAKE_TAX_KW * dt_s) / 3600.0

            if n_act >= 1:
                active_tier_step = twin_ref.tier_for_lte(n_act)
            else:
                active_tier_step = 1 if final_tier >= 1 else 0

        # 4. Execute Twin Physics Simulation
        if grid_up:
            site_dead_twin = False
            v_batt_twin = 54.0
            p_charge_t = batt_twin.charge(dt_s=dt_s) if batt_twin.soc < 1.0 else 0.0
            p_twin_dc = twin_ref.site_power_kw(active_tier_step, actual_demand, 0.58, "grid", n_waking=n_waking, n_waking_tax=n_tax)
            p_twin_ac = (p_twin_dc + p_charge_t) / RECTIFIER_EFF
            metrics["twin_ac_kwh"] += (p_twin_ac * dt_s) / 3600.0
            metrics["twin_recharge_kwh"] += (p_charge_t * dt_s) / 3600.0
            metrics["twin_grid_dc_kwh"] += (p_twin_dc * dt_s) / 3600.0
            metrics["twin_dc_kwh"] += (p_twin_dc * dt_s) / 3600.0
        else:
            p_charge_t = 0.0
            if not site_dead_twin:
                p_twin_dc = twin_ref.site_power_kw(active_tier_step, actual_demand, 0.0, "outage", n_waking=n_waking, n_waking_tax=n_tax)
                v_batt_twin = batt_twin.discharge(p_twin_dc, dt_s)
                metrics["twin_outage_dc_kwh"] += (p_twin_dc * dt_s) / 3600.0
                metrics["twin_dc_kwh"] += (p_twin_dc * dt_s) / 3600.0
            else:
                p_twin_dc = 0.0

        # Twin Thermal Coupling
        if grid_up:
            p_twin_equip = 350.0 + 110.0 + 20.0 * n_act
            p_twin_rec_loss = p_twin_ac * (1.0 - RECTIFIER_EFF) * 1000.0
            p_in_heat_t = p_twin_equip + p_twin_rec_loss
            i_chg_t = p_charge_t * 1000.0 / 54.0
            p_batt_heat_t = (i_chg_t ** 2) * batt_twin.R + (0.05 * i_chg_t * 54.0 if batt_twin.soc > 0.85 else 0.0)
        else:
            if site_dead_twin or active_tier_step == 0:
                p_twin_equip = 0.0
                p_in_heat_t = 0.0
                p_batt_heat_t = 0.0
            else:
                p_twin_equip = 350.0 + (60.0 if active_tier_step == 1 else 110.0) + 20.0 * n_act
                p_in_heat_t = p_twin_equip
                i_dis_t = p_twin_dc * 1000.0 / max(v_batt_twin, 30.0)
                p_batt_heat_t = (i_dis_t ** 2) * batt_twin.R

        thermal_twin.step(dt_s, grid_up, p_in_heat_t, p_batt_heat_t, t_amb, c_batt=240_000.0)
        batt_twin.T = thermal_twin.t_batt
        batt_twin.age_calendar(dt_s)

        if batt_twin.T >= T_WARN:
            metrics["thermal_warn_steps"] += 1

        # Track Temperature Extrema
        metrics["max_shelter_temp_base"] = max(metrics["max_shelter_temp_base"], thermal_base.t_shelter)
        metrics["max_shelter_temp_twin"] = max(metrics["max_shelter_temp_twin"], thermal_twin.t_shelter)
        metrics["max_batt_temp_base"] = max(metrics["max_batt_temp_base"], thermal_base.t_batt)
        metrics["max_batt_temp_twin"] = max(metrics["max_batt_temp_twin"], thermal_twin.t_batt)

        # Service Metrics Tracking
        if not grid_up and active_tier_step > 0:
            metrics["twin_any_srv_steps"] += 1
        if not grid_up and active_tier_step == 4:
            metrics["twin_full_srv_steps"] += 1

        cap = n_act
        if grid_up and actual_demand * BURST > cap:
            metrics["twin_viol_grid"] += 1
        if not grid_up:
            served_t = min(actual_demand, cap) if active_tier_step > 0 else 0.0
            metrics["twin_served_outage_cuh"] += served_t * dt_s / 3600.0
            metrics["twin_dropped_load_cu"] += (actual_demand - served_t) * dt_s / 3600.0

        # Tick carrier timers forward by dt_s
        for c_idx in range(3):
            if carrier_tax_rem[c_idx] > 0:
                carrier_tax_rem[c_idx] = max(0.0, carrier_tax_rem[c_idx] - (dt_s / 60.0))
            if carrier_states[c_idx] == "waking":
                carrier_wake_rem[c_idx] = max(0.0, carrier_wake_rem[c_idx] - (dt_s / 60.0))
                if carrier_wake_rem[c_idx] == 0.0:
                    carrier_states[c_idx] = "active"

        if return_history:
            history_rows.append({
                "timestamp": row["timestamp"],
                "site_id": site_id,
                "grid_status": int(row["grid_status"]),
                "actual_demand": actual_demand,
                "base_tier": tier_base,
                "twin_tier": active_tier_step,
                "base_soc": batt_base.soc,
                "twin_soc": batt_twin.soc,
                "base_v": v_batt_base,
                "twin_v": v_batt_twin,
                "t_shelter_base": thermal_base.t_shelter,
                "t_shelter_twin": thermal_twin.t_shelter,
                "t_batt_base": thermal_base.t_batt,
                "t_batt_twin": thermal_twin.t_batt,
                "p_base_ac": p_base_ac if grid_up else 0.0,
                "p_twin_ac": p_twin_ac if grid_up else 0.0,
                "p_base_dc": p_base_dc,
                "p_twin_dc": p_twin_dc,
                "shield_intervention": bool(final_tier != proposal),
                "shield_why": ";".join(why) if (final_tier != proposal or "conflict:UB-wins" in why) else "",
            })

    metrics["baseline_cal_damage"] = batt_base.cal_damage
    metrics["twin_cal_damage"] = batt_twin.cal_damage

    # ==========================================
    # FINAL REPORTING
    # ==========================================
    print("\n" + "█"*60)
    print(" DIGITAL TWIN MVP: SIMULATION RESULTS")
    print("█"*60)
    print(f" Site ID:          {site_id}")
    print(f" Test Duration:    {days:.1f} days ({metrics['outage_steps'] * 5 / 60:.1f} hours of total blackouts)")
    print(f" Battery Health:   {initial_soh*100:.1f}% (VRLA)")
    print(f" Carrier Wake Latency: {wake_latency_min:.1f} min (warm sync baseline)")
    print(f" Rectifier Efficiency: {RECTIFIER_EFF:.2f}")

    print("\n [ ENERGY SAVINGS (Grid UP) ]")
    saved_kwh = metrics['baseline_ac_kwh'] - metrics['twin_ac_kwh']
    pct_saved = (saved_kwh / metrics['baseline_ac_kwh']) * 100 if metrics['baseline_ac_kwh'] > 0 else 0
    print(f"  Baseline AC Draw: {metrics['baseline_ac_kwh']:.1f} kWh (Grid DC: {metrics['baseline_grid_dc_kwh']:.1f} kWh, Recharge DC: {metrics['baseline_recharge_kwh']:.1f} kWh)")
    print(f"  Twin AC Draw:     {metrics['twin_ac_kwh']:.1f} kWh (Grid DC: {metrics['twin_grid_dc_kwh']:.1f} kWh, Recharge DC: {metrics['twin_recharge_kwh']:.1f} kWh, Wake Tax: {metrics['twin_wake_tax_kwh']:.2f} kWh)")
    print(f"  Total Saved:     {saved_kwh:.1f} kWh ({pct_saved:.1f}% reduction)")
    print(f"  Total Site DC Delivered: baseline={metrics['baseline_dc_kwh']:.1f} kWh  twin={metrics['twin_dc_kwh']:.1f} kWh")
    print(f"  QoS violations (demand x{BURST} > capacity): baseline={metrics['base_viol_grid']}  twin={metrics['twin_viol_grid']}")

    print("\n [ OUTAGE SURVIVAL (Grid DOWN) ]")
    b_any_h = metrics['baseline_any_srv_steps'] * 5 / 60.0
    t_any_h = metrics['twin_any_srv_steps'] * 5 / 60.0
    up_gain = t_any_h - b_any_h
    pct_gain = (up_gain / b_any_h) * 100 if b_any_h > 0 else float('inf')
    out_h = metrics['outage_steps'] * 5 / 60.0

    print(f"  Baseline Any-Service: {b_any_h:.1f} / {out_h:.1f} hrs")
    print(f"  Twin Any-Service:     {t_any_h:.1f} / {out_h:.1f} hrs (voice/CSFB layer)")
    print(f"  Autonomy Gained:     {up_gain:+.1f} hrs ({pct_gain:+.1f}%)")
    print(f"  Traffic served during outages (CU·h of {metrics['outage_demand_cuh']:.1f} demanded): "
          f"baseline={metrics['base_served_outage_cuh']:.1f}  twin={metrics['twin_served_outage_cuh']:.1f}")

    print("\n [ SHELTER THERMAL DYNAMICS & BATTERY TEMPERATURE ]")
    print(f"  Peak Shelter Air:    Baseline={metrics['max_shelter_temp_base']:.1f}°C  Twin={metrics['max_shelter_temp_twin']:.1f}°C")
    print(f"  Peak Battery Temp:   Baseline={metrics['max_batt_temp_base']:.1f}°C  Twin={metrics['max_batt_temp_twin']:.1f}°C")
    print(f"  Thermal Warning:     {metrics['thermal_warn_steps']} steps (T_batt >= {T_WARN}°C, IEEE derating)")
    print(f"  Thermal Shutdowns:   {metrics['thermal_crit_interventions']} steps (T_batt >= {T_CRIT}°C, UB=1 clamped)")
    print(f"  Calendar Degradation: Baseline={metrics['baseline_cal_damage']*1e4:.2f} bp  Twin={metrics['twin_cal_damage']*1e4:.2f} bp")

    print("\n [ SAFETY SHIELD DIAGNOSTICS ]")
    print(f"  Proposal overridden by shield: {metrics['shield_interventions']} steps "
          f"({metrics['shield_ub_conflicts']} were safety UB-over-LB conflicts)")
    print("█"*60 + "\n")

    if return_history:
        return metrics, pd.DataFrame(history_rows)
    return metrics


if __name__ == "__main__":
    run_simulation(site_id="SITE_1", days=30)
