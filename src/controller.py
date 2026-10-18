import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

# Suppress lightgbm feature name warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure imports work from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.forecaster import engineer_features, split_by_site
from src.grid_client import BargheManClient
from src.thermal import ShelterThermalModel, ambient_temperature, T_WARN, T_CRIT
from src.carriers import CarrierBank
from src.conformal import ConformalConfig, rolling_conformal_margin
from tests.reference import twin_ref

BURST = 1.25   # sub-5-minute burst factor used in violation checks (Parameter Freeze)
RECTIFIER_EFF = getattr(twin_ref, "RECTIFIER_EFF", 0.96)
WAKE_TAX_KW = getattr(twin_ref, "WAKE_TAX_KW", 0.085)

DATA_PATH = "data/synthetic_telecom_data.csv"
MODEL_PATH = "models/lgbm_q95_forecaster.txt"

# Failure-handling constants (5-minute steps).
BACKHAUL_DEBOUNCE_STEPS = 3   # backhaul must be down 15 min before capacity carriers are shed (flap filter)
LEASE_TIMEOUT_STEPS = 3       # central lease lapses 15 min after the control link is lost (dead-man's switch)

POLICIES = ("shield", "none", "timer", "reactive")


@dataclass(frozen=True)
class Fault:
    """A failure-drill event on the *twin's* control path, in step indices of the simulated window
    (start inclusive, end exclusive).

    kind:
      "telemetry_loss" - PM telemetry stops arriving (fail-static after 15 min of age).
      "backhaul_down"  - transport to the site is down (debounced, then capacity carriers are shed).
      "lease_loss"     - the central control link/lease is lost (dead-man's switch after 15 min).
    """
    kind: str
    start: int
    end: int

    def __post_init__(self):
        if self.kind not in ("telemetry_loss", "backhaul_down", "lease_loss"):
            raise ValueError(f"unknown fault kind {self.kind!r}")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("fault needs 0 <= start < end")


def _fault_mask(faults: Optional[Sequence[Fault]], kind: str, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for f in faults or ():
        if f.kind == kind:
            m[f.start:min(f.end, n)] = True
    return m


def _tier_from_load(x: float, util: float = 0.6) -> int:
    n = max(1, min(3, int(np.ceil(x / util - 1e-9))))
    return twin_ref.tier_for_lte(n)


@lru_cache(maxsize=4)
def _load_features(data_path: str, mtime: float):
    df_full = pd.read_csv(data_path, parse_dates=["timestamp"])
    df_feat, feature_cols = engineer_features(df_full)
    return df_feat, feature_cols


@lru_cache(maxsize=2)
def _load_model(model_path: str, mtime: float):
    return lgb.Booster(model_file=model_path)


def simulate_site(
    site_df: pd.DataFrame,
    predicted_peaks: np.ndarray,
    *,
    site_id="SITE_1",
    wake_latency_min=5.0,
    initial_soh=1.0,
    grid_client=None,
    ambient_c=None,
    init_shelter_c=25.0,
    init_batt_c=25.0,
    return_history=False,
    policy="shield",
    bound_margin: Optional[np.ndarray] = None,
    faults: Optional[Sequence[Fault]] = None,
    backhaul_debounce_steps: int = BACKHAUL_DEBOUNCE_STEPS,
    lease_timeout_steps: int = LEASE_TIMEOUT_STEPS,
):
    """Closed-loop twin vs baseline on one site window.

    `policy` only changes behaviour while the grid is UP (energy-saving mode); during an outage every
    policy uses the receding-horizon planner + arbiter:
      shield   - forecast bound -> proposal -> Safety Arbiter (the product)
      none     - always FULL (no energy saving)
      timer    - static night timer (tier 2 from 00:00-06:00, FULL otherwise), no arbiter
      reactive - tier from the load measured *now* (util 0.6), no forecast, no arbiter
    `bound_margin` (optional) is added to the forecaster's q95 (conformal calibration).
    `faults` injects telemetry/backhaul/lease failures (shield policy only).
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    n_steps = len(site_df)
    predicted_peaks = np.asarray(predicted_peaks, dtype=float)
    if len(predicted_peaks) != n_steps:
        raise ValueError("predicted_peaks must align with site_df rows")
    margin_arr = np.zeros(n_steps) if bound_margin is None else np.asarray(bound_margin, dtype=float)
    if len(margin_arr) != n_steps:
        raise ValueError("bound_margin must align with site_df rows")

    # 3. Initialize Physical Twins & Thermal Models
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

    # Column views (avoids per-row pandas overhead in the hot loop)
    ts_arr = list(site_df["timestamp"])
    hour_arr = site_df["hour"].to_numpy(dtype=float)
    grid_arr = site_df["grid_status"].to_numpy()
    demand_arr = site_df["organic_load_cu"].to_numpy(dtype=float)
    sched_flag_arr = site_df["blackout_schedule_flag"].to_numpy()

    tele_loss = _fault_mask(faults, "telemetry_loss", n_steps)
    bh_fault = _fault_mask(faults, "backhaul_down", n_steps)
    lease_fault = _fault_mask(faults, "lease_loss", n_steps)

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
        "fail_static_steps": 0,
        "backhaul_shed_steps": 0,
        "lease_lock_steps": 0,
        "twin_transitions": 0,
        "twin_dropped_grid_cuh": 0.0,
    }

    current_tier_twin = 4
    steps_since_sleep = 0
    sleeps_left_today = 4
    site_dead_base = False
    site_dead_twin = False
    v_batt_base = 54.0
    v_batt_twin = 54.0

    # Carrier state tracking (wake latency & signalling tax) lives in the shared CarrierBank.
    bank = CarrierBank(wake_latency_min)

    # Failure-handling state
    tele_age_min = 0.0
    last_good_bound = float(predicted_peaks[0] + margin_arr[0]) if n_steps else 0.0
    bh_run = 0
    lease_run = 0

    p_base_ac = 0.0
    p_twin_ac = 0.0
    history_rows = []

    for i in range(n_steps):
        grid_up = bool(grid_arr[i] == 1)
        actual_demand = demand_arr[i]
        hour_i = hour_arr[i]

        # Calculate ambient temperature (user override or Iranian diurnal profile)
        t_amb = float(ambient_c) if ambient_c is not None else ambient_temperature(hour_i)

        if not grid_up:
            metrics["outage_steps"] += 1
            metrics["outage_demand_cuh"] += actual_demand * dt_s / 3600.0

        # Reset hardware fatigue cap at midnight
        if hour_i < 0.1 and i > 0 and hour_arr[i - 1] > 23.0:
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

        # --- control-path health (telemetry age, debounced backhaul, lease)
        if tele_loss[i]:
            tele_age_min += dt_s / 60.0
        else:
            tele_age_min = 0.0
        stale = tele_age_min > twin_ref.STALE_MIN
        bh_run = bh_run + 1 if bh_fault[i] else 0
        lease_run = lease_run + 1 if lease_fault[i] else 0
        backhaul_eff = bh_run >= backhaul_debounce_steps and bh_run > 0
        lease_ok = not (lease_run >= lease_timeout_steps and lease_run > 0)

        # 1. AI Forecast (pre-computed in batch by the caller); conformal margin only widens the bound
        if stale:
            predicted_peak = last_good_bound            # cannot refresh the forecast without telemetry
        else:
            predicted_peak = float(predicted_peaks[i] + margin_arr[i])
            last_good_bound = predicted_peak

        # 2. Propose Action
        if grid_up:
            # Energy Saver Mode
            if policy == "none":
                proposal = 4
            elif policy == "timer":
                proposal = 2 if hour_i < 6 else 4
            elif policy == "reactive":
                proposal = _tier_from_load(actual_demand)
            else:
                if predicted_peak <= 1.0:
                    proposal = 2
                elif predicted_peak <= 2.0:
                    proposal = 3
                else:
                    proposal = 4

                # Emergency hardware override (bypass AI if actual surge happens now)
                if actual_demand > twin_ref.TIER_N_LTE[proposal]:
                    proposal = 4

                # Fail-static (Arch 8.10): stale telemetry -> freeze the last verified safe state
                if stale:
                    metrics["fail_static_steps"] += 1
                    proposal = current_tier_twin
        else:
            # Outage Survival Mode (Receding-Horizon Planner)
            end_ts = sched_end_map.get(i) or sched_end_map.get(ts_arr[i])
            if sched_flag_arr[i] == 1 and end_ts is not None:
                # Read published Bargh-e Man schedule window without peeking future grid_status
                remaining_h = max(dt_s / 3600.0, (end_ts - ts_arr[i]).total_seconds() / 3600.0)
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
        if grid_up and policy != "shield":
            # Naive comparison policies act without the shield.
            final_tier, why = proposal, ["policy:no-shield"]
        else:
            arb_inp = twin_ref.ArbiterInput(
                grid_up=grid_up, current=current_tier_twin, proposal=proposal, bound=predicted_peak,
                measured=actual_demand, telemetry_age_min=tele_age_min, steps_since_sleep=steps_since_sleep,
                sleeps_left_today=sleeps_left_today, lease_valid=lease_ok, backhaul_down=backhaul_eff,
                v_batt=v_batt_twin, v_backstop=45.0, eff_soc=batt_twin.soc - 0.10, t_batt=batt_twin.T
            )
            final_tier, why = twin_ref.arbitrate(arb_inp)
        if final_tier != proposal:
            metrics["shield_interventions"] += 1
        if "conflict:UB-wins" in why:
            metrics["shield_ub_conflicts"] += 1
        if "UB:thermal-critical" in why:
            metrics["thermal_crit_interventions"] += 1
        if "UB:backhaul-down" in why:
            metrics["backhaul_shed_steps"] += 1
        if "UB:lease-expired-on-battery" in why or "LB:lease-expired->all-on" in why:
            metrics["lease_lock_steps"] += 1

        # Update hardware fatigue tracking
        if final_tier != current_tier_twin:
            steps_since_sleep = 0
            metrics["twin_transitions"] += 1
            if final_tier < current_tier_twin:
                sleeps_left_today -= 1
        else:
            steps_since_sleep += 1
        current_tier_twin = final_tier

        # Carrier state management (Wake-Up Latency & Signaling Tax)
        if not grid_up and (site_dead_twin or v_batt_twin < batt_twin.c["v_blvd"]):
            site_dead_twin = True
            view = bank.kill()
        else:
            view = bank.command(final_tier)
            metrics["twin_wake_tax_kwh"] += (view.n_tax * WAKE_TAX_KW * dt_s) / 3600.0
        n_act, n_waking, n_tax, active_tier_step = view.n_act, view.n_waking, view.n_tax, view.active_tier

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
        if grid_up:
            metrics["twin_dropped_grid_cuh"] += max(0.0, actual_demand - cap) * dt_s / 3600.0
        if not grid_up:
            served_t = min(actual_demand, cap) if active_tier_step > 0 else 0.0
            metrics["twin_served_outage_cuh"] += served_t * dt_s / 3600.0
            metrics["twin_dropped_load_cu"] += (actual_demand - served_t) * dt_s / 3600.0

        # Tick carrier timers forward by dt_s
        bank.tick(dt_s)

        if return_history:
            history_rows.append({
                "timestamp": ts_arr[i],
                "site_id": site_id,
                "grid_status": int(grid_arr[i]),
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
                "commanded_tier": final_tier,
                "pred_q95": float(predicted_peaks[i]),
                "conformal_margin": float(margin_arr[i]),
                "bound": predicted_peak,
                "fail_static": bool(grid_up and stale),
                "backhaul_shed": bool(backhaul_eff),
                "lease_lost": bool(not lease_ok),
            })

    metrics["baseline_cal_damage"] = batt_base.cal_damage
    metrics["twin_cal_damage"] = batt_twin.cal_damage

    if return_history:
        return metrics, pd.DataFrame(history_rows)
    return metrics


def print_report(metrics, *, site_id, days, initial_soh, wake_latency_min, faults=None, conformal_on=False):
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
    if faults or conformal_on:
        print(f"  Conformal calibration: {'ON' if conformal_on else 'off'}")
        print(f"  Failure drills: fail-static holds={metrics['fail_static_steps']}  "
              f"backhaul-shed steps={metrics['backhaul_shed_steps']}  lease-lock steps={metrics['lease_lock_steps']}")
    print("█"*60 + "\n")


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
    policy="shield",
    conformal=False,
    faults=None,
    verbose=True,
    data_path=DATA_PATH,
    model_path=MODEL_PATH,
):
    """Load the dataset + frozen forecaster, take the held-out test window of `site_id`, and simulate.

    conformal: False | True (default ConformalConfig) | ConformalConfig - rolling conformal calibration of the
               q95 bound (margin computed causally; the pool is warmed up with the site's pre-test history).
    """
    log = print if verbose else (lambda *a, **k: None)
    log(f"--- Starting MVP Simulation Loop for {site_id} (up to {days} days, OUT-OF-SAMPLE window) ---")

    # 1. Load Data & Engineer Features
    if not os.path.exists(data_path):
        log("Error: Synthetic data not found.")
        return None
    log("Pre-computing historical features for Forecaster...")
    df_feat, feature_cols = _load_features(data_path, os.path.getmtime(data_path))

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
    if not os.path.exists(model_path):
        log("Error: Forecaster model not found. Run src/forecaster.py first.")
        return None
    model = _load_model(model_path, os.path.getmtime(model_path))
    # Batch inference (one call instead of one per 5-min step)
    predicted_peaks = model.predict(site_df[feature_cols])
    log(f"Test window: {site_df['timestamp'].iloc[0]} -> {site_df['timestamp'].iloc[-1]} ({days:.1f} days)")

    bound_margin = None
    if conformal:
        cfg = conformal if isinstance(conformal, ConformalConfig) else ConformalConfig()
        # Calibrate on the site's FULL history so the residual pool is warm at the start of the test window.
        # Residual labels (the realised next-60-min peak) are observable 60 min later, so this stays causal.
        site_full = df_feat[df_feat["site_id"] == site_id].sort_values("timestamp").reset_index(drop=True)
        pred_full = model.predict(site_full[feature_cols])
        margin_full = rolling_conformal_margin(site_full["target_peak_60m"].values, pred_full,
                                               site_full["hour"].values, cfg)
        margin_by_ts = pd.Series(margin_full, index=site_full["timestamp"])
        bound_margin = margin_by_ts.reindex(site_df["timestamp"]).fillna(0.0).values

    initial_soh = float(soh) if soh is not None else site_df["battery_soh"].iloc[0]

    out = simulate_site(
        site_df, predicted_peaks, site_id=site_id, wake_latency_min=wake_latency_min, initial_soh=initial_soh,
        grid_client=grid_client, ambient_c=ambient_c, init_shelter_c=init_shelter_c, init_batt_c=init_batt_c,
        return_history=return_history, policy=policy, bound_margin=bound_margin, faults=faults,
    )
    metrics = out[0] if return_history else out

    if verbose:
        log("Running Event Loop (Ticking physics engine step-by-step)...")
        print_report(metrics, site_id=site_id, days=days, initial_soh=initial_soh,
                     wake_latency_min=wake_latency_min, faults=faults, conformal_on=bool(conformal))

    return out


if __name__ == "__main__":
    run_simulation(site_id="SITE_1", days=30)
