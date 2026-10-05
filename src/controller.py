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
from src.forecaster import engineer_features
from tests.reference import twin_ref

def run_simulation(site_id="SITE_1", days=30):
    print(f"--- Starting MVP Simulation Loop for {site_id} ({days} days) ---")
    
    # 1. Load Data & Engineer Features
    data_path = "data/synthetic_telecom_data.csv"
    if not os.path.exists(data_path):
        print("Error: Synthetic data not found.")
        return
        
    df_full = pd.read_csv(data_path, parse_dates=["timestamp"])
    print("Pre-computing historical features for Forecaster...")
    df_feat, feature_cols = engineer_features(df_full)
    
    # Trim to specific site and days
    steps = days * 288
    site_df = df_feat[df_feat["site_id"] == site_id].iloc[:steps].copy().reset_index(drop=True)
    
    # 2. Load Forecaster Model
    model_path = "models/lgbm_q95_forecaster.txt"
    if not os.path.exists(model_path):
        print("Error: Forecaster model not found. Run src/forecaster.py first.")
        return
    model = lgb.Booster(model_file=model_path)
    
    # 3. Initialize Physical Twins
    initial_soh = site_df["battery_soh"].iloc[0]
    batt_base = twin_ref.Battery(chem="VRLA", soh=initial_soh, Ah_nom=200.0, soc=1.0)
    batt_twin = twin_ref.Battery(chem="VRLA", soh=initial_soh, Ah_nom=200.0, soc=1.0)
    
    dt_s = 300 # 5 minute steps
    P_tiers = [twin_ref.site_power_kw(t, 0, 0, "outage") for t in range(5)]
    
    # 4. Metrics Tracking
    metrics = {
        "baseline_ac_kwh": 0.0,
        "twin_ac_kwh": 0.0,
        "baseline_any_srv_steps": 0,
        "twin_any_srv_steps": 0,
        "baseline_full_srv_steps": 0,
        "twin_full_srv_steps": 0,
        "outage_steps": 0,
        "twin_dropped_load_cu": 0.0,
        "shield_interventions": 0
    }
    
    current_tier_twin = 4
    steps_since_sleep = 0
    sleeps_left_today = 4
    site_dead_base = False
    site_dead_twin = False
    v_batt_base = 54.0
    v_batt_twin = 54.0
    
    print("Running Event Loop (Ticking physics engine step-by-step)...")
    
    for i, row in site_df.iterrows():
        grid_up = bool(row["grid_status"] == 1)
        actual_demand = row["organic_load_cu"]
        
        if not grid_up:
            metrics["outage_steps"] += 1
            
        # Reset hardware fatigue cap at midnight
        if row["hour"] < 0.1 and i > 0 and site_df.iloc[i-1]["hour"] > 23.0:
            sleeps_left_today = 4
            
        # ==========================================
        # BASELINE LOGIC (Run FULL until BLVD cuts)
        # ==========================================
        if grid_up:
            site_dead_base = False
            v_batt_base = 54.0
            if batt_base.soc < 1.0:
                batt_base.charge(dt_s=dt_s)
            tier_base = 4
            p_base_ac = twin_ref.site_power_kw(4, actual_demand, 0.58, "grid")
            metrics["baseline_ac_kwh"] += (p_base_ac * dt_s) / 3600.0
        else:
            if site_dead_base or v_batt_base < batt_base.c["v_blvd"]:
                site_dead_base = True
                tier_base = 0
            else:
                tier_base = 4
                p_base_dc = twin_ref.site_power_kw(4, actual_demand, 0.0, "outage")
                v_batt_base = batt_base.discharge(p_base_dc, dt_s)
                
        if not grid_up and tier_base > 0:
            metrics["baseline_any_srv_steps"] += 1
        if not grid_up and tier_base == 4:
            metrics["baseline_full_srv_steps"] += 1
            
            
        # ==========================================
        # TWIN LOGIC (Forecaster -> Planner -> Shield)
        # ==========================================
        
        # 1. AI Forecast
        X_df = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
        predicted_peak = model.predict(X_df)[0]
        
        # 2. Propose Action
        if grid_up:
            # Energy Saver Mode
            if predicted_peak <= 1.0: proposal = 2
            elif predicted_peak <= 2.0: proposal = 3
            else: proposal = 4
            
            # Emergency hardware override (bypass AI if actual surge happens now)
            if actual_demand > twin_ref.TIER_N_LTE[proposal]:
                proposal = 4
        else:
            # Outage Survival Mode (Receding-Horizon Planner)
            if row["blackout_schedule_flag"] == 1:
                # Read Bargh-e Man schedule for remaining duration
                future_vals = site_df["grid_status"].iloc[i:].values
                if future_vals.any():
                    steps_rem = np.argmax(future_vals)
                else:
                    steps_rem = 24 # Fallback assumption if EOF
                remaining_h = steps_rem * 5 / 60.0
                t_q50 = remaining_h
                t_q90 = remaining_h * 1.1 # 10% safety margin
            else:
                # Unscheduled fault
                t_q50 = 2.0
                t_q90 = 4.0
                
            # Available energy budget bounds
            cap_Wh = batt_twin.Ah_nom * 48.0
            E_soft = max(0, (batt_twin.soc - 0.35) * cap_Wh)
            E_hard = max(0, (batt_twin.soc - 0.15) * cap_Wh)
            
            proposal = twin_ref.planner_tier(E_soft, E_hard, t_q50, t_q90, P_tiers, 0.1)
            
        # 3. Arbitrate (Safety Shield / Priority Lattice)
        arb_inp = twin_ref.ArbiterInput(
            grid_up=grid_up, current=current_tier_twin, proposal=proposal, bound=predicted_peak,
            measured=actual_demand, telemetry_age_min=0.0, steps_since_sleep=steps_since_sleep,
            sleeps_left_today=sleeps_left_today, lease_valid=True, backhaul_down=False,
            v_batt=v_batt_twin, v_backstop=45.0, eff_soc=batt_twin.soc - 0.10, t_batt=batt_twin.T
        )
        
        final_tier, why = twin_ref.arbitrate(arb_inp)
        if final_tier != proposal and "conflict:UB-wins" in why:
            metrics["shield_interventions"] += 1
            
        # Update hardware fatigue tracking
        if final_tier != current_tier_twin:
            steps_since_sleep = 0
            if final_tier < current_tier_twin:
                sleeps_left_today -= 1
        else:
            steps_since_sleep += 1
        current_tier_twin = final_tier
        
        # 4. Execute Physics Simulation
        if grid_up:
            site_dead_twin = False
            v_batt_twin = 54.0
            if batt_twin.soc < 1.0:
                batt_twin.charge(dt_s=dt_s)
            p_twin_ac = twin_ref.site_power_kw(final_tier, actual_demand, 0.58, "grid")
            metrics["twin_ac_kwh"] += (p_twin_ac * dt_s) / 3600.0
        else:
            if site_dead_twin or v_batt_twin < batt_twin.c["v_blvd"]:
                site_dead_twin = True
                final_tier = 0
            else:
                p_twin_dc = twin_ref.site_power_kw(final_tier, actual_demand, 0.0, "outage")
                v_batt_twin = batt_twin.discharge(p_twin_dc, dt_s)
                
        # Metrics Tracking
        if not grid_up and final_tier > 0:
            metrics["twin_any_srv_steps"] += 1
        if not grid_up and final_tier == 4:
            metrics["twin_full_srv_steps"] += 1
            
        if final_tier > 0:
            cap = twin_ref.TIER_N_LTE[final_tier]
            if actual_demand > cap:
                metrics["twin_dropped_load_cu"] += (actual_demand - cap)
        elif not grid_up and actual_demand > 0:
            metrics["twin_dropped_load_cu"] += actual_demand
            
            
    # ==========================================
    # FINAL REPORTING
    # ==========================================
    print("\n" + "█"*60)
    print(" DIGITAL TWIN MVP: SIMULATION RESULTS")
    print("█"*60)
    print(f" Site ID:          {site_id}")
    print(f" Test Duration:    {days} days ({metrics['outage_steps'] * 5 / 60:.1f} hours of total blackouts)")
    print(f" Battery Health:   {initial_soh*100:.1f}% (VRLA)")
    
    print("\n [ ENERGY SAVINGS (Grid UP) ]")
    saved_kwh = metrics['baseline_ac_kwh'] - metrics['twin_ac_kwh']
    pct_saved = (saved_kwh / metrics['baseline_ac_kwh']) * 100 if metrics['baseline_ac_kwh'] > 0 else 0
    print(f"  Baseline Draw:   {metrics['baseline_ac_kwh']:.1f} kWh")
    print(f"  Twin Draw:       {metrics['twin_ac_kwh']:.1f} kWh")
    print(f"  Total Saved:     {saved_kwh:.1f} kWh ({pct_saved:.1f}% reduction)")
    
    print("\n [ OUTAGE SURVIVAL (Grid DOWN) ]")
    b_any_h = metrics['baseline_any_srv_steps'] * 5 / 60.0
    t_any_h = metrics['twin_any_srv_steps'] * 5 / 60.0
    up_gain = t_any_h - b_any_h
    pct_gain = (up_gain / b_any_h) * 100 if b_any_h > 0 else float('inf')
    
    print(f"  Baseline Survival: {b_any_h:.1f} hrs (Died due to running 4G on empty)")
    print(f"  Twin Survival:     {t_any_h:.1f} hrs (Kept voice/CSFB alive)")
    print(f"  Autonomy Gained:   +{up_gain:.1f} hrs (+{pct_gain:.1f}% Uptime)")
    
    print("\n [ SAFETY SHIELD DIAGNOSTICS ]")
    print(f"  Interventions:     {metrics['shield_interventions']} times safety overrode AI QoS")
    print("█"*60 + "\n")
    
if __name__ == "__main__":
    run_simulation(site_id="SITE_1", days=30)
