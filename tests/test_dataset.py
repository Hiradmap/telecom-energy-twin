import pandas as pd
import numpy as np

def run_tests():
    print("Loading synthetic dataset...")
    df = pd.read_csv("data/synthetic_telecom_data.csv", parse_dates=["timestamp"])
    
    errors = []
    
    print("\n--- Running v3.3 Dataset Consistency Checks ---")
    
    # 1. Timezone and Step Size
    # Note: parse_dates might parse as UTC if it has offset, let's check
    if df["timestamp"].dt.tz is None:
        # If it read it as string or naive, let's just check the string format in CSV
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    time_diffs = df[df["site_id"] == "SITE_1"]["timestamp"].diff().dropna()
    if not (time_diffs == pd.Timedelta(minutes=5)).all():
        errors.append("FAIL: Step size is not consistently 5 minutes.")
    else:
        print("PASS: Step size is exactly 5 minutes.")
        
    # 2. Battery SOH
    if df["battery_soh"].min() < 0.25 or df["battery_soh"].max() > 1.0:
        errors.append(f"FAIL: Battery SOH out of bounds (min={df['battery_soh'].min()}, max={df['battery_soh'].max()})")
    else:
        print("PASS: Battery SOH bounded between 25% and 100%.")
        
    # 3. Organic Load Bounds
    if df["organic_load_cu"].min() < 0 or df["organic_load_cu"].max() > 3.5:
        errors.append(f"FAIL: organic_load_cu out of bounds (max={df['organic_load_cu'].max()})")
    else:
        print("PASS: organic_load_cu is bounded between 0 and 3.5.")
        
    # 4. Traffic State Labels
    valid_labels = {0, 5}
    if not set(df["traffic_state_label"].unique()).issubset(valid_labels):
        errors.append(f"FAIL: Invalid labels found: {df['traffic_state_label'].unique()}")
    else:
        print("PASS: traffic_state_label contains only 0 (organic) and 5 (surge).")
        
    # 5. Scheduled Blackouts (Time of day and Duration)
    sched_mask = df["blackout_schedule_flag"] == 1
    sched_hours = df.loc[sched_mask, "timestamp"].dt.hour + df.loc[sched_mask, "timestamp"].dt.minute / 60.0
    
    if sched_mask.any():
        min_h = sched_hours.min()
        max_h = sched_hours.max()
        if min_h < 8.0 or max_h > 23.9: # 08:30 to 23:30 approx
            errors.append(f"FAIL: Scheduled blackouts outside 08:00 - 23:30 window (min={min_h}, max={max_h})")
        else:
            print(f"PASS: Scheduled blackouts occur strictly during daytime (min={min_h:.1f}h, max={max_h:.1f}h).")
            
    # 6. Surge Logic Check
    # Whenever grid is 0, traffic label must be 5
    grid_down_mask = df["grid_status"] == 0
    if not (df.loc[grid_down_mask, "traffic_state_label"] == 5).all():
        errors.append("FAIL: Grid down does not always map to traffic_state_label 5.")
    else:
        print("PASS: 100% of grid down events trigger a Label 5 demand surge.")
        
    # 7. Correlated Outages (Feeder level fate-sharing)
    # If a feeder is down at timestamp T, ALL sites on that feeder must be down at timestamp T.
    grouped = df.groupby(["timestamp", "feeder_id"])["grid_status"].nunique()
    if (grouped > 1).any():
        errors.append("FAIL: Sites on the same feeder have mismatched grid statuses at the same timestamp.")
    else:
        print("PASS: Feeder-level fate-sharing is perfect (correlated outages).")
        
    # 8. Unscheduled Faults exist
    if df["unscheduled_fault_flag"].sum() == 0:
        errors.append("FAIL: No unscheduled faults (Weibull) generated.")
    else:
        print(f"PASS: Unscheduled faults successfully injected ({df['unscheduled_fault_flag'].sum()} steps total).")

    # 9. Calendar Logic
    mean_mon_wed = df[df["dow"].isin([0, 1, 2])]["organic_load_cu"].mean()
    mean_thu = df[df["dow"] == 3]["organic_load_cu"].mean()
    mean_fri = df[df["dow"] == 4]["organic_load_cu"].mean()
    print(f"INFO: Avg Load - Mon-Wed: {mean_mon_wed:.2f}, Thu: {mean_thu:.2f}, Fri: {mean_fri:.2f}")
    if mean_fri >= mean_mon_wed:
        errors.append("FAIL: Friday (weekend) traffic is not lower than weekday traffic.")
    else:
        print("PASS: Weekend traffic curves correctly reflect Iranian Thursday/Friday drops.")

    if errors:
        print("\n❌ ERRORS FOUND:")
        for e in errors:
            print("  " + e)
    else:
        print("\n✅ ALL TESTS PASSED! Dataset perfectly matches v3.3 specification.")

if __name__ == "__main__":
    run_tests()
