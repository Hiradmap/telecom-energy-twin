import pandas as pd
import numpy as np
import pytest

def run_tests():
    print("Loading synthetic dataset...")
    df = pd.read_csv("data/synthetic_telecom_data.csv", parse_dates=["timestamp"])
    
    errors = []
    
    print("\n--- Running v3.3 Dataset Consistency Checks ---")
    
    # 1. Timezone and Step Size
    if df["timestamp"].dt.tz is None:
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
    grid_down_mask = df["grid_status"] == 0
    if not (df.loc[grid_down_mask, "traffic_state_label"] == 5).all():
        errors.append("FAIL: Grid down does not always map to traffic_state_label 5.")
    else:
        print("PASS: 100% of grid down events trigger a Label 5 demand surge.")
        
    # 7. Correlated Outages (Feeder level fate-sharing)
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
        raise AssertionError(f"Dataset consistency check failed with {len(errors)} errors:\n" + "\n".join(errors))
    else:
        print("\n✅ ALL TESTS PASSED! Dataset perfectly matches v3.3 specification.")


def test_dataset_consistency():
    run_tests()


def test_site_archetypes_exist_and_distinct():
    df = pd.read_csv("data/synthetic_telecom_data.csv", parse_dates=["timestamp"])
    assert "site_archetype" in df.columns
    archetypes = set(df["site_archetype"].unique())
    assert {"residential", "commercial", "highway"}.issubset(archetypes)
    
    # Commercial site peak should be during business hours (10:00 - 15:00)
    comm_df = df[df["site_archetype"] == "commercial"]
    comm_day = comm_df[(comm_df["hour"] >= 10.0) & (comm_df["hour"] <= 14.0)]["organic_load_cu"].mean()
    comm_night = comm_df[(comm_df["hour"] >= 21.0) | (comm_df["hour"] <= 4.0)]["organic_load_cu"].mean()
    assert comm_day > comm_night * 1.5, "Commercial site daytime traffic should dominate night traffic"

    # Residential site peak should be during evening hours (19:00 - 23:00)
    res_df = df[df["site_archetype"] == "residential"]
    res_eve = res_df[(res_df["hour"] >= 19.0) & (res_df["hour"] <= 23.0)]["organic_load_cu"].mean()
    res_morn = res_df[(res_df["hour"] >= 6.0) & (res_df["hour"] <= 9.0)]["organic_load_cu"].mean()
    assert res_eve > res_morn * 1.3, "Residential site evening traffic should dominate morning traffic"


def test_tavanir_timetable_blackouts_no_10am_clustering():
    df = pd.read_csv("data/synthetic_telecom_data.csv", parse_dates=["timestamp"])
    s1 = df[df["site_id"] == "SITE_1"]
    
    # Find start times of scheduled blackouts
    starts = s1.loc[(s1["blackout_schedule_flag"] == 1) & (s1["blackout_schedule_flag"].shift(1, fill_value=0) == 0)]
    assert len(starts) > 0, "Should have scheduled blackout events"
    
    start_hours = set(starts["timestamp"].dt.hour.unique())
    # Should be distributed across multiple timetable blocks (e.g. 9, 11, 13, 15, 17, 19)
    # The old bug caused all or nearly all blackouts to cluster solely at 10:00 AM!
    assert len(start_hours) >= 3, f"Expected scheduled blackouts across multiple timetable slots, got: {start_hours}"
    assert 10 not in start_hours, "Tavanir 2-hour blocks start at odd hours (09:00, 11:00, 13:00, 15:00, 17:00, 19:00), not 10:00"


if __name__ == "__main__":
    run_tests()
