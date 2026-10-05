import numpy as np
import pandas as pd
import argparse
import os

def generate_site_data(site_id, feeder_id, days, rng):
    periods = days * 288
    t = pd.date_range("2026-01-01", periods=periods, freq="5min", tz="Asia/Tehran")
    df = pd.DataFrame({"timestamp": t, "site_id": site_id, "feeder_id": feeder_id})
    df["hour"] = df.timestamp.dt.hour + df.timestamp.dt.minute / 60.0
    df["dow"] = df.timestamp.dt.dayofweek
    
    is_friday = (df["dow"] == 4).astype(float)
    is_thursday = (df["dow"] == 3).astype(float)
    
    # Base daily profile
    base_traffic = 0.4 + 1.2 * np.exp(-((df["hour"] - 21.0) ** 2) / 8) \
                       + 0.8 * np.exp(-((df["hour"] - 13.0) ** 2) / 10) \
                       + 0.5 * np.exp(-((df["hour"] - 9.5) ** 2) / 6)
                       
    # Weekend adjustments (Thursday afternoon drops, Friday day-shift)
    base_traffic = base_traffic * (1 - 0.15 * is_friday - 0.10 * is_thursday) \
                   + 0.4 * is_friday * np.exp(-((df["hour"] - 15) ** 2) / 12) \
                   + 0.2 * is_thursday * np.exp(-((df["hour"] - 11) ** 2) / 8)
                   
    noise = rng.normal(0, 0.15, periods) * (0.6 + base_traffic / 3)
    spikes = np.zeros(periods)
    
    spike_starts = np.where(rng.random(periods) < 0.02 / 12)[0]
    for idx in spike_starts:
        duration = rng.integers(3, 12)
        spikes[idx:idx + duration] += rng.uniform(0.3, 0.8)
        
    df["organic_load_cu"] = np.clip(base_traffic + noise + spikes, 0, 3.5)
    
    # SOH drawn from a defensible distribution (25% to 100%)
    df["battery_soh"] = rng.uniform(0.25, 1.0)
    
    return df

def generate_telecom_data(days=90, num_sites=3, seed=42):
    rng = np.random.default_rng(seed)
    print(f"Generating {days} days of synthetic data for {num_sites} sites...")
    
    # Assign feeders (e.g., probability that neighbors share a feeder)
    feeders = [f"F_{rng.integers(1, 3)}" for _ in range(num_sites)]
    
    site_dfs = []
    for i in range(num_sites):
        site_dfs.append(generate_site_data(f"SITE_{i+1}", feeders[i], days, rng))
        
    df = pd.concat(site_dfs, ignore_index=True)
    
    # Generate Blackouts at the FEEDER level
    df["grid_status"] = 1
    df["blackout_schedule_flag"] = 0
    df["surge_flag"] = 0
    df["unscheduled_fault_flag"] = 0
    
    periods = days * 288
    unique_feeders = df["feeder_id"].unique()
    
    for feeder in unique_feeders:
        feeder_mask = df["feeder_id"] == feeder
        
        # 1. Scheduled Blackouts (Daytime)
        current_idx = 0
        while current_idx < periods:
            jump = rng.integers(1 * 288, 4 * 288)
            current_idx += jump
            if current_idx >= periods: break
                
            hour = (current_idx % 288) * 5 / 60.0
            if hour < 8 or hour > 21:
                current_idx += int((10 - hour) * 12)
                if current_idx >= periods: break
                
            duration = rng.integers(18, 36) # 1.5 to 3 hours
            end_idx = min(current_idx + duration, periods)
            
            ts_start = df["timestamp"].unique()[current_idx]
            ts_end = df["timestamp"].unique()[end_idx - 1]
            
            mask = feeder_mask & (df["timestamp"] >= ts_start) & (df["timestamp"] <= ts_end)
            df.loc[mask, "grid_status"] = 0
            df.loc[mask, "blackout_schedule_flag"] = 1
            df.loc[mask, "surge_flag"] = 1
            df.loc[mask, "organic_load_cu"] *= 1.15
            
            current_idx = end_idx
            
        # 2. Unscheduled Faults (Weibull distribution)
        fault_idx = int(rng.weibull(1.5) * 288 * 30) 
        while fault_idx < periods:
            duration = rng.integers(6, 24) # 30 min to 2 hours
            end_idx = min(fault_idx + duration, periods)
            
            ts_start = df["timestamp"].unique()[fault_idx]
            ts_end = df["timestamp"].unique()[end_idx - 1]
            
            mask = feeder_mask & (df["timestamp"] >= ts_start) & (df["timestamp"] <= ts_end)
            df.loc[mask, "grid_status"] = 0
            df.loc[mask, "unscheduled_fault_flag"] = 1
            df.loc[mask, "surge_flag"] = 1
            df.loc[mask, "organic_load_cu"] *= 1.15
            
            fault_idx += int(rng.weibull(1.5) * 288 * 30)

    df["organic_load_cu"] = np.clip(df["organic_load_cu"], 0, 3.5)
    df["traffic_state_label"] = np.where(df["surge_flag"] == 1, 5, 0)
    df = df.drop(columns=["surge_flag"])
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic telecom data.")
    parser.add_argument("--days", type=int, default=90, help="Number of days to generate")
    parser.add_argument("--sites", type=int, default=3, help="Number of neighboring sites")
    parser.add_argument("--out", type=str, default="data/synthetic_telecom_data.csv", help="Output file path")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = generate_telecom_data(days=args.days, num_sites=args.sites)
    df.to_csv(args.out, index=False)
    print(f"Data successfully written to {args.out}")
    print(df.head())
    print("\nSummary Statistics:")
    print(f"Total rows (5-min steps): {len(df)}")
    print(f"Grid downtime overall: {(1 - df['grid_status'].mean()) * 100:.2f}%")
    print(f"Unscheduled fault rate: {df['unscheduled_fault_flag'].mean() * 100:.2f}%")
    print(f"Max load: {df['organic_load_cu'].max():.2f} carrier units")
