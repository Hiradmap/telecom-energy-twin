import numpy as np
import pandas as pd
import argparse
import os

def generate_telecom_data(days=90, seed=42):
    """
    Generates synthetic Iranian telecom data for the MVP.
    Includes organic traffic, scheduled blackouts, and demand surges.
    """
    rng = np.random.default_rng(seed)
    print(f"Generating {days} days of synthetic data...")
    
    # Base timeframe: 5-minute intervals
    periods = days * 288
    t = pd.date_range("2026-01-01", periods=periods, freq="5min", tz="Asia/Tehran")
    
    df = pd.DataFrame({"timestamp": t})
    df["hour"] = df.timestamp.dt.hour + df.timestamp.dt.minute / 60.0
    df["dow"] = df.timestamp.dt.dayofweek # Monday=0, Friday=4, Sunday=6
    
    # Iranian Weekend (Friday = dow 4, Thursday half-day = dow 3)
    is_friday = (df["dow"] == 4).astype(float)
    is_thursday = (df["dow"] == 3).astype(float)
    
    # Base daily profile (peaks around 21:00, 13:00, 09:30)
    # Scaled to max out around 3.0 (representing 3 carriers of capacity)
    base_traffic = 0.4 + 1.2 * np.exp(-((df["hour"] - 21.0) ** 2) / 8) \
                       + 0.8 * np.exp(-((df["hour"] - 13.0) ** 2) / 10) \
                       + 0.5 * np.exp(-((df["hour"] - 9.5) ** 2) / 6)
    
    # Weekend adjustments
    base_traffic = base_traffic * (1 - 0.15 * is_friday) + 0.4 * is_friday * np.exp(-((df["hour"] - 15) ** 2) / 12)
    
    # Noise and spikes
    noise = rng.normal(0, 0.15, periods) * (0.6 + base_traffic / 3)
    spikes = np.zeros(periods)
    
    # Random +10-25% spikes (2% probability per hour)
    spike_starts = np.where(rng.random(periods) < 0.02 / 12)[0]
    for idx in spike_starts:
        duration = rng.integers(3, 12) # 15 to 60 mins
        spikes[idx:idx + duration] += rng.uniform(0.3, 0.8)
        
    df["organic_load_cu"] = np.clip(base_traffic + noise + spikes, 0, 3.5)
    
    # ---------------------------------------------------------
    # Blackout Generation (Grid Status)
    # ---------------------------------------------------------
    df["grid_status"] = 1
    df["blackout_schedule_flag"] = 0
    df["surge_flag"] = 0
    
    # Inject rolling blackouts (e.g. 2 hours every ~2-3 days)
    # Iran district schedules: typically between 08:30 and 23:30
    current_idx = 0
    while current_idx < periods:
        # Jump ahead 1 to 4 days
        jump = rng.integers(1 * 288, 4 * 288)
        current_idx += jump
        
        if current_idx >= periods:
            break
            
        # Ensure it happens during daytime (08:30 - 23:30)
        hour = df.iloc[current_idx]["timestamp"].hour
        if hour < 8 or hour > 21:
            current_idx += (10 - hour) * 12 # Shift to ~10 AM
            if current_idx >= periods: break
            
        # Duration: 1.5 to 3 hours (18 to 36 steps)
        duration_steps = rng.integers(18, 36)
        end_idx = min(current_idx + duration_steps, periods)
        
        # Apply blackout
        df.loc[current_idx:end_idx, "grid_status"] = 0
        df.loc[current_idx:end_idx, "blackout_schedule_flag"] = 1 # Known in advance
        
        # Apply +15% surge during blackout due to Wi-Fi failure
        df.loc[current_idx:end_idx, "surge_flag"] = 1
        df.loc[current_idx:end_idx, "organic_load_cu"] *= 1.15
        
        current_idx = end_idx

    # Final cleanup
    df["organic_load_cu"] = np.clip(df["organic_load_cu"], 0, 3.5)
    
    # Feature for the forecaster: target label
    # 0 = normal organic, 5 = demand surge due to blackout
    df["traffic_state_label"] = np.where(df["surge_flag"] == 1, 5, 0)
    
    # Drop intermediate columns
    df = df.drop(columns=["surge_flag"])
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic telecom data.")
    parser.add_argument("--days", type=int, default=90, help="Number of days to generate")
    parser.add_argument("--out", type=str, default="data/synthetic_telecom_data.csv", help="Output file path")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = generate_telecom_data(days=args.days)
    df.to_csv(args.out, index=False)
    print(f"Data successfully written to {args.out}")
    print(df.head())
    print("\nSummary Statistics:")
    print(f"Total rows (5-min steps): {len(df)}")
    print(f"Grid downtime: {(1 - df['grid_status'].mean()) * 100:.2f}%")
    print(f"Max load: {df['organic_load_cu'].max():.2f} carrier units")
