import numpy as np
import pandas as pd
import argparse
import os

ARCHETYPES = ["residential", "commercial", "highway"]

# 2-hour Tavanir planned rolling blackout timetable blocks (09:00 - 21:00)
TAVANIR_BLOCKS = [
    (9.0, 11.0),
    (11.0, 13.0),
    (13.0, 15.0),
    (15.0, 17.0),
    (17.0, 19.0),
    (19.0, 21.0),
]


def generate_site_data(site_id, feeder_id, days, rng, archetype="residential"):
    periods = days * 288
    t = pd.date_range("2026-01-01", periods=periods, freq="5min", tz="Asia/Tehran")
    df = pd.DataFrame({
        "timestamp": t,
        "site_id": site_id,
        "feeder_id": feeder_id,
        "site_archetype": archetype,
    })
    df["hour"] = df.timestamp.dt.hour + df.timestamp.dt.minute / 60.0
    df["dow"] = df.timestamp.dt.dayofweek

    is_friday = (df["dow"] == 4).astype(float)
    is_thursday = (df["dow"] == 3).astype(float)
    hour = df["hour"].values

    # Distinct archetype daily load patterns
    if archetype == "commercial":
        # Business district: strong daytime working hours (09:00 - 17:00), quiet evenings,
        # sharp Thursday afternoon half-day and Friday full-weekend drops
        base_traffic = 0.20 + 1.25 * np.exp(-((hour - 10.5) ** 2) / 6.0) \
                            + 1.10 * np.exp(-((hour - 14.0) ** 2) / 8.0) \
                            + 0.40 * np.exp(-((hour - 16.5) ** 2) / 6.0)
        base_traffic = base_traffic * (1.0 - 0.55 * is_friday.values - 0.30 * is_thursday.values)
    elif archetype == "highway":
        # Transit / Highway corridor: morning rush (07:30 - 09:30), evening commute (16:30 - 19:30),
        # outbound getaway on Thursday afternoon, inbound return on Friday evening
        base_traffic = 0.25 + 1.10 * np.exp(-((hour - 8.0) ** 2) / 5.0) \
                            + 1.20 * np.exp(-((hour - 17.5) ** 2) / 6.0) \
                            + 0.50 * np.exp(-((hour - 13.0) ** 2) / 12.0)
        base_traffic = base_traffic * (1.0 - 0.20 * is_friday.values - 0.10 * is_thursday.values) \
                            + 0.40 * is_thursday.values * np.exp(-((hour - 16.0) ** 2) / 6.0) \
                            + 0.40 * is_friday.values * np.exp(-((hour - 20.0) ** 2) / 6.0)
    else:
        # Residential (default): quiet morning, active afternoon, high evening peak (19:00 - 23:30)
        base_traffic = 0.30 + 1.25 * np.exp(-((hour - 21.0) ** 2) / 8.0) \
                            + 0.70 * np.exp(-((hour - 13.5) ** 2) / 10.0) \
                            + 0.45 * np.exp(-((hour - 9.5) ** 2) / 7.0)
        base_traffic = base_traffic * (1.0 - 0.15 * is_friday.values - 0.10 * is_thursday.values) \
                            + 0.35 * is_friday.values * np.exp(-((hour - 16.0) ** 2) / 10.0) \
                            + 0.25 * is_thursday.values * np.exp(-((hour - 22.0) ** 2) / 8.0)

    noise = rng.normal(0, 0.12, periods) * (0.6 + base_traffic / 3.0)
    spikes = np.zeros(periods)

    spike_starts = np.where(rng.random(periods) < 0.02 / 12.0)[0]
    for idx in spike_starts:
        duration = rng.integers(3, 12)
        spikes[idx:idx + duration] += rng.uniform(0.3, 0.8)

    df["organic_load_cu"] = np.clip(base_traffic + noise + spikes, 0.05, 3.5)

    # Battery SOH drawn from a defensible distribution (25% to 100%)
    df["battery_soh"] = rng.uniform(0.25, 1.0)

    return df


def generate_telecom_data(days=90, num_sites=3, seed=42, archetypes=None):
    rng = np.random.default_rng(seed)
    print(f"Generating {days} days of synthetic data for {num_sites} sites...")

    if archetypes is None:
        archetypes = [ARCHETYPES[i % len(ARCHETYPES)] for i in range(num_sites)]

    # Assign feeders (e.g. probability that neighbors share a feeder)
    feeders = [f"F_{rng.integers(1, 3)}" for _ in range(num_sites)]

    site_dfs = []
    for i in range(num_sites):
        site_dfs.append(generate_site_data(f"SITE_{i+1}", feeders[i], days, rng, archetype=archetypes[i]))

    df = pd.concat(site_dfs, ignore_index=True)

    # Generate Blackouts at the FEEDER level
    df["grid_status"] = 1
    df["blackout_schedule_flag"] = 0
    df["unscheduled_fault_flag"] = 0

    periods = days * 288
    unique_feeders = df["feeder_id"].unique()
    unique_ts = pd.date_range("2026-01-01", periods=periods, freq="5min", tz="Asia/Tehran")

    for feeder in unique_feeders:
        feeder_mask = df["feeder_id"] == feeder

        # 1. Scheduled Blackouts: Realistic 2-hour Tavanir timetable blocks
        # Distributes 2-hour outages across morning, afternoon, and evening slots without clustering at 10 AM
        day = 0 if days <= 2 else int(rng.integers(0, 2))
        while day < days:
            block_idx = int(rng.integers(0, len(TAVANIR_BLOCKS)))
            start_hour, end_hour = TAVANIR_BLOCKS[block_idx]

            start_idx = day * 288 + int(start_hour * 12)
            duration_steps = int((end_hour - start_hour) * 12)  # exactly 24 steps = 2 hours
            end_idx = min(start_idx + duration_steps, periods)

            if start_idx < periods:
                ts_start = unique_ts[start_idx]
                ts_end = unique_ts[end_idx - 1]

                mask = feeder_mask & (df["timestamp"] >= ts_start) & (df["timestamp"] <= ts_end)
                df.loc[mask, "grid_status"] = 0
                df.loc[mask, "blackout_schedule_flag"] = 1

            day += int(rng.integers(1, 4))

        # 2. Unscheduled Faults (Weibull distribution)
        fault_offset = int(rng.integers(12, 48))
        fault_idx = (int(rng.weibull(1.2) * 288 * 3) + fault_offset) if days > 3 else int(rng.integers(36, 120))
        while fault_idx < periods:
            duration = int(rng.integers(6, 25))  # 30 min to ~2 hours
            end_idx = min(fault_idx + duration, periods)

            ts_start = unique_ts[fault_idx]
            ts_end = unique_ts[end_idx - 1]

            mask = feeder_mask & (df["timestamp"] >= ts_start) & (df["timestamp"] <= ts_end)
            # Only apply if not already part of a scheduled blackout
            not_sched = mask & (df["blackout_schedule_flag"] == 0)
            if not_sched.any():
                df.loc[not_sched, "grid_status"] = 0
                df.loc[not_sched, "unscheduled_fault_flag"] = 1

            fault_idx += int(rng.weibull(1.2) * 288 * 6) + int(rng.integers(24, 72))

    # Apply outage demand surge (+15% mobile load when fixed broadband fails)
    outage_mask = df["grid_status"] == 0
    df.loc[outage_mask, "organic_load_cu"] = np.clip(df.loc[outage_mask, "organic_load_cu"] * 1.15, 0.0, 3.5)

    # Traffic state label: 5 for outage surge, 0 for organic
    df["traffic_state_label"] = np.where(df["grid_status"] == 0, 5, 0)

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
    print(f"Scheduled blackout rate: {df['blackout_schedule_flag'].mean() * 100:.2f}%")
    print(f"Unscheduled fault rate: {df['unscheduled_fault_flag'].mean() * 100:.2f}%")
    print(f"Max load: {df['organic_load_cu'].max():.2f} carrier units")
