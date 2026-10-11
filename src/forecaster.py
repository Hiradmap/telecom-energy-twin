import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_pinball_loss
import argparse
import os

def engineer_features(df):
    """
    Creates lag features, rolling windows, and time encodings.
    Protects against data leakage by ONLY using safe features.
    """
    df = df.sort_values(by=["site_id", "timestamp"]).copy()
    
    # Target: Next-60-minute peak load (max over the next 12 steps)
    # Group by site, shift backwards
    df["target_peak_60m"] = df.groupby("site_id")["organic_load_cu"].apply(
        lambda x: x.rolling(window=12, min_periods=1).max().shift(-12)
    ).reset_index(level=0, drop=True)
    
    # Feature Engineering
    features = []
    
    # Time encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["is_friday"] = (df["dow"] == 4).astype(int)
    df["is_thursday"] = (df["dow"] == 3).astype(int)
    features.extend(["hour_sin", "hour_cos", "dow", "is_friday", "is_thursday"])
    
    # Lags (1 step = 5 mins)
    lags = [1, 2, 3, 6, 12, 36, 144, 288]
    for lag in lags:
        col_name = f"lag_{lag}"
        df[col_name] = df.groupby("site_id")["organic_load_cu"].shift(lag)
        features.append(col_name)
        
    # Rolling stats (1h=12, 6h=72, 24h=288)
    windows = [12, 72, 288]
    for w in windows:
        # Shift by 1 to ensure we don't leak current step if we consider current step "now"
        # Actually organic_load_cu at step t is known, so we can include it in rolling
        roll = df.groupby("site_id")["organic_load_cu"].rolling(window=w, min_periods=1)
        
        col_mean = f"roll_mean_{w}"
        col_max = f"roll_max_{w}"
        
        df[col_mean] = roll.mean().reset_index(level=0, drop=True)
        df[col_max] = roll.max().reset_index(level=0, drop=True)
        
        features.extend([col_mean, col_max])
        
    # Same time last week (7 days = 2016 steps)
    df["lag_1_week"] = df.groupby("site_id")["organic_load_cu"].shift(2016)
    features.append("lag_1_week")
    
    # Drop NaNs caused by lagging/shifting
    df = df.dropna()
    
    return df, features

TRAIN_FRAC = 0.7
HORIZON_STEPS = 12   # next-60-min target; used as purge gap between train and test


def split_by_site(df_feat, train_frac=TRAIN_FRAC, gap=HORIZON_STEPS):
    """Chronological train/test split applied to EVERY site with the same calendar cut-off.

    The previous split cut a site-sorted concatenated frame, so train held whole sites (and the same
    dates as the 'test' site) and the controller later evaluated on data the model had been trained on.
    Here: all sites share one cut-off timestamp; `gap` steps before it are purged so no training target
    window overlaps the test period. Returns (train_df, test_df).
    """
    ts = df_feat["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    cut = ts.iloc[int(len(ts) * train_frac)]
    purge_start = cut - pd.Timedelta(minutes=5 * gap)
    train = df_feat[df_feat["timestamp"] < purge_start]
    test = df_feat[df_feat["timestamp"] >= cut]
    return train, test


def train_and_evaluate():
    print("Loading synthetic dataset...")
    df = pd.read_csv("data/synthetic_telecom_data.csv", parse_dates=["timestamp"])
    
    print("Engineering features...")
    df_feat, feature_cols = engineer_features(df)
    
    # Per-site chronological split (same cut-off date for all sites, purge gap at the boundary)
    train_df, test_df = split_by_site(df_feat)
    
    X_train = train_df[feature_cols]
    y_train = train_df["target_peak_60m"]
    
    X_test = test_df[feature_cols]
    y_test = test_df["target_peak_60m"]
    
    print(f"Training set: {len(train_df)} rows (< {train_df['timestamp'].max()}). "
          f"Testing set: {len(test_df)} rows (>= {test_df['timestamp'].min()}).")
    print("Training Quantile LightGBM (alpha=0.95)...")
    
    model = lgb.LGBMRegressor(
        objective='quantile',
        alpha=0.95,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("Evaluating...")
    preds = model.predict(X_test)
    
    # Fair baseline: profile-q95 per (weekday, hour-of-day), fit on the same training period
    key = ["dow", "hour"]
    prof = train_df.groupby(key)["target_peak_60m"].quantile(0.95).rename("profile_q95")
    base_pred = test_df.join(prof, on=key)["profile_q95"].fillna(train_df["target_peak_60m"].quantile(0.95)).values
    base_pinball = mean_pinball_loss(y_test, base_pred, alpha=0.95)
    base_cov = np.mean(y_test.values <= base_pred)
    print(f"Baseline profile-q95: pinball={base_pinball:.4f} coverage={base_cov:.4f}")
    
    # Metrics
    # Pinball Loss
    pinball = mean_pinball_loss(y_test, preds, alpha=0.95)
    
    # Empirical Coverage (Percentage of times the true peak was <= our prediction)
    coverage = np.mean(y_test <= preds)
    
    print("\n--- Forecaster Results ---")
    print(f"Target: Next-60-Minute Peak Load")
    print(f"Alpha Target: 0.95 (95% Safety Bound)")
    print(f"Pinball Loss: {pinball:.4f}")
    print(f"Empirical Coverage: {coverage:.4f} ({coverage*100:.2f}%)")
    
    if coverage < 0.95:
        print("\n⚠️ WARNING: Empirical coverage is below 95%.")
        print("This means the AI is slightly under-predicting the peaks, which could lead to QoS violations.")
        print("As per v3.3 spec, we will need to wrap this in an EnbPI/Block-Conformal calibrator in the final pipeline.")
    else:
        print("\n✅ SUCCESS: Empirical coverage meets or exceeds the 95% safety bound.")
        
    # Feature Importance
    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(importances.head(5).to_string(index=False))

    # Save the model
    os.makedirs("models", exist_ok=True)
    model.booster_.save_model("models/lgbm_q95_forecaster.txt")
    print("\nModel saved to models/lgbm_q95_forecaster.txt")

if __name__ == "__main__":
    train_and_evaluate()
