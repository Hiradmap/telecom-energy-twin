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

def train_and_evaluate():
    print("Loading synthetic dataset...")
    df = pd.read_csv("data/synthetic_telecom_data.csv", parse_dates=["timestamp"])
    
    print("Engineering features...")
    df_feat, feature_cols = engineer_features(df)
    
    # Chronological Split (Train: 70%, Test: 30%)
    split_idx = int(len(df_feat) * 0.7)
    train_df = df_feat.iloc[:split_idx]
    test_df = df_feat.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df["target_peak_60m"]
    
    X_test = test_df[feature_cols]
    y_test = test_df["target_peak_60m"]
    
    print(f"Training set: {len(train_df)} rows. Testing set: {len(test_df)} rows.")
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
