"""Tests for src/benchmark.py reproducible out-of-sample evaluation suite."""
import pandas as pd
from src.benchmark import benchmark_conformal_shift, benchmark_grid_policies, benchmark_outage_survival


def test_benchmark_grid_policies_invariants():
    """Verify grid policy benchmark runs on unseen seeds and forecast+shield drastically reduces violations vs reactive."""
    df = benchmark_grid_policies(days=5, seeds=(101,), latencies_min=(0.0, 15.0))
    assert isinstance(df, pd.DataFrame)
    assert set(df["policy"]) == {"none", "timer", "reactive", "shield", "shield_conformal"}

    # At 15 min wake latency, shield/shield_conformal must have far fewer violations than reactive
    sub15 = df[df["wake_latency_min"] == 15.0].set_index("policy")
    assert sub15.loc["none", "saved_pct_mean"] == 0.0
    assert sub15.loc["shield", "viol_total"] < sub15.loc["reactive", "viol_total"]
    assert sub15.loc["shield_conformal", "viol_total"] <= sub15.loc["shield", "viol_total"]


def test_benchmark_outage_survival_invariants():
    """Verify outage survival benchmark shows twin outliving naive baseline on degraded batteries."""
    df = benchmark_outage_survival(days=7, seeds=(101,), soh_levels=(1.0, 0.40), outage_durations_h=(2.0, 4.0))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4

    # On degraded SOH (0.40) and 4h cuts, twin survival % must exceed baseline survival %
    deg = df[(df["soh"] == 0.40) & (df["outage_duration_h"] == 4.0)].iloc[0]
    assert deg["twin_survival_pct"] > deg["base_survival_pct"]
    assert deg["autonomy_gain_pct"] > 0.0


def test_benchmark_conformal_shift_repairs_coverage():
    """Verify rolling conformal calibration restores coverage on shifted regimes."""
    df = benchmark_conformal_shift(days=5, seed=303, num_sites=2)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    for _, row in df.iterrows():
        assert row["cal_overall_cov"] >= row["raw_overall_cov"]
        assert row["cal_evening_cov"] >= row["raw_evening_cov"]
        assert row["cal_overall_cov"] >= 0.90
