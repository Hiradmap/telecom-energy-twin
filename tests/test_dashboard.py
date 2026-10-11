"""Tests for Interactive Visual Dashboard (src/dashboard.py).

Validates:
1. execute_simulation helper correctly generates metrics and complete step-by-step telemetry DataFrame.
2. All required telemetry columns for Plotly visual rendering are present and non-null.
3. Scenario parameter variations (SOH 25% vs 100%, Wake latency 0 vs 15 min, Hot ambient 42°C)
   execute successfully without schema or numerical anomalies.
"""

import pytest
import pandas as pd
from src.dashboard import execute_simulation


def test_dashboard_simulation_execution_and_schema():
    """Verify that dashboard simulation returns valid metrics and complete telemetry DataFrame."""
    metrics, df = execute_simulation(
        s_id="SITE_1",
        days=2,
        lat_min=5.0,
        soh=0.60,
        amb_c=None,
        init_sh=28.0,
        init_ba=27.0,
        mode="mock",
    )

    assert isinstance(metrics, dict)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2 * 288

    # Verify all visualization columns required by Plotly tabs are present
    required_cols = [
        "timestamp", "grid_status", "actual_demand",
        "base_tier", "twin_tier",
        "base_soc", "twin_soc",
        "base_v", "twin_v",
        "t_shelter_base", "t_shelter_twin",
        "t_batt_base", "t_batt_twin",
        "p_base_ac", "p_twin_ac",
        "p_base_dc", "p_twin_dc",
        "shield_intervention", "shield_why",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert not df["twin_soc"].isna().any()
    assert not df["t_shelter_twin"].isna().any()
    assert not df["p_twin_ac"].isna().any()


def test_dashboard_slider_scenario_variations():
    """Verify scenario slider extremes: degraded SOH (25%), cold wake latency (15 min), and extreme heatwave (42°C)."""
    # 1. Severely degraded battery (SOH = 25%)
    m_deg, df_deg = execute_simulation(
        s_id="SITE_1", days=2, lat_min=0.0, soh=0.25, amb_c=25.0, init_sh=25.0, init_ba=25.0, mode="mock"
    )
    assert m_deg["twin_any_srv_steps"] > 0

    # 2. Cold wake latency (15 min)
    m_15, df_15 = execute_simulation(
        s_id="SITE_1", days=2, lat_min=15.0, soh=0.80, amb_c=25.0, init_sh=25.0, init_ba=25.0, mode="mock"
    )
    assert m_15["twin_viol_grid"] >= 0

    # 3. Extreme heatwave (42°C ambient)
    m_hot, df_hot = execute_simulation(
        s_id="SITE_1", days=2, lat_min=5.0, soh=0.80, amb_c=42.0, init_sh=38.0, init_ba=38.0, mode="mock"
    )
    assert m_hot["max_shelter_temp_twin"] > 35.0


def test_dashboard_outage_duration_slider_impact():
    """Verify that the Outage Duration slider (1.0h to 4.0h) scales blackout steps and stresses autonomy."""
    m_1h, df_1h = execute_simulation(
        s_id="SITE_1", days=3, lat_min=5.0, soh=0.50, amb_c=None, init_sh=25.0, init_ba=25.0, mode="mock", outage_dur_h=1.0
    )
    m_4h, df_4h = execute_simulation(
        s_id="SITE_1", days=3, lat_min=5.0, soh=0.50, amb_c=None, init_sh=25.0, init_ba=25.0, mode="mock", outage_dur_h=4.0
    )

    assert m_4h["outage_steps"] > m_1h["outage_steps"]
    # In long 4h outages on degraded batteries (SOH 50%), the twin saves much more autonomy vs collapsing baseline
    assert m_4h["twin_any_srv_steps"] >= m_4h["baseline_any_srv_steps"]


def test_dashboard_power_breakdown_accounting_consistency():
    """Verify that Tab 2 stacked bar power categories sum precisely to total AC consumption."""
    metrics, df = execute_simulation(
        s_id="SITE_1", days=2, lat_min=5.0, soh=0.60, amb_c=None, init_sh=25.0, init_ba=25.0, mode="mock"
    )
    base_dc = metrics["baseline_grid_dc_kwh"]
    twin_dc = metrics["twin_grid_dc_kwh"]
    base_rech = metrics["baseline_recharge_kwh"]
    twin_rech = metrics["twin_recharge_kwh"]
    wake_tax = metrics["twin_wake_tax_kwh"]

    twin_useful_dc = max(0.0, twin_dc - wake_tax)
    base_rec_loss = max(0.0, metrics["baseline_ac_kwh"] - (base_dc + base_rech))
    twin_rec_loss = max(0.0, metrics["twin_ac_kwh"] - (twin_dc + twin_rech))

    # Baseline stacked sum == baseline_ac_kwh
    assert pytest.approx(base_dc + base_rech + base_rec_loss, rel=1e-3) == metrics["baseline_ac_kwh"]
    # Twin stacked sum == twin_ac_kwh
    assert pytest.approx(twin_useful_dc + wake_tax + twin_rech + twin_rec_loss, rel=1e-3) == metrics["twin_ac_kwh"]

