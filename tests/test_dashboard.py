"""Tests for Interactive Visual Dashboard (src/dashboard.py).

Validates:
1. execute_simulation helper correctly generates metrics and complete step-by-step telemetry DataFrame.
2. All required telemetry columns for Plotly visual rendering are present and non-null.
3. Scenario parameter variations (SOH 25% vs 100%, Wake latency 0 vs 15 min, Hot ambient 42°C)
   execute successfully without schema or numerical anomalies.
4. Outage duration scaling and power breakdown stacked accounting consistency.
5. Outage intervals extraction, power breakdown math helper, timetable formatting, sparkline downsampling.
6. query_saapa_timetable caching and mock ingestion.
7. Headless AppTest execution validating that the full Streamlit UI renders without unhandled exceptions.
8. Two-way reactive synchronization between presets and sidebar parameter controls.
"""

from __future__ import annotations
import os
import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit.testing.v1 import AppTest

from src.dashboard import (
    execute_simulation,
    find_outage_intervals,
    calculate_power_breakdown,
    format_outage_timetable,
    downsample_series,
    apply_twin_chart_theme,
    query_saapa_timetable,
    SCENARIO_PRESETS,
)


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
    breakdown = calculate_power_breakdown(metrics)
    base_b = breakdown["baseline"]
    twin_b = breakdown["twin"]

    # Baseline stacked sum == baseline_ac_kwh
    base_sum = base_b["useful_dc"] + base_b["wake_tax"] + base_b["recharge"] + base_b["rectifier_loss"]
    assert pytest.approx(base_sum, rel=1e-3) == metrics["baseline_ac_kwh"]

    # Twin stacked sum == twin_ac_kwh
    twin_sum = twin_b["useful_dc"] + twin_b["wake_tax"] + twin_b["recharge"] + twin_b["rectifier_loss"]
    assert pytest.approx(twin_sum, rel=1e-3) == metrics["twin_ac_kwh"]


def test_find_outage_intervals():
    """Verify continuous blackout intervals extraction from grid status series."""
    # Empty DataFrame
    assert find_outage_intervals(pd.DataFrame()) == []

    # Synthetic series with 2 blackout blocks
    timestamps = pd.date_range("2026-03-01 00:00", periods=10, freq="5min")
    # Grid: UP, DOWN, DOWN, UP, UP, DOWN, DOWN, DOWN, UP, UP
    grid_status = [1, 0, 0, 1, 1, 0, 0, 0, 1, 1]
    df = pd.DataFrame({"timestamp": timestamps, "grid_status": grid_status})

    intervals = find_outage_intervals(df)
    assert len(intervals) == 2
    assert intervals[0] == (timestamps[1], timestamps[3])
    assert intervals[1] == (timestamps[5], timestamps[8])

    # Trailing blackout till end of series
    df_trailing = pd.DataFrame({"timestamp": timestamps, "grid_status": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]})
    int_trailing = find_outage_intervals(df_trailing)
    assert len(int_trailing) == 1
    assert int_trailing[0] == (timestamps[2], timestamps[9])


def test_format_outage_timetable():
    """Verify SAAPA raw dictionary timetable parsing into clean tabular representation."""
    empty_df = format_outage_timetable([])
    assert isinstance(empty_df, pd.DataFrame)
    assert len(empty_df) == 0

    mock_records = [
        {
            "bill_id": "SITE_1",
            "outage_id": "SAAPA-20260307-1300-F1",
            "feeder_id": "F_1",
            "district_code": "THR-02",
            "start_time": "2026-03-07T13:00:00+03:30",
            "end_time": "2026-03-07T15:00:00+03:30",
            "duration_minutes": 120,
            "reason_code": "LOAD_MANAGEMENT_TAVANIR",
            "reason_fa": "مدیریت بار",
            "status": "CONFIRMED",
        }
    ]
    df = format_outage_timetable(mock_records)
    assert len(df) == 1
    assert df.loc[0, "Duration (h)"] == 2.0
    assert "مدیریت بار" in df.loc[0, "Reason"]
    assert df.loc[0, "District"] == "THR-02"


def test_downsample_series():
    """Verify metric sparkline downsampling preserves range and bounded length."""
    # Empty
    assert downsample_series(pd.Series([])) == []

    # Short series
    short = pd.Series([1.5, 2.5, 3.5])
    assert downsample_series(short, 10) == [1.5, 2.5, 3.5]

    # Long series
    long_s = pd.Series(np.linspace(10.0, 50.0, 500))
    down = downsample_series(long_s, 20)
    assert len(down) == 20
    assert down[0] == 10.0
    assert down[-1] == 50.0


def test_apply_twin_chart_theme():
    """Verify Plotly chart theme application sets executive dark palette and fonts."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[10, 20, 30]))
    themed_fig = apply_twin_chart_theme(fig, height=500, title="Test Chart", yaxis_title="Voltage (V)")

    assert themed_fig.layout.paper_bgcolor == "#0B0F19"
    assert themed_fig.layout.plot_bgcolor == "#111827"
    assert themed_fig.layout.height == 500
    assert themed_fig.layout.title.text == "Test Chart"
    assert themed_fig.layout.yaxis.title.text == "Voltage (V)"


def test_query_saapa_timetable_caching():
    """Verify cached query_saapa_timetable retrieves mock timetables cleanly."""
    mock_df = pd.DataFrame({
        "site_id": ["SITE_1"] * 10,
        "timestamp": pd.date_range("2026-03-07 09:00", periods=10, freq="5min"),
        "grid_status": [1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        "blackout_schedule_flag": [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    })
    outages1 = query_saapa_timetable("mock", "SITE_1", mock_df)
    outages2 = query_saapa_timetable("mock", "SITE_1", mock_df)
    assert isinstance(outages1, list)
    assert outages1 == outages2
    assert len(outages1) >= 1
    assert outages1[0]["bill_id"] == "SITE_1"


def test_dashboard_headless_app_execution():
    """Headless AppTest validation: verify whole UI executes without exceptions, KeyError, or rendering glitches."""
    dashboard_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/dashboard.py"))
    at = AppTest.from_file(dashboard_path, default_timeout=35).run()

    # Must run completely cleanly without unhandled exceptions
    assert not at.exception, f"App execution failed with: {at.exception}"

    # Verify title and header
    assert len(at.title) >= 1
    assert "Telecom Energy Digital Twin" in at.title[0].value

    # Verify 5 KPI metric cards are present and populated with non-trivial values
    assert len(at.metric) >= 5
    metric_labels = [m.label for m in at.metric]
    assert "AC Energy Saved" in metric_labels
    assert "Outage Autonomy" in metric_labels
    assert "Served Traffic (Outage)" in metric_labels
    assert "Peak Battery Temp" in metric_labels
    assert "Safety Interventions" in metric_labels

    for m in at.metric:
        assert m.value is not None
        assert m.delta != 0  # No raw awkward 0 integer deltas

    # Verify essential sidebar widgets exist
    assert len(at.sidebar.selectbox) >= 2  # Preset and Archetype
    assert len(at.sidebar.slider) >= 3     # Horizon, Outage duration, SOH


def test_dashboard_two_way_preset_synchronization():
    """Verify that selecting a preset updates controls, and modifying a slider resets preset to Custom."""
    dashboard_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/dashboard.py"))
    at = AppTest.from_file(dashboard_path, default_timeout=35).run()

    # Initially Custom Configuration
    assert at.sidebar.selectbox[0].value == "Custom Configuration"

    # Select Severe Blackout Stress (index 2 in presets)
    preset_names = list(SCENARIO_PRESETS.keys())
    severe_idx = preset_names.index("Severe Blackout Stress (50% SOH, 4h Outage, 15m Wake)")
    at.sidebar.selectbox[0].select_index(severe_idx).run(timeout=35)

    assert at.sidebar.selectbox[0].value == preset_names[severe_idx]
    # SOH should now be 50, Outage duration 4.0
    assert at.sidebar.slider[2].value == 50
    assert at.sidebar.slider[1].value == 4.0

    # User alters SOH slider to 80 -> preset should automatically switch to Custom Configuration
    at.sidebar.slider[2].set_value(80).run(timeout=35)
    assert at.sidebar.selectbox[0].value == "Custom Configuration"
    assert at.sidebar.slider[2].value == 80
