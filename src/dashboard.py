"""Interactive Visual Dashboard for Telecom Energy Twin (src/dashboard.py).

Features:
  - Executive KPI cards with Material Symbols, sparklines, and status badges.
  - Two-Way Reactive Scenario Presets: Tehran Summer, Severe Outage, Extreme Heatwave, Lab Benchmark.
  - Controls: Battery SOH (25% to 100%), Carrier Wake Latency (0, 5, 15 min),
    Outage Duration Stress (1.0 to 4.0h), Ambient Regimes, and SAAPA Grid Mode.
  - Bespoke Slate Dark Plotly Visualizations:
      1. Outage Survival vs Baseline Collapse (SOC %, voltage vs BLVD, step-based active tier, shaded blackouts).
      2. Power & Rectifier Losses Breakdown (AC stacked bar, dual-axis power vs demand, rectifier losses, wake tax).
      3. Coupled Shelter Thermal Dynamics (two-node physics, shaded HVAC blackouts, 40°C derate & 50°C shutdown bands).
      4. Safety Arbiter Diagnostics & Bargh-e Man SAAPA Timetable Inspector (clean tables, JSON view, CSV export).

Run with:
  .venv/bin/streamlit run src/dashboard.py
"""

from __future__ import annotations
import sys
import os
import datetime
from typing import Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure project imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.controller import run_simulation, RECTIFIER_EFF, WAKE_TAX_KW
from src.grid_client import BargheManClient, TAVANIR_BLOCKS, BargheManNetworkError, BargheManAPIError
from src.thermal import T_WARN, T_CRIT
from src.translation_layer import (
    VENDORS,
    TIER_NAMES,
    TranslationError,
    translate,
    translate_plan,
    tier_transitions,
    summarize_command_counts,
)

VENDOR_LABELS: dict[str, str] = {
    "huawei_mml": "Huawei MML (U2000/MAE)",
    "ericsson_cli": "Ericsson ENM CLI",
    "oran_netconf": "O-RAN O1 NETCONF",
}


def default_preview_transition(
    timestamps: pd.Series,
    transitions: list[tuple[int, int, int]],
    outage_intervals: list[tuple[Any, Any]],
) -> int:
    """Pick the first enforced-tier transition at or after the first outage start (else 0)."""
    if not transitions or not outage_intervals or timestamps.empty:
        return 0
    first_outage_start = outage_intervals[0][0]
    for idx, (step, _, _) in enumerate(transitions):
        if 0 <= step < len(timestamps) and timestamps.iloc[step] >= first_outage_start:
            return idx
    return 0


def format_transition_label(
    timestamps: pd.Series,
    transition: tuple[int, int, int],
    idx: int,
) -> str:
    """Format a tier transition tuple (step, from_tier, to_tier) for the selectbox."""
    step, t_from, t_to = transition
    ts_str = str(timestamps.iloc[step])[:16] if 0 <= step < len(timestamps) else f"step {step}"
    return f"#{idx + 1} · {ts_str} · {TIER_NAMES[t_from]} → {TIER_NAMES[t_to]}"


# =============================================================================
# Preset Configurations
# =============================================================================

SCENARIO_PRESETS: dict[str, dict[str, Any] | None] = {
    "Custom Configuration": None,
    "Tehran Summer Baseline (60% SOH, 2h Outage, 5m Wake)": {
        "soh": 60,
        "outage_dur": 2.0,
        "latency": 5.0,
        "ambient": "Diurnal Summer Profile (26°C - 42°C)",
    },
    "Severe Blackout Stress (50% SOH, 4h Outage, 15m Wake)": {
        "soh": 50,
        "outage_dur": 4.0,
        "latency": 15.0,
        "ambient": "Diurnal Summer Profile (26°C - 42°C)",
    },
    "Extreme Heatwave 42°C (75% SOH, 2.5h Outage, 5m Wake)": {
        "soh": 75,
        "outage_dur": 2.5,
        "latency": 5.0,
        "ambient": "Constant Extreme Heatwave (42°C)",
    },
    "Lab Benchmark (100% SOH, 1h Outage, Instant Wake)": {
        "soh": 100,
        "outage_dur": 1.0,
        "latency": 0.0,
        "ambient": "Nominal Lab (25°C)",
    },
}


# =============================================================================
# Helper Utilities & Caching
# =============================================================================

@st.cache_data(show_spinner=False)
def execute_simulation(
    s_id: str,
    days: int,
    lat_min: float,
    soh: float,
    amb_c: float | None,
    init_sh: float,
    init_ba: float,
    mode: str,
    outage_dur_h: float | None = None,
    conformal_on: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    """Executes digital twin physical simulation loop and returns metrics and full telemetry history."""
    g_client = BargheManClient(mode=mode)
    res = run_simulation(
        site_id=s_id,
        days=days,
        wake_latency_min=lat_min,
        soh=soh,
        grid_client=g_client,
        ambient_c=amb_c,
        init_shelter_c=init_sh,
        init_batt_c=init_ba,
        outage_duration_h=outage_dur_h,
        return_history=True,
        conformal=conformal_on,
    )
    return res


def find_outage_intervals(df: pd.DataFrame) -> list[tuple[Any, Any]]:
    """Extract continuous blackout intervals (grid_status == 0) from the telemetry DataFrame."""
    if df.empty or "grid_status" not in df.columns or "timestamp" not in df.columns:
        return []

    intervals: list[tuple[Any, Any]] = []
    in_outage = False
    start_ts = None

    for _, row in df.iterrows():
        status = row["grid_status"]
        ts = row["timestamp"]
        if status == 0 and not in_outage:
            in_outage = True
            start_ts = ts
        elif status != 0 and in_outage:
            in_outage = False
            intervals.append((start_ts, ts))

    if in_outage and start_ts is not None:
        intervals.append((start_ts, df["timestamp"].iloc[-1]))

    return intervals


def calculate_power_breakdown(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Computes categorical AC energy components (Useful DC, Wake Tax, Recharge, Rectifier Loss)."""
    base_dc = float(metrics.get("baseline_grid_dc_kwh", 0.0))
    twin_dc = float(metrics.get("twin_grid_dc_kwh", 0.0))
    base_rech = float(metrics.get("baseline_recharge_kwh", 0.0))
    twin_rech = float(metrics.get("twin_recharge_kwh", 0.0))
    wake_tax = float(metrics.get("twin_wake_tax_kwh", 0.0))
    twin_useful_dc = max(0.0, twin_dc - wake_tax)
    base_ac = float(metrics.get("baseline_ac_kwh", 0.0))
    twin_ac = float(metrics.get("twin_ac_kwh", 0.0))
    base_rec_loss = max(0.0, base_ac - (base_dc + base_rech))
    twin_rec_loss = max(0.0, twin_ac - (twin_dc + twin_rech))
    return {
        "baseline": {
            "useful_dc": base_dc,
            "wake_tax": 0.0,
            "recharge": base_rech,
            "rectifier_loss": base_rec_loss,
            "total_ac": base_ac,
        },
        "twin": {
            "useful_dc": twin_useful_dc,
            "wake_tax": wake_tax,
            "recharge": twin_rech,
            "rectifier_loss": twin_rec_loss,
            "total_ac": twin_ac,
        },
    }


def format_outage_timetable(outages_list: list[dict[str, Any]]) -> pd.DataFrame:
    """Formats raw SAAPA timetable dictionary into a clean display DataFrame."""
    if not outages_list:
        return pd.DataFrame(
            columns=["Outage ID", "District", "Feeder", "Start Time", "End Time", "Duration (h)", "Reason", "Status"]
        )

    rows = []
    for out in outages_list:
        dur_h = float(out.get("duration_minutes", 120)) / 60.0
        reason_fa = str(out.get("reason_fa", ""))
        reason_code = str(out.get("reason_code", "LOAD_MANAGEMENT"))
        reason_str = f"{reason_fa} ({reason_code})" if reason_fa else reason_code
        rows.append({
            "Outage ID": str(out.get("outage_id", "N/A")),
            "District": str(out.get("district_code", "THR")),
            "Feeder": str(out.get("feeder_id", "N/A")),
            "Start Time": str(out.get("start_time", ""))[:19].replace("T", " "),
            "End Time": str(out.get("end_time", ""))[:19].replace("T", " "),
            "Duration (h)": round(dur_h, 2),
            "Reason": reason_str,
            "Status": str(out.get("status", "CONFIRMED")),
        })
    return pd.DataFrame(rows)


def downsample_series(series: pd.Series, max_points: int = 28) -> list[float]:
    """Downsamples a pandas Series to a compact sequence for Streamlit metric sparklines."""
    if len(series) == 0:
        return []
    if len(series) <= max_points:
        return [float(x) for x in series.tolist()]
    indices = np.linspace(0, len(series) - 1, max_points, dtype=int)
    return [round(float(series.iloc[i]), 2) for i in indices]


def apply_twin_chart_theme(
    fig: go.Figure,
    height: int = 450,
    title: str | None = None,
    yaxis_title: str | None = None,
    show_legend: bool = True,
) -> go.Figure:
    """Applies modern slate dark styling and typography to a Plotly figure."""
    layout_update: dict[str, Any] = dict(
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#111827",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=12, color="#E2E8F0"),
        height=height,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#F8FAFC",
            bordercolor="#3B82F6",
        ),
        margin=dict(l=45, r=45, t=55 if title else 35, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(17, 24, 39, 0.80)",
            bordercolor="rgba(51, 65, 85, 0.6)",
            borderwidth=1,
            font=dict(size=11, color="#CBD5E1"),
        ) if show_legend else dict(visible=False),
    )
    if title:
        layout_update["title"] = dict(
            text=title,
            font=dict(size=14, color="#F8FAFC", family="Inter, sans-serif"),
            x=0.01,
            xanchor="left",
        )
    if yaxis_title:
        layout_update["yaxis_title"] = yaxis_title

    fig.update_layout(**layout_update)
    fig.update_xaxes(
        gridcolor="#1F2937",
        zerolinecolor="#374151",
        showline=True,
        linecolor="#334155",
        tickfont=dict(color="#94A3B8"),
    )
    fig.update_yaxes(
        gridcolor="#1F2937",
        zerolinecolor="#374151",
        showline=True,
        linecolor="#334155",
        tickfont=dict(color="#94A3B8"),
    )
    return fig


@st.cache_data(ttl=300, show_spinner=False)
def query_saapa_timetable(client_mode: str, site_id: str, df_mock: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    """Cached query for Bargh-e Man SAAPA timetable (prevents slow network blocking on live API)."""
    client = BargheManClient(mode=client_mode)
    if client_mode == "mock" and df_mock is not None:
        client.load_mock_from_dataframe(df_mock)
    return client.fetch_planned_outages(site_id)


# =============================================================================
# Interactive Fragments
# =============================================================================

@st.fragment
def render_power_timeseries(hist_df: pd.DataFrame, outage_intervals: list[tuple[Any, Any]]) -> None:
    """Renders instantaneous AC power vs cellular demand with reactive window filtering."""
    max_days = len(hist_df) / 288.0
    window_choice = st.segmented_control(
        "Display Time Window",
        options=["Full Horizon", "First 3 Days", "First 24 Hours"],
        default="First 3 Days" if max_days >= 3 else "Full Horizon",
        label_visibility="collapsed",
    )
    if window_choice == "First 24 Hours":
        sample_plot = hist_df.iloc[:288]
        window_title = "24-Hour Zoom"
    elif window_choice == "First 3 Days":
        sample_plot = hist_df.iloc[:min(len(hist_df), 288 * 3)]
        window_title = f"{min(3, int(max_days))}-Day Sample"
    else:
        sample_plot = hist_df
        window_title = f"{max_days:.1f}-Day Full Horizon"

    fig_power = make_subplots(specs=[[{"secondary_y": True}]])
    fig_power.add_trace(
        go.Scatter(
            x=sample_plot["timestamp"], y=sample_plot["p_base_ac"],
            name="Baseline AC Power (kW)",
            line=dict(color="#EF4444", dash="dash", width=1.8),
            hovertemplate="%{y:.2f} kW<extra>Baseline AC</extra>",
        ),
        secondary_y=False,
    )
    fig_power.add_trace(
        go.Scatter(
            x=sample_plot["timestamp"], y=sample_plot["p_twin_ac"],
            name="Twin AC Power (kW)",
            line=dict(color="#10B981", width=2.2),
            hovertemplate="%{y:.2f} kW<extra>Twin AC</extra>",
        ),
        secondary_y=False,
    )
    fig_power.add_trace(
        go.Scatter(
            x=sample_plot["timestamp"], y=sample_plot["actual_demand"],
            name="Cellular Demand (CU)",
            line=dict(color="#8B5CF6", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(139, 92, 246, 0.08)",
            hovertemplate="%{y:.2f} CU<extra>Demand</extra>",
        ),
        secondary_y=True,
    )
    if "bound" in sample_plot.columns:
        fig_power.add_trace(
            go.Scatter(
                x=sample_plot["timestamp"], y=sample_plot["bound"],
                name="AI 60m Safety Bound (q95 + Conformal)",
                line=dict(color="#F59E0B", width=1.6, dash="dot"),
                hovertemplate="%{y:.2f} CU<extra>AI Bound</extra>",
            ),
            secondary_y=True,
        )

    # Shading for blackout intervals within selected sample window
    sample_start = sample_plot["timestamp"].iloc[0]
    sample_end = sample_plot["timestamp"].iloc[-1]
    for start_ts, end_ts in outage_intervals:
        if end_ts >= sample_start and start_ts <= sample_end:
            fig_power.add_vrect(
                x0=max(start_ts, sample_start),
                x1=min(end_ts, sample_end),
                fillcolor="rgba(239, 68, 68, 0.12)",
                layer="below",
                line_width=1,
                line_dash="dot",
                line_color="rgba(239, 68, 68, 0.35)",
            )

    fig_power.update_layout(
        title=f"Instantaneous AC Mains Draw vs Traffic Demand ({window_title})",
    )
    fig_power.update_yaxes(title_text="AC Power (kW)", secondary_y=False)
    fig_power.update_yaxes(title_text="Demand Load (CU)", secondary_y=True, showgrid=False)
    apply_twin_chart_theme(fig_power, height=430)
    st.plotly_chart(fig_power, theme=None, width="stretch", alt="Dual axis power vs traffic chart")


# =============================================================================
# Main Application
# =============================================================================

def main():
    st.set_page_config(
        page_title="Telecom Energy Digital Twin (v3.3)",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize reactive session state keys
    if "preset_selector" not in st.session_state:
        st.session_state["preset_selector"] = "Custom Configuration"
    if "soh_slider" not in st.session_state:
        st.session_state["soh_slider"] = 60
    if "outage_slider" not in st.session_state:
        st.session_state["outage_slider"] = 2.0
    if "latency_control" not in st.session_state:
        st.session_state["latency_control"] = 5.0
    if "ambient_radio" not in st.session_state:
        st.session_state["ambient_radio"] = "Diurnal Summer Profile (26°C - 42°C)"

    def sync_from_preset():
        p_name = st.session_state.get("preset_selector")
        if p_name and p_name in SCENARIO_PRESETS and SCENARIO_PRESETS[p_name] is not None:
            p_cfg = SCENARIO_PRESETS[p_name]
            st.session_state["soh_slider"] = int(p_cfg["soh"])
            st.session_state["outage_slider"] = float(p_cfg["outage_dur"])
            st.session_state["latency_control"] = float(p_cfg["latency"])
            st.session_state["ambient_radio"] = str(p_cfg["ambient"])

    def sync_to_custom():
        curr_preset = st.session_state.get("preset_selector", "Custom Configuration")
        if curr_preset != "Custom Configuration" and curr_preset in SCENARIO_PRESETS:
            cfg = SCENARIO_PRESETS[curr_preset]
            if cfg is not None:
                if (
                    int(st.session_state.get("soh_slider", 60)) != int(cfg["soh"])
                    or float(st.session_state.get("outage_slider", 2.0)) != float(cfg["outage_dur"])
                    or float(st.session_state.get("latency_control", 5.0)) != float(cfg["latency"])
                    or str(st.session_state.get("ambient_radio", "")) != str(cfg["ambient"])
                ):
                    st.session_state["preset_selector"] = "Custom Configuration"

    # -------------------------------------------------------------------------
    # Sidebar: Scenario Parameters & Controls
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### :material/tune: Scenario Configuration")

        st.selectbox(
            "Scenario Preset",
            options=list(SCENARIO_PRESETS.keys()),
            key="preset_selector",
            on_change=sync_from_preset,
            help="Select a calibrated scenario preset or customize individual parameters below.",
        )

        site_options = {
            "SITE_1 (Residential - Evening Peak)": "SITE_1",
            "SITE_2 (Commercial - Workday Daytime Peak)": "SITE_2",
            "SITE_3 (Highway / Transit Commute Corridor)": "SITE_3",
        }
        selected_site_label = st.selectbox(
            "Base Station Archetype",
            list(site_options.keys()),
            key="site_selector",
            help="Geographical archetype dictating diurnal cellular traffic demand profile.",
        )
        site_id = site_options[selected_site_label]

        sim_days = st.slider(
            "Simulation Horizon (Days)",
            min_value=1,
            max_value=30,
            value=7,
            step=1,
            key="horizon_slider",
            help="Duration of held-out test window to simulate step-by-step (5-minute resolution).",
        )

        outage_dur_choice = st.slider(
            "Scheduled Outage Duration (Hours)",
            min_value=1.0,
            max_value=4.0,
            step=0.5,
            key="outage_slider",
            on_change=sync_to_custom,
            help="Scheduled blackout duration to stress-test battery autonomy (1 to 4 hours per event).",
        )

        soh_pct = st.slider(
            "Battery Health (SOH %)",
            min_value=25,
            max_value=100,
            step=5,
            key="soh_slider",
            on_change=sync_to_custom,
            help="Degraded battery condition (25% to 100% aged capacity, affects internal resistance & Peukert loss).",
        )
        soh_val = float(soh_pct) / 100.0

        st.markdown("**Carrier Wake-Up Latency**")
        wake_latency_choice = st.segmented_control(
            "Carrier Wake-Up Latency",
            options=[0.0, 5.0, 15.0],
            format_func=lambda x: {0.0: "0m Instant", 5.0: "5m Warm Sync", 15.0: "15m Cold Reboot"}.get(x, f"{x}m"),
            label_visibility="collapsed",
            key="latency_control",
            on_change=sync_to_custom,
            help="0 min = Instant, 5 min = Warm baseband sync baseline, 15 min = Cold hardware reboot.",
        )
        if wake_latency_choice is None:
            wake_latency_choice = 5.0

        st.markdown("**Ambient Temperature Regime**")
        ambient_options = [
            "Diurnal Summer Profile (26°C - 42°C)",
            "Constant Extreme Heatwave (42°C)",
            "Nominal Lab (25°C)",
        ]
        ambient_mode = st.radio(
            "Ambient Temperature Regime",
            options=ambient_options,
            label_visibility="collapsed",
            key="ambient_radio",
            on_change=sync_to_custom,
        )

        if "Constant" in str(ambient_mode):
            ambient_c = 42.0
            init_shelter = 38.0
            init_batt = 38.0
            ambient_label = "42°C Heatwave"
        elif "Nominal" in str(ambient_mode):
            ambient_c = 25.0
            init_shelter = 25.0
            init_batt = 25.0
            ambient_label = "25°C Lab"
        else:
            ambient_c = None
            init_shelter = 28.0
            init_batt = 27.0
            ambient_label = "26-42°C Diurnal"

        st.markdown("**Bargh-e Man (SAAPA) Mode**")
        grid_mode = st.segmented_control(
            "Bargh-e Man (SAAPA) Mode",
            options=["mock", "live"],
            default="mock",
            key="grid_mode_control",
            format_func=lambda x: "Offline Timetable (Mock)" if x == "mock" else "Live API (uiapi.saapa.ir)",
            label_visibility="collapsed",
            help="Select offline simulated timetable or connect to SAAPA distribution grid endpoint.",
        )
        client_mode = grid_mode if grid_mode is not None else "mock"

        conformal_on = st.toggle(
            "Causal Conformal Calibration (q95 + ACI)",
            value=True,
            key="conformal_toggle",
            help="Applies 7-day rolling split-conformal margin (12-step causal delay, 4h diurnal buckets) to LightGBM q95.",
        )

        if st.button("Clear Cache & Recompute", icon=":material/refresh:", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    # -------------------------------------------------------------------------
    # Header Banner & Live Status Strip
    # -------------------------------------------------------------------------
    st.title("⚡ Telecom Energy Digital Twin (v3.3)", anchor=False)
    st.caption(
        "Tehran District Grid Resilience & O-RAN Carrier Power Orchestration · "
        "Receding-Horizon Reserve Planning · Bargh-e Man SAAPA Adapter · Coupled Shelter Thermal Dynamics"
    )

    # High-contrast executive status badges
    health_badge = f":green-badge[Battery Health: {soh_pct}% SOH]" if soh_pct >= 80 else (
        f":orange-badge[Battery Health: {soh_pct}% SOH]" if soh_pct >= 50 else f":red-badge[Battery Health: {soh_pct}% SOH]"
    )
    grid_badge = ":green-badge[Grid Mode: MOCK TIMETABLE]" if client_mode == "mock" else ":blue-badge[Grid Mode: LIVE SAAPA API]"
    conf_badge = ":green-badge[AI Calibrator: CONFORMAL ON]" if conformal_on else ":gray-badge[AI Calibrator: RAW q95]"

    st.markdown(
        f":blue-badge[Archetype: {site_id}] "
        f"{grid_badge} "
        f"{conf_badge} "
        f"{health_badge} "
        f":purple-badge[Outage Duration: {outage_dur_choice:.1f}h] "
        f":gray-badge[Horizon: {sim_days} Days] "
        f":red-badge[Ambient: {ambient_label}]"
    )

    # -------------------------------------------------------------------------
    # Simulation Execution
    # -------------------------------------------------------------------------
    with st.spinner("Ticking physical digital twin & solving Priority Lattice..."):
        sim_result = execute_simulation(
            site_id,
            sim_days,
            wake_latency_choice,
            soh_val,
            ambient_c,
            init_shelter,
            init_batt,
            client_mode,
            outage_dur_choice,
            conformal_on,
        )

    if sim_result is None:
        st.error("Failed to run simulation. Check that models/ and data/ exist.")
        st.stop()

    metrics, hist_df = sim_result

    # -------------------------------------------------------------------------
    # Executive KPI Metric Cards
    # -------------------------------------------------------------------------
    saved_ac_kwh = metrics["baseline_ac_kwh"] - metrics["twin_ac_kwh"]
    pct_saved = (saved_ac_kwh / metrics["baseline_ac_kwh"] * 100) if metrics["baseline_ac_kwh"] > 0 else 0.0
    b_any_h = metrics["baseline_any_srv_steps"] * 5 / 60.0
    t_any_h = metrics["twin_any_srv_steps"] * 5 / 60.0
    autonomy_gain = t_any_h - b_any_h
    pct_autonomy = ((autonomy_gain / b_any_h) * 100) if b_any_h > 0 else 0.0

    blvd_trips_base = int((hist_df["base_v"] <= 42.0).sum())
    blvd_trips_twin = int((hist_df["twin_v"] <= 42.0).sum())
    blvd_prevented = max(0, blvd_trips_base - blvd_trips_twin)

    # Executive Hero Callout Strip
    with st.container(border=True):
        sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
        with sc_col1:
            st.markdown(f"**:material/verified: Autonomy Extended**  \n`+{autonomy_gain:.1f} hrs` ({pct_autonomy:+.1f}%)")
        with sc_col2:
            st.markdown(f"**:material/energy_savings_leaf: AC Grid Reduction**  \n`{saved_ac_kwh:.1f} kWh` ({pct_saved:+.1f}%)")
        with sc_col3:
            st.markdown(f"**:material/power_off: BLVD Cutoffs Averted**  \n`{blvd_prevented} steps` ({blvd_prevented * 5} min)")
        with sc_col4:
            st.markdown(f"**:material/speed: Simulation Window**  \n`{len(hist_df):,} steps` ({sim_days} days)")

    # Sparkline series
    spark_ac = downsample_series(hist_df["p_twin_ac"], 30)
    spark_soc = downsample_series(hist_df["twin_soc"] * 100, 30)
    spark_demand = downsample_series(hist_df["actual_demand"], 30)
    spark_temp = downsample_series(hist_df["t_batt_twin"], 30)
    spark_shield = downsample_series(hist_df["shield_intervention"].astype(int).cumsum(), 30)

    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

    with kpi_col1:
        st.metric(
            label="AC Energy Saved",
            value=f"{saved_ac_kwh:.1f} kWh",
            delta=f"{pct_saved:+.1f}%" if abs(pct_saved) > 0.01 else "0.0%",
            delta_description="Grid AC reduction vs base",
            border=True,
            icon=":material/bolt:",
            chart_data=spark_ac,
            chart_type="line",
        )
    with kpi_col2:
        st.metric(
            label="Outage Autonomy",
            value=f"{t_any_h:.1f} hrs",
            delta=f"+{autonomy_gain:.1f} hrs" if autonomy_gain > 0.05 else "Parity (0.0 hrs)",
            delta_description=f"vs. baseline ({b_any_h:.1f} hrs)",
            border=True,
            icon=":material/battery_charging_full:",
            chart_data=spark_soc,
            chart_type="area",
        )
    with kpi_col3:
        served_pct = (
            (metrics['twin_served_outage_cuh'] / metrics['outage_demand_cuh'] * 100)
            if metrics.get('outage_demand_cuh', 0) > 0 else 100.0
        )
        st.metric(
            label="Served Traffic (Outage)",
            value=f"{metrics['twin_served_outage_cuh']:.1f} CU·h",
            delta=f"{served_pct:.0f}% delivered",
            delta_description=f"Demanded: {metrics['outage_demand_cuh']:.1f} CU·h",
            delta_color="off",
            border=True,
            icon=":material/cell_tower:",
            chart_data=spark_demand,
            chart_type="bar",
        )
    with kpi_col4:
        warn_cnt = metrics['thermal_warn_steps']
        st.metric(
            label="Peak Battery Temp",
            value=f"{metrics['max_batt_temp_twin']:.1f} °C",
            delta=f"{warn_cnt} warn steps" if warn_cnt > 0 else "Nominal (<40°C)",
            delta_description="IEEE 484 limit (≥40°C)",
            delta_color="inverse" if warn_cnt > 0 else "off",
            border=True,
            icon=":material/device_thermostat:",
            chart_data=spark_temp,
            chart_type="line",
        )
    with kpi_col5:
        ub_conflicts = metrics['shield_ub_conflicts']
        st.metric(
            label="Safety Interventions",
            value=f"{metrics['shield_interventions']}",
            delta=f"{ub_conflicts} UB conflicts" if ub_conflicts > 0 else "0 hard conflicts",
            delta_description="Priority Lattice hard bounds",
            delta_color="inverse" if ub_conflicts > 0 else "off",
            border=True,
            icon=":material/shield:",
            chart_data=spark_shield,
            chart_type="line",
        )

    # Pre-extract outage blackout intervals for chart annotations
    outage_intervals = find_outage_intervals(hist_df)

    # -------------------------------------------------------------------------
    # Main Visualization Tabs
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Outage Survival vs Baseline Collapse",
        "⚡ Power & Rectifier Losses Breakdown",
        "🌡️ Shelter Thermal & Battery Degradation",
        "🛡️ Safety Arbiter & Bargh-e Man Logs",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Outage Survival vs Baseline Collapse
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("Battery State-of-Charge & Service Tier Autonomy", anchor=False)
        with st.container(border=True):
            st.markdown(
                "**Physical Autonomy Invariant:** The Naive Baseline operates at Tier 4 FULL until the battery abruptly "
                "collapses to 0% and BLVD trips (42.0 V), triggering complete site blackout. The Digital Twin "
                "Receding-Horizon planner down-scales to Tier 1 (Voice/SMS CSFB), safeguarding the 15% SOC floor "
                "and extending battery life throughout the entire scheduled outage."
            )

        fig_survival = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=[
                "Battery State of Charge (SOC %)",
                "Battery Terminal Voltage (V) vs BLVD Cutoff (42.0 V)",
                "Active Service Tier (0: OFF, 1: SURVIVAL, 2: BASIC, 3: ANCHOR+1, 4: FULL)",
            ],
        )

        # 1. SOC plot
        fig_survival.add_trace(
            go.Scatter(
                x=hist_df["timestamp"], y=hist_df["base_soc"] * 100,
                name="Baseline SOC", line=dict(color="#EF4444", width=2, dash="dash"),
                hovertemplate="%{y:.1f}%<extra>Baseline SOC</extra>",
            ),
            row=1, col=1,
        )
        fig_survival.add_trace(
            go.Scatter(
                x=hist_df["timestamp"], y=hist_df["twin_soc"] * 100,
                name="Twin SOC", line=dict(color="#10B981", width=2.5),
                hovertemplate="%{y:.1f}%<extra>Twin SOC</extra>",
            ),
            row=1, col=1,
        )
        fig_survival.add_hline(
            y=15.0, line=dict(color="#F59E0B", dash="dot", width=1.5),
            annotation_text="SOC Floor (15%)", annotation_position="top right",
            annotation_font=dict(color="#F59E0B", size=10),
            row=1, col=1,
        )
        fig_survival.add_hrect(
            y0=0.0, y1=15.0,
            fillcolor="rgba(239, 68, 68, 0.07)",
            layer="below", line_width=0,
            row=1, col=1,
        )

        # 2. Voltage plot
        fig_survival.add_trace(
            go.Scatter(
                x=hist_df["timestamp"], y=hist_df["base_v"],
                name="Baseline Voltage", line=dict(color="#EF4444", width=1.8, dash="dash"),
                hovertemplate="%{y:.2f} V<extra>Baseline Voltage</extra>",
            ),
            row=2, col=1,
        )
        fig_survival.add_trace(
            go.Scatter(
                x=hist_df["timestamp"], y=hist_df["twin_v"],
                name="Twin Voltage", line=dict(color="#06B6D4", width=2.2),
                hovertemplate="%{y:.2f} V<extra>Twin Voltage</extra>",
            ),
            row=2, col=1,
        )
        fig_survival.add_hline(
            y=42.0, line=dict(color="#F43F5E", dash="dot", width=1.5),
            annotation_text="BLVD Trip (42.0 V)", annotation_position="top right",
            annotation_font=dict(color="#F43F5E", size=10),
            row=2, col=1,
        )
        fig_survival.add_hrect(
            y0=38.0, y1=42.0,
            fillcolor="rgba(244, 63, 94, 0.08)",
            layer="below", line_width=0,
            row=2, col=1,
        )

        # 3. Discrete Service Tier plot (step function with descriptive labels)
        tier_names = {
            0: "Tier 0: OFF (BLVD Disconnect)",
            1: "Tier 1: SURVIVAL (Voice/SMS CSFB)",
            2: "Tier 2: BASIC (1 Carrier 10MHz)",
            3: "Tier 3: ANCHOR+1 (2 Carriers CA)",
            4: "Tier 4: FULL (3 Carriers MIMO)",
        }
        base_tier_labels = [tier_names.get(int(t), f"Tier {t}") for t in hist_df["base_tier"]]
        twin_tier_labels = [tier_names.get(int(t), f"Tier {t}") for t in hist_df["twin_tier"]]

        fig_survival.add_trace(
            go.Scatter(
                x=hist_df["timestamp"], y=hist_df["base_tier"],
                name="Baseline Tier", line=dict(color="#EF4444", width=1.8, dash="dash", shape="hv"),
                customdata=base_tier_labels,
                hovertemplate="%{customdata}<extra>Baseline Tier</extra>",
            ),
            row=3, col=1,
        )
        fig_survival.add_trace(
            go.Scatter(
                x=hist_df["timestamp"], y=hist_df["twin_tier"],
                name="Twin Tier", line=dict(color="#8B5CF6", width=2.2, shape="hv"),
                customdata=twin_tier_labels,
                hovertemplate="%{customdata}<extra>Twin Tier</extra>",
            ),
            row=3, col=1,
        )

        # Add vertical shaded regions for blackouts across rows
        for s_idx, (start_ts, end_ts) in enumerate(outage_intervals):
            for r in [1, 2, 3]:
                fig_survival.add_vrect(
                    x0=start_ts,
                    x1=end_ts,
                    fillcolor="rgba(239, 68, 68, 0.12)",
                    layer="below",
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(239, 68, 68, 0.35)",
                    annotation_text="Blackout" if (r == 1 and s_idx == 0) else None,
                    annotation_position="top left",
                    annotation_font=dict(size=10, color="#F87171"),
                    row=r, col=1,
                )

        # Client-side quick range zoom buttons
        fig_survival.update_xaxes(
            rangeselector=dict(
                buttons=[
                    dict(count=24, label="24 Hours", step="hour", stepmode="backward"),
                    dict(count=3, label="3 Days", step="day", stepmode="backward"),
                    dict(count=7, label="7 Days", step="day", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#1E293B",
                activecolor="#3B82F6",
                font=dict(color="#F1F5F9", size=10),
                x=0.0,
                y=1.08,
            ),
            row=1, col=1,
        )

        fig_survival.update_yaxes(title_text="SOC (%)", range=[0, 105], row=1, col=1)
        fig_survival.update_yaxes(title_text="Terminal (V)", range=[38, 56], row=2, col=1)
        fig_survival.update_yaxes(
            title_text="Service Tier",
            tickmode="array",
            tickvals=[0, 1, 2, 3, 4],
            ticktext=["0: OFF", "1: SURV", "2: BASIC", "3: ANCH+1", "4: FULL"],
            range=[-0.2, 4.3],
            row=3, col=1,
        )

        apply_twin_chart_theme(fig_survival, height=750, show_legend=True)
        st.plotly_chart(fig_survival, theme=None, width="stretch", alt="Outage survival vs baseline collapse charts")

    # -------------------------------------------------------------------------
    # TAB 2: Power Breakdown & Rectifier Losses
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("AC Mains & Telecom Rectifier Energy Accounting", anchor=False)
        st.caption(
            f"Rectifier Efficiency η = {RECTIFIER_EFF:.2f}. Total AC power includes site DC load and battery recharge "
            f"divided by 0.96. Carrier wake-up signaling tax adds {WAKE_TAX_KW*1000:.0f} W (+10%) for 5 minutes post-wake."
        )

        c1, c2 = st.columns([1, 1.1])

        breakdown = calculate_power_breakdown(metrics)
        base_cat = breakdown["baseline"]
        twin_cat = breakdown["twin"]

        with c1:
            with st.container(border=True):
                categories = ["Baseline", "Digital Twin"]
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    name="Useful DC Load", x=categories,
                    y=[base_cat["useful_dc"], twin_cat["useful_dc"]],
                    marker_color="#3B82F6",
                    texttemplate="%{y:.1f}",
                    textposition="inside",
                    insidetextanchor="middle",
                    hovertemplate="%{y:.1f} kWh<extra>Useful DC</extra>",
                ))
                fig_bar.add_trace(go.Bar(
                    name="Carrier Wake Tax (+10%)", x=categories,
                    y=[base_cat["wake_tax"], twin_cat["wake_tax"]],
                    marker_color="#F97316",
                    texttemplate="%{y:.2f}",
                    textposition="inside",
                    insidetextanchor="middle",
                    hovertemplate="%{y:.2f} kWh<extra>Wake Tax</extra>",
                ))
                fig_bar.add_trace(go.Bar(
                    name="Battery Recharge", x=categories,
                    y=[base_cat["recharge"], twin_cat["recharge"]],
                    marker_color="#10B981",
                    texttemplate="%{y:.1f}",
                    textposition="inside",
                    insidetextanchor="middle",
                    hovertemplate="%{y:.1f} kWh<extra>Recharge</extra>",
                ))
                fig_bar.add_trace(go.Bar(
                    name="Rectifier Loss (4%)", x=categories,
                    y=[base_cat["rectifier_loss"], twin_cat["rectifier_loss"]],
                    marker_color="#64748B",
                    texttemplate="%{y:.1f}",
                    textposition="inside",
                    insidetextanchor="middle",
                    hovertemplate="%{y:.1f} kWh<extra>Rectifier Loss</extra>",
                ))

                # Display total annotations on top of each bar
                fig_bar.add_annotation(
                    x="Baseline",
                    y=base_cat["total_ac"],
                    text=f"Total: {base_cat['total_ac']:.1f} kWh",
                    showarrow=False,
                    yshift=14,
                    font=dict(size=12, color="#F8FAFC", family="Inter, sans-serif"),
                )
                fig_bar.add_annotation(
                    x="Digital Twin",
                    y=twin_cat["total_ac"],
                    text=f"Total: {twin_cat['total_ac']:.1f} kWh",
                    showarrow=False,
                    yshift=14,
                    font=dict(size=12, color="#10B981", family="Inter, sans-serif"),
                )

                fig_bar.update_layout(
                    barmode="stack",
                    title="Total AC Mains Energy Consumption Breakdown (kWh)",
                    yaxis_title="Energy (kWh)",
                )
                apply_twin_chart_theme(fig_bar, height=430)
                st.plotly_chart(fig_bar, theme=None, width="stretch", alt="AC Mains stacked energy breakdown chart")

                st.caption(
                    f"Digital Twin delivers **{pct_saved:.1f}% net AC energy reduction** "
                    f"(-{saved_ac_kwh:.1f} kWh) by orchestrating carrier power during low traffic and managing battery recharge."
                )

        with c2:
            with st.container(border=True):
                # Interactive zoom window selector isolated in a fast fragment
                render_power_timeseries(hist_df, outage_intervals)

    # -------------------------------------------------------------------------
    # TAB 3: Shelter Thermal & Battery Degradation
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Coupled Shelter Thermal Model & Battery Temperature Dynamics", anchor=False)
        st.caption(
            "Two-node physical model: When grid fails, HVAC drops offline. Remote Radio Units reject heat outdoors on the mast, "
            "while in-shelter equipment (BBU, transport, aux DC) and battery discharge losses accumulate inside the shelter. "
            "Battery temperature tracks shelter air with a ~3.7h thermal lag due to its 240 kJ/K thermal mass. "
            "Arrhenius degradation accelerates exponentially (doubling every 8°C above 25°C)."
        )

        fig_thermal = go.Figure()
        fig_thermal.add_trace(
            go.Scatter(
                x=hist_df["timestamp"], y=hist_df["t_shelter_base"],
                name="Baseline Shelter Air (°C)",
                line=dict(color="#94A3B8", dash="dash", width=1.8),
                hovertemplate="%{y:.1f} °C<extra>Baseline Shelter</extra>",
            )
        )
        fig_thermal.add_trace(
            go.Scatter(
                x=hist_df["timestamp"], y=hist_df["t_shelter_twin"],
                name="Twin Shelter Air (°C)",
                line=dict(color="#38BDF8", width=2.2),
                hovertemplate="%{y:.1f} °C<extra>Twin Shelter</extra>",
            )
        )
        fig_thermal.add_trace(
            go.Scatter(
                x=hist_df["timestamp"], y=hist_df["t_batt_base"],
                name="Baseline Battery Temp (°C)",
                line=dict(color="#F43F5E", dash="dash", width=1.8),
                hovertemplate="%{y:.1f} °C<extra>Baseline Battery</extra>",
            )
        )
        fig_thermal.add_trace(
            go.Scatter(
                x=hist_df["timestamp"], y=hist_df["t_batt_twin"],
                name="Twin Battery Temp (°C)",
                line=dict(color="#10B981", width=2.5),
                hovertemplate="%{y:.1f} °C<extra>Twin Battery</extra>",
            )
        )

        fig_thermal.add_hline(
            y=T_WARN, line=dict(color="#F59E0B", dash="dot", width=1.5),
            annotation_text="IEEE 484 Derating (40°C)", annotation_position="top right",
            annotation_font=dict(color="#F59E0B", size=10),
        )
        fig_thermal.add_hline(
            y=T_CRIT, line=dict(color="#EF4444", dash="dot", width=1.5),
            annotation_text="Critical Thermal Shutdown (50°C)", annotation_position="top right",
            annotation_font=dict(color="#EF4444", size=10),
        )

        # Danger zone bands
        fig_thermal.add_hrect(
            y0=T_WARN, y1=T_CRIT,
            fillcolor="rgba(245, 158, 11, 0.07)",
            layer="below", line_width=0,
        )
        fig_thermal.add_hrect(
            y0=T_CRIT, y1=65.0,
            fillcolor="rgba(239, 68, 68, 0.10)",
            layer="below", line_width=0,
        )

        # Add vertical shaded regions for blackout windows in thermal dynamics
        for s_idx, (start_ts, end_ts) in enumerate(outage_intervals):
            fig_thermal.add_vrect(
                x0=start_ts,
                x1=end_ts,
                fillcolor="rgba(239, 68, 68, 0.10)",
                layer="below",
                line_width=1,
                line_dash="dot",
                line_color="rgba(239, 68, 68, 0.35)",
                annotation_text="HVAC Offline" if s_idx == 0 else None,
                annotation_position="top left",
                annotation_font=dict(size=10, color="#F87171"),
            )

        apply_twin_chart_theme(
            fig_thermal,
            height=460,
            title="Shelter Air & Battery Temperature Trajectory (°C)",
            yaxis_title="Temperature (°C)",
        )
        st.plotly_chart(fig_thermal, theme=None, width="stretch", alt="Coupled thermal trajectories chart")

        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            with st.container(border=True):
                st.markdown("**:material/hourglass_empty: Arrhenius Calendar Aging**")
                b_bp = metrics['baseline_cal_damage'] * 1e4
                t_bp = metrics['twin_cal_damage'] * 1e4
                diff_bp = b_bp - t_bp
                st.markdown(
                    f"• Baseline damage: **{b_bp:.2f} bp**  \n"
                    f"• Digital Twin damage: **{t_bp:.2f} bp**  \n"
                    f"• Lifetime protection: **{diff_bp:+.2f} bp saved**"
                )
        with col_t2:
            with st.container(border=True):
                st.markdown("**:material/thermostat: Inertia & Heat Dissipation**")
                st.markdown(
                    f"• Battery thermal mass: **240 kJ/K** (~3.7h lag)  \n"
                    f"• Peak battery temp: **{metrics['max_batt_temp_twin']:.1f}°C** (Twin) vs "
                    f"**{metrics['max_batt_temp_base']:.1f}°C** (Base)  \n"
                    f"• Warn steps (≥40°C): **{metrics['thermal_warn_steps']}**"
                )
        with col_t3:
            with st.container(border=True):
                st.markdown("**:material/security: Thermal Safety Protection**")
                st.markdown(
                    "• **T ≥ 40°C**: Recharge current throttled (0.5× $I_{lim}$) to prevent float thermal runaway.  \n"
                    "• **T ≥ 50°C**: Safety Arbiter clamps Upper Bound UB = 1 (SURVIVAL), de-energizing all LTE carriers."
                )

    # -------------------------------------------------------------------------
    # TAB 4: Safety Arbiter Diagnostics & Bargh-e Man Logs
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("Safety Arbiter Decision Log & Bargh-e Man (SAAPA) Inspector", anchor=False)

        sub1, sub2 = st.columns([1.1, 0.9])

        with sub1:
            with st.container(border=True):
                st.markdown("##### :material/shield: Safety Arbiter Interventions")
                interventions = hist_df[hist_df["shield_intervention"] | (hist_df["shield_why"] != "")].copy()

                if len(interventions) > 0:
                    interventions["display_status"] = interventions["grid_status"].apply(
                        lambda s: "🟢 Grid UP" if s == 1 else "🔴 Blackout"
                    )

                    filter_choice = st.segmented_control(
                        "Intervention Filter",
                        options=["All", "UB Conflicts", "Clamped Proposals"],
                        default="All",
                        label_visibility="collapsed",
                    )
                    if filter_choice is None:
                        filter_choice = "All"

                    filtered_df = interventions
                    if filter_choice == "UB Conflicts":
                        filtered_df = interventions[interventions["shield_why"].str.contains("conflict:UB-wins", na=False)]
                    elif filter_choice == "Clamped Proposals":
                        filtered_df = interventions[interventions["shield_why"].str.contains("clamp", na=False)]

                    if len(filtered_df) > 0:
                        show_cols = ["timestamp", "display_status", "actual_demand", "twin_tier", "shield_why"]
                        st.dataframe(
                            filtered_df[show_cols].head(150),
                            column_config={
                                "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
                                "display_status": st.column_config.TextColumn("Grid"),
                                "actual_demand": st.column_config.NumberColumn("Demand (CU)", format="%.2f"),
                                "twin_tier": st.column_config.NumberColumn("Enforced Tier", format="%d"),
                                "shield_why": st.column_config.TextColumn("Safety Rule / Trigger"),
                            },
                            hide_index=True,
                            width="stretch",
                            height=340,
                        )
                        csv_data = filtered_df[show_cols].to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=f"Download Interventions CSV ({len(filtered_df)} records)",
                            data=csv_data,
                            file_name=f"interventions_{site_id}.csv",
                            mime="text/csv",
                            icon=":material/download:",
                        )
                    else:
                        st.info(f"No interventions match filter '{filter_choice}'. All proposals satisfied lattice rules.")
                else:
                    st.success("No shield interventions required in this window (all planner proposals satisfied safety lattice).")

        with sub2:
            with st.container(border=True):
                st.markdown("##### :material/calendar_today: Bargh-e Man Scheduled Blackouts (SAAPA)")
                client_status_badge = "🟢 Offline Mock Timetable" if client_mode == "mock" else "🌐 Live API (uiapi.saapa.ir)"
                st.caption(f"Status: {client_status_badge} · Bill ID: `{site_id}`")

                outages = []
                try:
                    df_mock = hist_df.copy()
                    if "blackout_schedule_flag" not in df_mock.columns and "grid_status" in df_mock.columns:
                        df_mock["blackout_schedule_flag"] = (df_mock["grid_status"] == 0).astype(int)
                    outages = query_saapa_timetable(client_mode, site_id, df_mock)
                except (BargheManNetworkError, BargheManAPIError, Exception) as exc:
                    st.warning(f"Could not connect to live SAAPA endpoint ({exc}). Switched to cached timetable.")
                    df_mock = hist_df.copy()
                    df_mock["blackout_schedule_flag"] = (df_mock["grid_status"] == 0).astype(int)
                    outages = query_saapa_timetable("mock", site_id, df_mock)

                if outages:
                    outages_df = format_outage_timetable(outages)
                    st.dataframe(
                        outages_df[["Start Time", "End Time", "Duration (h)", "Reason", "District", "Feeder", "Status"]],
                        column_config={
                            "Start Time": st.column_config.TextColumn("Start"),
                            "End Time": st.column_config.TextColumn("End"),
                            "Duration (h)": st.column_config.NumberColumn("Duration (h)", format="%.1f"),
                            "Reason": st.column_config.TextColumn("Reason"),
                            "District": st.column_config.TextColumn("District"),
                            "Feeder": st.column_config.TextColumn("Feeder"),
                            "Status": st.column_config.TextColumn("Status"),
                        },
                        hide_index=True,
                        width="stretch",
                        height=280,
                    )
                    st.caption(f"Showing {len(outages_df)} scheduled timetable windows for {site_id}.")
                    with st.expander("Raw SAAPA JSON API Payload", icon=":material/code:"):
                        st.json(outages[:3])
                else:
                    st.info(f"No scheduled timetable blackouts found for {site_id}.")

        # ---------------------------------------------------------------------
        # Vendor Command Preview (dry-run output of src/translation_layer.py)
        # ---------------------------------------------------------------------
        with st.container(border=True):
            st.markdown("##### :material/terminal: Vendor Command Preview")
            st.caption(
                "Illustrative dry-run output only: these vendor commands are never executed, and their syntax must be "
                "validated against the operator's MOM / MML release before any field use."
            )
            transitions = tier_transitions(hist_df["twin_tier"])
            if not transitions:
                st.info("The enforced (arbiter) tier never changed in this window, so no vendor commands are generated.")
            else:
                pick1, pick2 = st.columns([1, 1.4])
                with pick1:
                    vp_vendor = st.segmented_control(
                        "Vendor / target interface",
                        options=list(VENDORS),
                        format_func=lambda v: VENDOR_LABELS[v],
                        default="huawei_mml",
                        key="vp_vendor",
                    )
                    if vp_vendor is None:
                        vp_vendor = "huawei_mml"
                with pick2:
                    vp_default = default_preview_transition(hist_df["timestamp"], transitions, outage_intervals)
                    vp_idx = st.selectbox(
                        "Tier transition",
                        options=list(range(len(transitions))),
                        index=vp_default,
                        format_func=lambda i: format_transition_label(hist_df["timestamp"], transitions[i], i),
                        key="vp_transition",
                        help="Defaults to the first enforced-tier change after the first outage starts.",
                    )
                step, t_from, t_to = transitions[vp_idx]
                try:
                    vp_cmds = translate(
                        site_id, t_from, t_to, vp_vendor, reason="safety_arbiter: enforced tier change (twin_tier)"
                    )
                except TranslationError as exc:
                    vp_cmds = []
                    st.error(f"Translation refused: {exc}")

                if vp_cmds:
                    lang = "xml" if vp_vendor == "oran_netconf" else "text"
                    st.markdown(
                        f"**{TIER_NAMES[t_from]} → {TIER_NAMES[t_to]}** · {len(vp_cmds)} ordered action(s) · "
                        f"{'sleep (anchor locked last)' if t_to < t_from else 'wake (GSM, then anchor first)'}"
                    )
                    show_extra = st.toggle("Show read-back (verify) and rollback commands", value=False, key="vp_show_extra")
                    for n, c in enumerate(vp_cmds, start=1):
                        st.caption(f"{n}. {c.op.value} · {c.layer} · target `{c.target}`")
                        st.code(c.command, language=lang, wrap_lines=True)
                        if show_extra:
                            st.caption("Read-back")
                            st.code(c.verify, language=lang, wrap_lines=True)
                            st.caption("Rollback")
                            st.code(c.rollback, language=lang, wrap_lines=True)

                counts = summarize_command_counts(translate_plan(site_id, hist_df["twin_tier"], vp_vendor))
                row = counts.get(vp_vendor, {"commands": 0, "statements": 0})
                st.caption(
                    f"Whole run: {len(transitions)} tier change(s) → {row['commands']} {VENDOR_LABELS[vp_vendor]} "
                    f"action(s) / {row['statements']} statement(s) (dry run)."
                )


    st.caption("Developed for Cellular Network Energy Digital Twin v3.3 · Compliant with 3GPP TR 36.887 & IEEE 484 standards.")


if __name__ == "__main__":
    main()
