"""Interactive Visual Dashboard for Telecom Energy Twin (src/dashboard.py).

Features:
  - Scenario Sliders: Battery SOH (25% to 100%), Carrier Wake Latency (0, 5, 15 min),
    Outage Duration Stress, and Ambient Temperature.
  - Visual Plots (Plotly):
      1. Digital Twin Survival vs Baseline Collapse Curves (SOC %, terminal voltage vs BLVD, active tier).
      2. Power & Energy Breakdown (AC mains, DC load, rectifier losses 0.96, recharge draw, wake tax).
      3. Shelter Thermal Dynamics & Battery Temperature (diurnal profile, 40°C derate, 50°C shutdown).
      4. Safety Arbiter Diagnostics & Bargh-e Man SAAPA Timetable Inspector.
Run with:
  .venv/bin/streamlit run src/dashboard.py
"""

from __future__ import annotations
import sys
import os
import datetime
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
from src.grid_client import BargheManClient, TAVANIR_BLOCKS
from src.thermal import T_WARN, T_CRIT


def execute_simulation(s_id, days, lat_min, soh, amb_c, init_sh, init_ba, mode, outage_dur_h=None):
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
    )
    return res


def main():
    # Page setup
    st.set_page_config(
        page_title="Telecom Energy Twin (v3.3)",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("⚡ Telecom Energy Digital Twin (v3.3)")
    st.markdown(
        "**Tehran District Grid Resilience & O-RAN Carrier Power Orchestration**  \n"
        "*Receding-Horizon Reserve Planning · Bargh-e Man Timetable Adapter · "
        "Safety Arbiter Priority Lattice · Coupled Shelter Thermal Dynamics*"
    )

    # -------------------------------------------------------------------------
    # Sidebar: Scenario Parameters & Controls
    # -------------------------------------------------------------------------
    st.sidebar.header("🕹️ Scenario Configuration")

    site_options = {
        "SITE_1 (Residential - Evening Peak)": "SITE_1",
        "SITE_2 (Commercial - Workday Daytime Peak)": "SITE_2",
        "SITE_3 (Highway / Transit Commute Corridor)": "SITE_3",
    }
    selected_site_label = st.sidebar.selectbox("Base Station Archetype", list(site_options.keys()))
    site_id = site_options[selected_site_label]

    sim_days = st.sidebar.slider("Simulation Horizon (Days)", min_value=1, max_value=30, value=7, step=1)

    outage_dur_choice = st.sidebar.slider(
        "Scheduled Outage Duration (Hours)",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.5,
        help="Scheduled blackout duration to stress-test battery autonomy (1 to 4 hours).",
    )

    soh_pct = st.sidebar.slider(
        "Battery Health (SOH %)",
        min_value=25,
        max_value=100,
        value=60,
        step=5,
        help="Degraded battery condition (25% to 100% aged capacity).",
    )
    soh_val = soh_pct / 100.0

    wake_latency_choice = st.sidebar.select_slider(
        "Carrier Wake-Up Latency (min)",
        options=[0.0, 5.0, 15.0],
        value=5.0,
        help="0 min = Instant, 5 min = Warm baseband sync baseline, 15 min = Cold hardware reboot.",
    )

    ambient_mode = st.sidebar.radio(
        "Ambient Temperature Regime",
        options=["Diurnal Summer Profile (26°C - 42°C)", "Constant Extreme Heatwave (42°C)", "Nominal Lab (25°C)"],
        index=0,
    )
    if "Constant" in ambient_mode:
        ambient_c = 42.0
        init_shelter = 38.0
        init_batt = 38.0
    elif "Nominal" in ambient_mode:
        ambient_c = 25.0
        init_shelter = 25.0
        init_batt = 25.0
    else:
        ambient_c = None
        init_shelter = 28.0
        init_batt = 27.0

    grid_mode = st.sidebar.radio(
        "Bargh-e Man (SAAPA) Mode",
        options=["mock (Offline 2-Hour Timetable)", "live (uiapi.saapa.ir)"],
        index=0,
    )
    client_mode = "live" if "live" in grid_mode else "mock"

    with st.spinner("Ticking physical digital twin & solving Priority Lattice..."):
        sim_result = execute_simulation(
            site_id, sim_days, wake_latency_choice, soh_val, ambient_c, init_shelter, init_batt, client_mode, outage_dur_choice
        )

    if sim_result is None:
        st.error("Failed to run simulation. Check that models/ and data/ exist.")
        st.stop()

    metrics, hist_df = sim_result

    # -------------------------------------------------------------------------
    # KPI Metrics Row
    # -------------------------------------------------------------------------
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

    saved_ac_kwh = metrics["baseline_ac_kwh"] - metrics["twin_ac_kwh"]
    pct_saved = (saved_ac_kwh / metrics["baseline_ac_kwh"] * 100) if metrics["baseline_ac_kwh"] > 0 else 0
    b_any_h = metrics["baseline_any_srv_steps"] * 5 / 60.0
    t_any_h = metrics["twin_any_srv_steps"] * 5 / 60.0
    autonomy_gain = t_any_h - b_any_h

    with kpi_col1:
        st.metric(
            label="AC Energy Saved",
            value=f"{saved_ac_kwh:.1f} kWh",
            delta=f"-{pct_saved:.1f}% Grid AC",
        )
    with kpi_col2:
        st.metric(
            label="Outage Autonomy",
            value=f"{t_any_h:.1f} hrs",
            delta=f"+{autonomy_gain:.1f} hrs vs Base",
        )
    with kpi_col3:
        st.metric(
            label="Served Traffic (Outage)",
            value=f"{metrics['twin_served_outage_cuh']:.1f} CU·h",
            delta=f"Demanded: {metrics['outage_demand_cuh']:.1f}",
            delta_color="off",
        )
    with kpi_col4:
        st.metric(
            label="Peak Battery Temp",
            value=f"{metrics['max_batt_temp_twin']:.1f} °C",
            delta=f"{metrics['thermal_warn_steps']} warn steps (≥40°C)",
            delta_color="inverse" if metrics["thermal_warn_steps"] > 0 else "off",
        )
    with kpi_col5:
        st.metric(
            label="Safety Interventions",
            value=f"{metrics['shield_interventions']}",
            delta=f"{metrics['shield_ub_conflicts']} UB-over-LB wins",
            delta_color="off",
        )

    st.divider()

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
        st.subheader("Battery State-of-Charge & Service Tier Autonomy")
        st.caption(
            "Notice how the Naive Baseline operates at Tier 4 FULL until the battery abruptly collapses to 0% and BLVD trips, "
            "causing total blackout. The Digital Twin Receding-Horizon planner gracefully steps down carriers to protect the "
            "hard floor (SOC 15%), ensuring voice/emergency CSFB survival for the entire blackout duration."
        )

        fig_survival = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
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
                name="Baseline SOC", line=dict(color="#EF553B", width=2, dash="dash")
            ),
            row=1, col=1,
        )
        fig_survival.add_trace(
            go.Scatter(
                x=hist_df["timestamp"], y=hist_df["twin_soc"] * 100,
                name="Twin SOC", line=dict(color="#00CC96", width=2.5)
            ),
            row=1, col=1,
        )
        fig_survival.add_hline(y=15.0, line=dict(color="#FFA15A", dash="dot"), annotation_text="SOC Floor (15%)", row=1, col=1)

        # 2. Voltage plot
        fig_survival.add_trace(
            go.Scatter(
                x=hist_df["timestamp"], y=hist_df["base_v"],
                name="Baseline Voltage", line=dict(color="#EF553B", width=1.5, dash="dash")
            ),
            row=2, col=1,
        )
        fig_survival.add_trace(
            go.Scatter(
                x=hist_df["timestamp"], y=hist_df["twin_v"],
                name="Twin Voltage", line=dict(color="#00CC96", width=2)
            ),
            row=2, col=1,
        )
        fig_survival.add_hline(y=42.0, line=dict(color="#AB63FA", dash="dot"), annotation_text="BLVD Trip (42.0 V)", row=2, col=1)

        # 3. Service Tier plot
        fig_survival.add_trace(
            go.Scatter(
                x=hist_df["timestamp"], y=hist_df["base_tier"],
                name="Baseline Tier", line=dict(color="#EF553B", width=1.5, dash="dash")
            ),
            row=3, col=1,
        )
        fig_survival.add_trace(
            go.Scatter(
                x=hist_df["timestamp"], y=hist_df["twin_tier"],
                name="Twin Tier", line=dict(color="#636EFA", width=2)
            ),
            row=3, col=1,
        )

        fig_survival.update_layout(
            height=750,
            hovermode="x unified",
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_survival, width="stretch")

    # -------------------------------------------------------------------------
    # TAB 2: Power Breakdown & Rectifier Losses
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("AC Mains & Telecom Rectifier Energy Accounting")
        st.caption(
            f"Rectifier Efficiency η = {RECTIFIER_EFF:.2f}. AC power includes site DC load plus battery recharge "
            f"divided by 0.96. Carrier wake-up signaling tax adds {WAKE_TAX_KW*1000:.0f} W (+10%) for 5 minutes post-wake."
        )

        c1, c2 = st.columns([1, 1])

        with c1:
            base_dc = metrics["baseline_grid_dc_kwh"]
            twin_dc = metrics["twin_grid_dc_kwh"]
            base_rech = metrics["baseline_recharge_kwh"]
            twin_rech = metrics["twin_recharge_kwh"]
            wake_tax = metrics["twin_wake_tax_kwh"]
            twin_useful_dc = max(0.0, twin_dc - wake_tax)
            base_rec_loss = max(0.0, metrics["baseline_ac_kwh"] - (base_dc + base_rech))
            twin_rec_loss = max(0.0, metrics["twin_ac_kwh"] - (twin_dc + twin_rech))

            categories = ["Baseline", "Digital Twin"]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(name="Useful DC Load", x=categories, y=[base_dc, twin_useful_dc], marker_color="#636EFA"))
            fig_bar.add_trace(go.Bar(name="Carrier Wake Tax (+10%)", x=categories, y=[0.0, wake_tax], marker_color="#EF553B"))
            fig_bar.add_trace(go.Bar(name="Battery Recharge", x=categories, y=[base_rech, twin_rech], marker_color="#00CC96"))
            fig_bar.add_trace(go.Bar(name="Rectifier Loss (4%)", x=categories, y=[base_rec_loss, twin_rec_loss], marker_color="#FFA15A"))

            fig_bar.update_layout(
                barmode="stack",
                title="Total AC Mains Energy Consumption Breakdown (kWh)",
                yaxis_title="Energy (kWh)",
                height=420,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig_bar, width="stretch")

        with c2:
            sample_plot = hist_df.iloc[:288 * 3]  # first 3 days
            fig_power = go.Figure()
            fig_power.add_trace(
                go.Scatter(x=sample_plot["timestamp"], y=sample_plot["p_base_ac"], name="Baseline AC Power (kW)", line=dict(color="#EF553B", dash="dash"))
            )
            fig_power.add_trace(
                go.Scatter(x=sample_plot["timestamp"], y=sample_plot["p_twin_ac"], name="Twin AC Power (kW)", line=dict(color="#00CC96", width=2))
            )
            fig_power.add_trace(
                go.Scatter(x=sample_plot["timestamp"], y=sample_plot["actual_demand"], name="Demand Load (CU)", line=dict(color="#AB63FA", width=1.5))
            )
            fig_power.update_layout(
                title="Instantaneous AC Mains Draw vs Traffic Demand (3-Day Sample)",
                yaxis_title="Power (kW) / Traffic (CU)",
                hovermode="x unified",
                height=420,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig_power, width="stretch")

    # -------------------------------------------------------------------------
    # TAB 3: Shelter Thermal & Battery Degradation
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Coupled Shelter Thermal Model & Battery Temperature Dynamics")
        st.caption(
            "Two-node physical model: When grid fails, HVAC drops offline. Remote Radio Units reject heat outdoors on the mast, "
            "while in-shelter equipment (BBU, transport, aux DC) and battery discharge losses accumulate inside the shelter. "
            "Battery temperature tracks shelter air with a ~3.7h thermal lag due to its 240 kJ/K thermal mass. "
            "Arrhenius degradation accelerates exponentially (doubling every 8°C above 25°C)."
        )

        fig_thermal = go.Figure()
        fig_thermal.add_trace(
            go.Scatter(x=hist_df["timestamp"], y=hist_df["t_shelter_base"], name="Baseline Shelter Air (°C)", line=dict(color="#FFA15A", dash="dash"))
        )
        fig_thermal.add_trace(
            go.Scatter(x=hist_df["timestamp"], y=hist_df["t_shelter_twin"], name="Twin Shelter Air (°C)", line=dict(color="#FFA15A", width=2))
        )
        fig_thermal.add_trace(
            go.Scatter(x=hist_df["timestamp"], y=hist_df["t_batt_base"], name="Baseline Battery Temp (°C)", line=dict(color="#EF553B", dash="dash"))
        )
        fig_thermal.add_trace(
            go.Scatter(x=hist_df["timestamp"], y=hist_df["t_batt_twin"], name="Twin Battery Temp (°C)", line=dict(color="#00CC96", width=2.5))
        )

        fig_thermal.add_hline(y=T_WARN, line=dict(color="#FFA15A", dash="dot"), annotation_text="IEEE 484 Derating Warning (40°C)")
        fig_thermal.add_hline(y=T_CRIT, line=dict(color="#EF553B", dash="dot"), annotation_text="Critical Thermal Shutdown (50°C)")

        fig_thermal.update_layout(
            title="Shelter Air & Battery Temperature Trajectory (°C)",
            yaxis_title="Temperature (°C)",
            hovermode="x unified",
            height=450,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_thermal, width="stretch")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.info(
                f"**Arrhenius Calendar Aging (IEEE 484):**  \n"
                f"Baseline damage accrued: **{metrics['baseline_cal_damage']*1e4:.2f} bp**  \n"
                f"Digital Twin damage accrued: **{metrics['twin_cal_damage']*1e4:.2f} bp**  \n"
                f"Thermal Warning steps (T_batt ≥ 40°C): **{metrics['thermal_warn_steps']}** steps"
            )
        with col_t2:
            st.warning(
                f"**Thermal Safety Protection Rules:**  \n"
                f"• **T ≥ 40°C**: Recharge current halved (0.5× I_lim) to prevent float thermal runaway.  \n"
                f"• **T ≥ 50°C**: Safety Arbiter clamps Upper Bound UB = 1 (SURVIVAL), de-energizing all LTE carriers."
            )

    # -------------------------------------------------------------------------
    # TAB 4: Safety Arbiter Diagnostics & Bargh-e Man Logs
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("Safety Arbiter Decision Log & Bargh-e Man (SAAPA) Inspector")

        sub1, sub2 = st.columns([1.2, 0.8])

        with sub1:
            st.write("##### Safety Arbiter Interventions (Priority Lattice Rule Triggers)")
            interventions = hist_df[hist_df["shield_intervention"] | (hist_df["shield_why"] != "")].copy()
            if len(interventions) > 0:
                show_cols = ["timestamp", "grid_status", "actual_demand", "twin_tier", "shield_why"]
                st.dataframe(
                    interventions[show_cols].head(100),
                    width="stretch",
                    height=350,
                )
            else:
                st.success("No shield interventions required in this window (all planner proposals satisfied bounds).")

        with sub2:
            st.write("##### Bargh-e Man Scheduled Blackout Windows (SAAPA)")
            client = BargheManClient(mode=client_mode)
            if client_mode == "mock":
                client.load_mock_from_dataframe(hist_df)
            outages = client.fetch_planned_outages(site_id)
            if outages:
                st.json(outages[:3])
                st.caption(f"Showing 3 of {len(outages)} scheduled timetable windows for {site_id} (mode={client_mode}).")
            else:
                st.info(f"No scheduled timetable blackouts found for {site_id}.")

    st.caption("Developed for Cellular Network Energy Digital Twin v3.3 · Compliant with 3GPP TR 36.887 & IEEE 484 standards.")


if __name__ == "__main__":
    main()
