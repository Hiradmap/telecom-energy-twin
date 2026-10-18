"""
benchmark.py - Reproducible multi-seed out-of-sample evaluation suite for the Telecom Energy Digital Twin.

Every quantitative claim in README.md and docs/Architecture_v3.3.txt is produced by running:
    .venv/bin/python -m src.benchmark

Evaluates three dimensions on genuinely unseen out-of-sample data (fresh random seeds never seen during
LightGBM training):
  1. Grid-UP Energy Saving & QoS Violations across policies (none, timer, reactive, shield, shield_conformal)
     and carrier wake latencies (0, 5, 15 min).
  2. Outage Survival & Served Traffic across battery SOH (100%, 60%, 40%, 25%) and blackout durations (2h, 3h, 4h).
  3. Forecaster Coverage under Distribution Shift (in-distribution, Ramadan-like +30% evening surge,
     +30% global level shift, +0.25 noise) before and after rolling split-conformal calibration.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.conformal import ConformalConfig, coverage_by_regime, rolling_conformal_margin
from src.controller import MODEL_PATH, simulate_site
from src.data_generator import generate_telecom_data
from src.forecaster import engineer_features


def _ci95(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))


def _prepare_oos_sites(days: int = 30, seeds: Sequence[int] = (101, 202, 303), num_sites: int = 3):
    """Generate fresh out-of-sample site DataFrames with pre-computed features, raw predictions, and conformal margins."""
    booster = lgb.Booster(model_file=MODEL_PATH)
    cfg = ConformalConfig()
    runs = []
    # Generate extra 10 warm-up days so 7-day lag features and conformal residual pools are warm
    gen_days = days + 15
    for seed in seeds:
        raw_df = generate_telecom_data(days=gen_days, num_sites=num_sites, seed=seed)
        feat_df, feature_cols = engineer_features(raw_df)
        for site_id, g in feat_df.groupby("site_id"):
            g = g.sort_values("timestamp").reset_index(drop=True)
            pred_full = booster.predict(g[feature_cols])
            marg_full = rolling_conformal_margin(
                g["target_peak_60m"].values, pred_full, g["hour"].values, cfg
            )
            # Slice the last `days` as the evaluation window (after conformal pool warmup)
            eval_steps = min(len(g), days * 288)
            run_id = f"S{seed}_{site_id}"
            g_eval = g.iloc[-eval_steps:].copy().reset_index(drop=True)
            g_eval["site_id"] = run_id
            p_eval = pred_full[-eval_steps:]
            m_eval = marg_full[-eval_steps:]
            runs.append((run_id, g_eval, p_eval, m_eval))
    return runs


def benchmark_grid_policies(
    days: int = 21,
    seeds: Sequence[int] = (101, 202, 303),
    latencies_min: Sequence[float] = (0.0, 5.0, 15.0),
    runs: Sequence[tuple[str, pd.DataFrame, np.ndarray, np.ndarray]] | None = None,
) -> pd.DataFrame:
    """Compare grid-UP energy saving policies across wake latencies on unseen site-runs."""
    if runs is None:
        runs = _prepare_oos_sites(days=days, seeds=seeds)
    policies = ("none", "timer", "reactive", "shield", "shield_conformal")
    rows: list[dict[str, Any]] = []
    empty_client = type("_NoOutageClient", (), {"build_schedule_map": lambda *a, **k: {}})()

    for lat in latencies_min:
        per_pol: dict[str, list[dict[str, float]]] = {p: [] for p in policies}
        for run_id, site_df, pred, marg in runs:
            # Evaluate pure grid-UP behaviour by setting grid_status=1 on a copy
            g_up = site_df.copy()
            g_up["grid_status"] = 1
            g_up["blackout_schedule_flag"] = 0
            g_up["unscheduled_fault_flag"] = 0

            m_none = simulate_site(
                g_up, pred, site_id=run_id, wake_latency_min=lat, policy="none", grid_client=empty_client
            )
            base_ac = m_none["twin_ac_kwh"]

            for pol in policies:
                if pol == "none":
                    m = m_none
                elif pol == "shield_conformal":
                    m = simulate_site(
                        g_up, pred, site_id=run_id, wake_latency_min=lat,
                        policy="shield", bound_margin=marg, grid_client=empty_client,
                    )
                else:
                    m = simulate_site(
                        g_up, pred, site_id=run_id, wake_latency_min=lat, policy=pol, grid_client=empty_client
                    )

                saved_pct = 100.0 * (base_ac - m["twin_ac_kwh"]) / base_ac if base_ac > 0 else 0.0
                per_pol[pol].append({
                    "saved_pct": saved_pct,
                    "viol": float(m["twin_viol_grid"]),
                    "dropped_cuh": float(m["twin_dropped_grid_cuh"]),
                    "transitions": float(m["twin_transitions"]),
                })

        for pol in policies:
            sav_m, sav_ci = _ci95([x["saved_pct"] for x in per_pol[pol]])
            vio_m, vio_ci = _ci95([x["viol"] for x in per_pol[pol]])
            vio_tot = int(sum(x["viol"] for x in per_pol[pol]))
            drp_m, _ = _ci95([x["dropped_cuh"] for x in per_pol[pol]])
            trn_m, _ = _ci95([x["transitions"] for x in per_pol[pol]])
            rows.append({
                "wake_latency_min": lat,
                "policy": pol,
                "n_site_runs": len(runs),
                "saved_pct_mean": round(sav_m, 2),
                "saved_pct_ci95": round(sav_ci, 2),
                "viol_total": vio_tot,
                "viol_per_run_mean": round(vio_m, 1),
                "viol_per_run_ci95": round(vio_ci, 1),
                "dropped_cuh_mean": round(drp_m, 2),
                "transitions_mean": round(trn_m, 1),
            })

    return pd.DataFrame(rows)


def benchmark_outage_survival(
    days: int = 21,
    seeds: Sequence[int] = (101, 202),
    soh_levels: Sequence[float] = (1.0, 0.60, 0.40, 0.25),
    outage_durations_h: Sequence[float] = (2.0, 3.0, 4.0),
    runs: Sequence[tuple[str, pd.DataFrame, np.ndarray, np.ndarray]] | None = None,
) -> pd.DataFrame:
    """Compare Baseline vs Digital Twin outage autonomy and served traffic across SOH and blackout lengths."""
    if runs is None:
        runs = _prepare_oos_sites(days=days, seeds=seeds)
    rows: list[dict[str, Any]] = []

    for dur_h in outage_durations_h:
        target_steps = max(1, int(round(float(dur_h) * 12)))
        for soh in soh_levels:
            base_surv_pcts, twin_surv_pcts = [], []
            base_srv_h, twin_srv_h, out_tot_h = [], [], []
            base_cuh_list, twin_cuh_list, dem_cuh_list = [], [], []

            for run_id, site_df, pred, marg in runs:
                df_mod = site_df.copy()
                is_sched = df_mod["blackout_schedule_flag"].values
                starts = np.where((is_sched == 1) & (np.pad(is_sched[:-1], (1, 0), constant_values=0) == 0))[0]
                for s_idx in starts:
                    orig_end = s_idx
                    while orig_end < len(df_mod) and df_mod.at[orig_end, "blackout_schedule_flag"] == 1:
                        orig_end += 1
                    new_end = min(s_idx + target_steps, len(df_mod))
                    df_mod.loc[s_idx:new_end - 1, "grid_status"] = 0
                    df_mod.loc[s_idx:new_end - 1, "blackout_schedule_flag"] = 1
                    df_mod.loc[s_idx:new_end - 1, "unscheduled_fault_flag"] = 0
                    if new_end < orig_end:
                        df_mod.loc[new_end:orig_end - 1, "grid_status"] = 1
                        df_mod.loc[new_end:orig_end - 1, "blackout_schedule_flag"] = 0

                m = simulate_site(
                    df_mod, pred, site_id=run_id, wake_latency_min=5.0, initial_soh=soh, bound_margin=marg
                )
                out_h = m["outage_steps"] * 5.0 / 60.0
                b_h = m["baseline_any_srv_steps"] * 5.0 / 60.0
                t_h = m["twin_any_srv_steps"] * 5.0 / 60.0
                if out_h > 0:
                    base_surv_pcts.append(100.0 * b_h / out_h)
                    twin_surv_pcts.append(100.0 * t_h / out_h)
                    base_srv_h.append(b_h)
                    twin_srv_h.append(t_h)
                    out_tot_h.append(out_h)
                    base_cuh_list.append(m["base_served_outage_cuh"])
                    twin_cuh_list.append(m["twin_served_outage_cuh"])
                    dem_cuh_list.append(m["outage_demand_cuh"])

            b_pct_m, _ = _ci95(base_surv_pcts)
            t_pct_m, _ = _ci95(twin_surv_pcts)
            b_h_m = float(np.mean(base_srv_h))
            t_h_m = float(np.mean(twin_srv_h))
            out_h_m = float(np.mean(out_tot_h))
            gain_pct = 100.0 * (t_h_m - b_h_m) / b_h_m if b_h_m > 0 else 0.0

            rows.append({
                "outage_duration_h": dur_h,
                "soh": soh,
                "outage_hours_mean": round(out_h_m, 1),
                "base_any_srv_h": round(b_h_m, 1),
                "twin_any_srv_h": round(t_h_m, 1),
                "base_survival_pct": round(b_pct_m, 1),
                "twin_survival_pct": round(t_pct_m, 1),
                "autonomy_gain_pct": round(gain_pct, 1),
                "base_served_cuh": round(float(np.mean(base_cuh_list)), 1),
                "twin_served_cuh": round(float(np.mean(twin_cuh_list)), 1),
                "demanded_cuh": round(float(np.mean(dem_cuh_list)), 1),
            })

    return pd.DataFrame(rows)


def benchmark_conformal_shift(days: int = 20, seed: int = 2024, num_sites: int = 4) -> pd.DataFrame:
    """Evaluate raw LightGBM vs rolling split-conformal coverage under distribution shifts."""
    booster = lgb.Booster(model_file=MODEL_PATH)
    base_raw = generate_telecom_data(days=days + 15, num_sites=num_sites, seed=seed)
    cfg = ConformalConfig()

    def _ramadan(d: pd.DataFrame):
        d.loc[d["hour"].between(19, 24), "organic_load_cu"] *= 1.30
        d.loc[d["hour"] < 3, "organic_load_cu"] *= 1.20

    def _level30(d: pd.DataFrame):
        d["organic_load_cu"] *= 1.30

    def _noisy(d: pd.DataFrame):
        d["organic_load_cu"] += np.random.default_rng(seed + 7).normal(0, 0.25, len(d))

    regimes_spec = [
        ("In-distribution (OOS)", lambda d: None),
        ("Ramadan evening (+30%)", _ramadan),
        ("Global level (+30%)", _level30),
        ("High noise (+0.25 CU)", _noisy),
    ]

    rows: list[dict[str, Any]] = []
    for label, transform in regimes_spec:
        d = base_raw.copy()
        transform(d)
        d["organic_load_cu"] = d["organic_load_cu"].clip(0.05, 3.5)
        feat_df, fcols = engineer_features(d)
        feat_df = feat_df.reset_index(drop=True)

        ys, raw_b, cal_b, hs = [], [], [], []
        for _, g in feat_df.groupby("site_id"):
            g = g.sort_values("timestamp").reset_index(drop=True)
            p_g = booster.predict(g[fcols])
            m_g = rolling_conformal_margin(g["target_peak_60m"].values, p_g, g["hour"].values, cfg)
            # Evaluate after 8-day warmup so rolling 7-day pool is populated
            sl = slice(8 * 288, None)
            ys.append(g["target_peak_60m"].values[sl])
            raw_b.append(p_g[sl])
            cal_b.append((p_g + m_g)[sl])
            hs.append(g["hour"].values[sl])

        y_all = np.concatenate(ys)
        r_all = np.concatenate(raw_b)
        c_all = np.concatenate(cal_b)
        h_all = np.concatenate(hs)

        raw_reg = coverage_by_regime(y_all, r_all, h_all)
        cal_reg = coverage_by_regime(y_all, c_all, h_all)
        rows.append({
            "scenario": label,
            "raw_overall_cov": round(raw_reg["overall"]["coverage"], 3),
            "cal_overall_cov": round(cal_reg["overall"]["coverage"], 3),
            "raw_evening_cov": round(raw_reg["evening 19-24h"]["coverage"], 3),
            "cal_evening_cov": round(cal_reg["evening 19-24h"]["coverage"], 3),
            "raw_mean_bound": round(raw_reg["overall"]["mean_bound"], 3),
            "cal_mean_bound": round(cal_reg["overall"]["mean_bound"], 3),
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print(" TELECOM ENERGY DIGITAL TWIN v3.3 — REPRODUCIBLE OUT-OF-SAMPLE BENCHMARK")
    print("=" * 78)

    runs = _prepare_oos_sites(days=21, seeds=(101, 202, 303))

    print("\n[1/3] Grid-UP Policy Ablation (3 unseen seeds x 3 archetypes = 9 site-runs, 21 days each)")
    df_grid = benchmark_grid_policies(runs=runs)
    print(df_grid.to_string(index=False))

    print("\n[2/3] Outage Survival & Served Traffic vs Battery SOH and Cut Duration (9 unseen site-runs)")
    df_out = benchmark_outage_survival(runs=runs)
    print(df_out.to_string(index=False))

    print("\n[3/3] Forecaster Coverage Under Distribution Shift (Raw q95 vs Rolling Split-Conformal)")
    df_conf = benchmark_conformal_shift()
    print(df_conf.to_string(index=False))
    print("=" * 78)


if __name__ == "__main__":
    main()
