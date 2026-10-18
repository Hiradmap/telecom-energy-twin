# Cellular Network Energy & Outage Survival Digital Twin (v3.3)

**O-RAN-style energy saving, receding-horizon outage survival, and formal safety arbitration for macro cellular networks.**

## 🎯 Value Proposition & Measured Results

Many macro cellular sites face daily scheduled power cuts (2–4 hours) on aged VRLA battery banks. Running full multi-carrier LTE capacity during an outage rapidly drains degraded batteries to the Baseband Low-Voltage Disconnect (BLVD, 42.0 V), causing complete site blackouts. Conversely, myopic reactive carrier sleep during normal grid operation triggers severe QoS congestion when carriers require 5–15 minutes to wake up.

This Digital Twin acts as a closed-loop software controller with three verifiable guarantees, all reproducible via `.venv/bin/python -m src.benchmark`:

1. **Outage Survival (Grid DOWN):** A deterministic **Receding-Horizon Reserve Planner** coupled to published *Bargh-e Man* (SAAPA) schedules sheds capacity carriers early on degraded batteries to keep the anchor/voice layer alive:
   - **100% blackout survival** across 2-hour and 3-hour cuts down to **40% SOH** (vs. **48.2%** and **34.8%** for the naive baseline — **+106% to +187% autonomy**).
   - **+34.3% autonomy** on 60% SOH batteries (2-hour cuts) and **+92.6% autonomy** on the held-out `SITE_1` test window (43.5% SOH).
   - Explicitly tracks the broadband vs. voice survival trade-off (`served_cuh` vs. `any-service hours`), never claiming free energy on worn batteries.
2. **Low-Violation Energy Trimming (Grid UP):** When carrier wake-up takes **15 minutes**, a myopic reactive controller causes **694 QoS violation steps** across 9 unseen site-runs, whereas **Forecast + Conformal + Safety Shield** reduces violations by **15.4×** (to 45 total, vs. 5 physical burst violations when always on) while saving **3.2%–3.6% total AC mains energy** (including 0.96 rectifier efficiency, battery recharge draw, and +10% wake signaling tax).
3. **Shift-Resilient Conformal Calibration:** Wrapping the Quantile LightGBM (`q95`) forecaster in a causal **Rolling Split-Conformal Calibrator** (`src/conformal.py`) with 4-hour circular time-of-day buckets restores empirical coverage under severe distribution shifts:
   - **Ramadan-like +30% evening surge:** restores evening coverage from **76.9% → 96.4%** (overall **89.0% → 95.9%**).
   - **Global +30% load jump:** restores coverage from **74.0% → 95.3%**.
   - **High noise (+0.25 CU):** restores coverage from **60.5% → 94.6%**.

---

## 🏗️ Architecture

The system separates statistical forecasting from safety-critical hardware control through a four-stage pipeline:

1. **Forecaster & Conformal Calibrator (`src/forecaster.py`, `src/conformal.py`):** A Quantile LightGBM model predicts the next-60-minute organic peak load (`q95`) strictly from exogenous time and historical demand features (preventing *Observer Effect* causal leakage). A causal rolling split-conformal wrapper (`OnlineConformal` / `rolling_conformal_margin`) widens the bound per time-of-day regime and monitors raw coverage via a drift alarm (`drift_status`).
2. **Receding-Horizon Outage Planner (`tests/reference/twin_ref.py`, `src/grid_client.py`):** Queries scheduled blackout windows from the dual-mode *Bargh-e Man* (SAAPA) client (`mock` and `live` modes) without future data peeking, reserving aged battery capacity (`Ah_nom × SOH`) above the hard floor (15% SOC) for the `SURVIVAL` tier before allocating surplus energy above the soft floor (35% SOC) to LTE data tiers.
3. **Formal Safety Arbiter (`tests/reference/twin_ref.py`, `src/controller.py`):** A deterministic **Priority Lattice** (`final = clamp(proposal, LB, UB)` where Safety `UB` strictly overrides QoS `LB`). Enforces:
   - **Upper Bounds (Safety):** Thermal critical shutdown (`T_batt >= 50°C → UB = 1`), battery voltage/SOC backstop (`V < 45V` or `eff_soc < 15% → UB = 1`), dead-man's switch on battery (`lease_loss → UB = 1`), and 15-minute debounced backhaul fate-sharing (`backhaul_down → UB = 2`, shedding capacity carriers without power-cycling the anchor/voice layers).
   - **Lower Bounds (Service):** Anchor floor on grid (`LB = 2`), dead-man's switch on grid (`lease_loss → LB = 4`), forecast/emergency wake overrides, 15-minute dwell hysteresis, daily carrier fatigue budget (4 sleeps/day), fail-static hold on stale telemetry (`> 15 min`), and `max(bound, measured)` capacity clamping.
4. **Vendor Translation Layer (`src/translation_layer.py`):** Pure, dry-run translator mapping arbiter tier transitions (`0..4`) into ordered **Huawei MML** (`BLK/DEA/ACT/UBL CELL`, `BLK/UBL GTRX`), **Ericsson ENM CLI** (`cmedit` `EUtranCellFDD` + GSM `RXBLI/RXESE`), and **O-RAN O1 NETCONF XML** (`<edit-config>`) commands with read-back verification, rollback generation, and strict regex/XML injection hardening.

---

## 📊 Reproducible Benchmark Summary

Run `.venv/bin/python -m src.benchmark` to reproduce all three out-of-sample evaluation tables:

### 1. Grid-UP Policy Ablation (9 Unseen Site-Runs × 21 Days)
| Wake Latency | Policy | AC Energy Saved (%) | Total QoS Violations | Mean Violations / Run | Carrier Transitions / Run |
|---:|:---|---:|---:|---:|---:|
| **0 min** | `none` (Always FULL) | 0.00 ± 0.00% | 5 | 0.6 ± 0.7 | 0.0 |
| **0 min** | `reactive` (No Forecast) | 5.29 ± 0.40% | 5 | 0.6 ± 0.7 | 660.0 |
| **0 min** | `shield_conformal` | **3.39 ± 0.73%** | **5** | **0.6 ± 0.7** | **145.4** |
| **5 min** | `reactive` (No Forecast) | 5.02 ± 0.43% | 225 | 25.0 ± 4.6 | 660.0 |
| **5 min** | `shield_conformal` | **3.34 ± 0.73%** | **20** | **2.2 ± 1.4** | **145.4** |
| **15 min** | `timer` (Static Night) | 2.13 ± 0.03% | 151 | 16.8 ± 10.5 | 42.0 |
| **15 min** | `reactive` (No Forecast) | 4.76 ± 0.48% | 694 | 77.1 ± 11.8 | 660.0 |
| **15 min** | `shield` (Forecast + Arbiter) | **3.41 ± 0.75%** | **51** | **5.7 ± 2.9** | **147.8** |
| **15 min** | `shield_conformal` | **3.23 ± 0.74%** | **45** | **5.0 ± 3.0** | **145.4** |

### 2. Outage Survival vs. Battery SOH (9 Unseen Site-Runs)
| Cut Duration | Battery SOH | Baseline Survival (%) | Twin Survival (%) | Autonomy Gain (%) | Baseline Served (CU·h) | Twin Served (CU·h) |
|---:|---:|---:|---:|---:|---:|---:|
| **2.0 h** | **100%** | 100.0% | **100.0%** | +0.0% | 30.0 | 27.4 |
| **2.0 h** | **60%** | 74.3% | **100.0%** | **+34.3%** | 21.5 | 12.9 |
| **2.0 h** | **40%** | 48.2% | **100.0%** | **+106.4%** | 13.8 | 5.8 |
| **2.0 h** | **25%** | 29.0% | **100.0%** | **+242.0%** | 8.2 | 0.0 *(2G voice)* |
| **4.0 h** | **60%** | 41.7% | **100.0%** | **+139.4%** | 21.4 | 5.8 |
| **4.0 h** | **40%** | 27.1% | **100.0%** | **+267.7%** | 13.8 | 0.0 *(2G voice)* |
| **4.0 h** | **25%** | 16.3% | **76.1%** | **+363.5%** | 8.2 | 0.0 *(2G voice)* |

---

## 📂 Repository Structure

```text
.
├── docs/                           # Master architecture blueprints & roadmaps
│   ├── Architecture_v3.3.txt       # Engineering specification & parameter freeze
│   └── Production_Roadmap.md       # Telco-grade scale-out & safety roadmap
├── src/                            # Core implementation
│   ├── controller.py               # Closed-loop simulator, fault drills & physics loop
│   ├── forecaster.py               # Quantile LightGBM (q95) training & evaluation
│   ├── conformal.py                # Causal rolling split-conformal calibrator & drift alarm
│   ├── carriers.py                 # LTE carrier state machine (wake latency & signaling tax)
│   ├── translation_layer.py        # Huawei MML, Ericsson CLI & O-RAN NETCONF dry-run translator
│   ├── benchmark.py                # Multi-seed out-of-sample benchmark suite
│   ├── grid_client.py              # Dual-mode Bargh-e Man (SAAPA) outage schedule client
│   ├── thermal.py                  # Coupled two-node shelter & battery thermal model
│   ├── data_generator.py           # Tavanir 2-hr timetable & multi-archetype generator
│   └── dashboard.py                # Interactive Streamlit dashboard with Vendor Command Preview
├── tests/                          # Verification & property test suite (517 tests)
│   ├── test_physics_controller.py  # Physics, carrier state machine & failure-drill tests
│   ├── test_conformal.py           # Causal conformal calibration & drift alarm tests
│   ├── test_translation_layer.py   # Exhaustive vendor translation & injection hardening tests
│   ├── test_benchmark.py           # Out-of-sample benchmark invariant tests
│   ├── test_thermal.py             # Coupled shelter/battery thermal & IEEE derating tests
│   ├── test_grid_client.py         # Bargh-e Man SAAPA client & caching tests
│   ├── test_dataset.py             # Dataset consistency & archetype verification
│   ├── test_dashboard.py           # Headless Streamlit AppTest & preset sync verification
│   ├── test_invariants.py          # Hypothesis property tests (2,000 randomized arbiter states)
│   ├── arbiter_invariants.py       # Formal priority-lattice invariant specification
│   ├── verify_twin.py              # Standalone multi-stage twin verifier
│   └── reference/twin_ref.py       # Reference power, Peukert battery, arbiter & planner models
├── data/                           # Synthetic telemetry dataset (90 days, 3 archetypes)
├── models/                         # Serialized Quantile LightGBM booster (lgbm_q95_forecaster.txt)
├── archive/                        # Archived historical specification drafts
├── requirements.txt                # Top-level Python dependencies
└── requirements.lock.txt           # Locked reproducible environment dependencies
```

---

## 🚀 Quickstart Commands

All commands run offline using the local virtual environment (`.venv`):

```bash
# 1. Run the 30-day out-of-sample digital twin simulation
.venv/bin/python src/controller.py

# 2. Run the full multi-seed reproducible benchmark suite
.venv/bin/python -m src.benchmark

# 3. Launch the interactive Streamlit visual dashboard
.venv/bin/streamlit run src/dashboard.py

# 4. Run the complete automated test suite (517 unit, property & headless UI tests)
.venv/bin/pytest tests -q
```
