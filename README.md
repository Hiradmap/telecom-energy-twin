# Cellular Network Energy & Outage Survival MVP

**Digital-Twin PoC: O-RAN-style energy saving + outage survival for legacy macro networks.**

## 🎯 Value Proposition

Many macro sites face daily scheduled power cuts (1.5–4 hours) and sit on aged battery banks. Currently, sites run full capacity until batteries die, causing complete network blackouts. 

This Digital Twin MVP acts as a software-defined controller that:
1. **Trims Energy (Grid Up):** Uses predictive ML to safely put capacity carriers to sleep during low-demand periods, without hurting QoS.
2. **Survives Blackouts (Grid Down):** Employs a deterministic Receding-Horizon Reserve Planner to degrade capacity gracefully. By shedding data layers, we stretch marginal batteries to survive the entire blackout, guaranteeing that the **Anchor Voice/SMS** layer never goes dark.

**Headline Result:** Shedding capacity trades broadband throughput to extend "any-service" survival hours by ~40%, rescuing degraded batteries from complete failure.

---

## 🏗️ Architecture

The MVP completely removes speculative "AI control" from safety-critical hardware decisions. It relies on a three-stage pipeline:

1. **Forecaster (The Brain):** A Quantile LightGBM model predicts the next-60-minute organic load (q95). It is strictly protected from "Observer Effect" data leakage (it never sees hardware state or battery levels).
2. **Safety Arbiter (The Shield):** A formal mathematical **Priority Lattice** (`UB/LB clamp`). It evaluates all recommendations. Upper Bounds (Safety constraints like thermal alarms or battery voltage) absolutely override Lower Bounds (Service constraints).
3. **Outage Planner:** A deterministic **Receding-Horizon Reserve Policy** that uses published blackout schedules (*Bargh-e Man*) to reserve a hard energy budget for the SURVIVAL tier.

---

## 📂 Repository Structure

```text
.
├── docs/                      # Master architecture reports & blueprints
│   ├── Architecture_v3.3.txt  # Implementation-ready engineering blueprint
│   └── Production_Roadmap.md  # Telco-grade production scale-out architecture
├── src/                       # Core implementation
│   ├── controller.py          # Main digital-twin event loop & physics engine
│   ├── forecaster.py          # Quantile LightGBM forecaster (q95)
│   ├── grid_client.py         # Dual-mode Bargh-e Man (SAAPA) adapter
│   ├── thermal.py             # Coupled shelter & battery thermal dynamics
│   ├── data_generator.py      # Tavanir 2-hr timetable & archetype data generator
│   └── dashboard.py           # Interactive Streamlit UI dashboard
├── tests/                     # Comprehensive verification suites (61 tests)
│   ├── test_physics_controller.py # Controller accuracy & physics tests
│   ├── test_thermal.py        # Shelter & battery thermal dynamics tests
│   ├── test_grid_client.py    # Bargh-e Man SAAPA client & schedule tests
│   ├── test_dataset.py        # Dataset consistency & archetype checks
│   ├── test_dashboard.py      # Streamlit scenario & accounting tests
│   ├── test_invariants.py     # Hypothesis property tests across 2,000 states
│   ├── arbiter_invariants.py  # Priority lattice invariant checker
│   ├── verify_twin.py         # Multi-stage verification script
│   └── reference/twin_ref.py  # Reference mathematical physics engine
├── data/                      # Synthetic telemetry data (90 days, 3 archetypes)
├── models/                    # Serialized Quantile LightGBM booster weights
├── requirements.txt           # Top-level Python dependencies
└── README.md                  # System overview and quickstart guide
```

---

## 🚀 Quickstart & One-Line Commands

All commands run directly using the pre-configured local virtual environment (`.venv`):

### 1. Run the Physical Digital Twin Simulation
Executes the physical simulation loop against out-of-sample data with 0.96 rectifier efficiency, wake latency, and thermal tracking:
```bash
.venv/bin/python src/controller.py
```

### 2. Launch the Interactive Visual Dashboard
Launches the full Streamlit UI with scenario sliders for Battery SOH (25%–100%), Outage Duration (1–4h), Carrier Wake Latency (0, 5, 15 min), and interactive Plotly curves:
```bash
.venv/bin/streamlit run src/dashboard.py
```

### 3. Run the Comprehensive Verification Test Suite
Runs all 61 automated unit, invariant, and property tests:
```bash
.venv/bin/pytest tests -v
```

### 4. Regenerate Synthetic Telecom Data
Regenerates realistic 5-minute telemetry with 2-hour Tavanir timetable blocks and 3 distinct site archetypes:
```bash
.venv/bin/python src/data_generator.py --days 90 --sites 3
```

---

## ⚙️ Environment Setup

This project is built for **Python 3.14** and is **fully offline-capable** to guarantee zero reliance on external network calls during evaluation or competition judging.

### Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
```

