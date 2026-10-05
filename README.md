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
├── docs/               # Master architecture reports & documentation
│   └── Architecture_v3.3.txt  # The implementation-ready blueprint
├── src/                # Core implementation (Forecaster, Arbiter, Loop) - TO BE BUILT
├── tests/              # Verification suites and property tests
│   ├── verify_twin.py  # Validation script for physics & invariant tests
│   └── reference/      # Mathematical core (twin_ref.py)
├── data/               # Synthetic datasets & logs
├── archive/            # Previous iterations and draft files
├── requirements.txt    # Top-level Python dependencies
├── wheelhouse/         # Offline package mirror
└── README.md           # You are here
```

---

## ⚙️ Environment Setup

This project is built for **Python 3.14** and is **fully offline-capable** to protect against connectivity loss during development.

### Setup (Online)
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
```

### Setup (Offline, from the bundled mirror)
```bash
python3 -m venv .venv
.venv/bin/pip install --no-index --find-links wheelhouse -r requirements.lock.txt
```

*Note: Always run code using `.venv/bin/python` or by activating the environment (`source .venv/bin/activate`).*
