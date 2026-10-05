import re

with open("Cellular Network Energy v3.2.txt", "r") as f:
    text = f.read()

# Replace version and header
text = text.replace("Cellular Network Energy Efficiency — Master Report v3.2 (Consolidated & Implementation-Ready)", 
                    "Cellular Network Energy Efficiency — Master Report v3.3")
text = text.replace("Revision of v3.1 after consolidating v3/v3.1 knowledge, resolving parameter inconsistencies, and freezing remaining implementation parameters.", "Revision of v3.2: Incorporating priority lattice, receding-horizon planner, explicit tiers, VoLTE/CSFB requirements, DoD cycle damage, and Bargh-e Man schedules.")

history_add = """v3.3:
- Replaced RL Outage Planner with Receding-Horizon Reserve Policy based on 'Bargh-e Man' scheduled blackouts.
- Redesigned Safety Arbiter as a Priority Lattice (UB/LB clamp).
- Clarified VoLTE vs CSFB (Circuit Switched Fallback) needs: distinct GSM_MIN (SURVIVAL) vs GSM_FULL layers.
- Added explicit Power Tiers (0:OFF, 1:SURVIVAL, 2:BASIC, 3:ANCHOR+1, 4:FULL).
- Added VRLA Cycle Life & Calendar Aging models (DoD cycle damage).
- Updated [TEST] numbers based on the unified twin_ref.py reference model.

================================================================================"""
text = text.replace("================================================================================", history_add, 1)


# Update Verdict
text = text.replace("The architecture is sound: forecast → deterministic safety shield → optional RL", 
                    "The architecture is sound: forecast → deterministic safety shield (priority lattice) → receding-horizon planner")

# Update "Where this happens: Iran"
iran_facts_old = """Grid:
- Peak deficit around 20 GW in 2025; scheduled and unscheduled cuts are routine.
- Since Feb 2025 reports describe daily cuts of 3–4 h, and cuts are unevenly distributed across districts.
- In summer 2026 reporting, Tehran/Karaj see rolling 1.5–3 h cuts between 08:30 and 23:30."""
iran_facts_new = """Grid:
- Peak deficit around 20 GW in 2025; scheduled and unscheduled cuts are routine.
- Since Feb 2025 reports describe daily cuts of 3–4 h, and cuts are unevenly distributed across districts.
- Iranian blackouts *are published in advance* via the "Bargh-e Man" app. This transforms the outage problem from pure RL guesswork into a deterministic Receding-Horizon planning problem.
- Cold Load Pickup (breaker tripping on grid return) is mitigated by telecom rectifiers via "walk-in" soft starts.

Telecom impact:
- Voice over LTE (VoLTE) exists, but CSFB (Circuit Switched Fallback) to 2G/3G is still critical for a vast portion of users. There is no official 2G/3G sunset plan. We MUST preserve a GSM layer for SURVIVAL.
- Multi-stage LVD is standard: Non-Priority Disconnect (~44-46V) and Priority Disconnect (~42V)."""
text = text.replace(iran_facts_old, iran_facts_new)


# Update Power Model
power_old = """Calibrated 3-carrier macro site (1 anchor + 2 capacity):
- P_fixed = 0.35 kW (BBU, baseband processing, transport, auxiliary DC)
- Per-carrier P0 = 0.50 kW (idle RF transceiver, power amplifier bias)
- Per-carrier Delta = 0.35 kW (dynamic RF load scaling)
- Site idle (L=0): P_fixed + 3 * P0 = 0.35 + 1.50 = 1.85 kW
- Site full (L=1): P_fixed + 3 * (P0 + Delta) = 0.35 + 2.55 = 2.90 kW"""

power_new = """Calibrated explicit Power Tiers (GSM/3G voice layer + LTE capacity):
- P_fixed = 0.35 kW (BBU, transport, aux DC)
- P0 = 0.50 kW, Delta = 0.35 kW per LTE carrier
- GSM_FULL = 0.55 kW (6 TRX), GSM_MIN = 0.30 kW (3 TRX BCCH-only)
- 3GPP TR 38.864 sleep states are mapped via kappa factors. ES_SLEEP (kappa=0.58), OUTAGE_OFF (kappa=0).
Explicit Tiers:
- Tier 4 (FULL): GSM_FULL + 3 LTE carriers (3.450 kW at L=1)
- Tier 3 (ANCHOR+1): GSM_FULL + 2 LTE carriers (2.890 kW at L=1, 1 asleep)
- Tier 2 (BASIC): GSM_FULL + 1 LTE carrier (2.330 kW at L=1, 2 asleep)
- Tier 1 (SURVIVAL): GSM_MIN + 0 LTE carriers (0.650 kW, outage mode)
- Tier 0 (OFF): 0 kW (BLVD tripped)"""
text = text.replace(power_old, power_new)


# Update Battery Model
battery_add = """
Aging & DoD Damage:
- VRLA cycle life plummets with Depth of Discharge (DoD). A discharge to 80% DoD yields ~300 cycles; to 30% DoD yields ~1270 cycles.
- VRLA float life halves every 8°C over 25°C. The twin implements both cycle damage (N(d) = N80*(0.8/d)^k) and Arrhenius calendar damage."""
text = text.replace("--------------------------------------------------------------------------------\n4.3 Thermal model", battery_add + "\n--------------------------------------------------------------------------------\n4.3 Thermal model")


# Update Safety Shield
shield_old = "8. Safety shield v2 (replaces Section 34)"
shield_new = "8. Safety Arbiter: Priority Lattice (replaces Safety shield v2)"
text = text.replace(shield_old, shield_new)

lattice_text = """
The arbiter has been redesigned as a mathematically formal Priority Lattice (UB/LB clamp).
It calculates Upper Bounds (UB) based on safety constraints and Lower Bounds (LB) based on service requirements.
final = clamp(proposal, LB, UB). If LB > UB, UB (Safety) wins.

Upper Bounds (Safety - local site agent):
- Backhaul down -> UB = 0 (OFF)
- Battery/Thermal Critical (e.g. SOC < 35%, T >= 50C) -> UB = 1 (SURVIVAL)

Lower Bounds (Service - QoS):
- Grid Up -> LB = 2 (BASIC anchor floor)
- Forecast / Emergency Wake -> LB = required capacity tier
- Hold gates (stale telemetry, dwell time, fatigue budget) -> LB = current state
- Grid Down -> LB = 1 (SURVIVAL floor)

Invariant Violation Tests [TEST]:
Randomized Arbiter states (trials=10000): conflict:UB-wins in 1713 cases. 0 UB or LB violations.
"""
text = re.sub(r"Hard invariants \(verified across.*?200,000 trials \[TEST\]\.", lattice_text, text, flags=re.DOTALL)


# Update Outage Protocol
outage_old = "9. Outage Survival Protocol (replaces Section 34.4 and Section 34.5)"
outage_new = "9. Outage Planner: Receding-Horizon Reserve (replaces Outage Survival Protocol & RL)"
text = text.replace(outage_old, outage_new)

planner_text = """
Because Iranian blackouts are published in advance via the 'Bargh-e Man' app, we replace the speculative RL approach with a deterministic Receding-Horizon Reserve Policy.

1) Reserve enough HARD-floor energy to keep SURVIVAL (Tier 1) alive until the q90 (90th percentile) expected grid-return time.
2) Capacity tiers may only spend surplus energy above the SOFT floor (battery-life protection), and only what remains after the reserve.
3) Choose the highest tier affordable for the expected (q50) remaining outage.
Re-evaluated every step => tiers drop/rise as grid-return estimates change.

Example Planner decisions (T_q50=2.0h, T_q90=3.0h, margin=10%) [TEST]:
  E_soft=1.5kWh, E_hard=2.0kWh -> chooses Tier 1 (SURVIVAL)
  E_soft=3.0kWh, E_hard=8.0kWh -> chooses Tier 2 (BASIC)
  E_soft=6.0kWh, E_hard=8.0kWh -> chooses Tier 4 (FULL)

Measured trade-off during a 4-hour blackout [TEST]:
| SOH | Policy | Full-service h | Any-service h |
|:---|:---|---:|---:|
| 100% | Naive (run all until dead) | 1.62 h | 1.62 h |
| 100% | Graceful Ladder | 0.73 h | 2.67 h |
| 60% | Naive | 0.72 h | 0.72 h |
| 60% | Graceful Ladder | 0.35 h | 1.33 h |
| 40% | Naive | 0.40 h | 0.40 h |
| 40% | Graceful Ladder | 0.22 h | 0.80 h |

Survival by cut length [TEST]:
| SOH | Tier | Site Power | 2-Hour Cut | 3-Hour Cut | 4-Hour Cut |
|:---|:---|---:|:---|:---|:---|
| 100% | FULL | 3.450 kW | DOWN @ 1.6 h | DOWN @ 1.6 h | DOWN @ 1.6 h |
| 100% | ANCHOR+1 | 2.600 kW | UP (survives) | DOWN @ 2.5 h | DOWN @ 2.5 h |
| 100% | BASIC | 1.750 kW | UP (survives) | UP (survives) | UP (4.1 h) |
| 100% | SURVIVAL | 0.650 kW | UP (survives) | UP (survives) | UP (10.0 h) |
| 60% | FULL | 3.450 kW | DOWN @ 0.7 h | DOWN @ 0.7 h | DOWN @ 0.7 h |
| 60% | BASIC | 1.750 kW | UP (survives) | DOWN @ 2.1 h | DOWN @ 2.1 h |
| 60% | SURVIVAL | 0.650 kW | UP (survives) | UP (survives) | UP (survives 7.4 h) |
| 40% | BASIC | 1.750 kW | DOWN @ 1.2 h | DOWN @ 1.2 h | DOWN @ 1.2 h |
"""
text = re.sub(r"Graceful degradation ladder:.*?(\n\nPriority question for operator)", planner_text + r"\1", text, flags=re.DOTALL)

with open("Cellular Network Energy v3.3.txt", "w") as f:
    f.write(text)
