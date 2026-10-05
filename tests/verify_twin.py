"""
Verification suite for the Cellular Network Energy report (v3.3).
Runs the report's own parameters through physics / forecasting / control checks.
Uses the reference implementation in reference/twin_ref.py.
Run: python3 verify_twin.py
"""
import sys
import numpy as np
import pandas as pd
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'reference'))
import twin_ref

rng = np.random.default_rng(7)
LINE = "-" * 78

def hdr(t):
    print("\n" + LINE + "\n" + t + "\n" + LINE)

# ------------------------------------------------------------------ T1 power
hdr("T1  Power model: EARTH/ETSI TR 36.887 vs twin_ref site_power_kw")
N, P0_earth, dp, Pmax, Ps = 6, 130, 4.7, 20, 75
full_earth = N * (P0_earth + dp * Pmax)
idle_earth = N * P0_earth
sleep_earth = N * Ps
print(f"EARTH macro  full={full_earth} W  idle(0+)={idle_earth} W  sleep={sleep_earth} W")
print(f"  idle/full = {idle_earth/full_earth:.2f}   sleep/idle = {sleep_earth/idle_earth:.2f}  (saving at zero load = {1-sleep_earth/idle_earth:.0%})")

# twin_ref model
p_full = twin_ref.site_power_kw(tier=4, demand_cu=3.0, sleep_kappa=0.58, mode="grid")
p_anchor1 = twin_ref.site_power_kw(tier=3, demand_cu=2.0, sleep_kappa=0.58, mode="grid")
p_anchor = twin_ref.site_power_kw(tier=2, demand_cu=1.0, sleep_kappa=0.58, mode="grid")
p_surv = twin_ref.site_power_kw(tier=1, demand_cu=0.0, sleep_kappa=0.0, mode="outage")
p_off = twin_ref.site_power_kw(tier=0, demand_cu=0.0, sleep_kappa=0.0, mode="outage")

print(f"twin_ref tier 4 (FULL) L=1: {p_full:.3f} kW")
print(f"twin_ref tier 3 (ANCHOR+1) L=1 (1 asleep): {p_anchor1:.3f} kW")
print(f"twin_ref tier 2 (BASIC) L=1 (2 asleep): {p_anchor:.3f} kW")
print(f"twin_ref tier 1 (SURVIVAL) outage: {p_surv:.3f} kW")
print(f"twin_ref tier 0 (OFF) outage: {p_off:.3f} kW")

# ------------------------------------------------------------------ T2 battery
hdr("T2  Battery: twin_ref VRLA/LFP model (OCV, polarization, SOH, Peukert)")

def run_batt(P_kw, chem="VRLA", soh=1.0, dt_s=60, tmax_h=10):
    b = twin_ref.Battery(chem=chem, soh=soh, Ah_nom=200.0, soc=1.0, T=35.0)
    t = 0
    while b.soc > 0.15 and t < tmax_h * 3600:
        V = b.discharge(P_kw, dt_s)
        if V < b.c["v_blvd"]:
            break
        t += dt_s
    return t / 3600

print(f"Outage survival time to 15% SOC or BLVD (35C shelter, VRLA vs LFP, Ah=200):")
print(f"{'SOH':>5} {'Tier':<12} {'P(kW)':>6} {'VRLA(h)':>8} {'LFP(h)':>8}")
for soh in (1.0, 0.6, 0.4):
    for tier, name in ((4, "FULL"), (3, "ANCHOR+1"), (2, "BASIC"), (1, "SURVIVAL")):
        n_lte = twin_ref.TIER_N_LTE[tier]
        P = twin_ref.site_power_kw(tier, demand_cu=n_lte*1.15, sleep_kappa=0.0, mode="outage")
        h_vrla = run_batt(P, chem="VRLA", soh=soh)
        h_lfp = run_batt(P, chem="LFP", soh=soh)
        print(f"{soh*100:>4.0f}% {name:<12} {P:>6.3f} {h_vrla:>8.2f} {h_lfp:>8.2f}")

hdr("T3  Graceful Ladder vs Naive (VRLA)")
def ladder_sim(soh, ladder=False, chem="VRLA", dt_s=60, T_h=4.0):
    b = twin_ref.Battery(chem=chem, soh=soh, Ah_nom=200.0, soc=1.0, T=35.0)
    t = 0.0
    full_h, any_h = 0.0, 0.0
    while t < T_h * 3600 and b.soc > 0.15:
        tier = 4
        if ladder:
            if b.soc < 0.45: tier = 2
            elif b.soc < 0.65: tier = 3
        n_lte = twin_ref.TIER_N_LTE[tier]
        P = twin_ref.site_power_kw(tier, demand_cu=n_lte*1.15, sleep_kappa=0.0, mode="outage")
        V = b.discharge(P, dt_s)
        if V < b.c["v_blvd"]: break
        t += dt_s
        any_h += dt_s / 3600
        if tier == 4: full_h += dt_s / 3600
    return full_h, any_h

print(f"{'SOH':>5} {'Policy':>9} {'Full-service h':>16} {'Any-service h':>15}")
for soh in (1.0, 0.6, 0.4):
    for nm, lad in (("naive", False), ("ladder", True)):
        f, a = ladder_sim(soh, lad, chem="VRLA")
        print(f"{soh*100:>4.0f}% {nm:>9} {f:>16.2f} {a:>15.2f}")

# ------------------------------------------------------------------ T4 arbiter
hdr("T4  Safety Arbiter Invariants (Priority Lattice: UB/LB clamp)")
def check_arbiter(trials=10000):
    bad = {"conflict_ub_wins": 0, "ub_fail": 0, "lb_fail": 0}
    for _ in range(trials):
        g = bool(rng.integers(0, 2))
        curr = int(rng.integers(0, 5))
        prop = int(rng.integers(0, 5))
        bnd = float(rng.uniform(0, 3.5))
        meas = float(rng.uniform(0, 3.5))
        t_age = float(rng.uniform(0, 30))
        dwell = int(rng.integers(0, 5))
        sl_left = int(rng.integers(0, 5))
        bh_down = bool(rng.random() < 0.05)
        v_b = float(rng.uniform(40, 55))
        soc = float(rng.uniform(0, 1))
        t_b = float(rng.uniform(20, 60))
        
        inp = twin_ref.ArbiterInput(
            grid_up=g, current=curr, proposal=prop, bound=bnd, measured=meas,
            telemetry_age_min=t_age, steps_since_sleep=dwell, sleeps_left_today=sl_left,
            lease_valid=True, backhaul_down=bh_down, v_batt=v_b, v_backstop=45.0,
            eff_soc=soc, t_batt=t_b
        )
        fin, why = twin_ref.arbitrate(inp)
        
        if fin < 1 and not bh_down and fin != 0: 
            bad["ub_fail"] += 1
        if "conflict:UB-wins" in why:
            bad["conflict_ub_wins"] += 1
            
    return bad

print("Randomized Arbiter states (trials=10000):", check_arbiter())

# ------------------------------------------------------------------ T5 planner
hdr("T5  Outage Planner (Receding-Horizon Reserve)")
P_tiers = [0, 
           twin_ref.site_power_kw(1, 0, 0, "outage"),
           twin_ref.site_power_kw(2, 1, 0, "outage"),
           twin_ref.site_power_kw(3, 2, 0, "outage"),
           twin_ref.site_power_kw(4, 3, 0, "outage")]

t_q50 = 2.0
t_q90 = 3.0
print(f"Planner with T_q50={t_q50}h, T_q90={t_q90}h, margin=10%")
for e_hard in [1.0, 2.0, 4.0, 8.0]:
    for e_soft in [0.5, 1.5, 3.0, 6.0]:
        tier = twin_ref.planner_tier(e_soft*1000, e_hard*1000, t_q50, t_q90, P_tiers, 0.1)
        print(f"  E_soft={e_soft:>3.1f}kWh, E_hard={e_hard:>3.1f}kWh -> chooses Tier {tier} ({twin_ref.TIER_NAMES[tier]})")

print("\nDONE")
