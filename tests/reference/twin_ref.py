"""
twin_ref.py — Reference implementation of the v3.3 corrected models (NOT the production codebase).
Purpose: make every v3.3 [TEST] number reproducible and give the team an executable spec.

Contents
  1. Power model with explicit layers (GSM voice layer + LTE anchor + 2 LTE capacity) and
     three carrier states: ON / ES_SLEEP (kappa, grid mode) / OFF (outage power-off).
  2. Battery model: VRLA (24 cells) and LFP (15S); OCV + polarization + IR(SOH) + Peukert;
     CC-taper recharge with coulombic efficiency; conservative capacity estimator.
  3. Aging model: DoD cycle damage + Arrhenius calendar damage.
  4. Service tiers + Safety Arbiter (priority lattice: UB/LB clamp) used by BOTH grid-ES and outage.
  5. Outage energy-budget planner (receding-horizon reserve policy).
All parameters mirror Section 11 (Parameter Freeze v3.3) of the plan.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- 1. power
P_FIXED = 0.35          # kW  BBU + transport + aux DC
P0 = 0.50               # kW  per LTE carrier idle
DELTA = 0.35            # kW  per LTE carrier full-load increment
GSM_FULL = 0.55         # kW  GSM S222 (6 TRX @ ~90 W) [ASSUMP from 80-140 W/TRX]
GSM_MIN = 0.30          # kW  GSM S111 BCCH-only (3 TRX)            [ASSUMP]
VENDOR_MS = 0.20        # vendor micro-sleep: active-carrier power reduced by MS*(1-L)
KAPPA_OFF = 0.0         # outage power-off (RRU de-energised)

# Service tiers (index = order). n_lte = number of LTE carriers ON.
#   0 OFF        : all radios off
#   1 SURVIVAL   : GSM minimum (voice/SMS/emergency via 2G; CSFB) — LTE off
#   2 BASIC      : GSM full + LTE anchor            (== "anchor only")
#   3 ANCHOR+1   : GSM full + LTE anchor + 1 capacity
#   4 FULL       : GSM full + all 3 LTE carriers
TIER_NAMES = ["OFF", "SURVIVAL", "BASIC", "ANCHOR+1", "FULL"]
TIER_N_LTE = [0, 0, 1, 2, 3]


def lte_on_power(L: float, ms: float = VENDOR_MS) -> float:
    return (P0 + DELTA * L) * (1.0 - ms * (1.0 - L))


def site_power_kw(tier: int, demand_cu: float, sleep_kappa: float, mode: str = "grid",
                  ms: float = VENDOR_MS, n_waking: int = 0) -> float:
    """demand_cu: LTE offered load in carrier-capacity units.
    mode='grid': non-active LTE carriers are in ES_SLEEP (kappa); mode='outage': they are OFF."""
    if tier == 0:
        return 0.0
    n_on = TIER_N_LTE[tier]
    gsm = GSM_MIN if tier == 1 else GSM_FULL
    p = P_FIXED + gsm
    if n_on:
        L = min(1.0, demand_cu / n_on)
        p += n_on * lte_on_power(L, ms)
    n_idle = 3 - n_on
    k = sleep_kappa if mode == "grid" else KAPPA_OFF
    p += n_waking * P0 + max(0, n_idle - n_waking) * P0 * k
    return p


# --------------------------------------------------------------------------- 2. battery
CHEM = {
    #            cells  R0(ohm) n_fresh n_aged  eta_c  I_lim(C) soc_floor_hard  V_backstop V_LLVD V_BLVD
    "VRLA": dict(ncell=24, R0=0.015, n_fresh=1.20, n_aged=1.25, eta=0.90, ilim=0.10,
                 v_backstop=45.0, v_llvd=43.2, v_blvd=42.0, soc_soft=0.50, soc_hard=0.20,
                 est_unc=0.15, cyc_N80=300, cyc_k=1.47, cal_life_y=10.0),
    "LFP":  dict(ncell=15, R0=0.010, n_fresh=1.05, n_aged=1.15, eta=0.99, ilim=0.30,
                 v_backstop=46.5, v_llvd=45.0, v_blvd=42.0, soc_soft=0.20, soc_hard=0.08,
                 est_unc=0.05, cyc_N80=3000, cyc_k=1.30, cal_life_y=15.0),
}

_LFP_OCV = [(0.0, 2.80), (0.05, 3.05), (0.10, 3.20), (0.20, 3.25), (0.50, 3.29), (0.90, 3.33), (1.0, 3.45)]


def ocv_cell(chem: str, soc: float) -> float:
    soc = min(1.0, max(0.0, soc))
    if chem == "VRLA":                      # 1.95 V (empty) .. 2.13 V (full) per cell, ~linear (12.8 V/12 V bloc)
        return 1.95 + 0.18 * soc
    for (s0, v0), (s1, v1) in zip(_LFP_OCV, _LFP_OCV[1:]):
        if soc <= s1:
            return v0 + (v1 - v0) * (soc - s0) / (s1 - s0)
    return _LFP_OCV[-1][1]


@dataclass
class Battery:
    chem: str = "VRLA"
    soh: float = 1.0
    Ah_nom: float = 200.0
    soc: float = 1.0
    T: float = 25.0
    # bookkeeping
    ah_out: float = 0.0
    event_soc_start: float = 1.0
    event_soc_min: float = 1.0
    discharging: bool = False
    cyc_damage: float = 0.0
    cal_damage: float = 0.0
    dod_events: list = field(default_factory=list)

    def __post_init__(self):
        c = CHEM[self.chem]
        self.c = c
        self.C = self.Ah_nom * self.soh
        self.n = c["n_fresh"] if self.soh >= 0.8 else c["n_aged"]
        self.R = c["R0"] * (1.0 + 1.5 * (1.0 - self.soh))   # aged VRLA: ~1.6-2.1x at SOH 0.6-0.25 [ASSUMP]

    # polarization (per cell) grows with C-rate and near empty — makes the end-of-discharge knee
    def _eta(self, I: float) -> float:
        rate = I / (self.Ah_nom / 10.0)
        if self.chem == "VRLA":
            return 0.03 * rate * (1.0 + 0.1 / (self.soc + 0.02))
        return 0.01 * rate * (1.0 + 0.02 / (self.soc + 0.01))

    def v_terminal(self, I: float) -> float:
        return self.c["ncell"] * (ocv_cell(self.chem, self.soc) - self._eta(I)) - I * self.R

    def discharge(self, P_kw: float, dt_s: float) -> float:
        """Returns terminal voltage. Peukert drain referenced to C/10 of the aged capacity."""
        if P_kw <= 0:
            return self.v_terminal(0.0)
        if not self.discharging:
            self.discharging, self.event_soc_start, self.event_soc_min = True, self.soc, self.soc
        V = self.v_terminal(P_kw * 1000 / 50.0)
        for _ in range(3):
            I = P_kw * 1000.0 / max(V, 30.0)
            V = self.v_terminal(I)
        I_ref = self.C / 10.0
        dq = I * (I / I_ref) ** (self.n - 1.0) * dt_s / 3600.0
        self.soc = max(0.0, self.soc - dq / self.C)
        self.ah_out += I * dt_s / 3600.0
        self.event_soc_min = min(self.event_soc_min, self.soc)
        return V

    def charge(self, dt_s: float, ilim_C: float | None = None) -> float:
        """CC at ilim (fraction of nominal C) then taper above 85% SOC. Returns charge power kW."""
        self._close_event()
        ilim = (ilim_C if ilim_C is not None else self.c["ilim"]) * self.Ah_nom
        if self.T > 40.0:
            ilim *= 0.5                         # thermal charge derate
        I = ilim if self.soc < 0.85 else ilim * max(0.03, (1.0 - self.soc) / 0.15)
        self.soc = min(1.0, self.soc + self.c["eta"] * I * dt_s / 3600.0 / self.C)
        return I * 54.0 / 1000.0

    def _close_event(self):
        if self.discharging:
            d = self.event_soc_start - self.event_soc_min
            if d > 0.02:
                self.dod_events.append(d)
                self.cyc_damage += d_cycle_damage(self.chem, d)
            self.discharging = False

    def age_calendar(self, dt_s: float):
        L_h = self.c["cal_life_y"] * 8760.0 * 2.0 ** (-(self.T - 25.0) / 8.0)
        self.cal_damage += dt_s / 3600.0 / L_h


def d_cycle_damage(chem: str, dod: float) -> float:
    """Half-cycle-free simplification: one discharge event of depth dod consumes 1/N(dod) of life.
    N(d) = N80 * (0.8/d)^k ; VRLA fit to 300@80%, ~600@50%, ~1270@30% DoD."""
    c = CHEM[chem]
    return 1.0 / (c["cyc_N80"] * (0.8 / max(dod, 1e-3)) ** c["cyc_k"])


@dataclass
class CapacityEstimator:
    """Conservative SOC: capacity estimate has relative error e in [-u, +u]; controller uses C_est*(1-u)."""
    C_true: float
    e: float
    u: float
    soc0: float = 1.0
    ah0: float = 0.0

    def reset(self, soc_true: float, ah_out: float):
        self.soc0, self.ah0 = soc_true, ah_out

    def eff_soc(self, ah_out: float) -> float:
        C_cons = self.C_true * (1.0 + self.e) * (1.0 - self.u)
        return self.soc0 - (ah_out - self.ah0) / C_cons


# --------------------------------------------------------------------------- 4. safety arbiter
SLEEP_UTIL = 0.60       # sleep allowed if bound / n_target <= 0.60
WAKE_UTIL = 0.70        # wake required if bound / n_current > 0.70 (hysteresis band 0.60-0.70)
EMERG_UTIL = 0.85       # measured load per active carrier > 0.85 -> immediate wake
SOC_CRIT = 0.35         # effective SOC critical (grid down) -> UB = SURVIVAL
T_CRIT = 50.0           # battery temperature critical -> UB = SURVIVAL
DWELL_STEPS = 3         # 15 min between ES sleep actions (wakes never wait)
STALE_MIN = 15.0


@dataclass
class ArbiterInput:
    grid_up: bool
    current: int                 # current tier
    proposal: int                # tier proposed by policy / planner
    bound: float                 # q95 + conformal margin of next-60-min peak LTE demand (carrier units)
    measured: float              # current measured LTE demand (carrier units)
    telemetry_age_min: float
    steps_since_sleep: int
    sleeps_left_today: int       # ES transition budget (grid mode only)
    lease_valid: bool            # central ES lease still valid
    backhaul_down: bool
    v_batt: float
    v_backstop: float
    eff_soc: float
    t_batt: float


def tier_for_lte(n: int) -> int:
    return {1: 2, 2: 3, 3: 4}[max(1, min(3, n))]


def arbitrate(x: ArbiterInput) -> tuple[int, list[str]]:
    """Priority lattice. UB = min of upper bounds (safety), LB = max of lower bounds (service).
    final = clamp(proposal, LB, UB); if LB > UB, UB wins (battery/thermal safety beats QoS)."""
    why: list[str] = []
    prop = max(0, min(4, int(x.proposal)))
    UB, LB = 4, 0
    # ---- upper bounds (safety; local site agent; never depend on telemetry freshness)
    if x.backhaul_down:
        UB = min(UB, 0); why.append("UB:backhaul-down")
    if not x.grid_up and (x.v_batt < x.v_backstop or x.eff_soc < SOC_CRIT):
        UB = min(UB, 1); why.append("UB:battery-critical")
    if x.t_batt >= T_CRIT:
        UB = min(UB, 1); why.append("UB:thermal-critical")
    # ---- lower bounds (service)
    if x.grid_up:
        LB = max(LB, 2); why.append("LB:anchor-floor")
        if not x.lease_valid:
            LB = max(LB, 4); why.append("LB:lease-expired->all-on")
        n_cur = TIER_N_LTE[x.current] if x.current >= 2 else 1
        if x.bound / max(n_cur, 1) > WAKE_UTIL:
            LB = max(LB, tier_for_lte(math.ceil(x.bound / WAKE_UTIL - 1e-9))); why.append("LB:forecast-wake")
        if x.measured / max(n_cur, 1) > EMERG_UTIL:
            LB = max(LB, tier_for_lte(math.ceil(x.measured / EMERG_UTIL - 1e-9))); why.append("LB:emergency-wake")
        if prop < x.current:     # a sleep is being requested — every sleep gate is a LB = current
            gates = []
            if x.telemetry_age_min > STALE_MIN: gates.append("stale")
            if x.steps_since_sleep < DWELL_STEPS: gates.append("dwell")
            if x.sleeps_left_today <= 0: gates.append("fatigue-budget")
            if gates:
                LB = max(LB, x.current); why.append("LB:hold(" + ",".join(gates) + ")")
            else:   # largest safe reduction: n_target must satisfy bound/n <= SLEEP_UTIL
                n_safe = max(1, math.ceil(x.bound / SLEEP_UTIL - 1e-9))
                LB = max(LB, tier_for_lte(n_safe)); why.append("LB:capacity-clamp")
    else:
        LB = max(LB, 1); why.append("LB:survival-floor")   # outage: demand-based wakes do NOT apply
    if LB > UB:
        why.append("conflict:UB-wins")
        return UB, why
    return max(LB, min(UB, prop)), why


# --------------------------------------------------------------------------- 5. outage planner
def planner_tier(E_avail_wh_soft: float, E_avail_wh_hard: float, T_q50_h: float, T_q90_h: float,
                 P_tier_kw: list[float], margin: float = 0.10) -> int:
    """Receding-horizon reserve policy.
    1) Reserve enough HARD-floor energy to keep SURVIVAL alive until the q90 grid-return time.
    2) Capacity tiers may only spend energy above the SOFT floor (battery-life protection),
       and only what remains after the reserve; choose the highest tier affordable for the
       expected (q50) remaining outage. Re-evaluated every step => tiers drop/rise as info changes."""
    reserve = P_tier_kw[1] * 1000.0 * T_q90_h * (1.0 + margin)
    if E_avail_wh_hard < reserve:
        return 1
    surplus = min(E_avail_wh_soft, E_avail_wh_hard - reserve)
    best = 1
    for t in (2, 3, 4):
        extra = (P_tier_kw[t] - P_tier_kw[1]) * 1000.0 * T_q50_h
        if extra <= surplus:
            best = t
    return best
