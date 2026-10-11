"""Shared arbiter invariants. Used by pytest property tests AND tests/verify_twin.py.

Each invariant returns a violation name or None. They are written from the architecture spec (Sec. 8),
NOT from the implementation, so a wrong arbiter can fail them.
(The previous T4 check in verify_twin.py could never fail: its condition was `fin < 1 and fin != 0`
and `lb_fail` was never incremented.)
"""
from tests.reference import twin_ref as T

THERMAL_CRIT = T.T_CRIT


def violations(x: "T.ArbiterInput", fin: int) -> list[str]:
    bad: list[str] = []
    n_fin = T.TIER_N_LTE[fin] if 0 <= fin <= 4 else 0
    ub_backhaul = x.backhaul_down
    ub_thermal = x.t_batt >= THERMAL_CRIT
    ub_battery = (not x.grid_up) and (x.v_batt < x.v_backstop or x.eff_soc < T.SOC_CRIT)

    if not (0 <= fin <= 4):
        return ["tier-out-of-range"]
    if ub_backhaul and fin != 0:
        bad.append("backhaul-down-not-off")
    if not ub_backhaul and ub_thermal and fin > 1:
        bad.append("thermal-critical-not-limited")
    if not ub_backhaul and ub_battery and fin > 1:
        bad.append("battery-critical-not-limited")

    no_ub = not (ub_backhaul or ub_thermal or ub_battery)
    if no_ub and x.grid_up:
        if fin < 2:
            bad.append("anchor-floor-violated")
        if not x.lease_valid and fin != 4:
            bad.append("lease-expired-not-all-on")
        n_cur = max(n_fin, 1)
        # capacity: never leave fewer carriers than forecast / measured load need (unless already max)
        if fin != 4 and x.bound / n_cur > T.WAKE_UTIL + 1e-9:
            bad.append("forecast-overload-not-woken")
        if fin != 4 and x.measured / n_cur > T.EMERG_UTIL + 1e-9:
            bad.append("measured-overload-not-woken")
        if fin < x.current:      # this decision is a sleep: every sleep gate must be open, load must fit
            if x.telemetry_age_min > T.STALE_MIN:
                bad.append("sleep-on-stale-telemetry")
            if x.steps_since_sleep < T.DWELL_STEPS:
                bad.append("sleep-inside-dwell")
            if x.sleeps_left_today <= 0:
                bad.append("sleep-with-exhausted-fatigue-budget")
            if n_fin == 0 or x.bound / n_fin > T.SLEEP_UTIL + 1e-9:
                bad.append("sleep-into-forecast-overload")
            if n_fin == 0 or x.measured / n_fin > T.SLEEP_UTIL + 1e-9:
                bad.append("sleep-into-measured-overload")
    if no_ub and not x.grid_up and fin < 1:
        bad.append("survival-floor-violated")
    return bad
