"""
carriers.py - LTE carrier state machine (sleeping / waking / active) with wake latency and signalling tax.

Extracted from the controller so that the closed-loop simulator and the policy benchmark use *one*
implementation.  Semantics (unchanged from the validated controller):

* Putting a carrier to sleep is instantaneous (power-down needs no warm-up).
* Waking a sleeping carrier takes `wake_latency_min` (0 = instant).  While `waking` it draws idle power but
  cannot serve traffic, so the site's *serving* capacity lags the commanded tier.
* A freshly woken carrier pays a +10 % signalling tax for its first `TAX_MIN` minutes (WAKE_TAX_KW each).
* If the site is dead (BLVD tripped) every carrier is forced to sleeping and nothing is served.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from tests.reference import twin_ref

N_CARRIERS = 3
TAX_MIN = 5.0


@dataclass(frozen=True)
class CarrierView:
    n_act: int          # carriers able to serve traffic
    n_waking: int       # carriers powered but not yet serving
    n_tax: int          # carriers still paying the wake signalling tax
    active_tier: int    # tier actually being served right now (lags the commanded tier during wake)


class CarrierBank:
    def __init__(self, wake_latency_min: float = 5.0):
        if wake_latency_min < 0:
            raise ValueError("wake_latency_min must be >= 0")
        self.wake_latency_min = float(wake_latency_min)
        self.states: List[str] = ["active"] * N_CARRIERS
        self.wake_rem: List[float] = [0.0] * N_CARRIERS
        self.tax_rem: List[float] = [0.0] * N_CARRIERS

    def kill(self) -> CarrierView:
        """Site dead (battery exhausted): everything off."""
        self.states = ["sleeping"] * N_CARRIERS
        self.wake_rem = [0.0] * N_CARRIERS
        self.tax_rem = [0.0] * N_CARRIERS
        return CarrierView(0, 0, 0, 0)

    def command(self, final_tier: int) -> CarrierView:
        """Apply the arbiter's final tier and return what is physically active this step."""
        target_n = twin_ref.TIER_N_LTE[final_tier]
        for c in range(N_CARRIERS):
            if c < target_n:
                if self.states[c] == "sleeping":
                    if self.wake_latency_min <= 0:
                        self.states[c] = "active"
                        self.wake_rem[c] = 0.0
                    else:
                        self.states[c] = "waking"
                        self.wake_rem[c] = float(self.wake_latency_min)
                    self.tax_rem[c] = TAX_MIN
            else:
                self.states[c] = "sleeping"
                self.wake_rem[c] = 0.0
                self.tax_rem[c] = 0.0
        n_act = sum(1 for s in self.states if s == "active")
        n_waking = sum(1 for s in self.states if s == "waking")
        n_tax = sum(1 for t in self.tax_rem if t > 0.0)
        if n_act >= 1:
            active_tier = twin_ref.tier_for_lte(n_act)
        else:
            active_tier = 1 if final_tier >= 1 else 0
        return CarrierView(n_act, n_waking, n_tax, active_tier)

    def tick(self, dt_s: float) -> None:
        dt_min = dt_s / 60.0
        for c in range(N_CARRIERS):
            if self.tax_rem[c] > 0:
                self.tax_rem[c] = max(0.0, self.tax_rem[c] - dt_min)
            if self.states[c] == "waking":
                self.wake_rem[c] = max(0.0, self.wake_rem[c] - dt_min)
                if self.wake_rem[c] == 0.0:
                    self.states[c] = "active"
