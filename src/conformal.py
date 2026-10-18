"""
conformal.py - rolling conformal calibration + drift monitoring for the q95 peak-load forecaster.

What this is (and is not)
-------------------------
The LightGBM quantile model gives a point bound `q95_hat` for the next-60-min peak load.  It reaches ~95 %
coverage on the distribution it was trained on, but under demand shift (Ramadan evenings, a congested site,
a noisier site) coverage silently degrades and the Safety Arbiter then sleeps capacity it should not.

This module wraps the model with *rolling split-conformal calibration*:

    bound_t = q95_hat_t + margin_t
    margin_t = max(0, conformal_quantile_{1-alpha} of residuals r_j = y_j - q95_hat_j)

where the residual pool is a *sliding window* of the most recent calibration points that are already
observable at decision time (a label for step j is only known `label_delay_steps` = 12 steps (60 min) later,
because y_j is the max load over the next 60 minutes).  The pool is restricted to residuals from the same
time-of-day neighbourhood (`hour_bucket_h`) because shift is regime-dependent: my stress test showed a
+30 % evening surge reaches only ~0.51 evening coverage with the raw model, and a single global margin
cannot repair an evening-only error without wasting capacity all night.

It borrows the *sliding residual window* idea of EnbPI (Xu & Xie, ICML 2021) but it is NOT the full EnbPI
algorithm (there is no bootstrap ensemble / leave-one-out residuals; the point model is the single frozen
LightGBM).  Optionally an Adaptive-Conformal-Inference step (Gibbs & Candes, NeurIPS 2021) nudges the working
miscoverage level online so long-run coverage tracks the target even under persistent shift:

    alpha_{t+1} = clip(alpha_t + gamma * (alpha - err_t),  alpha_min, alpha_max)

(`err_t` = 1 if the bound missed the (delayed) realised peak.)  The margin is clamped at >= 0 on purpose: we only
ever WIDEN the safety bound, so calibration can never make the shield less safe than the raw model.

Everything here is causal: the margin at step k uses only residuals whose labels are observable at k.
`rolling_conformal_margin` (vectorised backtest form) and `OnlineConformal` (streaming form used in the
field) share the same selection rule and are tested to agree exactly.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np

STEPS_PER_DAY = 288


@dataclass(frozen=True)
class ConformalConfig:
    alpha: float = 0.05                    # target miscoverage (0.05 -> 95 % bound)
    window_steps: int = 7 * STEPS_PER_DAY  # residual pool length (7 days)
    label_delay_steps: int = 12            # 60 min until the next-60-min peak label is observable
    min_samples: int = 288                 # need >= 1 day of resolved labels before issuing a margin
    hour_bucket_h: float = 4.0             # same-time-of-day neighbourhood (full width, circular)
    min_bucket_samples: int = 60           # below this fall back to the global (all-hours) pool
    adaptive_gamma: float = 0.0            # ACI step size (0 = off)
    alpha_min: float = 0.005
    alpha_max: float = 0.20

    def __post_init__(self):
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if self.window_steps <= 0 or self.label_delay_steps < 0 or self.min_samples < 1:
            raise ValueError("window_steps/min_samples must be positive and label_delay_steps >= 0")
        if self.hour_bucket_h <= 0 or self.hour_bucket_h > 24:
            raise ValueError("hour_bucket_h must be in (0, 24]")
        if self.adaptive_gamma < 0:
            raise ValueError("adaptive_gamma must be >= 0")
        if not (0.0 < self.alpha_min <= self.alpha_max < 1.0):
            raise ValueError("need 0 < alpha_min <= alpha_max < 1")


def _circ_dist(h_a, h_b):
    d = np.abs(np.asarray(h_a, dtype=float) - h_b) % 24.0
    return np.minimum(d, 24.0 - d)


def _conformal_quantile(res: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile: the k-th smallest residual, k = ceil((n+1)(1-alpha)).
    If k > n (too few points for the requested coverage) the largest residual is returned (conservative)."""
    n = len(res)
    k = min(n, math.ceil((n + 1) * (1.0 - alpha) - 1e-12))
    return float(np.partition(res, k - 1)[k - 1])


def _margin_from_pool(res_pool: np.ndarray, hr_pool: np.ndarray, hour_now: float, alpha: float,
                      cfg: ConformalConfig) -> float:
    """Margin for one decision from an already-windowed residual pool (shared by offline + online code)."""
    n = len(res_pool)
    if n < cfg.min_samples:
        return 0.0
    near = _circ_dist(hr_pool, hour_now) <= cfg.hour_bucket_h / 2.0
    pool = res_pool[near] if near.sum() >= cfg.min_bucket_samples else res_pool
    return max(_conformal_quantile(pool, alpha), 0.0)


def rolling_conformal_margin(y, pred, hours, cfg: Optional[ConformalConfig] = None,
                             return_alpha: bool = False):
    """Causal margin series for ONE site (arrays in chronological 5-min order).

    margin[k] only uses residuals of origins j <= k - label_delay_steps (their labels are observable at k).
    """
    cfg = cfg or ConformalConfig()
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    hours = np.asarray(hours, dtype=float)
    if not (len(y) == len(pred) == len(hours)):
        raise ValueError("y, pred, hours must have equal length")
    if np.isnan(y).any() or np.isnan(pred).any():
        raise ValueError("NaN in y/pred: drop the unlabeled tail before calibrating")
    n = len(y)
    res = y - pred
    out = np.zeros(n)
    alpha_t = np.empty(n)
    a = cfg.alpha
    for k in range(n):
        hi = k - cfg.label_delay_steps          # exclusive upper origin index
        lo = max(0, hi - cfg.window_steps)
        if cfg.adaptive_gamma > 0 and hi >= 1:
            # origin hi-1 just became observable: did the *issued* bound at that origin miss?
            j = hi - 1
            miss = 1.0 if y[j] > pred[j] + out[j] else 0.0
            a = float(np.clip(a + cfg.adaptive_gamma * (cfg.alpha - miss), cfg.alpha_min, cfg.alpha_max))
        alpha_t[k] = a
        if hi - lo >= cfg.min_samples:
            out[k] = _margin_from_pool(res[lo:hi], hours[lo:hi], hours[k], a, cfg)
    return (out, alpha_t) if return_alpha else out


class OnlineConformal:
    """Streaming calibrator (what runs on the central service).

    Contract (identical to `rolling_conformal_margin`): at step k, first `add` origin j = k - label_delay_steps - 1
    (the newest origin whose 60-min label is safely observable; one step more conservative than strictly
    necessary), passing the margin that was *issued* at j, then call `margin(hour_k, k)`."""

    def __init__(self, cfg: Optional[ConformalConfig] = None):
        self.cfg = cfg or ConformalConfig()
        self._pool: deque = deque()            # (origin_step, hour, residual, issued_margin)
        self._alpha = self.cfg.alpha
        self.n_added = 0

    @property
    def alpha_working(self) -> float:
        return self._alpha

    def add(self, origin_step: int, hour: float, y_true: float, pred: float, issued_margin: float = 0.0):
        if any(isinstance(v, float) and math.isnan(v) for v in (hour, y_true, pred)):
            raise ValueError("NaN observation")
        if self.cfg.adaptive_gamma > 0:
            miss = 1.0 if y_true > pred + issued_margin else 0.0
            self._alpha = float(np.clip(self._alpha + self.cfg.adaptive_gamma * (self.cfg.alpha - miss),
                                        self.cfg.alpha_min, self.cfg.alpha_max))
        self._pool.append((int(origin_step), float(hour), float(y_true - pred)))
        self.n_added += 1

    def margin(self, hour_now: float, now_step: int) -> float:
        c = self.cfg
        lo = now_step - c.label_delay_steps - c.window_steps
        while self._pool and self._pool[0][0] < lo:
            self._pool.popleft()
        hi = now_step - c.label_delay_steps
        items = [(h, r) for (o, h, r) in self._pool if o < hi]
        if len(items) < c.min_samples:
            return 0.0
        hr = np.fromiter((h for h, _ in items), float, len(items))
        rs = np.fromiter((r for _, r in items), float, len(items))
        return _margin_from_pool(rs, hr, hour_now, self._alpha, c)


# --------------------------------------------------------------------------- monitoring
@dataclass(frozen=True)
class DriftConfig:
    window_steps: int = STEPS_PER_DAY      # last 24 h of resolved labels
    target: float = 0.95
    warn_below: float = 0.92
    alarm_below: float = 0.88
    min_samples: int = 144                 # half a day before the alarm can fire


def rolling_coverage(y, bound, window: int, label_delay_steps: int = 12):
    """Causal rolling empirical coverage of `bound` (NaN until `window` labels exist)."""
    y = np.asarray(y, dtype=float)
    hit = (y <= np.asarray(bound, dtype=float)).astype(float)
    n = len(y)
    cov = np.full(n, np.nan)
    csum = np.concatenate([[0.0], np.cumsum(hit)])
    for k in range(n):
        hi = k - label_delay_steps
        lo = hi - window
        if lo >= 0 and hi > lo:
            cov[k] = (csum[hi] - csum[lo]) / (hi - lo)
    return cov


def drift_status(y, raw_bound, cfg: Optional[DriftConfig] = None, label_delay_steps: int = 12):
    """Per-step status of the RAW (uncalibrated) bound: 0 = OK, 1 = WARN, 2 = ALARM (and the rolling coverage).

    Monitors the raw forecaster, not the calibrated bound, because the question the alarm answers is
    'has the model drifted away from the training distribution and needs retraining / review?'."""
    cfg = cfg or DriftConfig()
    cov = rolling_coverage(y, raw_bound, cfg.window_steps, label_delay_steps)
    status = np.zeros(len(cov), dtype=int)
    ok = ~np.isnan(cov) & (cfg.window_steps >= cfg.min_samples)
    status[ok & (cov < cfg.warn_below)] = 1
    status[ok & (cov < cfg.alarm_below)] = 2
    return status, cov


REGIMES = {
    "overall": lambda h, g: np.ones(len(h), bool),
    "night 00-06h": lambda h, g: (h >= 0) & (h < 6),
    "morning 06-12h": lambda h, g: (h >= 6) & (h < 12),
    "afternoon 12-19h": lambda h, g: (h >= 12) & (h < 19),
    "evening 19-24h": lambda h, g: (h >= 19) & (h < 24),
}


def coverage_by_regime(y, bound, hours, grid_status=None) -> Dict[str, Dict[str, float]]:
    """Empirical coverage + mean bound per regime (time of day, and grid up/down if provided)."""
    y = np.asarray(y, float)
    bound = np.asarray(bound, float)
    hours = np.asarray(hours, float)
    g = None if grid_status is None else np.asarray(grid_status)
    out: Dict[str, Dict[str, float]] = {}
    masks = {k: f(hours, g) for k, f in REGIMES.items()}
    if g is not None:
        masks["grid DOWN"] = g == 0
        masks["grid UP"] = g == 1
    for name, m in masks.items():
        if m.sum() == 0:
            continue
        out[name] = dict(n=int(m.sum()), coverage=float(np.mean(y[m] <= bound[m])),
                         mean_bound=float(np.mean(bound[m])))
    return out
