"""Four-level alert state machine for the Live Monitor (T8/T9).

Pure functions only — no I/O, no globals. Given the previous state and a
small ring of recent probabilities, return the next state.

Design rules (locked at the 2026-05-10 review with David):

- Bands by probability:
    NORMAL   p < 0.30
    WATCH    0.30 ≤ p < 0.50
    WARNING  0.50 ≤ p < 0.80
    ALERT    p ≥ 0.80

- Promotions (going up the ladder):
    NORMAL → WATCH      instant (1 reading in band)
    WATCH  → WARNING    sustained 3 readings ≥ 0.50
    *      → ALERT      instant (any reading ≥ 0.80, safety override)

- Demotions (going down the ladder, hysteresis):
    Drop one tier only after `DEMOTION_STREAK` consecutive readings
    strictly below the current tier's lower bound. Never skip tiers in
    one step — that prevents a single noise spike from masking a real
    rebound.

The shape is `next_state(prev_state, recent_probs) → new_state` where
`recent_probs` is a list of the last N probabilities ending with the
just-arrived value. The caller maintains the ring buffer.
"""
from __future__ import annotations

from typing import Sequence


States = ("NORMAL", "WATCH", "WARNING", "ALERT")
_ORDER = {s: i for i, s in enumerate(States)}

# Lower bound of each tier — entering a band means p ≥ this value.
_BAND_LOWER = {
    "NORMAL":  0.00,
    "WATCH":   0.30,
    "WARNING": 0.50,
    "ALERT":   0.80,
}

# Number of consecutive in-band readings required for non-instant promotions.
_PROMOTION_STREAK = {
    "WATCH":   1,   # instant
    "WARNING": 3,   # sustained
    "ALERT":   1,   # instant safety override
}

# Demotion hysteresis: this many consecutive readings strictly below the
# current tier's lower bound before dropping one tier.
DEMOTION_STREAK = 3


def band_for(prob: float) -> str:
    """Map a probability to its raw band name (no hysteresis)."""
    if prob >= _BAND_LOWER["ALERT"]:
        return "ALERT"
    if prob >= _BAND_LOWER["WARNING"]:
        return "WARNING"
    if prob >= _BAND_LOWER["WATCH"]:
        return "WATCH"
    return "NORMAL"


def next_state(prev_state: str, recent_probs: Sequence[float]) -> str:
    """Compute the new state after the latest probability is observed.

    `recent_probs[-1]` is the current tick. Older entries are used only
    for streak counting; a window of ~10 is more than enough.
    """
    if prev_state not in _ORDER:
        prev_state = "NORMAL"
    if not recent_probs:
        return prev_state

    current = recent_probs[-1]
    raw_band = band_for(current)
    prev_idx = _ORDER[prev_state]
    raw_idx = _ORDER[raw_band]

    # ── Safety override: any reading ≥ 0.80 → ALERT immediately ──
    if raw_band == "ALERT":
        return "ALERT"

    # ── Promotion path ──
    if raw_idx > prev_idx:
        target = States[prev_idx + 1]  # only climb one rung at a time
        required = _PROMOTION_STREAK[target]
        target_lower = _BAND_LOWER[target]
        tail = recent_probs[-required:]
        if len(tail) >= required and all(p >= target_lower for p in tail):
            return target
        return prev_state

    # ── Demotion path ──
    if raw_idx < prev_idx:
        prev_lower = _BAND_LOWER[prev_state]
        tail = recent_probs[-DEMOTION_STREAK:]
        if len(tail) >= DEMOTION_STREAK and all(p < prev_lower for p in tail):
            return States[prev_idx - 1]
        return prev_state

    # Same band as before — no change.
    return prev_state


__all__ = ["States", "band_for", "next_state", "DEMOTION_STREAK"]
