"""Pure-function tests for the four-level alert state machine (T8).

These tests pin down the promotion / hysteresis behaviour. If the
client asks us to change the alert thresholds after the demo, edit
`state_machine._BAND_LOWER` and these tests should still pass with
trivial updates.
"""
from app.inference.state_machine import band_for, next_state


def test_band_boundaries():
    assert band_for(0.0) == "NORMAL"
    assert band_for(0.29) == "NORMAL"
    assert band_for(0.30) == "WATCH"
    assert band_for(0.49) == "WATCH"
    assert band_for(0.50) == "WARNING"
    assert band_for(0.79) == "WARNING"
    assert band_for(0.80) == "ALERT"
    assert band_for(1.0) == "ALERT"


def test_alert_is_instant():
    """Safety override — any reading ≥ 0.8 jumps straight to ALERT."""
    assert next_state("NORMAL", [0.95]) == "ALERT"
    assert next_state("WATCH", [0.95]) == "ALERT"
    assert next_state("WARNING", [0.95]) == "ALERT"


def test_watch_promotion_is_instant():
    assert next_state("NORMAL", [0.35]) == "WATCH"


def test_warning_requires_3_sustained():
    # Single 0.55 reading from WATCH does NOT promote.
    assert next_state("WATCH", [0.55]) == "WATCH"
    # Two readings still not enough.
    assert next_state("WATCH", [0.55, 0.6]) == "WATCH"
    # Three sustained → promote.
    assert next_state("WATCH", [0.55, 0.6, 0.7]) == "WARNING"


def test_warning_streak_breaks_on_dip():
    # If one of the 3 readings drops back below 0.5, no promotion.
    assert next_state("WATCH", [0.55, 0.4, 0.6]) == "WATCH"


def test_one_step_promotion_only():
    """NORMAL with a 0.6 reading promotes to WATCH, not WARNING."""
    assert next_state("NORMAL", [0.6]) == "WATCH"


def test_demotion_requires_3_below():
    # WARNING with one reading below 0.5 — no demotion.
    assert next_state("WARNING", [0.4]) == "WARNING"
    # Two — still no demotion.
    assert next_state("WARNING", [0.4, 0.4]) == "WARNING"
    # Three consecutive below 0.5 — drop one tier to WATCH.
    assert next_state("WARNING", [0.4, 0.4, 0.4]) == "WATCH"


def test_demotion_resets_on_spike():
    # Two below + one above-0.5 → no demotion.
    assert next_state("WARNING", [0.4, 0.4, 0.6]) == "WARNING"


def test_alert_demotion_to_warning_only():
    """ALERT can only demote to WARNING, never skip to WATCH/NORMAL in one step."""
    assert next_state("ALERT", [0.5, 0.5, 0.5]) == "WARNING"
    assert next_state("ALERT", [0.0, 0.0, 0.0]) == "WARNING"


def test_empty_history_is_safe():
    assert next_state("NORMAL", []) == "NORMAL"
    assert next_state("WARNING", []) == "WARNING"


def test_unknown_prev_state_resets():
    assert next_state("BOGUS", [0.1]) == "NORMAL"
