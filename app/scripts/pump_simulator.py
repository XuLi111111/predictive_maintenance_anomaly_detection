"""Pump simulator — streams realistic SKAB-style sensor data.

Behaviour
---------
- Reads a real SKAB CSV as the baseline waveform (sample.csv by default —
  real valve2 data, which already has natural fluctuations).
- Picks a random starting row each launch, so successive demo runs see
  different segments of the file unless you pass ``--seed``.
- Adds per-channel multiplicative jitter (~±1.5%) so the data never
  looks "frozen" or perfectly periodic.
- Periodically injects an anomaly burst at random intervals
  (default 60–180 seconds apart) with random duration (15–40 samples)
  and random intensity. The burst envelope is a bell curve (sin-π),
  not a hard step — so the operator sees the probability ramp up,
  peak, then decay naturally.

In production this script is the seam where you'd swap in a real data
source — read from OPC-UA, Modbus, MQTT, etc. The downstream pipeline
(buffer → inference → WS → UI) is unchanged.

Usage
-----
    python app/scripts/pump_simulator.py                 # realistic, random
    python app/scripts/pump_simulator.py --speed 4       # 4× real-time
    python app/scripts/pump_simulator.py --no-burst      # only normal data
    python app/scripts/pump_simulator.py --seed 42       # reproducible
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    sys.exit(
        "Missing dependency: pandas. "
        "Install with: pip install pandas  (already in the backend venv)."
    )


SENSOR_COLS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[1] / "backend" / "artifacts" / "sample.csv"
)
DEFAULT_API = "http://localhost:8080/api/live/ingest"

# Anomaly burst dynamics — chosen so each demo run looks different.
BURST_INTERVAL_RANGE = (60, 180)   # samples between bursts
BURST_DURATION_RANGE = (15, 40)    # samples each burst lasts
BURST_INTENSITY_RANGE = (0.7, 1.4) # global scaling for the burst envelope

# Per-channel multiplicative jitter so successive samples are never
# identical — closer to real PLC noise / quantisation.
JITTER_SCALE = 0.015               # ±1.5%

# How each channel responds during an anomaly burst (multiplied by the
# bell-curve intensity, which goes 0 → 1 → 0 over the burst's lifetime).
BURST_SCALE_GAIN = {
    "Accelerometer1RMS": 2.0,   # x1 → x3 at peak
    "Accelerometer2RMS": 2.0,
    "Current":           0.8,   # x1 → x1.8 at peak
    "Pressure":          0.4,
}
BURST_TEMP_BUMP = 4.0   # additive (°C) at peak


# ──────────────────────────────────────────────────────────────────────

class BurstState:
    """Tracks whether we're currently inside an anomaly burst, when the
    next one is scheduled, and how intense / long it is.

    All randomness is funnelled through the injected RNG so a --seed
    arg fully reproduces a run."""

    def __init__(self, rng: random.Random, no_burst: bool = False) -> None:
        self.rng = rng
        self.no_burst = no_burst
        self.sample_n = 0
        self.in_burst = False
        self.burst_start = 0
        self.burst_duration = 0
        self.burst_intensity = 1.0
        self.next_burst_n = (
            float("inf") if no_burst else rng.randint(*BURST_INTERVAL_RANGE)
        )

    def advance(self) -> tuple[bool, bool]:
        """Move one sample forward. Returns (just_started, just_ended)."""
        self.sample_n += 1
        just_started = False
        just_ended = False

        if self.in_burst:
            if self.sample_n - self.burst_start >= self.burst_duration:
                self.in_burst = False
                just_ended = True
                if not self.no_burst:
                    self.next_burst_n = (
                        self.sample_n + self.rng.randint(*BURST_INTERVAL_RANGE)
                    )
        elif (not self.no_burst
              and self.sample_n >= self.next_burst_n):
            self.in_burst = True
            self.burst_start = self.sample_n
            self.burst_duration = self.rng.randint(*BURST_DURATION_RANGE)
            self.burst_intensity = self.rng.uniform(*BURST_INTENSITY_RANGE)
            just_started = True

        return just_started, just_ended

    @property
    def intensity(self) -> float:
        """Bell-curve envelope: 0 at start, 1 at midpoint, 0 at end,
        multiplied by this burst's random intensity."""
        if not self.in_burst:
            return 0.0
        progress = (self.sample_n - self.burst_start) / max(self.burst_duration, 1)
        progress = min(max(progress, 0.0), 1.0)
        return math.sin(progress * math.pi) * self.burst_intensity


def build_sensors(
    row: "pd.Series",
    state: BurstState,
    rng: random.Random,
) -> dict[str, float]:
    """Build one sensor payload from a baseline row + jitter + burst."""
    sensors: dict[str, float] = {
        c: float(row[c]) * (1.0 + rng.uniform(-JITTER_SCALE, JITTER_SCALE))
        for c in SENSOR_COLS
    }

    if state.in_burst:
        i = state.intensity
        for col, gain in BURST_SCALE_GAIN.items():
            sensors[col] *= (1.0 + gain * i)
        sensors["Temperature"] += BURST_TEMP_BUMP * i

    return sensors


# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE,
        help=f"Baseline SKAB-format CSV (default: {DEFAULT_SOURCE.name}).",
    )
    p.add_argument(
        "--api", default=DEFAULT_API,
        help=f"Ingest endpoint URL (default: {DEFAULT_API}).",
    )
    p.add_argument(
        "--speed", type=float, default=1.0,
        help="Samples per second (1.0 = real-time, 4 = 4× faster).",
    )
    p.add_argument(
        "--loop", action="store_true", default=True,
        help="Loop forever (default; pass --no-loop to stop at EOF).",
    )
    p.add_argument("--no-loop", dest="loop", action="store_false")
    p.add_argument(
        "--no-burst", action="store_true",
        help="Don't inject any anomaly bursts — stream only normal data.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed (default: random each run, so demos differ).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    if not args.source.is_file():
        print(f"ERROR: source CSV not found: {args.source}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.source, sep=";")
    missing = [c for c in SENSOR_COLS if c not in df.columns]
    if missing:
        print(f"ERROR: source CSV missing columns: {missing}", file=sys.stderr)
        return 2

    # Pick a random starting row so the demo doesn't always begin with
    # row 0 of the same file. If --seed is given the offset is also
    # deterministic.
    start_offset = rng.randint(0, max(len(df) - 1, 0))

    print(f"[simulator] reading   : {args.source.name} ({len(df)} rows)")
    print(f"[simulator] pushing to: {args.api}")
    print(f"[simulator] speed     : {args.speed} samples/sec "
          f"(interval {1.0 / max(args.speed, 0.01):.3f}s)")
    print(f"[simulator] seed      : {args.seed if args.seed is not None else 'random'}")
    print(f"[simulator] start row : #{start_offset}")
    if args.no_burst:
        print(f"[simulator] anomaly   : disabled (--no-burst)")
    else:
        print(f"[simulator] anomaly   : bursts every "
              f"{BURST_INTERVAL_RANGE[0]}–{BURST_INTERVAL_RANGE[1]} samples, "
              f"lasting {BURST_DURATION_RANGE[0]}–{BURST_DURATION_RANGE[1]} samples")

    state = BurstState(rng, no_burst=args.no_burst)
    interval = 1.0 / max(args.speed, 0.01)
    sent = 0
    failed = 0

    print("[simulator] streaming… (Ctrl+C to stop)")
    try:
        while True:
            for i in range(len(df)):
                row = df.iloc[(i + start_offset) % len(df)]
                started, ended = state.advance()
                if started:
                    print(f"  ⚡ anomaly burst #{state.sample_n} · "
                          f"duration {state.burst_duration}s · "
                          f"intensity {state.burst_intensity:.2f}")
                elif ended:
                    print(f"  ↩ burst ended #{state.sample_n}; "
                          f"next in ~{state.next_burst_n - state.sample_n}s")

                sensors = build_sensors(row, state, rng)
                payload = {
                    "timestamp": str(row["datetime"]),
                    "sensors": sensors,
                }
                body = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    args.api, data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        sent += 1
                        if sent % 30 == 0:
                            try:
                                emitted = json.loads(resp.read()).get("tick_emitted")
                            except Exception:  # noqa: BLE001
                                emitted = "?"
                            print(f"  · sent {sent} samples "
                                  f"(latest tick_emitted={emitted})")
                except urllib.error.HTTPError as exc:
                    failed += 1
                    if failed <= 3:
                        body_preview = exc.read()[:120].decode(
                            "utf-8", errors="replace",
                        )
                        print(f"  ! {exc.code}: {body_preview}", file=sys.stderr)
                except (urllib.error.URLError, TimeoutError) as exc:
                    failed += 1
                    if failed <= 3:
                        print(f"  ! request failed: {exc}", file=sys.stderr)

                time.sleep(interval)

            if not args.loop:
                print(f"[simulator] EOF reached, sent={sent}, failed={failed}.")
                return 0
    except KeyboardInterrupt:
        print(f"\n[simulator] stopped. sent={sent}, failed={failed}.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
