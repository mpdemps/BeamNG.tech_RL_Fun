"""
Record the centerline of West Coast USA's racetrack by polling the
vehicle's position while Mike drives a lap manually.

Plain-language overview, Mikey:
- The script opens BeamNG by itself, loads West Coast USA, and puts the
  ETK 800 on the racetrack.
- Dad takes the wheel and drives the racetrack once, normally.
- The script waits until the car starts moving, then writes down where
  the car is every half-second.
- It cleans up the list (drops duplicate points from when the car was
  stopped or barely moving) and saves the cleaned racing line to
  data/centerline_racetrack.py.
- The training script reads that file the next time we train, so the AI
  knows the shape of the track.

Run this ONCE for the project, then commit data/centerline_racetrack.py.
Future training (this machine or any other) reuses the file — no need to
re-record unless we switch tracks.

Usage:
    From the repo root: python scripts/record_centerline.py
    Then drive a lap and press Enter (in this terminal) when done.
"""

import argparse
import math
import os
import sys
import threading
from datetime import date
from pathlib import Path

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import State


# ---- Tunable constants ----
MAP_NAME = "west_coast_usa"
VEHICLE_MODEL = "etk800"
VEHICLE_ID = "ego"
# WCUSA racetrack start/finish line pose, recorded by Mike on 2026-05-12.
# Yaw is roughly 88.6°, rotated 180° from the previous attempt (which faced
# the wrong way down the straight).
SPAWN_POS = (394.70, -252.02, 145.16)
SPAWN_QUAT = (0.0, 0.0, 0.698, 0.716)

POLL_INTERVAL_FAST_S = 0.1     # motion-detection polling cadence
POLL_INTERVAL_RECORD_S = 0.5   # position-recording cadence once moving
MOTION_THRESHOLD_M_S = 2.0     # car must exceed this speed to start recording
DEDUPE_MIN_DIST_M = 1.0

DEFAULT_HOME = None
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 25252
OUTPUT_PATH = Path("data/centerline_racetrack.py")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--home", default=os.environ.get("BEAMNG_HOME", DEFAULT_HOME),
                   help="BeamNG.tech install directory.")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    return p.parse_args()


def setup_scenario(bng) -> Vehicle:
    """Build the West Coast USA racetrack scenario and start it. Return vehicle."""
    scenario = Scenario(MAP_NAME, "phase1_lap_recording")
    vehicle = Vehicle(VEHICLE_ID, model=VEHICLE_MODEL)
    # Attach our State sensor under "agent_state" — matches the convention
    # in envs/beamng_env.py and avoids collision with BeamNG's default
    # "state" sensor.
    vehicle.sensors.attach("agent_state", State())
    scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_QUAT)
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()
    # Intentionally do NOT call set_deterministic — Mike needs BeamNG to
    # run in normal real-time mode so he can drive with the wheel/keyboard.
    return vehicle


def record_loop(vehicle, points, complete_event):
    """Wait for first motion, then record positions until complete_event fires.

    Two phases:
      1. Motion detection at POLL_INTERVAL_FAST_S — poll speed, wait for it
         to exceed MOTION_THRESHOLD_M_S. Don't record anything yet.
      2. Position recording at POLL_INTERVAL_RECORD_S — append (x, y, z) to
         points each tick.

    Both phases exit cleanly the moment complete_event is set (Enter pressed
    in the main thread).
    """
    # Phase 1: motion detection.
    while not complete_event.is_set():
        try:
            vehicle.sensors.poll()
            vel = vehicle.sensors["agent_state"]["vel"]
            speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
            if speed > MOTION_THRESHOLD_M_S:
                print("Recording started!", flush=True)
                break
        except Exception as e:
            print(f"Poll error (continuing): {e}", file=sys.stderr)
        if complete_event.wait(POLL_INTERVAL_FAST_S):
            return  # Enter pressed before the car moved — nothing to record.

    # Phase 2: position recording.
    while not complete_event.is_set():
        try:
            vehicle.sensors.poll()
            pos = vehicle.sensors["agent_state"]["pos"]
            points.append((float(pos[0]), float(pos[1]), float(pos[2])))
        except Exception as e:
            print(f"Poll error (continuing): {e}", file=sys.stderr)
        if complete_event.wait(POLL_INTERVAL_RECORD_S):
            return


def dedupe(points, min_dist):
    """Drop consecutive points closer than min_dist meters."""
    if not points:
        return []
    out = [points[0]]
    for p in points[1:]:
        if _dist3(p, out[-1]) >= min_dist:
            out.append(p)
    return out


def total_length(points):
    """Sum of distances between consecutive points (in meters)."""
    return sum(_dist3(points[i], points[i + 1])
               for i in range(len(points) - 1))


def _dist3(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                     + (a[2] - b[2]) ** 2)


def write_centerline(path: Path, points, length_m: float):
    """Write the recorded points to a Python module the env can import."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        '"""',
        'Recorded centerline for West Coast USA racetrack.',
        '',
        f'Recorded: {date.today().isoformat()}',
        f'Total length: {length_m:.1f} m',
        f'Points: {len(points)}',
        '',
        'Regenerate with: python scripts/record_centerline.py',
        '"""',
        '',
        'CENTERLINE = [',
    ]
    body = [f'    ({x:.2f}, {y:.2f}, {z:.2f}),' for x, y, z in points]
    footer = [']', '']
    path.write_text('\n'.join(header + body + footer), encoding='utf-8')


def main():
    args = parse_args()
    if not args.home:
        raise SystemExit("Set BEAMNG_HOME or pass --home (path to the BeamNG.tech install).")

    print("Launching BeamNG (West Coast USA, ETK 800)...")
    bng = BeamNGpy(args.host, args.port, home=args.home)
    bng.open(launch=True)

    vehicle = setup_scenario(bng)

    print()
    print("=" * 64)
    print("BeamNG is ready.")
    print("Take the wheel and drive a clean lap of the racetrack.")
    print("Stay in the middle of the lane.")
    print("Recording will start automatically when the car begins moving.")
    print("Press Enter (in this terminal) when the lap is complete.")
    print("=" * 64)
    print()

    points: list[tuple[float, float, float]] = []
    complete_event = threading.Event()
    recorder = threading.Thread(
        target=record_loop,
        args=(vehicle, points, complete_event),
        daemon=True,
    )
    recorder.start()

    try:
        input()
    except KeyboardInterrupt:
        print("\nInterrupted — saving what we have so far.")
    finally:
        complete_event.set()
        recorder.join(timeout=2.0)

    if not points:
        print("No points recorded. Did the car move? "
              f"(motion threshold is {MOTION_THRESHOLD_M_S} m/s)",
              file=sys.stderr)
        sys.exit(1)

    print(f"\nRaw points recorded: {len(points)}")
    deduped = dedupe(points, DEDUPE_MIN_DIST_M)
    print(f"After dedupe (>= {DEDUPE_MIN_DIST_M} m apart): {len(deduped)}")
    length = total_length(deduped)
    print(f"Total lap length: {length:.1f} m")

    write_centerline(OUTPUT_PATH, deduped, length)
    print(f"\nWritten to {OUTPUT_PATH}")
    print("Next step: git add data/centerline_racetrack.py && commit.")


if __name__ == "__main__":
    main()
