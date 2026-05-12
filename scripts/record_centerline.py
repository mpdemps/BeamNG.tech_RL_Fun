"""
Record the centerline of West Coast USA's racetrack by polling the
vehicle's position while Mike drives a lap manually.

Plain-language overview, Mikey:
- Dad drives the car around the racetrack once, normally.
- This script writes down where the car is every half-second.
- It cleans up the list (drops duplicate points from when the car was
  stopped or barely moving).
- It saves the cleaned racing line to data/centerline_racetrack.py.
- The training script reads that file the next time we train, so the AI
  knows the shape of the track.

Run this ONCE for the project, then commit data/centerline_racetrack.py.
Future training (this machine or any other) reuses the file — no need to
re-record unless we switch tracks.

Usage:
    1. Start BeamNG.tech, load West Coast USA, spawn the ETK 800 on the
       racetrack starting line.
    2. From the repo root: python scripts/record_centerline.py
    3. Follow the on-screen instructions, then drive a lap.
    4. Press Enter when you cross the finish line.
"""

import argparse
import math
import os
import sys
import threading
from datetime import date
from pathlib import Path

from beamngpy import BeamNGpy
from beamngpy.sensors import State


POLL_INTERVAL_S = 0.5
DEDUPE_MIN_DIST_M = 1.0
DEFAULT_HOME = r"C:\BeamNG\BeamNG.tech.v0.38.5.0"
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


def get_vehicles_dict(bng):
    """Return {vid: Vehicle} across BeamNGpy API versions."""
    if hasattr(bng, "vehicles") and hasattr(bng.vehicles, "get_current"):
        return bng.vehicles.get_current()
    # Older BeamNGpy fallback.
    return bng.get_current_vehicles()


def pick_vehicle(bng):
    """Return the player's vehicle from the running BeamNG instance."""
    vehicles = get_vehicles_dict(bng)
    if not vehicles:
        raise RuntimeError(
            "No vehicles found in BeamNG. Spawn the ETK 800 on the "
            "racetrack first, then re-run this script."
        )
    if len(vehicles) == 1:
        return next(iter(vehicles.values()))
    # Multiple vehicles — prefer an explicit player getter if the BeamNGpy
    # version exposes one, otherwise take the first and warn.
    for attr in ("get_player_vehicle", "get_active"):
        try:
            v = getattr(bng.vehicles, attr)()
            if v is not None:
                return v
        except Exception:
            continue
    v = next(iter(vehicles.values()))
    print(f"WARNING: multiple vehicles found ({list(vehicles.keys())}); "
          f"using '{v.vid}'. Despawn the others if this is wrong.",
          file=sys.stderr)
    return v


def poll_loop(vehicle, points, interval, stop_event):
    """Background polling: appends (x, y, z) until stop_event fires."""
    while not stop_event.is_set():
        try:
            vehicle.sensors.poll()
            pos = vehicle.sensors["agent_state"]["pos"]
            points.append((float(pos[0]), float(pos[1]), float(pos[2])))
        except Exception as e:
            print(f"Poll error (continuing): {e}", file=sys.stderr)
        # Sleep, but wake immediately if stop_event is set during the wait.
        if stop_event.wait(interval):
            break


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

    print("Connecting to BeamNG (must already be running)...")
    bng = BeamNGpy(args.host, args.port, home=args.home)
    bng.open(launch=False)

    vehicle = pick_vehicle(bng)
    print(f"Using vehicle: {vehicle.vid}")

    # Attach our State sensor under "agent_state" — matches the convention
    # in envs/beamng_env.py and avoids collision with BeamNG's default
    # "state" sensor.
    vehicle.sensors.attach("agent_state", State())

    print()
    print("=" * 64)
    print("Drive a full lap of the racetrack at moderate speed (40-60 mph).")
    print("Stay in the middle of the lane.")
    print("Press Enter when finished.")
    print("=" * 64)
    print()

    points: list[tuple[float, float, float]] = []
    stop_event = threading.Event()
    poller = threading.Thread(
        target=poll_loop,
        args=(vehicle, points, POLL_INTERVAL_S, stop_event),
        daemon=True,
    )
    poller.start()

    try:
        input()
    except KeyboardInterrupt:
        print("\nInterrupted — saving what we have so far.")
    finally:
        stop_event.set()
        poller.join(timeout=2.0)

    if not points:
        print("No points recorded. Aborting (did the vehicle move?).",
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
