"""
Extract the racetrack centerline directly from BeamNG's DecalRoad geometry.

Plain-language overview, Mikey:
- BeamNG knows exactly where every road in the game is — it has to, in
  order to draw them. This script asks the game for that information.
- We pull the list of all drivable roads, find the one that's about 4 km
  long (the racetrack), and save its centerline to a file.
- That file replaces the wobbly hand-driven centerline we recorded earlier.

Two-pass workflow:
    1. python scripts/extract_centerline.py
       (prints a list of drivable roads sorted by length; pick the racetrack)
    2. python scripts/extract_centerline.py --road-id <picked_id>
       (writes data/centerline_racetrack_builtin.py)

The output module has the same `CENTERLINE = [(x, y, z), ...]` shape as
the recorded version, so the env can switch by changing one import line.
"""

import argparse
import math
import os
import sys
from datetime import date
from pathlib import Path

from beamngpy import BeamNGpy, Scenario, Vehicle


MAP_NAME = "west_coast_usa"
VEHICLE_MODEL = "etk800"
VEHICLE_ID = "ego_extract"
# Spawn just to start the scenario — this script never drives the car.
SPAWN_POS = (394.70, -252.02, 145.16)
SPAWN_QUAT = (0.0, 0.0, 0.0, 1.0)

DEFAULT_HOME = r"C:\BeamNG\BeamNG.tech.v0.38.5.0"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 25252
OUTPUT_PATH = Path("data/centerline_racetrack_builtin.py")
LIST_MIN_LENGTH_M = 1000.0  # only roads at least this long are candidates


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--home", default=os.environ.get("BEAMNG_HOME", DEFAULT_HOME),
                   help="BeamNG.tech install directory.")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--road-id", default=None,
                   help="If given, extract this road's centerline. Otherwise "
                        "just list candidates.")
    p.add_argument("--no-launch", dest="launch", action="store_false",
                   help="Attach to a running BeamNG instead of launching a "
                        "fresh one. Rarely useful; default is launch.")
    p.set_defaults(launch=True)
    return p.parse_args()


def setup_scenario(bng) -> None:
    """Load WCUSA with a dummy vehicle so the scenario is in the 'started' state."""
    scenario = Scenario(MAP_NAME, "extract_centerline")
    vehicle = Vehicle(VEHICLE_ID, model=VEHICLE_MODEL)
    scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_QUAT)
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()


def length_of_middle(edges):
    """Sum of distances between consecutive middle points (centerline length)."""
    total = 0.0
    for i in range(len(edges) - 1):
        a = edges[i]["middle"]
        b = edges[i + 1]["middle"]
        total += math.sqrt(
            (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
        )
    return total


def average_width(edges):
    """Mean distance from middle to one edge — gives a feel for road width."""
    if not edges:
        return 0.0
    widths = []
    for e in edges:
        l, m, r = e["left"], e["middle"], e["right"]
        # Use the average of (middle-to-left) and (middle-to-right) so the
        # number isn't sensitive to which side the road authoring favored.
        d_left = math.sqrt(sum((m[i] - l[i]) ** 2 for i in range(3)))
        d_right = math.sqrt(sum((m[i] - r[i]) ** 2 for i in range(3)))
        widths.append(d_left + d_right)
    return sum(widths) / len(widths)


def list_candidates(network):
    """Print drivable roads sorted by length, longest first.

    All rows shown are candidates (length >= LIST_MIN_LENGTH_M). Rows whose
    road ID contains "race" or "track" get a ★ next to their length so they
    stand out visually — but the racetrack might have a cryptic ID with no
    keyword match, so Mike should scan the whole list, not just stars.
    """
    rows = []
    for road_id, data in network.items():
        edges = data.get("edges", [])
        if not edges:
            continue
        length = length_of_middle(edges)
        if length < LIST_MIN_LENGTH_M:
            continue
        rows.append((
            road_id,
            length,
            len(edges),
            data.get("oneWay", False),
            data.get("drivability"),
            average_width(edges),
        ))
    rows.sort(key=lambda r: -r[1])

    print(f"\nDrivable roads >= {LIST_MIN_LENGTH_M:.0f} m, longest first "
          f"(all are candidates; ★ = name contains 'race' or 'track'):\n")
    hdr = (f"  {'road id':<45} {'length (m)':>11}  "
           f"{'points':>7} {'one-way':>8} {'drive':>6} {'width (m)':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for road_id, length, n_pts, oneway, drive, width in rows:
        name_l = str(road_id).lower()
        star = "★ " if ("race" in name_l or "track" in name_l) else "  "
        print(f"  {str(road_id):<45} {length:>11.1f}{star}{n_pts:>7} "
              f"{str(oneway):>8} {drive!s:>6} {width:>10.2f}")
    print("\nRe-run with --road-id <id> to extract one of these.")


def write_centerline(road_id, edges):
    """Save the road's middle points to data/centerline_racetrack_builtin.py."""
    centerline = [(e["middle"][0], e["middle"][1], e["middle"][2])
                  for e in edges]
    length = length_of_middle(edges)
    width = average_width(edges)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = [
        '"""',
        "Centerline for West Coast USA racetrack, extracted directly from",
        "BeamNG's DecalRoad geometry — the authoritative game data, with no",
        "recording wobble, dedup artifacts, or unclosed loops.",
        '',
        f'Extracted: {date.today().isoformat()}',
        f'Source road ID: {road_id}',
        f'Total length: {length:.1f} m',
        f'Average road width: {width:.2f} m',
        f'Points: {len(centerline)}',
        '',
        'Regenerate with: python scripts/extract_centerline.py --road-id <id>',
        '"""',
        '',
        'CENTERLINE = [',
    ]
    body = [f'    ({x:.2f}, {y:.2f}, {z:.2f}),' for x, y, z in centerline]
    footer = [']', '']
    OUTPUT_PATH.write_text('\n'.join(header + body + footer), encoding='utf-8')
    print(f"\nWrote {len(centerline)} points ({length:.1f} m, "
          f"~{width:.1f} m wide) to {OUTPUT_PATH}")


def main():
    args = parse_args()

    print(f"Connecting to BeamNG at {args.host}:{args.port}...")
    bng = BeamNGpy(args.host, args.port, home=args.home)
    bng.open(launch=args.launch)

    print(f"Loading scenario on {MAP_NAME} (needed for road-data API)...")
    setup_scenario(bng)

    print("Fetching road network (this can take 5-15 seconds)...")
    network = bng.scenario.get_road_network(include_edges=True,
                                            drivable_only=True)
    # Normalize road-ID keys to strings — BeamNGpy returns them as floats/ints
    # in some versions, which breaks `.lower()` and `args.road_id in network`
    # lookups against a CLI string.
    network = {str(k): v for k, v in network.items()}
    print(f"Found {len(network)} drivable roads.")

    if args.road_id is None:
        list_candidates(network)
        return

    if args.road_id not in network:
        print(f"\nERROR: road ID '{args.road_id}' not in the network. Run "
              "without --road-id to see the candidate list.", file=sys.stderr)
        sys.exit(1)

    edges = network[args.road_id].get("edges", [])
    if not edges:
        print(f"\nERROR: road '{args.road_id}' has no edges to extract.",
              file=sys.stderr)
        sys.exit(1)

    write_centerline(args.road_id, edges)


if __name__ == "__main__":
    main()
