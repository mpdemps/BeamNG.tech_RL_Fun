"""run20 prerequisite: extract per-node LEFT/RIGHT road edges for the racetrack.

The centerline file saved only the road 'middle' + an average width (11.41 m). The
min-curvature racing line needs the actual edges to stay inside the track, so this pulls
the same DecalRoad geometry extract_centerline.py used (get_road_network include_edges)
and saves left/middle/right per node to data/track_edges_builtin.py, ALIGNED to CENTERLINE.

It verifies the extracted 'middle' matches the saved CENTERLINE (same count, tiny max
deviation) so the edges are registered to the path the env already drives. If road 59564
isn't found (road IDs were unstable in run17), it falls back to the longest drivable road
and reports the match so we can see whether to trust it or synthesize constant-width edges.

Usage: python scripts/extract_edges.py   (BEAMNG_HOME set; launches headless)"""
import math, os, sys
from datetime import date
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from data.centerline_racetrack_builtin import CENTERLINE

MAP_NAME = "west_coast_usa"
VEHICLE_MODEL = "etk800"
SPAWN_POS = (394.70, -252.02, 145.16)
SPAWN_QUAT = (0.0, 0.0, 0.0, 1.0)
SOURCE_ROAD_ID = "59564.0"
OUTPUT_PATH = Path("data/track_edges_builtin.py")


def setup_scenario(bng):
    scenario = Scenario(MAP_NAME, "extract_edges")
    vehicle = Vehicle("ego_extract", model=VEHICLE_MODEL)
    scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_QUAT)
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()


def middle_len(edges):
    return sum(math.dist(edges[i]["middle"], edges[i + 1]["middle"])
               for i in range(len(edges) - 1))


def main():
    home = os.environ["BEAMNG_HOME"]
    bng = BeamNGpy("localhost", 25252, home=home, headless=True, nogpu=True)
    bng.open(launch=True)
    try:
        print(f"Loading {MAP_NAME}...")
        setup_scenario(bng)
        print("Fetching road network...")
        net = {str(k): v for k, v in
               bng.scenario.get_road_network(include_edges=True, drivable_only=True).items()}
        print(f"Found {len(net)} drivable roads.")

        rid = SOURCE_ROAD_ID
        if rid not in net:
            # fall back to the longest road and report
            rid = max(net, key=lambda k: middle_len(net[k].get("edges", [])) if net[k].get("edges") else 0)
            print(f"Road {SOURCE_ROAD_ID} NOT found; falling back to longest road {rid}")
        edges = net[rid].get("edges", [])
        print(f"Road {rid}: {len(edges)} nodes, middle length {middle_len(edges):.1f} m")

        mids = np.array([e["middle"] for e in edges])
        cl = np.array(CENTERLINE)
        # registration check vs saved CENTERLINE
        if len(mids) == len(cl):
            dev = np.linalg.norm(mids - cl, axis=1)
            print(f"REGISTRATION vs CENTERLINE: same count ({len(cl)}); "
                  f"middle deviation max={dev.max():.3f}m mean={dev.mean():.3f}m")
        else:
            print(f"REGISTRATION: COUNT MISMATCH extracted={len(mids)} vs CENTERLINE={len(cl)} "
                  f"-> edges NOT aligned; inspect before trusting (may need constant-width fallback)")

        halfw = np.array([(math.dist(e["middle"], e["left"]) + math.dist(e["middle"], e["right"])) / 2.0
                          for e in edges])
        print(f"half-width: min={halfw.min():.2f} mean={halfw.mean():.2f} max={halfw.max():.2f} m")

        lines = ['"""', "Per-node LEFT/RIGHT road edges for the WCUSA racetrack, from BeamNG DecalRoad",
                 "geometry (get_road_network include_edges). Aligned to CENTERLINE for the run20",
                 "min-curvature racing line. left/right are the drivable-surface boundaries.", "",
                 f"Extracted: {date.today().isoformat()}", f"Source road ID: {rid}",
                 f"Nodes: {len(edges)}", f"half-width min/mean/max: {halfw.min():.2f}/{halfw.mean():.2f}/{halfw.max():.2f} m",
                 '"""', "", "LEFT_EDGE = ["]
        lines += [f"    ({e['left'][0]:.2f}, {e['left'][1]:.2f}, {e['left'][2]:.2f})," for e in edges]
        lines += ["]", "", "RIGHT_EDGE = ["]
        lines += [f"    ({e['right'][0]:.2f}, {e['right'][1]:.2f}, {e['right'][2]:.2f})," for e in edges]
        lines += ["]", ""]
        OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nWrote {OUTPUT_PATH} (LEFT_EDGE + RIGHT_EDGE, {len(edges)} nodes)")
    finally:
        bng.close()


if __name__ == "__main__":
    main()
