"""run17 §1 measurement: the racetrack road half-width along the lap, to set
OFF_TRACK_THRESHOLD = max single-side half-width + margin (end the episode the instant the
car leaves the road, but never false-terminate on the widest legal road). The env's
center_off is perp distance to the local centerline, so the relevant number is how far the
EDGE sits from the middle (per side). No training; reads the DecalRoad edges. Port 25252."""
import math, os, sys
from beamngpy import BeamNGpy, Scenario, Vehicle

MAP = "west_coast_usa"; ROAD_ID = "59564.0"
SPAWN_POS = (394.70, -252.02, 145.16); SPAWN_QUAT = (0.0, 0.0, 0.0, 1.0)


def main():
    home = os.environ["BEAMNG_HOME"]
    bng = BeamNGpy("localhost", 25252, home=home, headless=True, nogpu=True)
    bng.open(launch=True)
    sc = Scenario(MAP, "width_probe")
    sc.add_vehicle(Vehicle("ego_w", model="etk800"), pos=SPAWN_POS, rot_quat=SPAWN_QUAT)
    sc.make(bng); bng.scenario.load(sc); bng.scenario.start()
    net = {str(k): v for k, v in bng.scenario.get_road_network(include_edges=True, drivable_only=True).items()}
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.centerline_racetrack_builtin import CENTERLINE
    # the racetrack may be split into several road segments this session; collect edges
    # from EVERY road whose middle points lie on our saved CENTERLINE (geometric match,
    # robust to id/segmentation changes). sample CENTERLINE for a fast nearest test.
    cl = [(p[0], p[1]) for p in CENTERLINE[::8]]

    def on_track(es):
        hit = 0
        for e in es:
            m = e["middle"]
            if min((m[0]-c[0])**2 + (m[1]-c[1])**2 for c in cl) < 9.0:  # within 3m of centerline
                hit += 1
        return hit / max(len(es), 1)

    edges = []
    matched = []
    for rid, d in net.items():
        es = d.get("edges", [])
        if es and on_track(es) > 0.6:
            edges += es; matched.append((rid, len(es)))
    print(f"matched {len(matched)} racetrack road segment(s): {matched[:8]}")
    print(f"total {len(edges)} edge pts on our centerline\n")
    if not edges:
        print("NO geometric match — dumping road lengths for diagnosis:")
        for rid, d in sorted(net.items(), key=lambda x: -len(x[1].get("edges", []) or [])):
            es = d.get("edges", [])
            if len(es) > 30:
                print(f"  id={rid} pts={len(es)} overlap={on_track(es):.2f}")
        bng.close(); return

    dl = []; dr = []
    for e in edges:
        l, m, r = e["left"], e["middle"], e["right"]
        dl.append(math.sqrt(sum((m[i]-l[i])**2 for i in range(3))))
        dr.append(math.sqrt(sum((m[i]-r[i])**2 for i in range(3))))
    halfmax = [max(a, b) for a, b in zip(dl, dr)]   # worst single-side half-width per point
    import statistics as st
    print(f"left half-width  (m): min {min(dl):.2f}  mean {st.mean(dl):.2f}  max {max(dl):.2f}")
    print(f"right half-width (m): min {min(dr):.2f}  mean {st.mean(dr):.2f}  max {max(dr):.2f}")
    print(f"max single-side half-width along lap: {max(halfmax):.2f} m (mean {st.mean(halfmax):.2f}, p95 {sorted(halfmax)[int(len(halfmax)*0.95)]:.2f})")
    print(f"full width: mean {st.mean([a+b for a,b in zip(dl,dr)]):.2f} m (matches saved 11.41)")
    for margin in (1.0, 1.5, 2.0):
        print(f"  OFF_TRACK_THRESHOLD = max_half_width + {margin:.1f}m = {max(halfmax)+margin:.1f} m")
    bng.close()


if __name__ == "__main__":
    main()
