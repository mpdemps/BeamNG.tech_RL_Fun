"""run13 measure-first: the corner-rate FLOOR. The run12 policy spins on the opening
straight, so it has no clean corner data. Instead drive the centerline with a clean
pure-pursuit tracker (speed-scaled lookahead like run10's reference, curvature-based
speed target so sharp corners are taken slow / straights fast) and measure the
|Δsteer|/step the tracker USES to negotiate each corner. That rate is the floor
STEER_RATE must stay above. Compare to the run12 spin-slam (Δ1.31) ceiling.

Separate port 25253; run12 untouched."""
import bisect
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import SAC  # noqa
from envs.beamng_env import make_beamng_env, _shared, CENTERLINE

PORT = 25253
A_LAT = 9.0            # m/s^2 cornering grip target (~0.9g) for the speed setpoint
V_MIN, V_MAX = 8.0, 33.0
GAIN = 1.6             # pure-pursuit steer gain (bearing_err normalized by 45deg)
STEPS = 2600


def main():
    import numpy as np
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost",
                          port=PORT, launch=True, headless=True, nogpu=True)
    env.reset()
    TRACK = env._track_length

    def yaw_at(arc):  # centerline tangent at an arc (for curvature)
        i = min(max(bisect.bisect_right(env._cum_arc, arc % TRACK) - 1, 0), len(CENTERLINE) - 1)
        return env._smoothed_forward_yaw(i)

    def curv_ahead(arc):  # |dθ| over the next 25 m -> curvature
        d = abs((yaw_at(arc + 25) - yaw_at(arc) + math.pi) % (2 * math.pi) - math.pi)
        return d / 25.0

    rows = []
    prev_steer = 0.0
    for step in range(STEPS):
        s = _shared["vehicle"].sensors["agent_state"]
        pos = s["pos"]; vel = s["vel"]; fdir = s.get("dir", (1.0, 0.0, 0.0))
        v = math.hypot(vel[0], vel[1])
        arc = env._cur_centerline_dist
        # pure-pursuit aim point (speed-scaled lookahead)
        L = min(45.0, max(12.0, 1.5 * v))
        aim = env._point_at_arc(arc + L)
        car_yaw = math.atan2(fdir[1], fdir[0])
        bearing = math.atan2(aim[1] - pos[1], aim[0] - pos[0])
        err = (bearing - car_yaw + math.pi) % (2 * math.pi) - math.pi
        steer = max(-1.0, min(1.0, GAIN * err / (math.pi / 4)))
        # curvature-based speed setpoint (min radius over next 40 m)
        kappa = max(curv_ahead(arc), curv_ahead(arc + 20), 1e-4)
        v_t = max(V_MIN, min(V_MAX, math.sqrt(A_LAT / kappa)))
        thr = max(0.0, min(0.6, 0.35 + 0.25 * (v_t - v)))
        brake = 0.4 if v > v_t + 4 else 0.0
        act_thr = thr if brake == 0 else 0.0
        env.step([steer, act_thr if brake == 0 else -brake])
        rows.append(dict(step=step, arc=arc, v=v, steer=steer,
                         dsteer=abs(steer - prev_steer), v_t=v_t,
                         coff=env._center_off, R=1.0 / kappa if kappa > 1e-4 else 9999))
        prev_steer = steer
    with open("logs/run13_steer_rate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    env.close()
    try: _shared["bng"].close()
    except Exception: pass

    # did it complete a clean lap? track-keeping quality
    import statistics as st
    maxarc = max(r["arc"] for r in rows)
    offs = [abs(r["coff"]) for r in rows]
    print(f"\n=== pure-pursuit tracker drive: reached arc {maxarc:.0f}m / {TRACK:.0f}m; "
          f"|center_off| median={st.median(offs):.2f}m p90={sorted(offs)[int(len(offs)*0.9)]:.2f}m ===")
    print("  (low center_off = clean tracking -> the dsteer it uses is a valid corner-rate floor)")
    # per-corner |dsteer|/step (corners from track_reference apex arcs)
    CORNERS = [("T1", 338, 55), ("T2", 596, 108), ("T3", 746, 82), ("T4", 1004, 55),
               ("T5", 1274, 35), ("T6", 1336, 34), ("T7", 1610, 133), ("T8", 2990, 68),
               ("T9", 3280, 139), ("T10", 3390, 65), ("T11", 3574, 40), ("T12", 3610, 51),
               ("T15", 4000, 76), ("T16", 4058, 53)]
    print(f"\n{'corner':>6} {'R':>4} {'entryV':>6} {'max|dS|/step':>12} {'p90|dS|':>8} {'steer_pk':>8}")
    floor = 0.0
    for name, apex, R in CORNERS:
        seg = [r for r in rows if (apex - 60) <= (r["arc"] % TRACK) <= (apex + 20)]
        if not seg:
            print(f"{name:>6} {R:>4} {'(not reached)':>6}")
            continue
        entryv = seg[0]["v"]
        mx = max(r["dsteer"] for r in seg); p90 = sorted(r["dsteer"] for r in seg)[int(len(seg)*0.9)]
        spk = max(abs(r["steer"]) for r in seg)
        floor = max(floor, p90)
        print(f"{name:>6} {R:>4} {entryv:>6.1f} {mx:>12.3f} {p90:>8.3f} {spk:>8.2f}")
    print(f"\n  corner-rate FLOOR (max p90 across corners) ~ {floor:.3f} /step")
    print(f"  spin-slam CEILING (run12) = 1.31 /step ; straight-hold avg = 0.08 /step")


if __name__ == "__main__":
    main()
