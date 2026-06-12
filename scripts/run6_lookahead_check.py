"""Static validation of the run6 fixed-arc lookahead (spec item b). No BeamNG.

Exercises BeamNGRaceEnv._point_at_arc directly (pure geometry; env __init__
does not connect to the sim). Cases: long straight, R15 hairpin, seam-straddle,
sparse-spacing region (max segment ~27.8 m, proves interpolation), dense region
(min segment ~0.6 m, proves binary-search resolution), and the ahead-of-car
regression assert (anchoring at the true arc position, never behind the car).

For each case prints the 6 lookahead points' (target arc, euclidean dist,
bearing vs the local tangent at the anchor) and independently RE-DERIVES each
resolved point's arc position by global projection, asserting it matches the
requested target (validates search + interpolation end to end).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.beamng_env import (BeamNGRaceEnv, CENTERLINE, LOOKAHEAD_DISTANCES_M)

env = BeamNGRaceEnv(random_spawn=False)   # no connect; geometry only
n = len(CENTERLINE)
L = env._track_length
cum = env._cum_arc


def seg_len(i):
    if i < n - 1:
        return cum[i + 1] - cum[i]
    return L - cum[n - 1]


def idx_at_arc(arc):
    arc %= L
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cum[mid] <= arc:
            lo = mid
        else:
            hi = mid - 1
    return lo


def tangent_at(arc):
    i = idx_at_arc(arc)
    p = CENTERLINE[(i - 1) % n]
    q = CENTERLINE[(i + 1) % n]
    return math.atan2(q[1] - p[1], q[0] - p[0])


def arc_of_point(p):
    """Independent re-derivation: globally project p onto the centerline."""
    best, best_d = 0.0, 1e18
    for i in range(n):
        a = CENTERLINE[i]
        b = CENTERLINE[(i + 1) % n]
        ax, ay = b[0] - a[0], b[1] - a[1]
        seg2 = ax * ax + ay * ay
        if seg2 < 1e-12:
            continue
        t = max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / seg2))
        qx, qy = a[0] + t * ax, a[1] + t * ay
        d2 = (p[0] - qx) ** 2 + (p[1] - qy) ** 2
        if d2 < best_d:
            best_d = d2
            best = (cum[i] + t * seg_len(i)) % L
    return best


def wrap_diff(a, b):
    """Forward arc distance from b to a on the loop, in [0, L)."""
    return (a - b) % L


sparse_i = max(range(n), key=seg_len)
dense_i = min(range(n), key=seg_len)
cases = [
    ("LONG STRAIGHT (idx ~500)", cum[500]),
    ("R15 HAIRPIN (idx 355)", cum[355]),
    ("SEAM STRADDLE (L - 50 m)", L - 50.0),
    (f"SPARSE mid-seg (idx {sparse_i}, seg {seg_len(sparse_i):.1f} m)",
     cum[sparse_i] + 0.9 * seg_len(sparse_i)),   # 90% along the longest segment
    (f"DENSE (idx {dense_i}, seg {seg_len(dense_i):.2f} m)", cum[dense_i]),
]

fails = 0
for label, anchor in cases:
    anchor %= L
    car = env._point_at_arc(anchor)
    tan = tangent_at(anchor)
    print(f"\n=== {label}  anchor arc={anchor:.1f} m ===")
    prev_eu = -1.0
    for d_ahead in LOOKAHEAD_DISTANCES_M:
        target = (anchor + d_ahead) % L
        p = env._point_at_arc(anchor + d_ahead)
        eu = math.sqrt((p[0] - car[0]) ** 2 + (p[1] - car[1]) ** 2)
        bear = math.degrees((math.atan2(p[1] - car[1], p[0] - car[0]) - tan
                             + math.pi) % (2 * math.pi) - math.pi)
        rearc = arc_of_point(p)
        arc_err = min(wrap_diff(rearc, target), wrap_diff(target, rearc))
        ahead = wrap_diff(rearc, anchor)
        ok_arc = arc_err < 3.0
        ok_ahead = 0.0 < ahead < d_ahead + 3.0
        ok_chord = eu <= d_ahead + 0.5
        flag = "" if (ok_arc and ok_ahead and ok_chord) else "  <-- FAIL"
        if flag:
            fails += 1
        print(f"  +{d_ahead:5.0f}m -> target arc {target:7.1f}  euclid {eu:6.1f} m"
              f"  bearing {bear:+7.1f} deg  re-derived arc {rearc:7.1f}"
              f"  (err {arc_err:.2f} m, ahead {ahead:6.1f} m){flag}")
        prev_eu = eu

print(f"\n{'ALL CHECKS PASSED' if fails == 0 else f'{fails} FAILURES'}")
sys.exit(0 if fails == 0 else 1)
