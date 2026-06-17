"""run16 §5.1 offline probe: generate the braking-aware V_TARGET profile over the
centerline and confirm it is sane and braking-distance-aware. No BeamNG."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.centerline_racetrack_builtin import CENTERLINE
from envs.speed_profile import (compute_speed_profile, A_LAT_MAX, A_BRAKE, A_ACCEL, V_MAX)
import bisect, math

# corners (apex arc, radius) from docs/track_reference.md
CORNERS = [("T1", 338, 55), ("T2", 596, 108), ("T3", 746, 82), ("T4", 1004, 55),
           ("T5", 1274, 35), ("T6", 1336, 34), ("T7", 1610, 133), ("T8", 2990, 68),
           ("T9", 3280, 139), ("T10", 3390, 65), ("T11", 3574, 40), ("T12", 3610, 51)]


def main():
    v, R, cum, track, _kappa = compute_speed_profile(CENTERLINE)
    n = len(v)
    print(f"profile over {n} pts, track={track:.0f}m | A_LAT_MAX={A_LAT_MAX} A_BRAKE={A_BRAKE} "
          f"A_ACCEL={A_ACCEL} V_MAX={V_MAX}")
    print(f"v_target: min={min(v):.1f} max={max(v):.1f} m/s\n")

    def at(arc):  # interpolate v_target / R at an arc
        arc %= track
        i = min(bisect.bisect_right(cum, arc) - 1, n - 1)
        return v[i], R[i]

    print(f"{'corner':>6}{'apex':>6}{'R_ref':>6}{'R_meas':>7}{'v@apex':>8}{'sqrt(aR)':>9}")
    for name, apex, Rr in CORNERS:
        va, Ra = at(apex)
        print(f"{name:>6}{apex:>6}{Rr:>6}{Ra:>7.0f}{va:>8.1f}{math.sqrt(A_LAT_MAX*Rr):>9.1f}")

    # THE braking-aware check: T1 target must START FALLING before the 338 apex,
    # over the braking distance from straight speed (~33) to the T1 corner speed.
    print("\n=== T1 approach (apex 338m): target must fall BEFORE the apex ===")
    print(f"   {'arc':>5}{'v_target':>9}")
    for arc in range(280, 360, 5):
        va, _ = at(arc)
        print(f"   {arc:>5}{va:>9.1f}")
    v_corner = min(at(338)[0], at(330)[0])
    v_straight = at(295)[0]
    # where does it cross down through (v_straight - 1)?
    start = next((arc for arc in range(280, 340) if at(arc)[0] < v_straight - 1.0), None)
    need = (v_straight**2 - v_corner**2) / (2 * A_BRAKE) if v_straight > v_corner else 0
    print(f"\n   straight v(~295m)={v_straight:.1f}  T1 corner v={v_corner:.1f}  "
          f"braking dist needed={need:.0f}m -> target should start dropping by ~arc {338-need:.0f}")
    print(f"   measured: target starts dropping at ~arc {start}  "
          f"({'BRAKING-AWARE (before apex)' if start and start < 338 else 'FAIL: drops at/after apex'})")

    # straights should be at V_MAX
    print(f"\n   opening straight v(~150m)={at(150)[0]:.1f} (expect ~{V_MAX})")
    print(f"   hairpin T5 v(~1274m)={at(1274)[0]:.1f} (expect ~{math.sqrt(A_LAT_MAX*35):.1f})")


if __name__ == "__main__":
    main()
