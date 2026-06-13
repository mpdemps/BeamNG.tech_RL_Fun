"""Deterministic validation of the run8 anti-weave penalty + two requested checks.

A) Four signatures (reads info['weave_penalty'] from the reward path):
   straight weave -> fires; centered straight -> ~0; corner sweep -> ~0;
   drift-back-to-line -> ~0.
B) on_line risk: measure peak |center_off| during a full-amplitude nose weave on
   a straight, against WEAVE_OFF_DEAD=1.0m (does a weave swing the car past the
   gate's on-line band?).
C) turn-in timing: for the kink (R252), a median (R143) and the R40, report how
   far AHEAD of corner entry the straightness gate releases (want >=30m).

Pure env (geometry) + a few teleport-and-step sequences. Port 25252.
"""

import bisect
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.beamng_env import (
    make_beamng_env, _shared, CENTERLINE, _yaw_to_quat,
    SPAWN_QUAT_YAW_CORRECTION_DEG, SPAWN_Z_OFFSET_M,
    WEAVE_LOOK_M, WEAVE_BEND_DEAD, WEAVE_BEND_FULL, WEAVE_WEIGHT)


def teleport_arc(env, arc, heading_off=0.0, lateral_off=0.0):
    """Place the car at arc-position `arc`, optionally displaced `lateral_off`
    metres perpendicular to the centerline (for the drift-back case), facing
    down-track (+ heading_off rad), at rest."""
    n = len(CENTERLINE)
    i = min(max(bisect.bisect_right(env._cum_arc, arc % env._track_length) - 1,
                0), n - 1)
    tan = env._smoothed_forward_yaw(i)
    sent = -(tan + heading_off) - math.radians(SPAWN_QUAT_YAW_CORRECTION_DEG)
    p = CENTERLINE[i]
    nx, ny = -math.sin(tan), math.cos(tan)   # unit normal to the tangent (XY)
    px, py = p[0] + lateral_off * nx, p[1] + lateral_off * ny
    _shared["vehicle"].teleport((px, py, p[2] + SPAWN_Z_OFFSET_M),
                                rot_quat=_yaw_to_quat(sent), reset=True)
    # Sync the windowed progress tracker to the teleport target (it only searches
    # +/-20 idx around its last position, so a far jump would otherwise desync and
    # corrupt center_off / the lookahead).
    env._progress_idx = i
    env._cur_centerline_dist = env._cum_arc[i]
    env._last_centerline_dist = env._cum_arc[i]
    for _ in range(4):                       # let the teleport settle
        env.step([0.0, 0.0])
        env._progress_idx = i                # hold it through settle (car ~stationary)
        env._cur_centerline_dist = env._cum_arc[i]
        env._last_centerline_dist = env._cum_arc[i]


def main():
    home = os.environ.get("BEAMNG_HOME")
    if not home:
        raise SystemExit("Set BEAMNG_HOME.")
    env = make_beamng_env(random_spawn=False, home=home, launch=True,
                          headless=True, nogpu=True)
    env.reset()
    n = len(CENTERLINE)

    # ---- (C) turn-in release distance: pure geometry, no driving ----
    # For a corner apex index, the gate releases at the arc where bend-over-
    # WEAVE_LOOK_M first exceeds WEAVE_BEND_DEAD. Release distance = apex_arc -
    # that arc (how far ahead of the apex the gate opens).
    print(f"\n=== (C) turn-in: straightness gate release vs corner entry "
          f"(LOOK={WEAVE_LOOK_M:.0f}m, releases as bend>{math.degrees(WEAVE_BEND_DEAD):.0f}deg) ===",
          flush=True)

    def bend_at(arc):
        i = min(max(bisect.bisect_right(env._cum_arc, arc % env._track_length) - 1, 0), n - 1)
        j = min(max(bisect.bisect_right(env._cum_arc, (arc + WEAVE_LOOK_M) % env._track_length) - 1, 0), n - 1)
        return abs((env._smoothed_forward_yaw(j) - env._smoothed_forward_yaw(i)
                    + math.pi) % (2 * math.pi) - math.pi)

    for label, apex_idx in [("R252 kink", 534), ("R143 median", 369),
                            ("R40 first corner", 300), ("R15 hairpin", 355)]:
        apex_arc = env._cum_arc[apex_idx]
        rel = None
        for back in range(0, 200):           # scan backward from the apex
            a = apex_arc - back
            if bend_at(a) < WEAVE_BEND_DEAD:
                rel = back                    # gate is ON this far back; release is just inside
                break
        # release distance = how far before apex the gate first opens
        opened = None
        for back in range(200, 0, -1):
            if bend_at(apex_arc - back) >= WEAVE_BEND_DEAD:
                opened = back
                break
        print(f"  {label:18s} apex idx {apex_idx}: gate releases ~{opened}m "
              f"before apex" if opened else f"  {label}: (no release in 200m)",
              flush=True)

    # ---- (A) four signatures + (B) weave lateral excursion ----
    print("\n=== (A) signatures + (B) weave lateral excursion ===", flush=True)
    straight_arc = env._cum_arc[500]   # the long 694m straight

    # 1) STRAIGHT WEAVE: oscillate steer +/-0.8 on the straight, centered.
    teleport_arc(env, straight_arc)
    wp, off = [], []
    for step in range(20):
        st = 0.8 if (step // 1) % 2 == 0 else -0.8
        _, r, term, trunc, info = env.step([st, 0.2])
        wp.append(info['weave_penalty'])
        off.append(abs(env._center_off))
        if term or trunc:
            teleport_arc(env, straight_arc)
    print(f"  STRAIGHT WEAVE (+/-0.8): weave_penalty mean={sum(wp)/len(wp):+.3f} "
          f"min={min(wp):+.3f}  | peak |center_off|={max(off):.2f}m vs OFF_DEAD=1.0m",
          flush=True)

    # 2) CENTERED STRAIGHT: hold steer ~0.
    teleport_arc(env, straight_arc)
    wp = []
    for _ in range(10):
        _, r, t, tr, info = env.step([0.0, 0.2]); wp.append(info['weave_penalty'])
    print(f"  CENTERED STRAIGHT (steer 0): weave_penalty max={max(wp,key=abs):+.3f}", flush=True)

    # 3) CORNER SWEEP: at the R40 approach, sustained steer.
    teleport_arc(env, env._cum_arc[300] - 30)   # 30m before the R40 apex
    wp = []
    for _ in range(10):
        _, r, t, tr, info = env.step([0.6, 0.2]); wp.append(info['weave_penalty'])
        if t or tr: break
    print(f"  CORNER SWEEP (R40 approach, steer 0.6): weave_penalty max={max(wp,key=abs):+.3f}", flush=True)

    # 4) DRIFT-BACK: genuinely LATERALLY displaced 3m off the line, steer back.
    teleport_arc(env, straight_arc, lateral_off=3.0)
    wp, offs = [], []
    for _ in range(10):
        _, r, t, tr, info = env.step([-0.5, 0.2])     # steer back toward line
        wp.append(info['weave_penalty']); offs.append(abs(env._center_off))
        if t or tr: break
    print(f"  DRIFT-BACK (3m off-line, steer -0.5): weave_penalty max={max(wp,key=abs):+.3f} "
          f"  |center_off| range {min(offs):.2f}-{max(offs):.2f}m", flush=True)

    env.close()
    try:
        _shared["bng"].close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
