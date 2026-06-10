"""Empirical sweep of the four spawn-heading formula candidates.

No reasoning about quaternion handedness -- just spawn with each candidate,
settle, and MEASURE the resulting forward vector. For each formula we set the
sent yaw, teleport the car directly to idx=0 with that orientation, step ~10x at
zero throttle to settle, read the ACTUAL settled "dir" forward vector, and report
the angle between it and the centerline tangent. The winner is whichever yields
~0 deg. We then re-measure the winner at indices around the whole loop (a
handedness/mirror bug shows DIFFERENT errors at different headings, so a real fix
must be ~0 everywhere, not just at idx=0).

Candidates (deg), where `desired` is the centerline tangent heading:
    A: sent =  desired - 90
    B: sent =  desired + 90
    C: sent = -desired - 90   (current code, confirmed wrong on G14: faces left)
    D: sent = -desired + 90

Run from repo root with BEAMNG_HOME set:
    python scripts/spawn_formula_sweep.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.beamng_env import (
    make_beamng_env, _shared, CENTERLINE, _yaw_to_quat, SPAWN_Z_OFFSET_M,
)

SETTLE_STEPS = 10


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def tangent_deg(idx):
    n = len(CENTERLINE)
    prev_pt = CENTERLINE[(idx - 1) % n]
    next_pt = CENTERLINE[(idx + 1) % n]
    return math.degrees(math.atan2(next_pt[1] - prev_pt[1],
                                   next_pt[0] - prev_pt[0]))


FORMULAS = {
    "A: desired-90":  lambda d:  d - 90.0,
    "B: desired+90":  lambda d:  d + 90.0,
    "C: -desired-90": lambda d: -d - 90.0,
    "D: -desired+90": lambda d: -d + 90.0,
}


def spawn_and_measure(idx, sent_yaw_deg):
    """Teleport to idx with the given sent yaw (deg), settle, return settled
    forward yaw (deg)."""
    cx, cy, cz = CENTERLINE[idx]
    pos = (cx, cy, cz + SPAWN_Z_OFFSET_M)
    quat = _yaw_to_quat(math.radians(sent_yaw_deg))
    v = _shared["vehicle"]
    v.teleport(pos, rot_quat=quat, reset=True)
    for _ in range(SETTLE_STEPS):
        v.control(steering=0.0, throttle=0.0, brake=0.0)
        _shared["bng"].step(3)
    v.sensors.poll()
    d = v.sensors["agent_state"]["dir"]
    return math.degrees(math.atan2(d[1], d[0])), d


def main():
    home = os.environ.get("BEAMNG_HOME")
    if not home:
        raise SystemExit("Set BEAMNG_HOME.")

    env = make_beamng_env(
        random_spawn=False, home=home, host="localhost", port=25252,
        launch=True, headless=True, nogpu=True,
    )
    env.reset()  # connect + load scenario

    idx = 0
    tan = tangent_deg(idx)
    print(f"\n=== Four-formula sweep at idx={idx} "
          f"(centerline tangent = {tan:+.2f} deg) ===\n", flush=True)

    results = {}
    for label, fn in FORMULAS.items():
        sent = fn(tan)
        dir_yaw, d = spawn_and_measure(idx, sent)
        err = wrap_deg(dir_yaw - tan)
        results[label] = err
        print(f"{label:16s}: sent_yaw={sent:+8.2f}  settled dir_yaw={dir_yaw:+8.2f}"
              f"  err={err:+7.2f} deg  dir=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})",
              flush=True)

    print("\n(NOTE: at idx=0 the tangent is ~+-180, so desired-90 and "
          "-desired-90 collapse to nearly the same spawn -- idx=0 alone cannot "
          "distinguish them. The loop validation below is the real test.)",
          flush=True)

    # Validate ALL FOUR around the whole loop. Only a TRUE frame fix reads ~0 at
    # every heading; a formula that merely coincides at idx=0 diverges elsewhere.
    n = len(CENTERLINE)
    sweep_idx = [int(round(f * n)) % n
                 for f in (0.0, 0.125, 0.25, 0.375, 0.5,
                           0.625, 0.75, 0.875, 0.97)]
    print(f"\n=== Validating ALL FOUR formulas across {len(sweep_idx)} indices "
          f"===\n", flush=True)
    loop_stats = {}
    for label, fn in FORMULAS.items():
        errs = []
        for i in sweep_idx:
            t = tangent_deg(i)
            dir_yaw, _ = spawn_and_measure(i, fn(t))
            errs.append(wrap_deg(dir_yaw - t))
        mean = sum(errs) / len(errs)
        std = math.sqrt(sum((e - mean) ** 2 for e in errs) / len(errs))
        worst = max(errs, key=abs)
        loop_stats[label] = (mean, std, worst)
        print(f"{label:16s}: mean={mean:+8.2f}  std={std:6.2f}  "
              f"worst={worst:+8.2f} deg   "
              f"per-idx=[{', '.join(f'{e:+.0f}' for e in errs)}]", flush=True)

    winner = min(loop_stats, key=lambda k: abs(loop_stats[k][2]))  # smallest worst-case
    print(f"\n=== WINNER (smallest error across the whole loop): {winner} ===",
          flush=True)
    print("(Headless readback. Mike re-validates this winner visually on the "
          "G14 with UP-TO-DATE code before launch.)", flush=True)

    env.close()
    try:
        _shared["bng"].close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
