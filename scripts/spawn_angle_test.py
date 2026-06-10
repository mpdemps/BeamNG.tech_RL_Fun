"""Task 1 (mikey_run2): measure the ACTUAL settled spawn orientation.

Mirrors the real training spawn path: make_beamng_env(random_spawn=False),
reset() (spawns at centerline idx=0), step ~10 times at zero throttle so the
teleport settles, then read the car's settled forward vector ("dir") from the
state sensor and compute the signed angle between it and the centerline tangent
at idx=0. NOT the hand-tangent/intended-heading check -- that only validates the
math; this measures the outcome. Repeat N times and report the distribution.

Run from repo root with BEAMNG_HOME set:
    python scripts/spawn_angle_test.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.beamng_env import make_beamng_env, _shared, CENTERLINE

SETTLE_STEPS = 10
# Spread spawn indices around the whole 985-point loop so we prove the spawn
# fix is a GENERAL frame transform (err ~ 0 everywhere), not an idx=0-only tune.
N = len(CENTERLINE)
SPAWN_INDICES = [int(round(f * N)) % N
                 for f in (0.0, 0.125, 0.25, 0.375, 0.5,
                           0.625, 0.75, 0.875, 0.97)]


def wrap_deg(a):
    """Wrap an angle in degrees to [-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def tangent_yaw_deg(idx):
    """Centerline tangent (prev->next) at idx, in degrees -- same as the env."""
    n = len(CENTERLINE)
    prev_pt = CENTERLINE[(idx - 1) % n]
    next_pt = CENTERLINE[(idx + 1) % n]
    return math.degrees(math.atan2(next_pt[1] - prev_pt[1],
                                   next_pt[0] - prev_pt[0]))


def main():
    home = os.environ.get("BEAMNG_HOME")
    if not home:
        raise SystemExit("Set BEAMNG_HOME (path to the BeamNG.tech install).")

    env = make_beamng_env(
        random_spawn=False, home=home, host="localhost", port=25252,
        launch=True, headless=True, nogpu=True,
    )

    print(f"\n=== Spawn fix validation across {len(SPAWN_INDICES)} indices "
          f"(track has {N} points) ===\n", flush=True)

    errors = []
    for idx in SPAWN_INDICES:
        tan = tangent_yaw_deg(idx)
        env.reset(options={"spawn_idx": idx})
        # Step at zero throttle/brake/steer so the teleport settles.
        for _ in range(SETTLE_STEPS):
            env.step([0.0, 0.0])

        s = _shared["vehicle"].sensors["agent_state"]
        d = s.get("dir", None)
        pos = s["pos"]
        if d is None:
            print(f"idx {idx:4d}: no 'dir' in state; keys={list(s.keys())}",
                  flush=True)
            continue

        dir_yaw = math.degrees(math.atan2(d[1], d[0]))
        err = wrap_deg(dir_yaw - tan)   # +CCW, -CW relative to tangent
        errors.append(err)
        print(
            f"idx {idx:4d}: tangent={tan:+8.2f}  dir_yaw={dir_yaw:+8.2f}  "
            f"err(dir-tan)={err:+7.2f} deg   "
            f"pos=({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})",
            flush=True,
        )

    env.close()
    try:
        _shared["bng"].close()
    except Exception:
        pass

    if errors:
        mean = sum(errors) / len(errors)
        var = sum((e - mean) ** 2 for e in errors) / len(errors)
        std = math.sqrt(var)
        print("\n=== SUMMARY ===", flush=True)
        print(f"trials with data: {len(errors)}", flush=True)
        print(f"err mean = {mean:+.2f} deg, std = {std:.2f} deg", flush=True)
        print(f"err min  = {min(errors):+.2f}, max = {max(errors):+.2f}",
              flush=True)
        print("(err = settled forward yaw minus centerline tangent; "
              "negative = clockwise of track direction)", flush=True)


if __name__ == "__main__":
    main()
