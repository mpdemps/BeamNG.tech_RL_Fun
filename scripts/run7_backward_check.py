"""Show a backward termination logging reason=backward AND the -25 terminal,
together, in one info row (smoke part b). The fresh random smoke crawls and
ends 'stuck', so it never exercises the backward path; this forces it.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.beamng_env import (
    make_beamng_env, _shared, CENTERLINE, _yaw_to_quat,
    SPAWN_QUAT_YAW_CORRECTION_DEG, SPAWN_Z_OFFSET_M, BACKWARD_TERM_PENALTY)


def main():
    home = os.environ.get("BEAMNG_HOME")
    if not home:
        raise SystemExit("Set BEAMNG_HOME.")
    env = make_beamng_env(random_spawn=False, home=home, launch=True,
                          headless=True, nogpu=True)
    env.reset()
    idx = env._progress_idx
    tangent = env._smoothed_forward_yaw(idx)
    sent = -(tangent + math.pi) - math.radians(SPAWN_QUAT_YAW_CORRECTION_DEG)
    cx, cy, cz = CENTERLINE[idx]
    _shared["vehicle"].teleport((cx, cy, cz + SPAWN_Z_OFFSET_M),
                                rot_quat=_yaw_to_quat(sent), reset=True)
    print(f"\n=== forcing backward termination (BACKWARD_TERM_PENALTY="
          f"{BACKWARD_TERM_PENALTY}) ===", flush=True)
    for step in range(55):
        _, r, term, trunc, info = env.step([0.0, 0.0])
        if term:
            print(f"TERMINAL step {step}: reward={r:+.3f}  "
                  f"termination_reason={info['termination_reason']!r}  "
                  f"backward_steps(info)={info['backward_steps']}  "
                  f"heading_align={info['heading_align']:+.2f}  "
                  f"max_arc={info['max_arc']:.1f}m", flush=True)
            break
    env.close()
    try:
        _shared["bng"].close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
