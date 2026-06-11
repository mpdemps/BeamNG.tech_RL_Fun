"""Validate the run5 heading gate IN THE REWARD PATH (smoke part b, deterministic).

Reads info straight from env.step(). Three phases:
  FWD-FLOOR : floor from spawn -> launch wheelspin (spin_penalty fires), nose
              forward (heading_align > 0) so reward is NOT killed.
  FWD-CRUISE: gentle throttle -> hooked up, spin ~0, shows speed_reward @ 0.003.
  BACKWARD  : teleport the car to face BACKWARD (nose up-track) and hold ->
              heading_align ~ -1 -> reward must be 0 (kill-switch), and the
              episode must terminate at ~40 consecutive backward steps.

Run3/4 stopped, so this uses the default port 25252. Reward path read only.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.beamng_env import (
    make_beamng_env, _shared, CENTERLINE, _yaw_to_quat,
    SPAWN_QUAT_YAW_CORRECTION_DEG, SPAWN_Z_OFFSET_M,
    HEADING_KILL_THRESHOLD, BACKWARD_TERM_STEPS, SPEED_WEIGHT)


def row(tag, step, info, reward):
    print(f" {tag:9s} {step:3d} | head_align={info['heading_align']:+6.2f} | "
          f"reward={reward:+7.3f} | speed_rwd={info['speed_reward']:+.4f} | "
          f"spin_pen={info['spin_penalty']:+.3f} | slip={info['slip']:+5.1f} | "
          f"bwd={info['backward_steps']:2d} cp={info['checkpoints_reached']}",
          flush=True)


def main():
    home = os.environ.get("BEAMNG_HOME")
    if not home:
        raise SystemExit("Set BEAMNG_HOME.")
    env = make_beamng_env(random_spawn=False, home=home, launch=True,
                          headless=True, nogpu=True)
    env.reset()
    print(f"\n=== run5 reward-path check (kill thr={HEADING_KILL_THRESHOLD}, "
          f"term={BACKWARD_TERM_STEPS} steps, SPEED_WEIGHT={SPEED_WEIGHT}) ===",
          flush=True)

    print("\n-- FWD-FLOOR (launch spin; nose forward, reward NOT killed) --", flush=True)
    for step in range(8):
        _, r, term, trunc, info = env.step([0.0, 1.0])
        row("FWD-FLOOR", step, info, r)
        if term or trunc:
            env.reset()

    print("\n-- FWD-CRUISE (hooked up; speed_reward @ 0.003) --", flush=True)
    for step in range(14):
        _, r, term, trunc, info = env.step([0.0, 0.15])
        if step % 2 == 0:
            row("FWD-CRUISE", step, info, r)
        if term or trunc:
            env.reset()

    # BACKWARD: teleport the car to face up-track (nose backward) and hold.
    env.reset()
    idx = env._progress_idx
    tangent = env._smoothed_forward_yaw(idx)          # down-track heading
    backward_heading = tangent + math.pi              # nose up-track
    sent_yaw = -backward_heading - math.radians(SPAWN_QUAT_YAW_CORRECTION_DEG)
    cx, cy, cz = CENTERLINE[idx]
    _shared["vehicle"].teleport((cx, cy, cz + SPAWN_Z_OFFSET_M),
                                rot_quat=_yaw_to_quat(sent_yaw), reset=True)
    print("\n-- BACKWARD (teleported nose-backward; reward must be 0, "
          "terminate ~40) --", flush=True)
    terminated_at = None
    for step in range(55):
        _, r, term, trunc, info = env.step([0.0, 0.0])
        if step < 3 or step % 5 == 0 or term:
            row("BACKWARD", step, info, r)
        if term:
            terminated_at = (step, info["backward_steps"])
            print(f"   --> TERMINATED at probe-step {step}, "
                  f"backward_steps={info['backward_steps']}", flush=True)
            break
    if terminated_at is None:
        print("   --> NO termination in 55 steps (unexpected)", flush=True)

    env.close()
    try:
        _shared["bng"].close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
