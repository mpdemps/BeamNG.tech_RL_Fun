"""Validate run6 smoothness terms in the reward path (smoke part b, deterministic).

Reads info from env.step(). Phases (steer/throttle held forward-ish so the run5
heading kill-switch does not zero things):
  STEADY      : steer 0, throttle 0.3 held -> both smoothness penalties ~0.
  THR-CHATTER : steer 0, throttle alternating +1/-1 -> throttle_smooth fires
                (~ -0.05*2 = -0.1), steering smooth ~0.
  STEER-OSC   : steer alternating +/-0.9, throttle 0.3 -> steering smooth bites
                harder at SMOOTH_WEIGHT=0.2 (~ -0.2*1.8 = -0.36), throttle ~0.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.beamng_env import (make_beamng_env, _shared, SMOOTH_WEIGHT,
                             THROTTLE_SMOOTH_WEIGHT)


def row(tag, step, info, r):
    print(f" {tag:11s} {step:2d} | steer_smooth={info['smoothness_penalty']:+.3f} | "
          f"thr_smooth={info['throttle_smooth_penalty']:+.3f} | "
          f"speed_rwd={info['speed_reward']:+.4f} | spin={info['spin_penalty']:+.3f} | "
          f"head={info['heading_align']:+.2f} | r={r:+.3f}", flush=True)


def main():
    home = os.environ.get("BEAMNG_HOME")
    if not home:
        raise SystemExit("Set BEAMNG_HOME.")
    env = make_beamng_env(random_spawn=False, home=home, launch=True,
                          headless=True, nogpu=True)
    env.reset()
    print(f"\n=== run6 smoothness check (SMOOTH_WEIGHT={SMOOTH_WEIGHT}, "
          f"THROTTLE_SMOOTH_WEIGHT={THROTTLE_SMOOTH_WEIGHT}) ===", flush=True)

    print("\n-- STEADY (steer 0, throttle 0.3): both penalties ~0 --", flush=True)
    for step in range(8):
        _, r, term, trunc, info = env.step([0.0, 0.3])
        row("STEADY", step, info, r)
        if term or trunc:
            env.reset()

    print("\n-- THR-CHATTER (throttle +1/-1, steer 0): thr_smooth fires --", flush=True)
    for step in range(10):
        thr = 1.0 if step % 2 == 0 else -1.0
        _, r, term, trunc, info = env.step([0.0, thr])
        row("THR-CHATTER", step, info, r)
        if term or trunc:
            env.reset()

    env.reset()
    print("\n-- STEER-OSC (steer +/-0.9, throttle 0.3): steer_smooth bites @0.2 --", flush=True)
    for step in range(10):
        st = 0.9 if step % 2 == 0 else -0.9
        _, r, term, trunc, info = env.step([st, 0.3])
        row("STEER-OSC", step, info, r)
        if term or trunc:
            env.reset()

    env.close()
    try:
        _shared["bng"].close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
