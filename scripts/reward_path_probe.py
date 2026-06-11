"""Validate the run4 spin_penalty IN THE REWARD PATH (smoke part c).

Reads info["slip"] and info["spin_penalty"] straight out of env.step() -- i.e.
the actual reward-path computation, not a re-derivation. Floors throttle from
spawn (launch wheelspin: slip should hit +5-14, spin_penalty clearly negative),
then drives gently (cruise: slip < 0.3 under the 2.0 deadzone, spin_penalty ~0).

Run3 is stopped, so this uses the default port 25252. READ of the reward path
only; no training.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.beamng_env import make_beamng_env, SLIP_DEADZONE, SPIN_WEIGHT


def row(phase, step, info, reward):
    print(f" {phase:6s} {step:3d} | slip={info['slip']:+7.2f} | "
          f"spin_penalty={info['spin_penalty']:+7.3f} | "
          f"speed_rwd={info['speed_reward']:+.3f} | "
          f"align={info['alignment']:+.2f} | step_r={reward:+7.2f}",
          flush=True)


def main():
    home = os.environ.get("BEAMNG_HOME")
    if not home:
        raise SystemExit("Set BEAMNG_HOME.")
    env = make_beamng_env(random_spawn=False, home=home, launch=True,
                          headless=True, nogpu=True)
    env.reset()
    print(f"\n=== reward-path check (SLIP_DEADZONE={SLIP_DEADZONE}, "
          f"SPIN_WEIGHT={SPIN_WEIGHT}) ===", flush=True)
    print(" phase  stp |   slip    | spin_penalty | speed_rwd | align | step_r",
          flush=True)
    print("-" * 78, flush=True)

    # LAUNCH: floor it from the standstill spawn -> rear wheelspin.
    for step in range(18):
        _, reward, term, trunc, info = env.step([0.0, 1.0])
        row("LAUNCH", step, info, reward)
        if term or trunc:
            env.reset()

    # CRUISE: gentle throttle so the wheels hook up -> slip should fall to ~0.
    env.reset()
    for step in range(28):
        _, reward, term, trunc, info = env.step([0.0, 0.15])
        if step % 2 == 0 or step >= 20:
            row("CRUISE", step, info, reward)
        if term or trunc:
            env.reset()

    env.close()
    from envs.beamng_env import _shared
    try:
        _shared["bng"].close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
