"""Confirm the warm-start source DRIVES (smoke part a), deterministically.

Training samples stochastically, which surfaces run3's high-variance spin-out
mode; this runs the loaded policy DETERMINISTICALLY (mean action, like eval/G14)
so we see its actual driving capability from step 0 -- the unambiguous proof the
warm-start carries a real policy, not a fresh/flailing one. Run3 stopped, so
port 25252 is free.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from envs.beamng_env import make_beamng_env, _shared

SNAP = "checkpoints/mikey_run3/final.zip"


def main():
    home = os.environ.get("BEAMNG_HOME")
    if not home:
        raise SystemExit("Set BEAMNG_HOME.")
    env = make_beamng_env(random_spawn=False, home=home, launch=True,
                          headless=True, nogpu=True)
    model = SAC.load(SNAP, device="cpu")
    print(f"\nloaded {SNAP}: n_updates={model._n_updates} "
          f"(>>7k confirms it is run3's trained model, not fresh)", flush=True)

    for ep in range(2):
        obs, _ = env.reset()
        ep_r = 0.0
        max_cp = 0
        max_slip = 0.0
        steps = 0
        for _ in range(1500):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_r += float(r)
            steps += 1
            max_cp = max(max_cp, int(info["checkpoints_reached"]))
            max_slip = max(max_slip, float(info["slip"]))
            if term or trunc:
                break
        print(f"ep{ep} DETERMINISTIC: reached cp={max_cp}  reward={ep_r:+.1f}  "
              f"steps={steps}  max_slip={max_slip:.1f}", flush=True)

    env.close()
    try:
        _shared["bng"].close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
