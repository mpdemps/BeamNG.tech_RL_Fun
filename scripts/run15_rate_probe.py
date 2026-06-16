"""run15 behavioral probe (BeamNG): run the mature run14 policy DETERMINISTICALLY with
the run15 speed-scaled rate ON (steer_rate=0.5, steer_rate_hi=0.15) + the kept ESC. Confirm:
  - at HIGH speed (v>=V_HI=31) the applied |Δsteer|/step is capped at ~0.15 (the tightened
    rate binds; steer_ratehi_frac > 0),
  - at CORNER/low speed (v<=V_LO=27) |Δsteer| can still exceed 0.15 up to 0.5 (full agility),
  - the high-speed REVERSAL is stretched (slower yaw input) vs the run13/14 ~3-step slam,
  - composes with ESC/TC/Grad-CAPS, no NaN.
Reads env internals per step. Port 25252 (run14 stopped)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stable_baselines3 import SAC
from envs.beamng_env import make_beamng_env, _shared, STEER_RATE_V_LO, STEER_RATE_V_HI

PORT = 25252
CKPT = "checkpoints/mikey_run14/final.zip"
N_EP = 2
MAX_STEPS = 350


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost", port=PORT,
                          launch=True, headless=True, nogpu=True,
                          steer_rate=0.5, steer_rate_hi=0.15, esc_min=0.1)
    model = SAC.load(CKPT, device="cpu")
    print(f"loaded {CKPT}; deterministic; steer_rate=0.5 steer_rate_hi=0.15 esc_min=0.1\n")

    for ep in range(N_EP):
        obs, _ = env.reset()
        prev_app = 0.0
        rows = []
        hi_viol = 0   # steps at v>=V_HI where |Δapp| exceeded 0.15+eps (should be 0)
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            req = float(np.clip(action[0], -1.0, 1.0))
            v = env._last_speed_h
            obs, _, term, trunc, info = env.step(action)
            app = env._cur_steer
            dapp = abs(app - prev_app)
            if v >= STEER_RATE_V_HI and dapp > 0.15 + 1e-6:
                hi_viol += 1
            rows.append((step, v, req, app, dapp, info["beta_max"], info["heading_align"]))
            prev_app = app
            if term or trunc:
                break
        # summaries by speed regime
        lo = [r for r in rows if r[1] <= STEER_RATE_V_LO]
        hi = [r for r in rows if r[1] >= STEER_RATE_V_HI]
        maxd_lo = max((r[4] for r in lo), default=0.0)
        maxd_hi = max((r[4] for r in hi), default=0.0)
        print(f"=== EP{ep}: {len(rows)} steps, term={info.get('termination_reason')}, "
              f"steer_ratehi_frac={info['steer_ratehi_frac']:.3f} ===")
        print(f"  max |Δapplied_steer|/step:  v<=27: {maxd_lo:.3f} (can be up to 0.5)   "
              f"v>=31: {maxd_hi:.3f} (must be <=0.15)")
        print(f"  [{'ok' if hi_viol == 0 else 'FAIL'}] high-speed rate cap held: "
              f"{hi_viol} violations of 0.15 at v>=31 (of {len(hi)} high-speed steps)")
        # show a high-speed window where the policy slams and the rate stretches it
        hi_idx = next((i for i, r in enumerate(rows) if r[1] >= STEER_RATE_V_HI
                       and abs(r[2] - r[3]) > 0.2), None)
        if hi_idx is not None:
            print(f"   {'step':>5}{'v':>6}{'req':>7}{'app':>7}{'|dapp|':>7}{'head':>6}")
            for r in rows[max(0, hi_idx - 2):hi_idx + 10]:
                print(f"   {r[0]:>5}{r[1]:>6.1f}{r[2]:>7.2f}{r[3]:>7.2f}{r[4]:>7.3f}{r[6]:>6.2f}")
        print()
    env.close()
    try: _shared["bng"].close()
    except Exception: pass


if __name__ == "__main__":
    main()
