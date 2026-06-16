"""run14 ESC behavioral probe (BeamNG): run the mature run13 policy DETERMINISTICALLY
with ESC ON (esc_min=0.1) against its known opening-straight spin. Confirms beta is
computed correctly from real sensors and that ESC:
  - stays OFF (esc_factor=1.0, app_thr=req_thr) on the clean opening straight (beta<DEAD),
  - CUTS throttle (esc_factor<1, app_thr<req_thr) exactly when beta climbs into the slide,
  - and ideally lets the car RECOVER instead of spinning.

Reads env._last_beta / ._last_slip directly (per-step; the info dict only has aggregates).
Port 25252 (nothing training). Compares the same checkpoint the run13 trace used."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stable_baselines3 import SAC
from envs.beamng_env import (make_beamng_env, ESC_BETA_DEAD, ESC_BETA_FULL,
                             TC_SLIP_DEAD, TC_SLIP_FULL, TC_MIN_THR)

PORT = 25252
CKPT = "checkpoints/mikey_run13/rolling_500000_steps.zip"
ESC_MIN = 0.1
N_EP = 2
MAX_STEPS = 400


def esc_factor(beta):
    return max(ESC_MIN, min(1.0, 1.0 - (beta - ESC_BETA_DEAD) / (ESC_BETA_FULL - ESC_BETA_DEAD)))


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost",
                          port=PORT, launch=True, headless=True, nogpu=True,
                          esc_min=ESC_MIN)
    model = SAC.load(CKPT, device="cpu")
    print(f"loaded {CKPT}; deterministic; esc_min={ESC_MIN}; DEAD={ESC_BETA_DEAD} FULL={ESC_BETA_FULL}\n")

    for ep in range(N_EP):
        obs, _ = env.reset()
        prev_slip = 0.0
        clean_cuts = 0      # ESC cuts while beta < DEAD (should be ZERO -> not trigger-happy)
        clean_steps = 0
        slide_cuts = 0      # ESC cuts while beta >= DEAD (the intended behavior)
        slide_steps = 0
        recovered = False
        rows = []
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            req_thr = max(0.0, float(np.clip(action[1], -1.0, 1.0)))
            beta_used = env._last_beta          # last-step beta (what the gate uses)
            esc = esc_factor(beta_used) if ESC_MIN < 1.0 else 1.0
            tc = max(TC_MIN_THR, min(1.0, 1.0 - (prev_slip - TC_SLIP_DEAD) / (TC_SLIP_FULL - TC_SLIP_DEAD)))
            app_thr = req_thr * tc * esc
            obs, _, term, trunc, info = env.step(action)
            beta_now = env._last_beta
            head = info["heading_align"]
            cut = req_thr > 0.01 and esc < 0.999
            if beta_used < ESC_BETA_DEAD:
                clean_steps += 1; clean_cuts += int(cut)
            else:
                slide_steps += 1; slide_cuts += int(cut)
            # recovery: a slide (beta>DEAD) that returns to clean (beta<DEAD) with nose forward
            rows.append((step, info["max_arc"], info["mean_speed"], beta_now, esc, req_thr, app_thr,
                         info["slip"], head))
            prev_slip = info["slip"]
            if term or trunc:
                break
        # detect recovery: any beta excursion >DEAD that later drops <DEAD with head>0.8
        was_sliding = False
        for r in rows:
            if r[3] >= ESC_BETA_DEAD:
                was_sliding = True
            elif was_sliding and r[3] < ESC_BETA_DEAD and r[8] > 0.8:
                recovered = True; was_sliding = False
        maxbeta = max(r[3] for r in rows)
        term = info.get("termination_reason", "?")
        print(f"=== EP{ep}: {len(rows)} steps, reached arc {rows[-1][1]:.0f}m, term={term}, "
              f"max beta={maxbeta:.0f}deg ===")
        print(f"  CLEAN driving (beta<{ESC_BETA_DEAD:.0f}): {clean_steps} steps, ESC cut {clean_cuts} "
              f"({'GOOD: ESC quiet on clean' if clean_cuts == 0 else 'FLAG: ESC firing on clean!'})")
        print(f"  SLIDE (beta>={ESC_BETA_DEAD:.0f}):      {slide_steps} steps, ESC cut {slide_cuts} "
              f"({'GOOD: ESC fires on slides' if slide_cuts > 0 or slide_steps == 0 else 'FLAG'})")
        print(f"  recovered from a slide back to clean+forward: {recovered}")
        # show the slide window: first beta>DEAD onward
        onset = next((i for i, r in enumerate(rows) if r[3] >= ESC_BETA_DEAD), None)
        if onset is not None:
            print(f"  {'step':>5}{'arc':>7}{'kph':>6}{'beta':>6}{'esc':>6}{'req_thr':>8}{'app_thr':>8}{'slip':>6}{'head':>6}")
            for r in rows[max(0, onset - 4):onset + 12]:
                print(f"  {r[0]:>5}{r[1]:>7.0f}{r[2]*3.6:>6.0f}{r[3]:>6.0f}{r[4]:>6.2f}"
                      f"{r[5]:>8.2f}{r[6]:>8.2f}{r[7]:>6.1f}{r[8]:>6.2f}")
        print()

    env.close()
    try: env_close = _shared_close()  # noqa
    except Exception: pass


def _shared_close():
    from envs.beamng_env import _shared
    try: _shared["bng"].close()
    except Exception: pass


if __name__ == "__main__":
    main()
