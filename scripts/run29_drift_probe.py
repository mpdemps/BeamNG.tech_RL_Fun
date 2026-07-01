"""run29 grip-straights / drift-corners scaffolding probe (read-only, no training).

run28 watch: the car drifted the STRAIGHT (fishtailed, went off before T1) because beta_target was
speed-based -> read ~12 deg on the straight and the car chased it. run29 corner-gates the TARGET too
and penalizes straight slides, via a smooth corner_factor cf (0 straight .. 1 corner). This probe
checks the TRANSITION is clean:
  A. corner_factor ramp + coupling (deterministic): as cf rises, TARGET ramps UP and the STRAIGHT
     slip penalty ramps OFF together -- no discontinuity (drift can initiate at turn-in; straight
     slide still punished).
  B. obs[19] (target) ~0 on a straight, the band in a corner (sim).
  C. forced STRAIGHT FISHTAIL (sim): the straight slip penalty FIRES (r_slip<0, straight_slip_frac>0)
     and the drift reward stays ~0 (not rewarded on the straight).
Port 25253 (sim free)."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import envs.beamng_env as be
from envs.beamng_env import (make_beamng_env, _shared, _PROFILE_KAPPA, W_SLIP, BETA_SLIP_DEAD)
from envs.base_controller import BaseController
from envs.drift import (corner_factor, beta_target, BETA_TARGET_OBS_NORM,
                        CORNER_KAPPA_LO, CORNER_KAPPA_HI)
from envs.speed_profile import compute_speed_profile
from data.raceline_builtin import RACELINE


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost", port=25253,
                          launch=True, headless=True, nogpu=True, steer_rate=0.5, drift_mode=True)
    u = env.unwrapped
    cum = np.array(compute_speed_profile(RACELINE)[2])
    kabs = np.abs(np.array(_PROFILE_KAPPA))
    straight_idx = int(np.argmin(np.abs(cum - 1700.0)))          # mid back-straight
    corner_idx = int(np.argmax(kabs))                            # sharpest corner

    print("\n===== run29 grip-straights / drift-corners probe =====")
    print(f"obs {env.observation_space.shape}; corner ramp LO={CORNER_KAPPA_LO} HI={CORNER_KAPPA_HI}")

    # ---- A. transition coupling (deterministic) ----
    print("\n[A] TRANSITION @ 20 m/s (beta 18): target ramps UP as straight-slip-pen ramps OFF -- together:")
    for k in (0.004, 0.008, 0.012, 0.016, 0.020):
        cf = corner_factor(k); bt = cf * beta_target(20.0)
        sp = -(1 - cf) * W_SLIP * max(0.0, 18.0 - BETA_SLIP_DEAD)
        print(f"    |kappa|={k:.3f} cf={cf:.2f}  target={bt:4.1f} deg  straight_slip_pen={sp:+.2f}")
    print("    -> continuous crossfade, no gap: drift can initiate at turn-in, straight slide still punished.")

    # ---- B. obs[19] target gated: ~0 straight, band in corner ----
    ob_s, _ = env.reset(options={"spawn_idx": straight_idx})
    ks = kabs[straight_idx]
    ob_c, _ = env.reset(options={"spawn_idx": corner_idx})
    kc = kabs[corner_idx]
    print(f"\n[B] obs[19] (target/norm):  straight idx |kappa|={ks:.3f} -> {ob_s[19]:.2f} (~0 expected)")
    print(f"                            corner   idx |kappa|={kc:.3f} -> {ob_c[19]:.2f} (band expected, ~0.5 at rest)")

    # ---- C. forced straight fishtail: straight slip penalty must fire, drift reward ~0 ----
    print(f"\n[C] FORCED STRAIGHT FISHTAIL (mid-straight, oscillate steer): straight slip penalty must fire:")
    env.reset(options={"spawn_idx": straight_idx}); ctrl = BaseController(); ctrl.reset(); ctrl._idx = straight_idx
    for _ in range(40):                              # build speed on the straight
        s = _shared["vehicle"].sensors["agent_state"]; vel = s["vel"]; sp = math.hypot(vel[0], vel[1])
        st, th = ctrl.action(s["pos"], vel, s.get("dir", (1.0, 0.0, 0.0)))
        env.step(np.array([st, max(th, 0.5)], np.float32))
        if sp >= 15: break
    beta_pk = 0.0; nstep = 0; term = None
    for i in range(80):                              # fishtail: alternate hard steer to slide on the straight
        kick = 0.6 if (i // 4) % 2 == 0 else -0.6
        _, _, t, tr, info = env.step(np.array([kick, 0.7], np.float32))
        nstep += 1; beta_pk = max(beta_pk, u._last_beta)
        if t or tr: term = info.get("termination_reason"); break
    print(f"    steps={nstep}  peak|beta|={beta_pk:.0f} deg")
    print(f"    r_slip (straight slip penalty sum) = {float(info['r_slip']):.2f}  straight_slip_frac = {float(info['straight_slip_frac']):.2f}")
    print(f"    r_drift (should be ~0 on the straight) = {float(info['r_drift']):.2f}")
    print(f"    -> {'OK (straight slide punished, not rewarded as drift)' if float(info['r_slip']) < -0.01 and float(info['r_drift']) < 0.5 else 'check'}")

    env.close()
    try: _shared["bng"].close()
    except Exception: pass
    print("===== run29 probe done =====")


if __name__ == "__main__":
    main()
