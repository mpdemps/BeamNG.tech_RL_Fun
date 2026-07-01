"""run30 strong-straight-grip probe (read-only, no training).

run29 verdict: start-line eval reached T1 in 0/34 evals -- the car fishtailed the opening straight
and died ~50m in, because the straight slip penalty (W_SLIP 0.05) was too weak (it ate the tiny cost
and slid anyway). run30's ONE change: a dedicated, much stronger straight slip weight
(W_DRIFT_STRAIGHT_SLIP 0.2) so GRIPPING the straight clearly beats sliding. This probe checks:
  A. ECONOMICS (deterministic): on the straight, does the slip penalty now OUTWEIGH the forward-
     progress reward for a fishtail? (run29 0.05: sliding was still net-POSITIVE -> paid to slide;
     run30 0.2: net-NEGATIVE -> gripping wins.)
  B. TRANSITION still clean (deterministic): the stronger straight penalty still rings OFF into the
     corner via (1-cf), so a turn-in drift can still initiate (no block).
  C. Phase 1 untouched: the shared W_SLIP is still 0.05 (grip-lap env unchanged, 19-dim).
  D. SIM forced straight fishtail: the penalty now bites ~4x harder; drift reward still 0 on straight.
Port 25253 (sim free)."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import envs.beamng_env as be
from envs.beamng_env import (make_beamng_env, _shared, _PROFILE_KAPPA, W_SLIP, W_DRIFT_STRAIGHT_SLIP,
                             W_PROG, BETA_SLIP_DEAD, DT_S)
from envs.base_controller import BaseController
from envs.drift import corner_factor, beta_target, BETA_TARGET_OBS_NORM
from envs.speed_profile import compute_speed_profile
from data.raceline_builtin import RACELINE

RUN29_WSLIP = 0.05  # what run29 used (for the before/after comparison)


def main():
    print("\n===== run30 strong-straight-grip probe =====")
    print(f"[const] straight slip weight: run29 W_SLIP={RUN29_WSLIP} -> run30 W_DRIFT_STRAIGHT_SLIP={W_DRIFT_STRAIGHT_SLIP}")
    print(f"        Phase 1 shared W_SLIP still {W_SLIP} (unchanged) -> {'OK' if W_SLIP == 0.05 else 'FAIL'}")

    # ---- A. economics on the straight (deterministic) ----
    sp = 15.0                       # m/s on the opening straight
    align = 0.95                    # a mild fishtail still travels mostly down-track
    prog = sp * DT_S * align        # raw_progress * gated_alignment ~ forward-progress reward /step (W_PROG=1.0)
    prog_reward = W_PROG * prog
    print(f"\n[A] STRAIGHT economics @ {sp:.0f} m/s: forward-progress reward ~ +{prog_reward:.2f}/step (grip)")
    print(f"    {'beta':>5}{'slip_pen(0.05)':>16}{'net_29':>9}{'slip_pen(0.20)':>16}{'net_30':>9}")
    for beta in (12, 18, 25):
        excess = max(0.0, beta - BETA_SLIP_DEAD)
        p29 = -RUN29_WSLIP * excess; p30 = -W_DRIFT_STRAIGHT_SLIP * excess
        print(f"    {beta:>5}{p29:>16.2f}{prog_reward + p29:>+9.2f}{p30:>16.2f}{prog_reward + p30:>+9.2f}")
    print("    -> run29: sliding stays net-POSITIVE (paid to slide). run30: net-NEGATIVE -> gripping wins.")

    # ---- B. transition still clean (deterministic) ----
    print(f"\n[B] TRANSITION @ 20 m/s (beta 18): stronger straight penalty still rings OFF into corner:")
    for k in (0.004, 0.008, 0.012, 0.016, 0.020):
        cf = corner_factor(k); bt = cf * beta_target(20.0)
        spn = -(1 - cf) * W_DRIFT_STRAIGHT_SLIP * max(0.0, 18.0 - BETA_SLIP_DEAD)
        print(f"    |kappa|={k:.3f} cf={cf:.2f}  target={bt:4.1f} deg  straight_slip_pen={spn:+.2f}")
    print("    -> penalty 0 in the corner (cf=1): turn-in drift still initiates; straight (cf=0) fully punished.")

    # ---- C+D. sim: Phase-1 unchanged + forced straight fishtail bites harder ----
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost", port=25253,
                          launch=True, headless=True, nogpu=True, steer_rate=0.5, drift_mode=True)
    u = env.unwrapped
    cum = np.array(compute_speed_profile(RACELINE)[2])
    straight_idx = int(np.argmin(np.abs(cum - 1700.0)))
    print(f"\n[C] drift env obs {env.observation_space.shape} (20 expected)")
    print(f"\n[D] FORCED STRAIGHT FISHTAIL (mid-straight): straight slip penalty should bite ~4x run29's -31:")
    env.reset(options={"spawn_idx": straight_idx}); ctrl = BaseController(); ctrl.reset(); ctrl._idx = straight_idx
    for _ in range(40):
        s = _shared["vehicle"].sensors["agent_state"]; vel = s["vel"]; spd = math.hypot(vel[0], vel[1])
        st, th = ctrl.action(s["pos"], vel, s.get("dir", (1.0, 0.0, 0.0)))
        env.step(np.array([st, max(th, 0.5)], np.float32))
        if spd >= 15: break
    beta_pk = 0.0; nstep = 0; info = {}
    for i in range(80):
        kick = 0.6 if (i // 4) % 2 == 0 else -0.6
        _, _, t, tr, info = env.step(np.array([kick, 0.7], np.float32))
        nstep += 1; beta_pk = max(beta_pk, u._last_beta)
        if t or tr: break
    print(f"    steps={nstep} peak|beta|={beta_pk:.0f}  r_slip(straight pen sum)={float(info['r_slip']):.1f}  "
          f"straight_slip_frac={float(info['straight_slip_frac']):.2f}  r_drift={float(info['r_drift']):.2f}")
    print(f"    -> {'OK (stronger bite; straight slide punished, r_drift~0)' if float(info['r_slip']) < -60 and float(info['r_drift']) < 0.5 else 'check'}")

    env.close()
    try: _shared["bng"].close()
    except Exception: pass
    print("===== run30 probe done =====")


if __name__ == "__main__":
    main()
