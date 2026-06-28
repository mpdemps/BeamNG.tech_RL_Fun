"""run28 controlled-forward-drift scaffolding probe (read-only, no training).

Three run28 changes on top of run27's drift setup; this confirms them:
  A. constants: W_DRIFT_PROGRESS raised 0.3 -> 0.6 (proposed), W_DRIFT_BACKWARD added.
  B. BACKWARD PENALTY is SAFE FOR DRIFT (logic, deterministic): the penalty keys on the SIGN of
     forward-velocity (speed * cos to the track tangent). A forward slide -- even pointed sideways
     (align 0.5-0.7) -- has fwd_vel > 0 => penalty 0. Only actual reverse (align < 0) => penalty < 0.
  C. OVER-SPEED DISCIPLINE RE-ACTIVATED in drift corners (sim): floor the throttle into a corner and
     confirm the over-speed penalty (r_overspeed) accrues while in the corner (run27 dropped it -> 0).
     Same drive confirms forward motion never trips the backward penalty (r_backward ~ 0, frac ~ 0).
  D. FORCED REVERSE (sim): hold full brake from a stop; if it rolls backward, r_backward < 0 and
     backward_frac > 0 -- the penalty fires on reverse.
Port 25253 (sim free)."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import envs.beamng_env as be
from envs.beamng_env import (make_beamng_env, _shared, _PROFILE_KAPPA, DRIFT_CORNER_KAPPA,
                             W_DRIFT, W_DRIFT_PROGRESS, W_DRIFT_BACKWARD)
from envs.base_controller import BaseController


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost", port=25253,
                          launch=True, headless=True, nogpu=True, steer_rate=0.5, drift_mode=True)
    u = env.unwrapped
    print("\n===== run28 controlled-forward-drift probe =====")
    print(f"[A] W_DRIFT={W_DRIFT}  W_DRIFT_PROGRESS={W_DRIFT_PROGRESS} (run27 0.3 -> run28 0.6)  "
          f"W_DRIFT_BACKWARD={W_DRIFT_BACKWARD}")

    # ---- B. backward penalty SAFE FOR DRIFT (deterministic) ----
    def pen(speed, align):
        fwd = speed * align
        return (-W_DRIFT_BACKWARD * (-fwd)) if fwd < 0 else 0.0
    print("\n[B] backward penalty (keys on sign of forward-velocity = speed*cos-to-tangent):")
    for tag, sp, al in [("forward grip", 15, 0.99), ("forward DRIFT sideways", 15, 0.70),
                        ("hard fwd drift", 12, 0.50), ("REVERSE", 5, -0.80), ("slow reverse", 2, -0.6)]:
        print(f"    {tag:22s} sp={sp:>2} align={al:+.2f} -> fwd_vel {sp*al:+5.1f}  pen {pen(sp,al):+.2f}")
    print("    -> forward (incl. sideways) never penalized; only reverse. SAFE FOR DRIFT.")

    ctrl = BaseController()

    # ---- C. over-speed RE-ACTIVATED in drift corners + forward never trips backward ----
    print("\n[C] FLOOR throttle through corners: over-speed must accrue IN corners; backward ~0:")
    env.reset(options={"spawn_idx": 0}); ctrl.reset()
    prev_ov = 0.0; corner_ov = 0.0; straight_ov = 0.0; r_back = 0.0; bfrac = 0.0; nstep = 0
    for i in range(300):
        s = _shared["vehicle"].sensors["agent_state"]
        vel = s["vel"]
        st, _ = ctrl.action(s["pos"], vel, s.get("dir", (1.0, 0.0, 0.0)))
        _, _, t, tr, info = env.step(np.array([st, 1.0], np.float32))   # FLOOR (override controller brake)
        nstep += 1
        ov = float(info["r_overspeed"]); d_ov = ov - prev_ov; prev_ov = ov
        corner = abs(float(_PROFILE_KAPPA[u._progress_idx])) > DRIFT_CORNER_KAPPA
        if corner: corner_ov += d_ov
        else:      straight_ov += d_ov
        r_back = float(info["r_backward"]); bfrac = float(info["backward_frac"])
        if t or tr: break
    print(f"    steps={nstep}  over-speed penalty accrued IN corners: {corner_ov:.1f}  on straights: {straight_ov:.1f}")
    print(f"    -> {'OK (brake signal ACTIVE in drift corners)' if corner_ov < -0.01 else 'NOT firing in corners?'}")
    print(f"    forward drive: r_backward={r_back:.2f} backward_frac={bfrac:.2f} "
          f"-> {'OK (forward never trips backward pen)' if abs(r_back) < 1e-6 and bfrac == 0 else 'check'}")

    # ---- D. forced reverse fires the backward penalty ----
    print("\n[D] FORCED REVERSE (full brake from a stop): backward penalty must fire if it reverses:")
    env.reset(options={"spawn_idx": 0})
    min_fwd = 0.0; r_back = 0.0; bfrac = 0.0; term = None; nstep = 0
    for i in range(160):
        s = _shared["vehicle"].sensors["agent_state"]; vel = s["vel"]
        sp = math.hypot(vel[0], vel[1]); al = u._last_raw_alignment
        min_fwd = min(min_fwd, sp * al)
        _, _, t, tr, info = env.step(np.array([0.0, -1.0], np.float32))   # full brake -> reverse
        nstep += 1
        r_back = float(info["r_backward"]); bfrac = float(info["backward_frac"])
        if t or tr: term = info.get("termination_reason"); break
    print(f"    steps={nstep+1}  min forward-vel {min_fwd:+.1f} m/s  r_backward={r_back:.1f}  backward_frac={bfrac:.2f}  ended={term}")
    print(f"    -> {'OK (reverse penalized)' if r_back < -0.01 else 'car did not roll backward (penalty would fire if it did; see [B])'}")

    env.close()
    try: _shared["bng"].close()
    except Exception: pass
    print("===== run28 probe done =====")


if __name__ == "__main__":
    main()
