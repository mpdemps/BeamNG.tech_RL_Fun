"""run27 Phase 2 DRIFT scaffolding probe (read-only, no training).

Confirms the scaffolding on the GTS drift car, after Mikey's reward-design changes:
  A. CONFIG read-back: gts_drift.pc applied -> rear diff = differential_R_race (LSD), drive mode
     defaults to DRIFT.
  B. obs is 20-dim and obs[19]*BETA_TARGET_OBS_NORM = beta_target (now the SMALL ~20 deg start).
  D. REWARD SHAPE (logic, no sim): the over-drift side is gentle with NO penalty out to the ~60 deg
     spin; and the terminator discriminates a controlled in-band drift (never fires) from a spin.
  E. CORNER GATE (sim): drive the base controller a lap-ish; the drift-match reward accrues only in
     corners (|kappa| > DRIFT_CORNER_KAPPA) and reads exactly 0 on straights.
  C. FORCED SPIN (sim): full-lock at speed -> the drift terminator fires.
Port 25253 (sim free)."""
import math, os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import envs.beamng_env as be
from envs.beamng_env import (make_beamng_env, _shared, _PROFILE_KAPPA, DRIFT_CORNER_KAPPA)
from envs.base_controller import BaseController
from envs.speed_profile import compute_speed_profile
from envs.drift import (beta_target, drift_reward, is_drift_spin, SPIN_BETA_DEG,
                        BETA_TARGET_OBS_NORM, DRIFT_SCALE, DRIFT_SPIN_STEPS, DRIFT_NO_PROGRESS_STEPS)
from data.raceline_builtin import RACELINE


def _drive_to_speed(env, u, veh, ctrl, target, max_steps=200):
    for _ in range(max_steps):
        s = veh.sensors["agent_state"]; vel = s["vel"]; sp = math.hypot(vel[0], vel[1])
        st, th = ctrl.action(s["pos"], vel, s.get("dir", (1.0, 0.0, 0.0)))
        _, _, t, tr, info = env.step(np.array([st, max(th, 0.6)], np.float32))
        if sp >= target:
            return True, sp
        if t or tr:
            return False, sp
    return False, sp


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost", port=25253,
                          launch=True, headless=True, nogpu=True, steer_rate=0.5, drift_mode=True)
    u = env.unwrapped
    cum = np.array(compute_speed_profile(RACELINE)[2])
    straight_idx = int(np.argmin(np.abs(cum - 1700.0)))

    print("\n===== run27 DRIFT scaffolding probe (Mikey reward changes) =====")
    print(f"obs space: {env.observation_space.shape} (expect (20,));  DRIFT_SCALE={DRIFT_SCALE}")

    # ---- A. config read-back ----
    env.reset(options={"spawn_idx": 0}); veh = _shared["vehicle"]
    blob = str(veh.get_part_config())
    m = re.search(r"'partConfigFilename': '([^']*)'", blob)
    cfg = m.group(1) if m else "?"
    def chosen(key):
        i = blob.find("'" + key + "'")
        mm = re.search(r"'chosenPartName': '([^']*)'", blob[i:i + 800]) if i >= 0 else None
        return mm.group(1) if mm else "?"
    print(f"\n[A] partConfigFilename = {cfg}")
    print(f"    rear diff = {chosen('scintilla_differential_R')}  drivemode = {chosen('scintilla_DSE_drivemodes_default')}")

    # ---- B. obs[19] = beta_target (SMALL ~20 deg start), anchored to the big-regime norm ----
    obs, _ = env.reset(options={"spawn_idx": straight_idx})
    print(f"\n[B] at speed~0: obs[19]={obs[19]:.3f} -> beta_target={obs[19]*BETA_TARGET_OBS_NORM:.1f} deg "
          f"(expect ~{beta_target(0):.0f}; obs anchored to {BETA_TARGET_OBS_NORM:.0f})")

    # ---- D. reward shape + terminator discrimination (logic, no sim) ----
    print(f"\n[D] over-drift side (target 20): GENTLE, NO penalty out to the spin ({SPIN_BETA_DEG:.0f}):")
    for b in (20, 30, 40, 50, 60):
        print(f"      beta {b:>2}: drift_reward={drift_reward(b, 20.0):.2f}")
    print(f"      min reward over beta 0..60 = {min(drift_reward(b,20.0) for b in range(0,61)):.2f} (>=0 => no penalty)")
    def fires_at(trace):
        c = 0
        for k, (bb, hh) in enumerate(trace):
            c = c + 1 if is_drift_spin(bb, hh) else 0
            if c >= DRIFT_SPIN_STEPS: return k
        return None
    controlled = [(18.0, 0.9)] * 100      # in-band drift held 5 s
    spin = [(95.0, -0.3)] * 40
    print(f"    terminator: in-band drift fires at {fires_at(controlled)} (None=allowed); "
          f"spin fires at {fires_at(spin)} (expect {DRIFT_SPIN_STEPS-1})")

    # ---- E. corner gate (sim): drift-match accrues ONLY in corners ----
    print(f"\n[E] CORNER GATE: base controller drive; r_drift must be 0 on straights, active in corners")
    env.reset(options={"spawn_idx": 0}); ctrl = BaseController(); ctrl.reset()
    prev = u._drift_sum; c_steps = c_active = s_steps = s_active = 0
    for i in range(450):
        s = veh.sensors["agent_state"]; vel = s["vel"]
        st, th = ctrl.action(s["pos"], vel, s.get("dir", (1.0, 0.0, 0.0)))
        _, _, t, tr, info = env.step(np.array([st, th], np.float32))
        inc = u._drift_sum - prev; prev = u._drift_sum
        corner = abs(float(_PROFILE_KAPPA[u._progress_idx])) > DRIFT_CORNER_KAPPA
        if corner: c_steps += 1; c_active += (inc > 1e-9)
        else:      s_steps += 1; s_active += (inc > 1e-9)
        if t or tr: break
    print(f"    corner steps {c_steps}: r_drift active {c_active}  ({100*c_active/max(c_steps,1):.0f}%)")
    print(f"    straight steps {s_steps}: r_drift active {s_active}  -> "
          f"{'OK (0 on straights -> gated)' if s_active == 0 else 'FAIL (leaked onto straights)'}")

    # ---- C. forced spin: terminator MUST fire ----
    print(f"\n[C] FORCED SPIN (controller to ~20 m/s, then full-lock):")
    env.reset(options={"spawn_idx": straight_idx}); ctrl.reset(); ctrl._idx = straight_idx
    ok, sp0 = _drive_to_speed(env, u, veh, ctrl, 20.0)
    term = None; nstep = 0; bpk = 0.0
    for i in range(200):
        _, _, t, tr, info = env.step(np.array([1.0, 1.0], np.float32))
        nstep += 1; bpk = max(bpk, u._last_beta)
        if t or tr: term = info.get("termination_reason"); break
    print(f"    reached {sp0:.1f} m/s; steps={nstep} peak|beta|={bpk:.0f}; ended: {term}  -> "
          f"{'OK (fired)' if term in ('loss_of_control','backward') else 'check'}")

    env.close()
    try: _shared["bng"].close()
    except Exception: pass
    print("===== drift probe done =====")


if __name__ == "__main__":
    main()
