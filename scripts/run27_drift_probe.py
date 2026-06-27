"""run27 Phase 2 DRIFT scaffolding probe (read-only, no training).

Confirms the four scaffolding pieces on the GTS drift car:
  A. CONFIG read-back: gts_drift.pc applied -> rear diff = differential_R_race (LSD) and the
     drive mode defaults to DRIFT.
  B. obs is 20-dim and obs[19] = beta_target/45 tracks the speed (sanity vs envs.drift.beta_target).
  C. TERMINATOR re-tune: drive a sustained power-slide (high slip/yaw while still going down-track)
     and log BOTH counters -- the OLD yaw terminator (loss_of_control_steps, would fire >=8) vs the
     NEW drift terminator (drift_spin_steps). If the car keeps progressing, the new counter stays ~0
     (drift ALLOWED) even while the old one climbs past 8 (would have killed it). Then force a real
     SPIN (full steer+throttle from a stop, no correction) and confirm the new terminator FIRES.
Reports measured beta/yaw so the SPIN_BETA / DRIFT_SPIN_STEPS / NO_PROGRESS thresholds are grounded.
Port 25253 (sim free)."""
import math, os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import envs.beamng_env as be
from envs.beamng_env import make_beamng_env, _shared
from envs.base_controller import BaseController
from envs.speed_profile import compute_speed_profile
from envs.drift import beta_target, SPIN_BETA_DEG, DRIFT_SPIN_STEPS, DRIFT_NO_PROGRESS_STEPS
from data.raceline_builtin import RACELINE


def _drive_to_speed(env, u, veh, ctrl, target, max_steps=200):
    """Use the base controller (steering + throttle) to bring the car up to `target` m/s ON the
    racing line, so beta is live (>8 m/s ESC floor) and the car is not off-road. Returns (ok, sp)."""
    for _ in range(max_steps):
        s = veh.sensors["agent_state"]; vel = s["vel"]; sp = math.hypot(vel[0], vel[1])
        st, th = ctrl.action(s["pos"], vel, s.get("dir", (1.0, 0.0, 0.0)))
        _, _, t, tr, info = env.step(np.array([st, max(th, 0.6)], np.float32))  # min throttle to spool up
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
    straight_idx = int(np.argmin(np.abs(cum - 1700.0)))   # mid back-straight, aligned

    print("\n===== run27 DRIFT scaffolding probe =====")
    print(f"obs space: {env.observation_space.shape} (expect (20,))")

    # ---- A. config read-back ----
    env.reset(options={"spawn_idx": 0})
    veh = _shared["vehicle"]
    blob = str(veh.get_part_config())
    m = re.search(r"'partConfigFilename': '([^']*)'", blob)
    cfg = m.group(1) if m else "?"
    def chosen(key):
        i = blob.find("'" + key + "'")
        mm = re.search(r"'chosenPartName': '([^']*)'", blob[i:i + 800]) if i >= 0 else None
        return mm.group(1) if mm else "?"
    diff = chosen("scintilla_differential_R")
    dmode = chosen("scintilla_DSE_drivemodes_default")
    coil = chosen("scintilla_coilover_F")
    print(f"\n[A] partConfigFilename = {cfg}")
    print(f"    rear diff           = {diff}   -> {'OK (LSD)' if 'race' in diff else 'FAIL (open)'}")
    print(f"    drivemode default   = {dmode}  -> {'OK (drift)' if 'drift' in dmode else 'FAIL'}")
    print(f"    coilover_F          = {coil} (expect adaptive: GTS base)")

    # ---- B. obs[19] beta_target tracks speed ----
    print(f"\n[B] obs[19] = beta_target/45 vs envs.drift.beta_target(speed):")
    obs, _ = env.reset(options={"spawn_idx": straight_idx})
    print(f"    at spawn: speed~0 -> obs[19]*45 = {obs[19]*45:.1f} deg (expect ~{beta_target(0):.0f})")

    # ---- D. terminator DISCRIMINATION (deterministic, no sim) ----
    # Hand-flying a clean 30 deg drift open-loop is exactly the skill we're training the RL to learn,
    # so prove the terminator LOGIC directly: feed it an in-band drift trace vs a spin trace and check
    # it ends the spin but never the drift. (is_drift_spin + the consecutive-step counter, as the env
    # runs them.)
    from envs.drift import is_drift_spin
    def fires_at(trace):
        c = 0
        for k, (b, h) in enumerate(trace):
            c = c + 1 if is_drift_spin(b, h) else 0
            if c >= DRIFT_SPIN_STEPS:
                return k
        return None
    controlled = [(30.0, 0.85)] * 100   # in-band slide (beta 30 deg, nose forward), held 5 s
    flick_then_hold = [(72.0, 0.4)] * 6 + [(30.0, 0.85)] * 100  # brief initiation overshoot, then settle
    spin = [(95.0, -0.3)] * 40          # past the band + nose backward
    print(f"\n[D] terminator discrimination (logic, no sim; fires at >= {DRIFT_SPIN_STEPS} consec):")
    print(f"    controlled in-band drift (beta 30, 100 steps): fires at {fires_at(controlled)}  -> "
          f"{'OK (never -> drift ALLOWED)' if fires_at(controlled) is None else 'FAIL'}")
    print(f"    initiation flick (beta 72 x6) then settle:      fires at {fires_at(flick_then_hold)}  -> "
          f"{'OK (flick ridden out)' if fires_at(flick_then_hold) is None else 'FAIL'}")
    print(f"    true spin (beta 95, backward):                  fires at {fires_at(spin)}  -> "
          f"{'OK (fires)' if fires_at(spin) is not None else 'FAIL'}")

    ctrl = BaseController()

    # ---- C1. real-dynamics over-the-limit slide: confirms beta>60 detection in the sim ----
    print(f"\n[C1] HARD SLIDE in the sim (controller to ~20 m/s, then power-on oversteer kick):")
    env.reset(options={"spawn_idx": straight_idx}); ctrl.reset(); ctrl._idx = straight_idx
    ok, sp0 = _drive_to_speed(env, u, veh, ctrl, 20.0)
    print(f"    reached {sp0:.1f} m/s on the line (ok={ok})")
    old_max = 0; new_max = 0; beta_pk = 0.0; prog_steps = 0; term = None; nstep = 0; betas = []
    for i in range(120):                          # controller steering + FLOOR + extra steer kick
        s = veh.sensors["agent_state"]; vel = s["vel"]
        st, _ = ctrl.action(s["pos"], vel, s.get("dir", (1.0, 0.0, 0.0)))
        st = max(-1.0, min(1.0, st + 0.4 * (1 if st >= 0 else -1)))  # over-steer kick to break the rear
        obs, r, t, tr, info = env.step(np.array([st, 1.0], np.float32))
        nstep += 1
        old_max = max(old_max, u._loss_of_control_steps)
        new_max = max(new_max, u._drift_spin_steps)
        beta_pk = max(beta_pk, u._last_beta); betas.append(u._last_beta)
        if u._steps_since_progress == 0: prog_steps += 1
        if t or tr:
            term = info.get("termination_reason"); break
    beta_hi = sorted(betas)[int(len(betas) * 0.9)] if betas else 0.0
    print(f"    steps={nstep}  peak|beta|={beta_pk:.0f} deg (p90 {beta_hi:.0f})  progressing {prog_steps}/{nstep} steps")
    print(f"    OLD yaw counter peak = {old_max}  (the Phase-1 yaw terminator fires at 8)")
    print(f"    NEW drift counter peak = {new_max}  (fires at {DRIFT_SPIN_STEPS}; beta>{SPIN_BETA_DEG:.0f} this hard => spin)")
    print(f"    ended: {term or 'survived'}  -> "
          f"{'OK (a real over-the-limit slide is caught)' if term in ('loss_of_control','off_track','backward') else 'check'}")

    # ---- C2. forced spin: new terminator MUST fire ----
    print(f"\n[C2] FORCED SPIN (controller to ~20 m/s, then full-lock + full throttle, no correction):")
    env.reset(options={"spawn_idx": straight_idx}); ctrl.reset(); ctrl._idx = straight_idx
    ok, sp0 = _drive_to_speed(env, u, veh, ctrl, 20.0)
    print(f"    reached {sp0:.1f} m/s (ok={ok}); now yanking full lock")
    beta_pk = 0.0; term = None; nstep = 0; new_max = 0; back_max = 0
    for i in range(200):
        obs, r, t, tr, info = env.step(np.array([1.0, 1.0], np.float32))
        nstep += 1; beta_pk = max(beta_pk, u._last_beta)
        new_max = max(new_max, u._drift_spin_steps); back_max = max(back_max, u._backward_steps)
        if t or tr:
            term = info.get("termination_reason"); break
    print(f"    steps={nstep}  peak|beta|={beta_pk:.0f} deg  drift_spin_peak={new_max}  backward_peak={back_max}")
    print(f"    ended: {term}  -> {'OK (terminator fired)' if term in ('loss_of_control','backward') else 'check: '+str(term)}")

    print(f"\n    proposed thresholds: SPIN_BETA={SPIN_BETA_DEG} DRIFT_SPIN_STEPS={DRIFT_SPIN_STEPS} "
          f"NO_PROGRESS={DRIFT_NO_PROGRESS_STEPS}")
    env.close()
    try: _shared["bng"].close()
    except Exception: pass
    print("===== drift probe done =====")


if __name__ == "__main__":
    main()
