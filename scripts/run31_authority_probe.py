"""run31 controller-backed-drift AUTHORITY probe (read-only, no training, no file changes).

The make-or-break for the run31 architecture pivot (controller backbone + corner-gated RL residual):
can the BASE CONTROLLER'S STEERING stay active through a corner while a forced slide (extra throttle
breaking the rear loose) is NOT nulled back to grip? If yes -> the clean split is "controller STEERS,
RL owns THROTTLE in corners" (option b). If the controller steering re-grips and kills the slide ->
the RL also needs steer authority (option a).

Two passes through the SAME corner (R~44 off the long back straight, ~arc 2979), controller steering
throughout:
  A. BASELINE  -- controller throttle too (grip): beta should stay low, car tracks the line.
  B. FORCED SLIDE -- floor the throttle ONLY in the corner (cf>0.3), controller keeps steering:
     does beta build + SUSTAIN through the corner (drift held) with the car on-track and nose still
     roughly forward (heading_align not collapsing)? That is option (b) working.
Port 25253 (sim free)."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from envs.beamng_env import make_beamng_env, _shared, _PROFILE_KAPPA, OFF_TRACK_THRESHOLD_M
from envs.base_controller import BaseController
from envs.drift import corner_factor
from envs.speed_profile import compute_speed_profile
from data.raceline_builtin import RACELINE

CORNER_ARC = 2979.0     # R~44 corner just after the long back straight
SPAWN_ARC = 2820.0      # ~160 m before it, on the straight -> controller builds speed then enters


def run_pass(env, u, ctrl, spawn_idx, force_slide):
    ctrl.reset(); ctrl._idx = spawn_idx
    env.reset(options={"spawn_idx": spawn_idx})
    ctrl._idx = spawn_idx
    veh = _shared["vehicle"]
    in_corner = []      # per-step (beta, heading_align, offtrack_frac) while cf>0.3
    term = None; nstep = 0; reached = 0.0
    for i in range(400):
        s = veh.sensors["agent_state"]; vel = s["vel"]; sp = math.hypot(vel[0], vel[1])
        st, th = ctrl.action(s["pos"], vel, s.get("dir", (1.0, 0.0, 0.0)))
        cf = corner_factor(abs(float(_PROFILE_KAPPA[u._progress_idx])))
        if force_slide and cf > 0.3:
            th = 1.0                                  # RL-style: break the rear loose in the corner
        _, _, t, tr, info = env.step(np.array([st, th], np.float32))
        nstep += 1; reached = float(u._cur_centerline_dist)
        cf2 = corner_factor(abs(float(_PROFILE_KAPPA[u._progress_idx])))
        if cf2 > 0.3 and sp > 8.0:
            in_corner.append((u._last_beta, u._last_heading_align))
        if t or tr:
            term = info.get("termination_reason"); break
    betas = [b for b, _ in in_corner]; heads = [h for _, h in in_corner]
    return {
        "n_corner": len(betas),
        "beta_max": max(betas) if betas else 0.0,
        "beta_mean": float(np.mean(betas)) if betas else 0.0,
        "beta_band_steps": sum(1 for b in betas if 15 <= b <= 50),   # sustained drift-band steps
        "head_min": min(heads) if heads else 1.0,                    # nose stays forward? (1=aligned)
        "term": term, "reached": reached, "nstep": nstep,
    }


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost", port=25253,
                          launch=True, headless=True, nogpu=True, steer_rate=0.5, drift_mode=True)
    u = env.unwrapped
    cum = np.array(compute_speed_profile(RACELINE)[2])
    spawn_idx = int(np.argmin(np.abs(cum - SPAWN_ARC)))
    ctrl = BaseController()
    print(f"\n===== run31 authority probe: corner ~{CORNER_ARC:.0f}m, spawn arc {cum[spawn_idx]:.0f}m =====")

    a = run_pass(env, u, ctrl, spawn_idx, force_slide=False)
    print(f"\n[A] BASELINE (controller steer + controller throttle = grip):")
    print(f"    corner steps {a['n_corner']}: beta max {a['beta_max']:.0f} mean {a['beta_mean']:.0f}  "
          f"band-steps {a['beta_band_steps']}  head_min {a['head_min']:.2f}  reached {a['reached']:.0f}m  term {a['term']}")

    b = run_pass(env, u, ctrl, spawn_idx, force_slide=True)
    print(f"\n[B] FORCED SLIDE (controller steer + FLOORED throttle in corner):")
    print(f"    corner steps {b['n_corner']}: beta max {b['beta_max']:.0f} mean {b['beta_mean']:.0f}  "
          f"band-steps {b['beta_band_steps']}  head_min {b['head_min']:.2f}  reached {b['reached']:.0f}m  term {b['term']}")

    print(f"\n[VERDICT]")
    sustained = b['beta_band_steps'] >= 8 and b['head_min'] > -0.2 and b['reached'] > a['reached'] - 30
    print(f"    forced slide built beta {b['beta_mean']:.0f} deg (vs baseline {a['beta_mean']:.0f}) and held "
          f"{b['beta_band_steps']} band-steps with nose fwd (head_min {b['head_min']:.2f})")
    if sustained:
        print(f"    -> SUSTAINED: controller STEERING held the car through the corner while the forced")
        print(f"       slide was NOT nulled to grip. Option (b) viable: controller steers, RL owns throttle.")
    else:
        print(f"    -> NOT sustained (slide collapsed / spun / went off): controller steering alone can't")
        print(f"       hold it -> RL likely needs STEER authority too (option a). See beta/head/term above.")

    env.close()
    try: _shared["bng"].close()
    except Exception: pass
    print("===== run31 authority probe done =====")


if __name__ == "__main__":
    main()
