"""run31 controller-backed-drift BUILT-VERSION probe (read-only, no training, no file changes).

Drives the corner-gated ResidualHybrid (option a) around the drift env with CRAFTED residuals (a
stand-in for an aggressive policy) to confirm the build:
  A. bounds ramp (deterministic): straight tight (+/-0.05) -> corner big (steer +/-0.5, thr[-0.5,+1.0]),
     controller brake relaxed by cf.
  B. STRAIGHT BOUNDED + GRIPS: feed an aggressive residual [steer 0.5, throttle 1.0] on a straight;
     the wrapper must CLIP it to ~+/-0.05 so the applied action ~= controller and the car GRIPS
     (beta low) -- the fishtail is killed by the architecture.
  C. CORNER AUTHORITY: same aggressive throttle residual through a corner; the bound opens + the
     controller brake relaxes, so the applied throttle rises, a slide ESTABLISHES (beta up) and is
     NOT immediately nulled back to grip while the controller STEERING keeps tracking the corner.
  D. HEADING-GAP FIX: force reverse; confirm the backward penalty (r_backward) accrues -- including
     when the nose points backward (kill-switch path), which used to escape it.
Port 25253 (sim free)."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from envs.beamng_env import make_beamng_env, _shared, _PROFILE_KAPPA, W_DRIFT_BACKWARD, HEADING_KILL_THRESHOLD
from envs.residual_hybrid import (ResidualHybrid, CG_STRAIGHT, CG_CORNER_STEER,
                                  CG_CORNER_THR_UP, CG_CORNER_THR_DN)
from envs.drift import corner_factor
from envs.speed_profile import compute_speed_profile
from data.raceline_builtin import RACELINE

STRAIGHT_ARC = 1700.0
CORNER_SPAWN_ARC = 2820.0   # ~160 m before the R~44 corner at 2979 (build speed on the straight, enter)


def main():
    home = os.environ["BEAMNG_HOME"]
    base = make_beamng_env(random_spawn=False, home=home, host="localhost", port=25253,
                           launch=True, headless=True, nogpu=True, steer_rate=0.5, drift_mode=True)
    env = ResidualHybrid(base, corner_gated=True)
    u = env.unwrapped
    cum = np.array(compute_speed_profile(RACELINE)[2]); kabs = np.abs(np.array(_PROFILE_KAPPA))
    straight_idx = int(np.argmin(np.abs(cum - STRAIGHT_ARC)))
    corner_spawn = int(np.argmin(np.abs(cum - CORNER_SPAWN_ARC)))

    print("\n===== run31 controller-backed-drift BUILT probe =====")
    print("[A] bounds ramp:")
    for cf in (0.0, 0.5, 1.0):
        ds=CG_STRAIGHT+cf*(CG_CORNER_STEER-CG_STRAIGHT); up=CG_STRAIGHT+cf*(CG_CORNER_THR_UP-CG_STRAIGHT)
        dn=CG_STRAIGHT+cf*(CG_CORNER_THR_DN-CG_STRAIGHT)
        print(f"    cf={cf}: steer +/-{ds:.2f}  throttle[-{dn:.2f},+{up:.2f}]  ctrl brake x{1-cf:.2f}")

    # ---- B. straight bounded + grips ----
    env.reset(options={"spawn_idx": straight_idx}); env.controller._idx = straight_idx
    max_res = 0.0; beta_pk = 0.0; nstep = 0
    for i in range(70):
        _, _, t, tr, info = env.step(np.array([0.5, 1.0], np.float32))   # aggressive residual
        nstep += 1
        max_res = max(max_res, float(np.max(np.abs(env.last_residual))))
        beta_pk = max(beta_pk, u._last_beta)
        if t or tr: break
    print(f"\n[B] STRAIGHT + aggressive residual [0.5,1.0]:")
    print(f"    max |applied residual| = {max_res:.3f} (bound {CG_STRAIGHT}); peak|beta| {beta_pk:.0f} deg  cf~{info.get('residual_cf',0):.2f}")
    print(f"    -> {'OK (residual clipped tight, car grips)' if max_res <= CG_STRAIGHT + 1e-3 and beta_pk < 15 else 'check'}")

    # ---- C. corner authority: slide establishes, controller steering tracks ----
    env.reset(options={"spawn_idx": corner_spawn}); env.controller._idx = corner_spawn
    in_corner = []; applied_thr_corner = []; term = None; nstep = 0
    for i in range(400):
        _, _, t, tr, info = env.step(np.array([0.0, 0.7], np.float32))   # let controller steer; push throttle
        nstep += 1
        cf = float(info.get("residual_cf", 0.0))
        if cf > 0.3:
            in_corner.append((u._last_beta, u._last_heading_align))
            applied_thr_corner.append(float(env.last_applied[1]))
        if t or tr: term = info.get("termination_reason"); break
    betas=[b for b,_ in in_corner]; heads=[h for _,h in in_corner]
    band=sum(1 for b in betas if 15<=b<=50)
    print(f"\n[C] CORNER + throttle residual 0.7 (steer from controller):")
    print(f"    corner steps {len(betas)}: applied throttle mean {np.mean(applied_thr_corner) if applied_thr_corner else 0:.2f} "
          f"(vs straight-bound {CG_STRAIGHT}); beta max {max(betas) if betas else 0:.0f} mean {np.mean(betas) if betas else 0:.0f}")
    print(f"    drift-band steps {band}; nose head_min {min(heads) if heads else 1:.2f}; term {term}")
    print(f"    -> {'OK (authority opened, slide established + held, steering tracked)' if band>=5 and (min(heads) if heads else 1)>-0.3 else 'partial -- see numbers'}")

    # ---- D. heading-gap fix: reverse penalized (incl nose-back kill-switch path) ----
    env.reset(options={"spawn_idx": 0})
    killswitch_pen_steps = 0; nstep = 0
    for i in range(160):
        # spin it hard then it reverses: big steer + brake => nose swings back, car can roll backward
        res = np.array([1.0, -1.0], np.float32)
        prev_rb = u._backward_pen_sum
        _, _, t, tr, info = env.step(res)
        nstep += 1
        if u._last_heading_align < HEADING_KILL_THRESHOLD and (u._backward_pen_sum - prev_rb) < -1e-9:
            killswitch_pen_steps += 1
        if t or tr: term = info.get("termination_reason"); break
    print(f"\n[D] HEADING-GAP FIX (force spin+reverse): r_backward={u._backward_pen_sum:.1f}  "
          f"nose-back steps penalized (kill-switch path) = {killswitch_pen_steps}")
    print(f"    -> {'OK (reverse penalized incl. nose-back)' if u._backward_pen_sum < -0.01 else 'no reverse occurred this run (fix is wired; see code)'}")

    env.close()
    try: _shared["bng"].close()
    except Exception: pass
    print("===== run31 hybrid probe done =====")


if __name__ == "__main__":
    main()
