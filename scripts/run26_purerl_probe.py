"""run26 PURE-RL pre-flight probe (read-only, no training).

run26 drops the controller + residual entirely: plain SAC, the policy outputs the FULL action.
Before spending a 7k smoke, confirm the pure-RL plumbing on the GTS:
  1. make_beamng_env with NO residual wrapper -> the raw env is what SAC sees (assert type).
  2. observation_space is 19-dim; obs is finite every step (no NaN feeding the net).
  3. active config reads back as GTS (gts.pc + adaptive coilover).
  4. the grip/off-track obs[18] is live (ramps up as the car runs wide).
  5. the loss-of-control spin terminator CAN fire: provoke a spin (hard steer + throttle from a
     stop) and confirm termination_reason == loss_of_control (or at least a clean terminal + reason).

Port 25252 (sim free)."""
import math, os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from envs.residual_hybrid import ResidualHybrid
from envs.beamng_env import make_beamng_env, _shared, VEHICLE_PART_CONFIG


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=True, home=home, host="localhost", port=25252,
                          launch=True, headless=True, nogpu=True, steer_rate=0.5)

    print("\n===== run26 PURE-RL pre-flight (GTS) =====")
    # 1. pure path: no residual wrapper
    is_resid = isinstance(env, ResidualHybrid)
    print(f"[1] residual wrapper present: {is_resid}  -> {'FAIL' if is_resid else 'OK (pure RL: policy outputs full action)'}")

    # 2. obs space
    shape = env.observation_space.shape
    print(f"[2] observation_space = {shape}  -> {'OK' if shape == (19,) else 'FAIL (expected (19,))'}")

    obs, _ = env.reset(options={"spawn_idx": 0})
    veh = _shared["vehicle"]

    # 3. config read-back
    blob = str(veh.get_part_config())
    m = re.search(r"'partConfigFilename': '([^']*)'", blob)
    cfg = m.group(1) if m else "?"
    i = blob.find("'scintilla_coilover_F/")
    mc = re.search(r"'chosenPartName': '([^']*)'", blob[i:i + 600]) if i > 0 else None
    coil = mc.group(1) if mc else "?"
    gts_ok = cfg.endswith("gts.pc") and "adaptive" in coil
    print(f"[3] partConfigFilename = {cfg}")
    print(f"    scintilla_coilover_F = {coil}  -> {'OK (GTS, adaptive)' if gts_ok else 'FAIL'}")
    print(f"    env VEHICLE_PART_CONFIG = {VEHICLE_PART_CONFIG}")

    # 4 + 5. step: floor throttle + hard steer from the start to provoke a spin; watch obs finite,
    # grip obs[18], and the terminator.
    print(f"[4/5] provoking a spin (steer=+1, throttle=1) -- watch obs[18] (grip) + terminator")
    finite_all = True; grip_max = 0.0; beta_max = 0.0; term_reason = None; nstep = 0
    for step in range(120):
        a = np.array([1.0, 1.0], np.float32)        # hard steer + full throttle: should spin/run wide
        obs, r, term, trunc, info = env.step(a)
        nstep += 1
        if not np.all(np.isfinite(obs)) or not np.isfinite(r):
            finite_all = False
            print(f"    !! NON-FINITE at step {step}: obs_finite={np.all(np.isfinite(obs))} r={r}")
        grip_max = max(grip_max, float(obs[18]))
        beta_max = max(beta_max, abs(float(obs[17])) * 45.0)
        if term or trunc:
            term_reason = info.get("termination_reason")
            break
    print(f"    steps={nstep}  obs+reward finite throughout: {finite_all}  -> {'OK' if finite_all else 'FAIL (NaN)'}")
    print(f"    grip obs[18] max = {grip_max:.2f} (0=on road, 1=at edge)  beta max ~{beta_max:.0f} deg")
    print(f"    terminated: reason = {term_reason}  -> "
          f"{'OK (terminator fired)' if term_reason else 'no terminal in 120 steps (spin not provoked)'}")

    env.close()
    try: _shared["bng"].close()
    except Exception: pass
    print("===== pre-flight done =====")


if __name__ == "__main__":
    main()
