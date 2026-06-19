"""run19 MEASUREMENT (the run8 lesson: measure the APPLIED penalty before touching a weight).

Replays run18's best_model DETERMINISTICALLY from the start line (idx 0, the eval condition)
through the T1 turn-in spin, and logs step-by-step across the spin:
  - beta            : lateral slip angle (deg), env._last_beta
  - slip_pen_applied: info["spin_penalty"] -- the slip penalty ACTUALLY added to reward
                      (NOTE: the heading kill-switch zeroes this once nose > ~101deg off)
  - slip_pen_wouldbe: -W_SLIP*max(0, beta-BETA_DEAD) recomputed from beta, IGNORING the
                      kill-switch -- shows what the slide "should" cost if not structurally zeroed
  - over_pen        : info["speed_reward"] -- the over-speed penalty actually applied
  - v / v_target    : speed vs braking-aware target at the car's position
  - thr / steer     : APPLIED throttle (>0) / brake (<0 in thr), applied steer
  - head_align      : heading gate value (kill-switch fires below -0.2)

THE KEY QUESTION (Mike's): at SPIN ONSET (rear breaks loose at turn-in), is the slip penalty
already NONZERO (beta > BETA_DEAD=9) -> raise W_SLIP; or is beta still < 9 -> lower BETA_DEAD.
Also: is the over-speed penalty firing at turn-in (flooring should exceed the corner's low
v_target)?  3x deterministic to confirm the spin is identical/reproducible. Port 25254."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stable_baselines3 import SAC
from envs.beamng_env import (make_beamng_env, _shared, V_TARGET_PROFILE,
                             W_SLIP, BETA_SLIP_DEAD)

CKPT = "checkpoints/mikey_run18/best_model/best_model.zip"
N_EP = 3
MAX_STEPS = 1500
T1_EXIT = 394.0


def wouldbe_slip_pen(beta):
    return -W_SLIP * max(0.0, beta - BETA_SLIP_DEAD)


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost", port=25254,
                          launch=True, headless=True, nogpu=True, steer_rate=0.5)
    model = SAC.load(CKPT, device="cpu")
    print(f"loaded {CKPT}; deterministic; start line (idx 0)")
    print(f"W_SLIP={W_SLIP} BETA_DEAD={BETA_SLIP_DEAD}deg; T1 exit {T1_EXIT}m\n")

    onsets = []
    for ep in range(N_EP):
        obs, _ = env.reset(options={"spawn_idx": 0})
        rows = []   # (step, arc, v, vt, thr, steer, beta, slip_app, slip_wb, over, head)
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            s = _shared["vehicle"].sensors["agent_state"]; vel = s["vel"]
            v = math.hypot(vel[0], vel[1])
            obs, _, term, trunc, info = env.step(action)
            arc = float(env._cur_centerline_dist)
            beta = float(env._last_beta)
            vt = float(V_TARGET_PROFILE[env._progress_idx])
            rows.append((step, arc, v, vt, float(env._cur_throttle), float(env._cur_steer),
                         beta, float(info["spin_penalty"]), wouldbe_slip_pen(beta),
                         float(info["speed_reward"]), float(info["heading_align"])))
            if term or trunc:
                break

        # spin onset = first step past 250m where beta first exceeds BETA_DEAD
        onset = next((i for i, r in enumerate(rows) if r[1] > 250.0 and r[6] > BETA_SLIP_DEAD), None)
        ma = info["max_arc"]
        cleared = "CLEARED" if ma > T1_EXIT else "SHORT OF T1"
        print(f"=== EP{ep}: len={len(rows)} term={info['termination_reason']} "
              f"max_arc={ma:.0f}m [{cleared}] checkpoints={int(float(info['checkpoints_reached']))} ===")
        if onset is None:
            print("  no beta>BETA_DEAD past 250m (no slide detected in T1 region)\n")
            continue
        onsets.append((rows[onset][1], rows[onset][6]))
        lo, hi = max(0, onset - 12), min(len(rows), onset + 28)
        print(f"  spin onset @ step {rows[onset][0]} arc={rows[onset][1]:.0f}m beta={rows[onset][6]:.1f}deg")
        print(f"  {'step':>4} {'arc':>5} {'v':>5} {'vtgt':>5} {'thr':>6} {'steer':>6} "
              f"{'beta':>5} {'slipAp':>7} {'slipWb':>7} {'overP':>7} {'head':>6}")
        for r in rows[lo:hi]:
            mark = " <-- ONSET" if r[0] == rows[onset][0] else (" KILL" if r[10] < -0.2 else "")
            print(f"  {r[0]:>4} {r[1]:>5.0f} {r[2]:>5.1f} {r[3]:>5.1f} {r[4]:>+6.2f} {r[5]:>+6.2f} "
                  f"{r[6]:>5.1f} {r[7]:>7.3f} {r[8]:>7.3f} {r[9]:>7.3f} {r[10]:>+6.2f}{mark}")
        print()

    if onsets:
        arcs = [o[0] for o in onsets]; betas = [o[1] for o in onsets]
        print(f"reproducibility: spin onset arc {min(arcs):.0f}-{max(arcs):.0f}m, "
              f"beta {min(betas):.1f}-{max(betas):.1f}deg across {len(onsets)} eps")
    env.close()
    try: _shared["bng"].close()
    except Exception: pass


if __name__ == "__main__":
    main()
