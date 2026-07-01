"""run30 backward-penalty verification (read-only, no training, no file changes).

The reverse flail reappeared in the run30 watch, so confirm the backward-motion penalty is still
wired + firing in the CURRENT drift reward path, and probe one nuance: the reward has an earlier
KILL-SWITCH early-return when the NOSE points backward (heading_align < -0.2) -> in that case the
reward is 0 and the backward_pen block is SKIPPED (relies on the 40-step backward terminator). So the
backward penalty bites a nose-forward reverse (rolling back while facing forward) but NOT a spun
nose-backward reverse. This forces a reverse and reports which path fires. Port 25253 (sim free)."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from envs.beamng_env import make_beamng_env, _shared, W_DRIFT_BACKWARD, HEADING_KILL_THRESHOLD


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost", port=25253,
                          launch=True, headless=True, nogpu=True, steer_rate=0.5, drift_mode=True)
    u = env.unwrapped
    print(f"\n===== run30 backward-penalty check =====")
    print(f"W_DRIFT_BACKWARD = {W_DRIFT_BACKWARD} (per m/s of reverse); HEADING_KILL_THRESHOLD = {HEADING_KILL_THRESHOLD}")

    env.reset(options={"spawn_idx": 0})
    print("\nFORCED REVERSE (full brake from a stop): watch fwd_vel<0, r_backward, heading_align")
    fired = 0; killswitch = 0; min_fwd = 0.0; nstep = 0
    for i in range(160):
        prev_rb = u._backward_pen_sum
        _, _, t, tr, info = env.step(np.array([0.0, -1.0], np.float32))
        nstep += 1
        s = _shared["vehicle"].sensors["agent_state"]; vel = s["vel"]
        sp = math.hypot(vel[0], vel[1]); fwd = sp * u._last_raw_alignment
        min_fwd = min(min_fwd, fwd)
        d_rb = u._backward_pen_sum - prev_rb
        if fwd < -0.2:                       # actually reversing this step
            if d_rb < -1e-9:
                fired += 1                   # backward penalty applied
            elif u._last_heading_align < HEADING_KILL_THRESHOLD:
                killswitch += 1              # nose-backward -> kill-switch pre-empted the penalty
        if t or tr:
            term = info.get("termination_reason"); break
    else:
        term = "ran"
    print(f"  steps={nstep}  min fwd_vel {min_fwd:+.1f} m/s")
    print(f"  r_backward (penalty sum) = {u._backward_pen_sum:.1f}   backward_frac = {float(info['backward_frac']):.2f}")
    print(f"  reverse steps PENALIZED (nose-forward): {fired}")
    print(f"  reverse steps via KILL-SWITCH (nose-backward, reward 0, penalty skipped): {killswitch}")
    print(f"  ended: {term}")
    verdict = "WIRED + FIRES" if u._backward_pen_sum < -0.01 else "did NOT fire (car didn't reverse, or all nose-backward)"
    print(f"  -> backward penalty {verdict}")
    if killswitch > 0:
        print(f"  NOTE: a nose-backward reverse (e.g. pinned against a wall) hits the kill-switch first,")
        print(f"        so the explicit backward penalty is SKIPPED there -- only the 40-step backward")
        print(f"        terminator + the 0 reward apply. Relevant to the wall-pin flail.")

    env.close()
    try: _shared["bng"].close()
    except Exception: pass
    print("===== done =====")


if __name__ == "__main__":
    main()
