"""run11 measure-first: characterize LEGITIMATE accel slip vs runaway slip on the
Scintilla as a VEHICLE property (the run10 policy never grips, so its traces can't
show it). Scripted straight-line launch (steer=0) from standstill at fixed throttle
levels; for each, the sustained slip and the acceleration achieved. The throttle that
maximizes acceleration defines the optimal/legit slip; more throttle that adds slip
without adding accel is runaway. Sets TC_SLIP_DEAD (>= legit) / TC_SLIP_FULL (< runaway).

Clean rollout, separate port 25253, does not disturb run10."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import SAC  # noqa (not used; keep import parity light)
from envs.beamng_env import make_beamng_env, _shared

PORT = 25253
LEVELS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
N = 45  # steps per launch (~2.25s)


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost",
                          port=PORT, launch=True, headless=True, nogpu=True)
    print(f"\n=== traction sweep: straight-line launch, steer=0, {N} steps/level ===", flush=True)
    print(f"{'thr':>4} {'v@end':>6} {'dist':>6} {'slip_mean':>9} {'slip_pk':>7} {'slip@v5-15':>10}", flush=True)
    results = []
    for T in LEVELS:
        env.reset()
        slips = []; speeds = []; accel_slips = []
        for i in range(N):
            obs, r, term, trunc, info = env.step([0.0, T])
            s = _shared["vehicle"].sensors["agent_state"]; vel = s["vel"]
            v = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
            slips.append(info["slip"]); speeds.append(v)
            if 5.0 < v < 15.0:                 # slip during the productive mid-launch accel band
                accel_slips.append(info["slip"])
            if term or trunc:
                break
        vend = speeds[-1] if speeds else 0.0
        dist = env._cur_centerline_dist
        body = [s for s in slips[5:]]          # drop initial transient
        sm = sum(body) / len(body) if body else 0.0
        pk = max(slips) if slips else 0.0
        accel_slip = (sum(accel_slips) / len(accel_slips)) if accel_slips else float('nan')
        results.append((T, vend, dist, sm, pk, accel_slip))
        print(f"{T:>4.1f} {vend:>6.1f} {dist:>6.0f} {sm:>9.2f} {pk:>7.1f} {accel_slip:>10.2f}", flush=True)

    env.close()
    try:
        _shared["bng"].close()
    except Exception:
        pass

    # interpret: best accel = max v@end; its slip = legit optimal; runaway = where more
    # throttle stops adding speed but keeps adding slip.
    best = max(results, key=lambda r: r[1])
    print(f"\n  best acceleration: throttle={best[0]} -> v@end={best[1]:.1f} m/s, "
          f"sustained slip~{best[3]:.1f}, mid-launch slip~{best[5]:.1f}", flush=True)
    print("  legit-accel slip = slip at/below the best-accel throttle; runaway = slip "
          "from throttle that adds slip without adding v@end.", flush=True)


if __name__ == "__main__":
    main()
