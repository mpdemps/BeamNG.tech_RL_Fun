"""GTS clean top-speed probe (read-only): the race-profile probe spun the GTS at T6 (~1615m)
before it ever reached the long back straight, so it never got a clean top-speed reading.

Fix: spawn the car ON the long back straight (past T6), aligned with the road, and FLOOR the
throttle down the ~1300 m straight with controller steering keeping it on the line. Measure peak
speed and whether it's still climbing at the far end (=> true top speed is even higher). This sets
V_MAX for the GTS profile. Port 25252 (sim free)."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from envs.beamng_env import make_beamng_env, _shared, V_TARGET_PROFILE, VEHICLE_PART_CONFIG
from envs.base_controller import BaseController
from envs.speed_profile import compute_speed_profile
from data.raceline_builtin import RACELINE

MAX_STEPS = 1400
STRAIGHT = (1625.0, 2938.0)   # the long back straight (run24 probe); spawn just inside it


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost", port=25252,
                          launch=True, headless=True, nogpu=True, steer_rate=0.5)
    u = env.unwrapped
    cum = np.array(compute_speed_profile(RACELINE)[2])
    spawn_idx = int(np.argmin(np.abs(cum - (STRAIGHT[0] + 35.0))))  # ~35 m into the straight
    ctrl = BaseController()
    env.reset(options={"spawn_idx": spawn_idx})
    ctrl._idx = spawn_idx   # sync the controller's nearest-point tracker to the mid-track spawn
                            # (its windowed search starts at _idx=0, else it's lost off the line)
    print(f"CONFIG = {VEHICLE_PART_CONFIG}")
    print(f"clean top-speed: spawn arc {cum[spawn_idx]:.0f}m (idx {spawn_idx}), FLOOR down the "
          f"straight to {STRAIGHT[1]:.0f}m\n")
    peak = 0.0; peak_arc = 0.0; samples = []
    for step in range(MAX_STEPS):
        s = _shared["vehicle"].sensors["agent_state"]
        vel = s["vel"]; speed = math.hypot(vel[0], vel[1])
        st, _ = ctrl.action(s["pos"], vel, s.get("dir", (1.0, 0.0, 0.0)))
        obs, _, term, trunc, info = env.step(np.array([st, 1.0], np.float32))  # FLOOR throttle
        arc = float(u._cur_centerline_dist)
        if speed > peak:
            peak, peak_arc = speed, arc
        samples.append((arc, speed))
        if arc >= STRAIGHT[1] - 20 or term or trunc:
            if term or trunc:
                print(f"  (ended early: {info.get('termination_reason')} at arc {arc:.0f}m, step {step+1})")
            break
    on = [(a, v) for a, v in samples if STRAIGHT[0] <= a <= STRAIGHT[1]]
    print(f"PEAK SPEED: {peak:.1f} m/s = {peak*3.6:.0f} kph  (at arc {peak_arc:.0f} m)")
    if on:
        on.sort()
        mx = max(v for _, v in on)
        print(f"straight {STRAIGHT[0]:.0f}-{STRAIGHT[1]:.0f}m: entry {on[0][1]:.1f} -> exit {on[-1][1]:.1f} m/s "
              f"({on[-1][1]*3.6:.0f} kph); max {mx:.1f} m/s")
        climbing = on[-1][1] >= mx - 0.3
        print(f"  {'still ACCELERATING at the far end -> true top speed is higher' if climbing else 'leveled off -> at/near top speed'}")
    env.close()
    try: _shared["bng"].close()
    except Exception: pass


if __name__ == "__main__":
    main()
