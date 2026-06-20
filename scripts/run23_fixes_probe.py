"""run23 STEP 1 verification: the three grip-awareness fixes on the NEW 19-dim env.
(The run22 policy is 18-dim and can't load here; we drive the lapping BASE CONTROLLER for the
clean-cornering checks and SCRIPTED actions to force the two failure modes.)

PHASE 1 -- clean controller lap (no policy): confirm NO false loss_of_control, yaw-rate stays
  low in every corner, and the grip obs[18] reads ~0 on the line (rising only where it runs wide).
PHASE 2 -- forced SPIN: from speed, hold steer=+1/throttle=+1 to power-oversteer into a donut;
  confirm loss_of_control terminates FAST and log the yaw-rate at termination.
PHASE 3 -- forced OFF-ROAD: steer steadily off the side into the grass; confirm off_track fires
  AND grip obs[18] saturates to ~1.0 just before (the signal the policy will learn from).
Port 25252, start line, steer_rate=0.5."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from envs.beamng_env import (make_beamng_env, _shared, LOSS_OF_CONTROL_YAWRATE_DEG_S,
                             LOSS_OF_CONTROL_STEPS, OFF_TRACK_THRESHOLD_M)
from envs.base_controller import BaseController

DT = 3.0 / 60.0


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost", port=25252,
                          launch=True, headless=True, nogpu=True, steer_rate=0.5)
    u = env.unwrapped
    print(f"loss-of-control: yaw>{LOSS_OF_CONTROL_YAWRATE_DEG_S:.0f} deg/s x{LOSS_OF_CONTROL_STEPS} steps; "
          f"off-track>{OFF_TRACK_THRESHOLD_M:.0f} m\n")

    # ---- PHASE 1: clean controller lap ----
    ctrl = BaseController()
    obs, _ = env.reset(options={"spawn_idx": 0})
    max_yaw = 0.0; max_loc = 0; grip_on_line = []; false_fire = False; info = {}
    for step in range(2000):
        s = _shared["vehicle"].sensors["agent_state"]
        a = np.array(ctrl.action(s["pos"], s["vel"], s.get("dir", (1.0, 0.0, 0.0))), np.float32)
        obs, _, term, trunc, info = env.step(a)
        max_yaw = max(max_yaw, u._yaw_rate_deg_s); max_loc = max(max_loc, u._loss_of_control_steps)
        grip_on_line.append(float(obs[18]))
        if info["termination_reason"] == "loss_of_control":
            false_fire = True; break
        if term or trunc:
            break
    print("=== PHASE 1: clean controller lap ===")
    print(f"  ended: {info.get('termination_reason')} max_arc={info.get('max_arc',0):.0f}m steps={step+1}")
    print(f"  FALSE loss_of_control fire: {false_fire} ({'BAD' if false_fire else 'OK -- none in clean cornering'})")
    print(f"  max yaw-rate in clean cornering: {max_yaw:.0f} deg/s (threshold {LOSS_OF_CONTROL_YAWRATE_DEG_S:.0f}); "
          f"max consec over-threshold: {max_loc} (need {LOSS_OF_CONTROL_STEPS})")
    print(f"  grip obs[18] on the racing line: mean={np.mean(grip_on_line):.3f} p95={np.percentile(grip_on_line,95):.3f} "
          f"max={max(grip_on_line):.3f} ({'~0 on-line OK' if np.mean(grip_on_line)<0.2 else 'high -- check'})\n")

    # ---- PHASE 2: forced CONTAINED donut from rest (burnout-style, stays within 8 m) ----
    # From near-rest, full throttle + full lock on the 748hp RWD car lights the rears and pivots
    # it IN PLACE (high yaw, low forward travel) -- the contained donut that no old terminator
    # caught. Building speed first would instead carve it off the edge (off_track), which is why
    # the first attempt mis-fired. max yaw + consec tracked to show the terminator engaging.
    obs, _ = env.reset(options={"spawn_idx": 0})
    spin_term_step = None; yaw_at_term = 0.0; max_yaw_spin = 0.0; max_consec = 0; off_first = None
    for step in range(400):
        obs, _, term, trunc, info = env.step(np.array([1.0, 1.0], np.float32))  # donut from rest
        max_yaw_spin = max(max_yaw_spin, u._yaw_rate_deg_s); max_consec = max(max_consec, u._loss_of_control_steps)
        if info["termination_reason"] == "loss_of_control" and spin_term_step is None:
            spin_term_step = step + 1; yaw_at_term = u._yaw_rate_deg_s
        if info["termination_reason"] == "off_track" and off_first is None:
            off_first = step + 1
        if term or trunc:
            break
    print("=== PHASE 2: forced contained donut (steer=+1, throttle=+1 from rest) ===")
    print(f"  spin yaw-rate: max={max_yaw_spin:.0f} deg/s, max consec >threshold={max_consec} (need {LOSS_OF_CONTROL_STEPS})")
    if spin_term_step is not None:
        print(f"  loss_of_control TERMINATED after {spin_term_step} steps (~{spin_term_step*DT:.1f}s); "
              f"yaw at term {yaw_at_term:.0f} deg/s -> OK (fast spin kill)")
    else:
        print(f"  did NOT terminate via loss_of_control (got {info.get('termination_reason')} "
              f"{'at step '+str(off_first) if off_first else ''}) -- CHECK")
    print()

    # ---- PHASE 3: forced off-road (scripted steer off the side) ----
    obs, _ = env.reset(options={"spawn_idx": 0})
    for _ in range(30):
        obs, _, term, trunc, info = env.step(np.array([0.0, 0.5], np.float32))
        if term or trunc: break
    grip_trace = []; off_step = None
    for step in range(300):
        obs, _, term, trunc, info = env.step(np.array([0.5, 0.3], np.float32))  # drift off one side
        grip_trace.append(float(obs[18]))
        if info["termination_reason"] == "off_track":
            off_step = step + 1; break
        if term or trunc:
            break
    print("=== PHASE 3: forced off-road (steer into the verge) ===")
    if off_step is not None:
        pre = grip_trace[-3:] if len(grip_trace) >= 3 else grip_trace
        print(f"  off_track FIRED after {off_step} steps; grip obs[18] just before = {np.round(pre,3)} "
              f"({'saturated ~1.0 OK' if max(pre) > 0.9 else 'did not saturate -- check'})")
    else:
        print(f"  off_track did NOT fire (got {info.get('termination_reason')}); grip max={max(grip_trace or [0]):.3f} -- CHECK")
    env.close()
    try: _shared["bng"].close()
    except Exception: pass


if __name__ == "__main__":
    main()
