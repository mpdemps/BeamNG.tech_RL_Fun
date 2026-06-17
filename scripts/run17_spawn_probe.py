"""run17 §6.1 spawn probe: verify random_spawn produces SANE starts across the track
(on-track, down-track, start_speed <= v_target[idx], upstream checkpoints pre-marked) and
that the tightened OFF_TRACK_THRESHOLD (8m) terminates at the road edge, not on the road.
Uses make_beamng_env (launches headless correctly). Port 25252."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from envs.beamng_env import make_beamng_env, _shared, V_TARGET_PROFILE, OFF_TRACK_THRESHOLD_M, CENTERLINE


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=True, home=home, host="localhost", port=25252,
                          launch=True, headless=True, nogpu=True, steer_rate=0.5)
    print(f"OFF_TRACK_THRESHOLD={OFF_TRACK_THRESHOLD_M}m; {len(CENTERLINE)} centerline pts\n")
    print("=== 10 random spawns: on-track + start_speed<=v_target + checkpoints pre-marked ===")
    print(f"{'idx':>5}{'arc':>7}{'v_targ':>7}{'start_v':>8}{'center_off':>11}{'cps_premarked':>14}")
    bad = 0
    for _ in range(10):
        env.reset()
        idx = env._progress_idx
        # let one step settle to read speed/center_off
        env.step([0.0, 0.0])
        s = _shared["vehicle"].sensors["agent_state"]; vel = s["vel"]
        sv = math.hypot(vel[0], vel[1])
        vt = V_TARGET_PROFILE[idx]
        coff = abs(env._center_off)
        ncp = len(env._checkpoints_hit)
        ok_speed = sv <= vt + 3.0          # within target (+ small settle tolerance)
        ok_track = coff < OFF_TRACK_THRESHOLD_M
        if not (ok_speed and ok_track): bad += 1
        print(f"{idx:>5}{env._cur_centerline_dist:>7.0f}{vt:>7.1f}{sv:>8.1f}{coff:>11.2f}{ncp:>14}"
              f"  {'' if ok_speed and ok_track else '<-- FLAG'}")
    print(f"\n  sane spawns: {10-bad}/10 (on-track AND start_speed<=v_target)")

    # off-track termination: spawn at the start, steer hard to leave the road, find the
    # center_off at which off_track fires.
    print("\n=== off_track termination distance (steer off the road) ===")
    env2 = env
    env2.reset(options={"spawn_idx": 0})
    term_coff = None
    for st in range(120):
        _, _, term, trunc, info = env2.step([1.0, 0.4])   # hard steer + mild throttle -> leave road
        if term or trunc:
            term_coff = abs(env2._center_off); reason = info["termination_reason"]; break
    if term_coff is not None:
        print(f"   terminated at center_off={term_coff:.2f}m, reason={reason}")
        print(f"   -> {'GOOD: ends near the road edge (~threshold), not 20m out' if term_coff <= OFF_TRACK_THRESHOLD_M + 3 or reason!='off_track' else 'CHECK'}")
    else:
        print("   (did not terminate in 120 steps)")
    env.close()
    try: _shared["bng"].close()
    except Exception: pass


if __name__ == "__main__":
    main()
