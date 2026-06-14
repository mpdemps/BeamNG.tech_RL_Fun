"""run10 gating measurement: does the run9 weave scale with speed the way pure-
pursuit instability predicts? Deterministic rollout of a clean run9 checkpoint,
log per-step speed/center_off/steer/arc, then measure zigzag half-swing duration
and amplitude vs speed. Pure-pursuit (fixed 10m lookahead) predicts: preview time
T=L/v shrinks with v, so oscillation frequency RISES with speed (half-swing
duration falls ~1/v) and amplitude grows once T drops below the damping threshold.
No training; nothing else is running."""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import SAC
from envs.beamng_env import make_beamng_env, _shared, LOOKAHEAD_DISTANCES_M

CKPT = "checkpoints/mikey_run9/rolling_75000_steps.zip"
PORT = 25252
OUT = "logs/run10_speedscale_trace.csv"
REV_AMP_DEAD = 0.10
DT = 1.0 / 20.0  # 20 Hz


def main():
    home = os.environ.get("BEAMNG_HOME")
    model = SAC.load(CKPT, device="cpu")
    env = make_beamng_env(random_spawn=False, home=home, host="localhost",
                          port=PORT, launch=True, headless=True, nogpu=True)
    obs, _ = env.reset()
    rows = []
    for step in range(400):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        s = _shared["vehicle"].sensors["agent_state"]
        vel = s["vel"]
        speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
        rows.append((step, speed, env._center_off, env._cur_steer,
                     env._cur_centerline_dist, info["heading_align"],
                     int(info["heading_align"] < -0.2)))
        if term or trunc:
            print(f"ep ended step {step}: {info['termination_reason']} arc={env._cur_centerline_dist:.0f}m", flush=True)
            break
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "speed", "center_off", "steer", "arc", "heading_align", "killswitch"])
        w.writerows(rows)
    env.close()
    try:
        _shared["bng"].close()
    except Exception:
        pass

    # ---- speed-scaling analysis (pre-killswitch only) ----
    pre = [r for r in rows if r[6] == 0]
    offs = [r[2] for r in pre]
    spds = [r[1] for r in pre]
    # zigzag reversals
    revs = []
    direction = 0
    extreme = offs[0] if offs else 0.0
    for i, o in enumerate(offs):
        if direction >= 0 and o > extreme: extreme = o
        elif direction <= 0 and o < extreme: extreme = o
        if direction >= 0 and extreme - o > REV_AMP_DEAD:
            if direction == 1: revs.append((i, extreme))
            direction = -1; extreme = o
        elif direction <= 0 and o - extreme > REV_AMP_DEAD:
            if direction == -1: revs.append((i, extreme))
            direction = 1; extreme = o
        elif direction == 0:
            direction = 1 if o >= extreme else -1
    print(f"\n=== run9 rolling_75000 rollout: {len(pre)} pre-killswitch steps, "
          f"{len(revs)} zigzag reversals ===")
    print(f"speed range {min(spds)*3.6:.0f}-{max(spds)*3.6:.0f} kph")
    # per half-swing: duration (steps) and amplitude (|extreme change|), tagged by mean speed
    print(f"\n half-swing#  spd(kph)  dur(steps)  dur(s)  amp(m)  preview_T=10m/v(s)")
    swings = []
    for k in range(1, len(revs)):
        i0, e0 = revs[k - 1]; i1, e1 = revs[k]
        dur = i1 - i0
        amp = abs(e1 - e0)
        vmean = sum(spds[i0:i1 + 1]) / max(1, (i1 - i0 + 1))
        swings.append((vmean, dur, amp))
        print(f"   {k:>3}        {vmean*3.6:>5.0f}     {dur:>5}      {dur*DT:>4.2f}  {amp:>5.2f}   {10.0/max(vmean,1):>5.2f}")
    # correlation: does duration fall and amplitude rise with speed?
    if len(swings) >= 3:
        import statistics as st
        lo = [s for s in swings if s[0] < st.median([x[0] for x in swings])]
        hi = [s for s in swings if s[0] >= st.median([x[0] for x in swings])]
        def avg(xs, j): return sum(x[j] for x in xs) / len(xs)
        print(f"\n  LOW-speed half-swings (n={len(lo)}, mean {avg(lo,0)*3.6:.0f}kph): "
              f"dur {avg(lo,1):.1f} steps ({avg(lo,1)*DT:.2f}s), amp {avg(lo,2):.2f}m")
        print(f"  HIGH-speed half-swings (n={len(hi)}, mean {avg(hi,0)*3.6:.0f}kph): "
              f"dur {avg(hi,1):.1f} steps ({avg(hi,1)*DT:.2f}s), amp {avg(hi,2):.2f}m")
        print(f"\n  pure-pursuit signature = HIGH-speed shorter duration (higher freq) AND/OR larger amplitude")


if __name__ == "__main__":
    main()
