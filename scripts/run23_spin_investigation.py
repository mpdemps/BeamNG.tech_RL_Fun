"""run23 STEP 1 investigation (measure before setting thresholds -- the run8 lesson).
Replays the run22 best_model FULL HYBRID (controller + clip(policy,+/-0.12)) deterministically
from the start line, into the T1-exit over-throttle spin/donut the watch exposed. Per step through
the spin it logs the quantities that decide all three run23 fixes:
  - dist_road : point-to-segment distance to ROAD_CENTERLINE (the off_track metric). Answers #2:
                does the donut stay <8 m (so off_track correctly doesn't fire) or is it off-road?
  - beta_raw  : slip angle (deg) computed UNPINNED (no 8 m/s ESC floor) -- is beta a reliable
                sustained-high spin signal, or does a slow donut pin it to ~0?
  - yawrate   : |d(nose heading)/dt| deg/s from State 'dir' -- the speed-independent spin signal.
  - speed, heading_align, backward_steps, steps_since_progress, term_reason -- why the existing
                backward(40)/stuck(200) terminators don't fire during a continuous spin.
Port 25252 (run stopped). steer_rate=0.5, start line."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stable_baselines3 import SAC
from envs.beamng_env import make_beamng_env, _shared, ROAD_CENTERLINE, _dist_to_road
from envs.residual_hybrid import ResidualHybrid

CKPT = "checkpoints/mikey_run22/best_model/best_model.zip"
MAX_STEPS = 1500
DT = 3.0 / 60.0


def main():
    home = os.environ["BEAMNG_HOME"]
    base = make_beamng_env(random_spawn=False, home=home, host="localhost", port=25252,
                           launch=True, headless=True, nogpu=True, steer_rate=0.5)
    env = ResidualHybrid(base, delta=0.12)
    model = SAC.load(CKPT, device="cpu")
    print(f"replay run22 hybrid {CKPT}; start line; capturing the T1-exit spin\n")
    obs, _ = env.reset(options={"spawn_idx": 0})
    rows = []
    prev_heading = None
    for step in range(MAX_STEPS):
        s = _shared["vehicle"].sensors["agent_state"]
        pos, vel, dr = s["pos"], s["vel"], s.get("dir", (1.0, 0.0, 0.0))
        speed = math.hypot(vel[0], vel[1])
        heading = math.atan2(dr[1], dr[0])
        # raw beta (unpinned): angle between velocity and nose
        hn = math.hypot(dr[0], dr[1])
        if speed > 0.3 and hn > 1e-6:
            cb = (vel[0]*dr[0] + vel[1]*dr[1]) / (speed*hn)
            beta_raw = math.degrees(math.acos(max(-1.0, min(1.0, cb))))
        else:
            beta_raw = 0.0
        yawrate = 0.0 if prev_heading is None else abs(math.degrees(
            (heading - prev_heading + math.pi) % (2*math.pi) - math.pi)) / DT
        prev_heading = heading
        arc = float(env.unwrapped._cur_centerline_dist)
        rows.append((step, arc, _dist_to_road(pos[0], pos[1]), beta_raw, yawrate, speed,
                     float(env.unwrapped._last_heading_align), int(env.unwrapped._backward_steps),
                     int(env.unwrapped._steps_since_progress)))
        residual, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(residual)
        if term or trunc:
            rows.append(("TERM", arc, info.get("termination_reason"), 0,0,0,0,0,0))
            break
    # find spin onset: first sustained beta_raw>40 or yawrate>120
    onset = next((i for i,r in enumerate(rows) if isinstance(r[0],int) and r[1]>380 and (r[3]>40 or r[4]>120)), None)
    last = rows[-1]
    print(f"ended: {'TERM='+str(last[2]) if last[0]=='TERM' else 'NO TERM (ran '+str(MAX_STEPS)+' steps)'}\n")
    print(f"{'step':>5}{'arc':>7}{'distRd':>7}{'betaRaw':>8}{'yawrate':>8}{'speed':>6}{'headAl':>7}{'bwd':>4}{'stuck':>6}")
    show = rows[max(0,(onset or 0)-5):] if onset is not None else rows
    for r in show[::2]:
        if r[0]=="TERM":
            print(f"  --> TERMINATED: reason={r[2]} at arc={r[1]:.0f}"); continue
        mark = " <-SPIN" if (r[3]>40 or r[4]>120) else ""
        print(f"{r[0]:>5}{r[1]:>7.0f}{r[2]:>7.1f}{r[3]:>8.1f}{r[4]:>8.0f}{r[5]:>6.1f}{r[6]:>+7.2f}{r[7]:>4}{r[8]:>6}{mark}")
    if onset is not None:
        spin = [r for r in rows if isinstance(r[0],int) and r[0]>=onset]
        dr_ = [r[2] for r in spin]; br = [r[3] for r in spin]; yr = [r[4] for r in spin]
        print(f"\nSPIN STATS (from onset step {rows[onset][0]}, arc {rows[onset][1]:.0f}, {len(spin)} steps):")
        print(f"  dist_road: min={min(dr_):.1f} max={max(dr_):.1f} m  (>8 = off-road; <8 = why off_track silent)")
        print(f"  beta_raw : min={min(br):.0f} max={max(br):.0f} mean={np.mean(br):.0f} deg")
        print(f"  yawrate  : min={min(yr):.0f} max={max(yr):.0f} mean={np.mean(yr):.0f} deg/s")
        print(f"  backward_steps max reached: {max(r[7] for r in spin)} (term needs 40 CONSECUTIVE)")
        print(f"  stuck steps max reached:    {max(r[8] for r in spin)} (term needs >200)")
    env.close()
    try: _shared["bng"].close()
    except Exception: pass


if __name__ == "__main__":
    main()
