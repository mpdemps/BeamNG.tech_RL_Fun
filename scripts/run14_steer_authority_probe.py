"""run14 measure-first: the REAL steering authority vs speed, and the reversal limit.

The run13 trace shows (a) action units do NOT map to ~0.5 rad road angle -- the car held
0.97 steer at 27 m/s while tracking (would be 14g at 0.5 rad), so BeamNG speed-limits the
steering input; and (b) the spins are transient REVERSALS, not sustained over-grip. Both
the corner-need table and the proposed cap curve assumed the wrong ratio. So measure the
real mapping directly instead of guessing.

Two measurements, port 25253 (run13 is stopped; nothing else running):

A. AUTHORITY SWEEP: hold a fixed steering value, floor throttle, and as the car
   accelerates record (speed, applied_steer, yaw_rate, lateral_g, effective_road_angle,
   slip, heading). yaw_rate from the heading (dir) delta per 50ms step;
   eff_angle = yaw_rate * L / v; lateral_g = v * yaw_rate / 9.81. Repeat per steer level.
   -> gives effective road angle vs speed per action level (is it speed-limited?), and
      the lateral-g grip ceiling. steer_rate=0 (rate limit OFF) so we see RAW authority.

B. REVERSAL TEST: build to ~30 m/s straight, then slam steering +X -> -X and watch slip
   and heading, for X in {0.35 (the proposed cap), 0.9 (the observed spin amplitude)}.
   -> does a capped +-0.35 reversal break the rear at speed, or only the +-0.9 slam?

Measure only. No policy, no centerline tracker (that one failed) -- direct open-loop
inputs with the env's own physics. Does not build/modify the env."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from envs.beamng_env import make_beamng_env, _shared

PORT = 25253
L = 2.6                      # Scintilla wheelbase (m), for eff road angle = yaw_rate*L/v
STEER_LEVELS = [0.15, 0.30, 0.50, 0.70, 1.00]
DT = 3.0 / 60.0             # 3 physics steps at 60 Hz


def state():
    s = _shared["vehicle"].sensors["agent_state"]
    vel = s["vel"]; fdir = s.get("dir", (1.0, 0.0, 0.0))
    v = math.hypot(vel[0], vel[1])
    yaw = math.atan2(fdir[1], fdir[0])
    return v, yaw


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost",
                          port=PORT, launch=True, headless=True, nogpu=True,
                          steer_rate=0.0)   # rate limit OFF: measure raw authority
    print(f"port {PORT}; steer_rate=0 (raw); wheelbase L={L}m\n")

    # ---------- A. authority sweep ----------
    print("=== A. AUTHORITY SWEEP: effective road angle & lateral-g vs speed, per action ===")
    print(f"{'steer':>6} {'speed':>6} {'yaw_rate':>9} {'eff_ang':>8} {'lat_g':>6} {'slip':>6} {'head':>6}")
    sweep = {}
    for lvl in STEER_LEVELS:
        env.reset()
        _shared["vehicle"].sensors.poll()
        v_prev, yaw_prev = state()
        rows = []
        for step in range(90):
            env.step([lvl, 1.0])          # fixed steer, floor throttle
            _shared["vehicle"].sensors.poll()
            v, yaw = state()
            dyaw = (yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi
            yaw_rate = dyaw / DT
            lat_g = abs(v * yaw_rate) / 9.81
            eff_ang = abs(yaw_rate) * L / v if v > 1.0 else 0.0
            slip = env._last_slip
            head = env._last_heading_align
            rows.append((v, yaw_rate, eff_ang, lat_g, slip, head))
            v_prev, yaw_prev = v, yaw
            if head < 0.0:               # spun; stop this level
                break
        sweep[lvl] = rows
        # sample at the speed bands of interest (nearest row >= target)
        for target in (15, 20, 25, 30, 33):
            cand = [r for r in rows if r[0] >= target - 1.0]
            if not cand:
                continue
            r = min(cand, key=lambda r: abs(r[0] - target))
            if abs(r[0] - target) <= 3.0:
                print(f"{lvl:>6.2f} {r[0]:>6.1f} {r[1]:>9.3f} {r[2]:>8.4f} {r[3]:>6.2f} {r[4]:>6.1f} {r[5]:>6.2f}")
        print()

    # ---------- B. reversal test ----------
    print("=== B. REVERSAL TEST at speed: does a capped +-0.35 reversal break the rear? ===")
    for amp in (0.35, 0.90):
        env.reset()
        # build speed straight
        for _ in range(60):
            env.step([0.0, 1.0])
            _shared["vehicle"].sensors.poll()
            v, _ = state()
            if v >= 30:
                break
        v0, _ = state()
        # slam + amp, hold, then - amp, hold; log slip/heading
        log = []
        for phase, st in ((f"+{amp}", amp), (f"-{amp}", -amp)):
            for k in range(8):
                env.step([st, 0.0])      # throttle 0 during the flick: isolate steering
                _shared["vehicle"].sensors.poll()
                v, _ = state()
                log.append((phase, v, env._last_slip, env._last_heading_align))
        worst_slip = max(abs(r[2]) for r in log)
        min_head = min(r[3] for r in log)
        spun = min_head < 0.5
        print(f"  reversal +-{amp:.2f} from {v0:.0f} m/s: peak|slip|={worst_slip:.1f}  "
              f"min_head={min_head:+.2f}  -> {'SPUN/LOST' if spun else 'held'}")
        for r in log:
            print(f"      {r[0]:>6} v={r[1]:4.1f} slip={r[2]:6.1f} head={r[3]:+.2f}")

    env.close()
    try: _shared["bng"].close()
    except Exception: pass


if __name__ == "__main__":
    main()
