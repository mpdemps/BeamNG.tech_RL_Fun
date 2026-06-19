"""run21 STEP 1 GATE: drive the BASE CONTROLLER ALONE (no policy) from the start line and check
it gets through T1 cleanly and laps. Tests a list of gain configs in ONE BeamNG session (reset
between), reports max_arc / lap / termination per config, and for the best config dumps the T1
trace (steer, speed vs v_target, beta) and a lap map. The controller drives THROUGH the env so
the real steer-rate limit (0.5) applies -- the same path the residual blend will take later.

Iterate gains here until it LAPS; that lap is the foundation the fade schedule stands on.
Port 25252. Env steer_rate=0.5, random_spawn=False (start line)."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from envs.beamng_env import make_beamng_env, _shared, V_TARGET_PROFILE
from envs.base_controller import BaseController
from data.raceline_builtin import RACELINE

DT = 3.0 / 60.0
MAX_STEPS = 6000
T1_EXIT = 394.0

# Diagnosis: arc 0-250 is near-STRAIGHT (R>320m); prior k_steer 22-90 saturated/spun on it
# (over-gain). Geometric estimate -> k_steer~5. Re-tune into the correct range with a stable
# speed-scaled lookahead (l_min=8, k_v=0.8, l_max=30). Trace the approach through T1 (arc 317).
# T1 SOLVED with k_steer~6 (clean brake-early/apex/gas-out, no spin). Dies ~1411m consistently
# across k_steer 6-10 -> a specific corner, not raw gain. Trace the death zone; try more braking
# (kp) + shorter l_max (tight corners aimed past the apex under-steer with a 30m lookahead).
# Deaths at 1411/1856 were SPURIOUS off-track (sparse-vertex metric); fixed _is_off_track to
# point-to-segment. Re-run the good-steering configs for a FULL LAP. T1-clean was k_steer~6.
CONFIGS = [
    dict(steer_sign=-1.0, k_steer=6.0, kp_speed=0.25, l_min=8.0, k_v=0.8, l_max=30.0, speed_factor=1.00),
    dict(steer_sign=-1.0, k_steer=7.0, kp_speed=0.35, l_min=6.0, k_v=0.7, l_max=22.0, speed_factor=1.00),
    dict(steer_sign=-1.0, k_steer=7.0, kp_speed=0.35, l_min=6.0, k_v=0.7, l_max=22.0, speed_factor=0.95),
    dict(steer_sign=-1.0, k_steer=6.0, kp_speed=0.30, l_min=7.0, k_v=0.7, l_max=24.0, speed_factor=0.95),
]
TRACE_LO, TRACE_HI = 280.0, 470.0


def drive(env, cfg):
    ctrl = BaseController(**cfg)
    obs, _ = env.reset(options={"spawn_idx": 0})
    path, trace = [], []
    info = {}
    lap_step = None
    for step in range(MAX_STEPS):
        s = _shared["vehicle"].sensors["agent_state"]
        pos, vel, dir_ = s["pos"], s["vel"], s.get("dir", (1.0, 0.0, 0.0))
        steer, thr = ctrl.action(pos, vel, dir_)
        obs, _, term, trunc, info = env.step(np.array([steer, thr], np.float32))
        v = math.hypot(vel[0], vel[1]); arc = float(env._cur_centerline_dist)
        path.append((pos[0], pos[1], v))
        if TRACE_LO <= arc <= TRACE_HI:
            trace.append((arc, v, float(V_TARGET_PROFILE[env._progress_idx]), steer, float(env._last_beta)))
        if info.get("lap_completed") and lap_step is None:
            lap_step = step + 1
        if term or trunc:
            break
    return dict(cfg=cfg, max_arc=info.get("max_arc", 0.0), term=info.get("termination_reason", "?"),
                steps=step + 1, lap_step=lap_step, path=path, trace=trace)


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost", port=25252,
                          launch=True, headless=True, nogpu=True, steer_rate=0.5)
    print(f"BASE CONTROLLER gate; start line; T1 exit {T1_EXIT}m; {len(CONFIGS)} configs\n")
    results = []
    for cfg in CONFIGS:
        r = drive(env, cfg)
        results.append(r)
        lap = f"LAP @ {r['lap_step']*DT:.1f}s" if r["lap_step"] else "no lap"
        print(f"  sign={cfg['steer_sign']:+.0f} k_steer={cfg['k_steer']:.0f} kp={cfg['kp_speed']:.2f}: "
              f"max_arc={r['max_arc']:.0f}m {'>=T1' if r['max_arc']>T1_EXIT else '<T1'} "
              f"term={r['term']} steps={r['steps']} {lap}")

    best = max(results, key=lambda r: (r["lap_step"] is not None, r["max_arc"]))
    print(f"\n=== BEST: {best['cfg']} -> max_arc={best['max_arc']:.0f}m "
          f"{'LAPPED '+format(best['lap_step']*DT,'.1f')+'s' if best['lap_step'] else 'no lap'} ===")
    if best["trace"]:
        print(" T1 trace:  arc    v   vtgt   steer   beta")
        for a, v, vt, st, b in best["trace"][::3]:
            print(f"          {a:5.0f} {v:5.1f} {vt:5.1f} {st:+6.2f} {b:5.1f}")

    # map of the best run, colored by speed
    p = np.array(best["path"])
    cl = np.array(RACELINE)
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.plot(cl[:, 0], cl[:, 1], "-", color="0.7", lw=0.8, label="racing line")
    sc = ax.scatter(p[:, 0], p[:, 1], c=p[:, 2], cmap="viridis", s=6, zorder=3)
    ax.scatter(cl[0, 0], cl[0, 1], c="black", s=60, marker="s", zorder=5, label="start")
    ax.set_aspect("equal"); ax.legend(); fig.colorbar(sc, ax=ax, label="speed (m/s)")
    ax.set_title(f"base controller alone: max_arc={best['max_arc']:.0f}m "
                 f"{'LAP '+format(best['lap_step']*DT,'.1f')+'s' if best['lap_step'] else '(no lap)'}")
    out = "docs/run21_controller_map.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); print(f"\nwrote {out}")
    env.close()
    try: _shared["bng"].close()
    except Exception: pass


if __name__ == "__main__":
    main()
