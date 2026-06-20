"""run22 STEP 1 probe: additive bounded residual hybrid.
  (A) OFFLINE math: applied = clip(controller + clip(residual, +/-delta), -1, 1).
      policy=0 -> applied == controller (exactly); policy=+/-delta -> applied = controller +/- delta
      (clamped to [-1,1]); out-of-range policy -> still bounded; residual authority == delta.
  (B) BeamNG: drive the FULL wrapper from the start line with a fixed "policy":
        - policy=0  -> must reproduce the controller-alone LAP (the fixed baseline).
        - policy=+delta and -delta (constant) -> stays sane/bounded (sanity, not required to lap).
      Logs max_arc / lap / mean|residual| per case + a map of the baseline lap.
Plain SAC is not needed here; we inject the residual directly through ResidualHybrid."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from envs.beamng_env import make_beamng_env, _shared
from envs.base_controller import BaseController
from envs.residual_hybrid import ResidualHybrid
from data.raceline_builtin import RACELINE

DT = 3.0 / 60.0
MAX_STEPS = 6000
DELTA = 0.12
T1_EXIT = 394.0


def offline_checks():
    print("=== (A) OFFLINE residual math (delta=%.2f) ===" % DELTA)
    ok = True
    ctrl = np.array([0.30, 0.80]);
    for res, name in [(np.array([0.0, 0.0]), "policy=0"),
                      (np.array([DELTA, DELTA]), "policy=+delta"),
                      (np.array([-DELTA, -DELTA]), "policy=-delta"),
                      (np.array([5.0, -5.0]), "policy out-of-range")]:
        clipped = np.clip(res, -DELTA, DELTA)
        applied = np.clip(ctrl + clipped, -1.0, 1.0)
        bounded = np.all(np.abs(applied) <= 1.0)
        auth = np.max(np.abs(clipped))
        tag = ""
        if name == "policy=0":
            eq = np.allclose(applied, ctrl); ok &= eq; tag = f"== controller: {eq}"
        ok &= bounded and auth <= DELTA + 1e-9
        print(f"  {name:20s} clipped_res={np.round(clipped,3)} applied={np.round(applied,3)} "
              f"bounded={bounded} |res|max={auth:.3f} {tag}")
    print(f"  authority bound: max|residual| never exceeds delta={DELTA} -> {'OK' if ok else 'FAIL'}\n")
    return ok


def drive(env, residual_vec, label):
    obs, _ = env.reset(options={"spawn_idx": 0})
    path = []; info = {}; lap_step = None; res_abs = 0.0
    for step in range(MAX_STEPS):
        s = _shared["vehicle"].sensors["agent_state"]; v = math.hypot(s["vel"][0], s["vel"][1])
        path.append((s["pos"][0], s["pos"][1], v))
        obs, _, term, trunc, info = env.step(residual_vec)
        res_abs = info.get("residual_abs", 0.0)
        if info.get("lap_completed") and lap_step is None:
            lap_step = step + 1
        if term or trunc:
            break
    ma = info.get("max_arc", 0.0)
    lap = f"LAP @ {lap_step*DT:.1f}s" if lap_step else "no lap"
    print(f"  {label:16s}: max_arc={ma:.0f}m {'>=T1' if ma>T1_EXIT else '<T1'} "
          f"term={info.get('termination_reason','?')} steps={step+1} {lap} mean|res|={res_abs:.4f}")
    return dict(path=path, max_arc=ma, lap_step=lap_step, term=info.get("termination_reason"))


def main():
    if not offline_checks():
        print("OFFLINE CHECKS FAILED -- stopping before BeamNG."); return
    home = os.environ["BEAMNG_HOME"]
    base = make_beamng_env(random_spawn=False, home=home, host="localhost", port=25252,
                           launch=True, headless=True, nogpu=True, steer_rate=0.5)
    env = ResidualHybrid(base, delta=DELTA)
    print("=== (B) BeamNG: full hybrid wrapper from start line ===")
    zero = drive(env, np.array([0.0, 0.0], np.float32), "policy=0 (BASELINE)")
    drive(env, np.array([DELTA, DELTA], np.float32), "policy=+delta")
    drive(env, np.array([-DELTA, -DELTA], np.float32), "policy=-delta")

    # map the baseline (policy=0) lap
    p = np.array(zero["path"]); cl = np.array(RACELINE)
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.plot(cl[:, 0], cl[:, 1], "-", color="0.7", lw=0.8, label="racing line")
    sc = ax.scatter(p[:, 0], p[:, 1], c=p[:, 2], cmap="viridis", s=6, zorder=3)
    ax.scatter(cl[0, 0], cl[0, 1], c="black", s=60, marker="s", zorder=5, label="start")
    ax.set_aspect("equal"); ax.legend(); fig.colorbar(sc, ax=ax, label="speed (m/s)")
    lap = f"LAP {zero['lap_step']*DT:.1f}s" if zero["lap_step"] else "(no lap)"
    ax.set_title(f"run22 baseline: hybrid @ policy=0 == controller, max_arc={zero['max_arc']:.0f}m {lap}")
    out = "docs/run22_baseline_map.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); print(f"\nwrote {out}")
    env.close()
    try: _shared["bng"].close()
    except Exception: pass


if __name__ == "__main__":
    main()
