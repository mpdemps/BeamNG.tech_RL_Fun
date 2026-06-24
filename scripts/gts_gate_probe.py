"""GTS STEP-1 GATE: the base controller ALONE must lap the GTS grip-aware profile cleanly --
braking from the straight speeds in time and surviving T11 at the reduced grip, no spin. Same
must-lap gate as run23/run25, now on the GTS (road) config + GTS-retuned profile/gains. Drives
BaseController (no policy, no training) from the start line; logs the full lap + a per-corner table
(slope / v_target / entry & apex speed) + a speed map. Port 25252 (sim free). steer_rate=0.5."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from envs.beamng_env import make_beamng_env, _shared, V_TARGET_PROFILE, VEHICLE_PART_CONFIG
from envs.base_controller import BaseController
from envs.speed_profile import compute_speed_profile, V_MAX
from data.raceline_builtin import RACELINE

DT = 3.0 / 60.0
MAX_STEPS = 6000


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost", port=25252,
                          launch=True, headless=True, nogpu=True, steer_rate=0.5)
    u = env.unwrapped
    v_prof, R, cum, tl, kap = (np.array(a) for a in compute_speed_profile(RACELINE))
    z = np.array(RACELINE)[:, 2]
    ctrl = BaseController()
    obs, _ = env.reset(options={"spawn_idx": 0})
    print(f"GATE: controller alone on the GTS profile (V_MAX={V_MAX:.0f}). config={VEHICLE_PART_CONFIG}")
    print(f"      lap=4326m, T1 exit 394m\n")
    samples = []; path = []; info = {}; lap_step = None; peak = 0.0
    for step in range(MAX_STEPS):
        s = _shared["vehicle"].sensors["agent_state"]; vel = s["vel"]; sp = math.hypot(vel[0], vel[1])
        a = np.array(ctrl.action(s["pos"], vel, s.get("dir", (1.0, 0.0, 0.0))), np.float32)
        obs, _, term, trunc, info = env.step(a)
        arc = float(u._cur_centerline_dist); peak = max(peak, sp)
        samples.append((arc, sp)); path.append((s["pos"][0], s["pos"][1], sp))
        if info.get("lap_completed") and lap_step is None:
            lap_step = step + 1
        if term or trunc:
            break
    ma = info.get("max_arc", 0.0); term_r = info.get("termination_reason")
    lap = f"LAP @ {lap_step*DT:.1f}s" if lap_step else "NO LAP"
    print(f"RESULT: {lap}  max_arc={ma:.0f}m  term={term_r}  steps={step+1}  peak_speed={peak:.1f} m/s ({peak*3.6:.0f} kph)\n")

    samples = np.array(samples)
    n = len(RACELINE)
    corner = R < 120.0; cs = []; i = 0
    while i < n:
        if corner[i]:
            j = i
            while j < n and corner[j]: j += 1
            cs.append(i + int(np.argmin(R[i:j]))); i = j
        else: i += 1

    def speed_at(arc_target):           # measured speed nearest an arc (first lap only)
        m = samples[samples[:, 0] <= tl]
        if len(m) == 0: return float("nan")
        return float(m[np.argmin(np.abs(m[:, 0] - arc_target)), 1])

    def grade(ap):
        i0 = int(np.argmin(np.abs(cum - (cum[ap] - 40))))
        return (z[ap] - z[i0]) / max(cum[ap] - cum[i0], 1e-6)

    print(f"{'Tn':>3}{'apex_arc':>9}{'R':>5}{'grade%':>7}{'v_tgt':>6}{'v_entry':>8}{'v_apex':>7}  status")
    for k, ap in enumerate(cs):
        ventry = speed_at(cum[ap] - 30); vapex = speed_at(cum[ap])
        over = vapex - v_prof[ap]
        st = "OK" if over < 3.0 else f"HOT +{over:.1f}"
        tag = "  <-- T11" if 3530 <= cum[ap] <= 3620 else ""
        print(f"{k:>3}{cum[ap]:>9.0f}{R[ap]:>5.0f}{grade(ap)*100:>7.1f}{v_prof[ap]:>6.1f}{ventry:>8.1f}{vapex:>7.1f}  {st}{tag}")

    # map: driven path colored by speed
    p = np.array(path); cl = np.array(RACELINE)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(cl[:, 0], cl[:, 1], "-", color="0.8", lw=0.8, label="racing line")
    sc = ax.scatter(p[:, 0], p[:, 1], c=p[:, 2], cmap="viridis", s=6, zorder=3)
    ax.scatter(cl[0, 0], cl[0, 1], c="black", s=60, marker="s", zorder=5, label="start")
    for k, ap in enumerate(cs):
        if 3530 <= cum[ap] <= 3620:
            ax.scatter(cl[ap, 0], cl[ap, 1], c="red", s=80, marker="*", zorder=6, label="T11")
    ax.set_aspect("equal"); ax.legend(); fig.colorbar(sc, ax=ax, label="speed (m/s)")
    ax.set_title(f"GTS gate: controller on GTS profile -> {lap}, max_arc={ma:.0f}m, peak {peak*3.6:.0f} kph")
    fig.savefig("docs/gts_gate_map.png", dpi=110); print("\nwrote docs/gts_gate_map.png")
    env.close()
    try: _shared["bng"].close()
    except Exception: pass


if __name__ == "__main__":
    main()
