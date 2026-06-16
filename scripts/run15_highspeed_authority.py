"""run15 measure-first: the REAL high-speed steering authority (action -> radius vs
speed), to set the cap's high-speed floor. The corner table shows T2 (R108, ~31 m/s),
T7 (R133, ~34 m/s), T9 (R139, ~35 m/s) are taken ABOVE the 27 m/s knee where the cap
bites -- so the cap must still let them turn. The sim speed-limits steering (~9x off the
0.5-rad assumption), so measure don't assume.

Method (safe: throttle-OFF steering held even at +-0.9 in the prior probe): for each
fixed steer level, floor straight to ~37 m/s, then THROTTLE 0 and hold the steer, coasting
down while logging per step: speed, yaw_rate (from the dir-vector delta / 50ms), the
achieved radius R = v/yaw_rate, effective road angle = yaw_rate*L/v, lateral g, beta, head.
Binning by speed gives action->R(v); then for each fast corner we read the MIN action that
achieves its radius at its entry speed -> the cap's high-speed floor must exceed that, and
full-lock (the spin slam) must exceed the cap.  Port 25252 (run14 stopped)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beamng_env import make_beamng_env, _shared

PORT = 25252
L = 2.6
DT = 3.0 / 60.0
STEER_LEVELS = [0.15, 0.25, 0.35, 0.50, 0.70, 1.00]
BUILD_V = 37.0
# corners taken above the 27 m/s knee (radius m, entry speed m/s)
FAST_CORNERS = [("T3", 82, 27), ("T2", 108, 31), ("T7", 133, 34), ("T9", 139, 35)]


def st():
    s = _shared["vehicle"].sensors["agent_state"]
    v = math.hypot(s["vel"][0], s["vel"][1])
    yaw = math.atan2(s.get("dir", (1, 0, 0))[1], s.get("dir", (1, 0, 0))[0])
    return v, yaw


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost",
                          port=PORT, launch=True, headless=True, nogpu=True)
    print(f"port {PORT}; all gates OFF (raw steering); L={L}m\n")

    # samples[steer] = list of (v, yaw_rate, R, eff_ang, lat_g)
    samples = {lvl: [] for lvl in STEER_LEVELS}
    for lvl in STEER_LEVELS:
        env.reset()
        for _ in range(80):
            env.step([0.0, 1.0]); _shared["vehicle"].sensors.poll()
            if st()[0] >= BUILD_V:
                break
        _, yaw_prev = st()
        for k in range(70):
            env.step([lvl, 0.0]); _shared["vehicle"].sensors.poll()
            v, yaw = st()
            dyaw = (yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi
            yaw_prev = yaw
            wr = abs(dyaw) / DT
            if v < 1.0 or k < 3:        # skip the initial yaw transient
                continue
            R = v / wr if wr > 1e-3 else 9999
            samples[lvl].append((v, wr, R, wr * L / v, v * wr / 9.81))
            if v < 16 or env._last_heading_align < 0.0:
                break

    def at_speed(lvl, vt, tol=2.5):
        cand = [s for s in samples[lvl] if abs(s[0] - vt) <= tol]
        if not cand:
            return None
        cand.sort(key=lambda s: abs(s[0] - vt))
        # median R over the closest few (steady-state, reject transient outliers)
        near = cand[:5]
        Rs = sorted(s[2] for s in near)
        return Rs[len(Rs) // 2]

    print("=== achieved RADIUS (m) per (steer, speed) -- smaller R = tighter turn ===")
    print(f"{'steer':>6}" + "".join(f"{v:>8}m/s" for v in (35, 34, 31, 27, 25)))
    for lvl in STEER_LEVELS:
        row = f"{lvl:>6.2f}"
        for v in (35, 34, 31, 27, 25):
            R = at_speed(lvl, v)
            row += f"{('%.0f' % R) if R else '   -':>11}"
        print(row)

    print("\n=== per fast corner: MIN action achieving its radius at entry speed ===")
    print(f"{'corner':>6}{'R':>5}{'v':>5}  needed_action   cap@v(0.35curve)  verdict")
    for name, R_c, v_c in FAST_CORNERS:
        need = None
        for lvl in STEER_LEVELS:                       # smallest steer achieving R<=R_c
            R = at_speed(lvl, v_c)
            if R is not None and R <= R_c:
                need = lvl
                break
        # cap value at v_c on the 27->31 / 0.35 curve
        if v_c <= 27:
            cap = 1.0
        elif v_c >= 31:
            cap = 0.35
        else:
            cap = 1.0 - (v_c - 27) / 4 * 0.65
        verdict = ("PASS" if (need is not None and need <= cap + 1e-9)
                   else ("STARVED" if need is not None else "need>fulllock?"))
        print(f"{name:>6}{R_c:>5}{v_c:>5}  {str(need):>12}   {cap:>14.2f}   {verdict}")

    # full-lock radius at the spin band (must be tighter than any corner -> cap blocks it)
    print("\n=== full-lock (1.0) achieved radius in the 33-36 m/s spin band ===")
    for v in (35, 34):
        R = at_speed(1.0, v)
        print(f"  v={v}: full-lock R={R:.0f}m" if R else f"  v={v}: (no sample)")
    env.close()
    try: _shared["bng"].close()
    except Exception: pass


if __name__ == "__main__":
    main()
