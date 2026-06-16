"""run15 skidpad: controlled-speed steering authority. The coast-down probe failed --
the opening straight only reaches ~32 m/s and the car leaves it within a few steps once
steering. Instead: reset (faces down the straight), set_velocity to a target, let it
settle, then hold a fixed steer for a few steps and measure steady yaw_rate -> achieved
radius R = v/yaw_rate and effective road angle. A few steps at 35 m/s cover <10 m, well
inside the 294 m straight. Gives action->R(v) to set the cap's high-speed floor so the
fast corners T2(R108,~31) T7(R133,~34) T9(R139,~35) still pass while the 0.9-1.0 slam is
blocked. All gates OFF (raw). Port 25252."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beamng_env import make_beamng_env, _shared

PORT = 25252
L = 2.6
DT = 3.0 / 60.0
SPEEDS = [25.0, 30.0, 34.0]
STEERS = [0.25, 0.35, 0.50, 0.70, 1.00]
SETTLE = 2
MEASURE = 5


def vyaw():
    s = _shared["vehicle"].sensors["agent_state"]
    return (math.hypot(s["vel"][0], s["vel"][1]),
            math.atan2(s.get("dir", (1, 0, 0))[1], s.get("dir", (1, 0, 0))[0]))


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost",
                          port=PORT, launch=True, headless=True, nogpu=True)
    print(f"port {PORT}; skidpad, raw steering; L={L}m\n")

    grid = {v: {} for v in SPEEDS}
    for vt in SPEEDS:
        for steer in STEERS:
            env.reset()                       # connects on first call; vehicle in _shared
            try:
                _shared["vehicle"].set_velocity(vt)
            except Exception:
                pass
            for _ in range(SETTLE):
                env.step([0.0, 0.0])
            _, yaw_prev = vyaw()
            wrs, vs = [], []
            for _ in range(MEASURE):
                env.step([steer, 0.0])
                v, yaw = vyaw()
                dyaw = (yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi
                yaw_prev = yaw
                wrs.append(abs(dyaw) / DT); vs.append(v)
            wrs.sort()
            wr = wrs[len(wrs) // 2]
            vm = sum(vs) / len(vs)
            R = vm / wr if wr > 1e-3 else 9999
            grid[vt][steer] = (R, vm, wr * L / vm, vm * wr / 9.81)

    print("=== achieved RADIUS R(m) per (speed, steer) [v_actual, eff_angle rad, lat_g] ===")
    for vt in SPEEDS:
        print(f"\n target v~{vt:.0f} m/s:")
        print(f"   {'steer':>6}{'R(m)':>8}{'v':>7}{'effang':>9}{'lat_g':>7}")
        for steer in STEERS:
            R, vm, ea, g = grid[vt][steer]
            print(f"   {steer:>6.2f}{R:>8.0f}{vm:>7.1f}{ea:>9.4f}{g:>7.2f}")

    print("\n=== fast-corner authority need vs the 0.35 curve ===")
    print(f"{'corner':>6}{'R':>5}{'v':>5}{'need_action':>12}{'cap@v':>8}  verdict")
    CORNERS = [("T8", 68, 25), ("T3", 82, 25), ("T2", 108, 30), ("T7", 133, 34), ("T9", 139, 34)]
    for name, Rc, vt in CORNERS:
        g = grid.get(vt, {})
        need = next((s for s in STEERS if g.get(s, (9999,))[0] <= Rc), None)
        cap = 1.0 if vt <= 27 else (0.35 if vt >= 31 else 1.0 - (vt - 27) / 4 * 0.65)
        verdict = ("PASS" if need is not None and need <= cap + 1e-9 else
                   "STARVED" if need is not None else "need>fulllock")
        print(f"{name:>6}{Rc:>5}{vt:>5}{str(need):>12}{cap:>8.2f}  {verdict}")
    env.close()
    try: _shared["bng"].close()
    except Exception: pass


if __name__ == "__main__":
    main()
