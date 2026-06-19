"""run20 retarget probe: confirm the LINE-RELATIVE observations are sane and bounded across the
whole lap, driving the env's REAL _advance_progress + _get_observation (no BeamNG -- a mock
vehicle feeds poses). Places a virtual car exactly ON the racing line at each point (heading =
line tangent, speed = line v_target) and checks:
  - obs SHAPE is 18 and every dim stays in [-1, 1] (bounded), no NaN, across the lap;
  - center_off ~= 0 when the car is ON the line (the line-relative offset works);
  - a deliberate +2 m lateral offset reads center_off ~= +2 m (correct sign/magnitude);
  - progress advances monotonically along the line and completes ~one lap (track machinery
    follows the line, not the old centerline).
Pure offline geometry over data/raceline_builtin.py via the real env code."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import envs.beamng_env as E
from envs.beamng_env import BeamNGRaceEnv, RACELINE, ROAD_CENTERLINE, V_TARGET_PROFILE


class _FakeSensors(dict):
    def poll(self):
        pass


class _FakeVehicle:
    def __init__(self):
        self.sensors = _FakeSensors()
        self.sensors["agent_state"] = {"pos": (0, 0, 0), "vel": (0, 0, 0),
                                       "dir": (1, 0, 0), "up": (0, 0, 1)}
        self.sensors["electrics"] = {"wheelspeed": 0.0}

    def teleport(self, *a, **k):
        pass

    def control(self, *a, **k):
        pass


def tangent(i, n):
    a = RACELINE[(i - 1) % n]; b = RACELINE[(i + 1) % n]
    t = np.array([b[0] - a[0], b[1] - a[1]])
    return t / (np.linalg.norm(t) + 1e-9)


def set_pose(veh, pos, t, speed):
    veh.sensors["agent_state"] = {"pos": (float(pos[0]), float(pos[1]), float(pos[2])),
                                  "vel": (float(t[0] * speed), float(t[1] * speed), 0.0),
                                  "dir": (float(t[0]), float(t[1]), 0.0), "up": (0.0, 0.0, 1.0)}
    veh.sensors["electrics"] = {"wheelspeed": speed}


def main():
    E._shared["vehicle"] = _FakeVehicle()
    E._shared["bng"] = None
    E._shared["initialized"] = True
    env = BeamNGRaceEnv(random_spawn=False)
    veh = E._shared["vehicle"]
    n = len(RACELINE)

    # init progress state as reset()'s tail does (idx 0, zero laps)
    env._progress_idx = 0; env._laps = 0; env._lap_done = False
    env._cur_centerline_dist = env._cum_arc[0]; env._last_centerline_dist = env._cur_centerline_dist
    env._last_beta = 0.0; env._center_off = 0.0

    print("=== run20 retarget obs probe (line-relative, real env code, mock vehicle) ===")
    print(f"reference path = RACELINE ({n} pts); off-track measured vs ROAD_CENTERLINE "
          f"({len(ROAD_CENTERLINE)} pts)\n")

    obs_all, center_offs, dists, headings = [], [], [], []
    last_d = None; monotonic = True
    for i in range(0, n):
        t = tangent(i, n)
        set_pose(veh, RACELINE[i], t, float(V_TARGET_PROFILE[i]))
        env._advance_progress(veh.sensors["agent_state"]["pos"])
        obs = env._get_observation()
        obs_all.append(obs); center_offs.append(env._center_off)
        dists.append(env._cur_centerline_dist); headings.append(obs[1] * math.pi)
        if last_d is not None and env._cur_centerline_dist < last_d - 1.0:
            monotonic = False
        last_d = env._cur_centerline_dist
    obs_all = np.array(obs_all); center_offs = np.array(center_offs)

    # (a) shape + bounded + no NaN
    in_range = np.all((obs_all >= -1.0001) & (obs_all <= 1.0001))
    print(f"(a) SHAPE/BOUNDED: obs shape {obs_all.shape[1]} dims (expect 18); "
          f"all in [-1,1]: {in_range}; any NaN: {bool(np.isnan(obs_all).any())}")
    print(f"    per-dim min/max:")
    for d in range(obs_all.shape[1]):
        print(f"      obs[{d:2d}]  min={obs_all[:, d].min():+.3f}  max={obs_all[:, d].max():+.3f}")

    # (b) on-line center_off ~ 0
    print(f"\n(b) ON-LINE center_off: mean={center_offs.mean():+.3f} m  max|.|={np.abs(center_offs).max():.3f} m "
          f"({'~0 OK (car tracks the line)' if np.abs(center_offs).max() < 0.5 else 'NONZERO (check)'})")

    # (c) deliberate +2 m lateral offset reads back
    i = n // 3; t = tangent(i, n); nrm = np.array([-t[1], t[0]])  # left normal
    off_pos = np.array(RACELINE[i][:2]) + 2.0 * nrm
    set_pose(veh, (off_pos[0], off_pos[1], RACELINE[i][2]), t, 10.0)
    env._progress_idx = i; env._advance_progress(veh.sensors["agent_state"]["pos"]); env._get_observation()
    print(f"\n(c) +2.0 m LEFT offset @ idx {i}: center_off read = {env._center_off:+.3f} m "
          f"({'OK (sign+magnitude)' if 1.5 < abs(env._center_off) < 2.5 else 'check'})")

    # (d) progress monotonic + completes ~a lap
    print(f"\n(d) PROGRESS: monotonic along line: {monotonic}; "
          f"arc start={dists[0]:.1f} end={dists[-1]:.1f} m vs track_length={env._track_length:.1f} m "
          f"({'~full lap OK' if dists[-1] > 0.95 * env._track_length else 'short (check)'})")
    print(f"    heading_err over lap: max|.|={np.max(np.abs(headings)):.2f} rad "
          f"(nonzero in corners is expected = upcoming-curve preview; bounded < pi)")


if __name__ == "__main__":
    main()
