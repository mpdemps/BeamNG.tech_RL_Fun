"""run24 STEP 1 probe (offline, no BeamNG, launches nothing): verify the ASYMMETRIC per-channel
residual bound and the per-channel residual_abs split.
  - bounds: steer in [-0.12,+0.12]; throttle in [-0.12,+0.05] (positive=gas capped at +0.05 =
    the over-throttle cut; negative=brake/lift keeps full -0.12).
  - policy=0 -> applied == controller exactly (still laps; baseline unchanged).
  - the THROTTLE CUT: a big +throttle residual clamps to +0.05, but a big -throttle clamps to
    -0.12 (asymmetry is the whole point).
  - residual_abs split: info residual_abs_steer / residual_abs_throttle track each channel.
  - applied always bounded to [-1,1]."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import gymnasium
from gymnasium import spaces
import envs.beamng_env as E
from envs.residual_hybrid import ResidualHybrid
from data.raceline_builtin import RACELINE

DELTA, THROTTLE_UP = 0.12, 0.05


class Stub(gymnasium.Env):
    observation_space = spaces.Box(-1, 1, (19,), np.float32)
    action_space = spaces.Box(-1, 1, (2,), np.float32)
    def reset(self, **k): return np.zeros(19, np.float32), {}
    def step(self, a):
        assert a.shape == (2,) and np.all(np.abs(a) <= 1.0 + 1e-6), f"applied out of [-1,1]: {a}"
        return np.zeros(19, np.float32), 0.0, False, False, {}


def main():
    # mock the shared vehicle so the controller reads a fixed on-line pose
    class FS(dict):
        def poll(self): pass
    v = type("V", (), {})(); v.sensors = FS()
    t = np.array(RACELINE[60][:2]) - np.array(RACELINE[58][:2]); t = t / np.linalg.norm(t)
    v.sensors["agent_state"] = {"pos": RACELINE[59], "vel": (t[0]*22, t[1]*22, 0.0), "dir": (t[0], t[1], 0.0)}
    E._shared["vehicle"] = v; E._shared["bng"] = None; E._shared["initialized"] = True

    env = ResidualHybrid(Stub(), delta=DELTA, throttle_up=THROTTLE_UP)
    ok = True
    print(f"=== run24 asymmetric residual bound: steer +/-{DELTA}, throttle [-{DELTA}, +{THROTTLE_UP}] ===")
    print(f"  wrapper.low={env.low}  wrapper.high={env.high}")
    ok &= np.allclose(env.low, [-DELTA, -DELTA]) and np.allclose(env.high, [DELTA, THROTTLE_UP])

    ctrl = env._controller_action()
    print(f"\n  controller action at on-line pose: steer={ctrl[0]:+.3f} thr={ctrl[1]:+.3f}")

    cases = [
        ("policy=0",            [0.0, 0.0],   [0.0, 0.0]),
        ("steer +big",          [1.0, 0.0],   [DELTA, 0.0]),
        ("steer -big",          [-1.0, 0.0],  [-DELTA, 0.0]),
        ("THROTTLE +big (CUT)", [0.0, 1.0],   [0.0, THROTTLE_UP]),
        ("throttle -big (lift)",[0.0, -1.0],  [0.0, -DELTA]),
        ("throttle +0.20 (CUT)",[0.0, 0.20],  [0.0, THROTTLE_UP]),
    ]
    print(f"\n  {'case':22s}{'residual':>16}{'-> clipped':>16}{'expect':>16}{'  applied':>16} ok")
    for name, res, exp in cases:
        env.reset()
        env.step(np.array(res, np.float32))
        clip = env.last_residual; appl = env.last_applied
        good = np.allclose(clip, exp, atol=1e-6) and np.all(np.abs(appl) <= 1.0 + 1e-6)
        if name == "policy=0":
            good &= np.allclose(appl, ctrl, atol=1e-6)   # applied == controller
        ok &= good
        print(f"  {name:22s}{str(np.round(res,2)):>16}{str(np.round(clip,3)):>16}"
              f"{str(np.round(exp,3)):>16}{str(np.round(appl,3)):>16}  {'OK' if good else 'FAIL'}")

    # per-channel residual_abs split: steer big, throttle small over a few steps
    env.reset()
    for _ in range(4):
        _, _, _, _, info = env.step(np.array([0.12, -0.03], np.float32))
    split_ok = (abs(info["residual_abs_steer"] - 0.12) < 1e-6 and
                abs(info["residual_abs_throttle"] - 0.03) < 1e-6)
    ok &= split_ok
    print(f"\n  residual_abs split: steer={info['residual_abs_steer']:.3f} (exp 0.120) "
          f"throttle={info['residual_abs_throttle']:.3f} (exp 0.030) combined={info['residual_abs']:.3f} "
          f"-> {'OK' if split_ok else 'FAIL'}")

    print(f"\n=== {'ALL PASS' if ok else 'FAILURES ABOVE'} ===")
    print("note: policy=0 path is identical to run22's verified controller lap (controller untouched); "
          "the only change is the asymmetric clip, fully covered here offline.")


if __name__ == "__main__":
    main()
