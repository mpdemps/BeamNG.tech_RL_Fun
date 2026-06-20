"""run22 STEP 1: additive bounded residual hybrid (controller-led, NOT run21's convex-blend floor).

The policy outputs a RESIDUAL; the env applies the base controller at FULL plus the clamped residual:

    applied = clip( controller(state) + clip(policy_residual, -delta, +delta), -1, 1 )

This is a gym env WRAPPER (not a sampler hook like run21's BlendSAC) on purpose: evaluation uses
policy.predict, which never touches the training sampler, so the only way EVAL can run the FULL
hybrid (controller + residual, per the run22 plan) is to put the residual in the environment.
Both train-time collection and predict-time eval then produce identical hybrid actions, and SAC's
buffer stores the policy's residual (what it controls) -- the standard residual-RL formulation.

delta is small (the controller already laps; the residual only trims the line/speed). With
policy=0 the applied action IS the controller, so the hybrid still laps (the fixed baseline).
The wrapper drops mean |applied residual| into info ('residual_abs') so we can watch how hard the
RL pushes (and whether it pins delta = wants more authority).
"""
import gymnasium
import numpy as np
from envs.beamng_env import _shared
from envs.base_controller import BaseController


class ResidualHybrid(gymnasium.Wrapper):
    def __init__(self, env, delta=0.12, controller=None):
        super().__init__(env)
        self.delta = float(delta)
        self.controller = controller if controller is not None else BaseController()
        self._res_sum = 0.0
        self._res_n = 0

    def reset(self, **kwargs):
        self.controller.reset()
        self._res_sum = 0.0
        self._res_n = 0
        return self.env.reset(**kwargs)

    def _controller_action(self):
        s = _shared["vehicle"].sensors["agent_state"]
        steer, thr = self.controller.action(s["pos"], s["vel"], s.get("dir", (1.0, 0.0, 0.0)))
        return np.array([steer, thr], dtype=np.float32)

    def step(self, residual):
        residual = np.asarray(residual, dtype=np.float32).reshape(-1)
        clipped = np.clip(residual, -self.delta, self.delta)         # bounded authority
        ctrl = self._controller_action()
        applied = np.clip(ctrl + clipped, -1.0, 1.0)                 # controller at FULL + residual
        obs, reward, terminated, truncated, info = self.env.step(applied)
        # track mean |applied residual| over the episode (how hard the RL pushes)
        self._res_sum += float(np.mean(np.abs(clipped)))
        self._res_n += 1
        info["residual_abs"] = self._res_sum / max(self._res_n, 1)
        return obs, reward, terminated, truncated, info
