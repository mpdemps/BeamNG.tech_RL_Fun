"""run21 STEP 2: guided residual RL with controller fade-out.

BlendSAC is plain SAC except that during DATA COLLECTION it applies a convex blend of the base
controller and the policy:

    applied = beta * base_controller(state) + (1 - beta) * policy(state)

with beta scheduled over training: beta=1 warmup (buffer fills with clean controller laps) ->
linear anneal to 0 over the middle -> beta=0 held for the final stretch (policy stands alone).

The blend lives in _sample_action (the rollout-collection hook), so:
  * the buffer stores the APPLIED (blended) action -> SAC's critic learns Q for the action
    actually taken (a valid Bellman backup); the actor update is unaffected (it samples fresh).
  * EVALUATION never calls _sample_action (EvalCallback uses policy.predict), so eval is ALWAYS
    at beta=0 -- the cards measure the STANDALONE policy. eval/max_arc clearing 394 m is the win.

The controller / schedule are set as attributes AFTER construction (so SAC.load works unchanged);
with controller unset BlendSAC behaves as plain SAC (beta=0). Reward and obs are untouched.
"""
import numpy as np
from stable_baselines3 import SAC
from envs.beamng_env import _shared


def beta_at(t, warmup, anneal_end):
    """beta=1 for t<warmup, linear 1->0 over [warmup, anneal_end), 0 after."""
    if anneal_end <= warmup:
        return 0.0
    if t < warmup:
        return 1.0
    if t >= anneal_end:
        return 0.0
    return 1.0 - (t - warmup) / (anneal_end - warmup)


def blend_action(beta, ctrl, policy_action):
    """Convex blend, clipped to the [-1,1] action box. beta=1 -> pure controller, 0 -> pure policy."""
    return np.clip(beta * np.asarray(ctrl) + (1.0 - beta) * np.asarray(policy_action), -1.0, 1.0)


class BlendSAC(SAC):
    # set as instance attributes after construction (train script); defaults = plain SAC.
    controller = None
    beta_warmup = 0
    beta_anneal_end = 0
    beta_offset = 0          # global steps already done before this segment (warm-restart safe)

    def _current_beta(self):
        if self.controller is None:
            return 0.0
        # num_timesteps resets to 0 each wrapper segment (reset_num_timesteps=True), so add the
        # offset of clean steps already done -> beta tracks GLOBAL progress across freeze-restarts.
        return beta_at(self.num_timesteps + self.beta_offset, self.beta_warmup, self.beta_anneal_end)

    def _controller_action(self):
        s = _shared["vehicle"].sensors["agent_state"]
        steer, thr = self.controller.action(s["pos"], s["vel"], s.get("dir", (1.0, 0.0, 0.0)))
        return np.array([steer, thr], dtype=np.float32)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        action, buffer_action = super()._sample_action(learning_starts, action_noise, n_envs)
        beta = self._current_beta()
        self.logger.record("train/beta", float(beta))
        if beta > 0.0 and self.controller is not None:
            ctrl = self._controller_action()                      # (2,) in [-1,1]
            blended = blend_action(beta, ctrl, action[0])
            action = blended.reshape(1, -1).astype(action.dtype)
            buffer_action = action.copy()                          # env action space is [-1,1] (scaled==unscaled)
        return action, buffer_action
