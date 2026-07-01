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
from envs.beamng_env import _shared, _PROFILE_KAPPA
from envs.base_controller import BaseController
from envs.drift import corner_factor

# run31 controller-backed drift: corner-gated residual authority (option a). On STRAIGHTS the residual
# is tight (controller grips clean, kills the fishtail); in CORNERS the RL gets BIG authority on BOTH
# channels (steer to counter-steer, throttle to break the rear loose), ramped in at turn-in by the
# same corner_factor the reward uses. Also relaxes the controller's throttle-BRAKING in corners
# (scaled by cf) so the RL's throttle isn't fighting the controller's grip-speed brake; the
# controller's STEERING stays at full (the backbone that keeps the car pointed through the corner).
CG_STRAIGHT = 0.05    # straight residual bound (both channels): tight -> grip, no fishtail
CG_CORNER_STEER = 0.5   # corner steer residual bound (+/-): room to add counter-steer
CG_CORNER_THR_UP = 1.0  # corner POSITIVE throttle residual: floor it to break the rear loose
CG_CORNER_THR_DN = 0.5  # corner NEGATIVE throttle residual: lift/brake to modulate the slide


class ResidualHybrid(gymnasium.Wrapper):
    """run24 THROTTLE-AUTHORITY CUT: the residual bound is now ASYMMETRIC PER CHANNEL.
    steer stays +/-delta; throttle is [-delta, +throttle_up] with throttle_up << delta. Throttle
    sign is positive=gas / negative=brake-lift (verified in base_controller), so capping the
    POSITIVE throttle residual at ~+0.05 limits how much EXTRA gas the policy can pile on the
    controller (the over-throttle-into-spin lever from run22/23), while -delta keeps full
    lift/brake authority to shed grip-killing speed. throttle_up=None -> symmetric (run22/23)."""

    def __init__(self, env, delta=0.12, throttle_up=None, controller=None, corner_gated=False):
        super().__init__(env)
        self.delta = float(delta)
        # per-channel [low, high] for [steer, throttle]; throttle high capped at throttle_up
        t_up = self.delta if throttle_up is None else float(throttle_up)
        self.low = np.array([-self.delta, -self.delta], dtype=np.float32)
        self.high = np.array([self.delta, t_up], dtype=np.float32)
        # run31: corner-gated authority (option a) -- the per-channel bounds RAMP with corner_factor
        # (tight straights / big corners) and the controller's braking is relaxed in corners. When
        # False, the fixed self.low/self.high above apply (run22-24 behavior, unchanged).
        self.corner_gated = bool(corner_gated)
        self.controller = controller if controller is not None else BaseController()
        self._steer_sum = 0.0
        self._thr_sum = 0.0
        self._thr_pos_sum = 0.0     # run24: sum of POSITIVE throttle residual (gas the policy adds)
        self._thr_sat_n = 0         # run24: steps the +throttle residual is AT the +cap (saturating)
        self._res_n = 0
        self.last_applied = None      # watcher display: the controller+residual action actually sent
        self.last_residual = None     # watcher display: the clipped residual the policy added

    def reset(self, **kwargs):
        self.controller.reset()
        self._steer_sum = 0.0
        self._thr_sum = 0.0
        self._thr_pos_sum = 0.0
        self._thr_sat_n = 0
        self._res_n = 0
        return self.env.reset(**kwargs)

    def _controller_action(self):
        s = _shared["vehicle"].sensors["agent_state"]
        steer, thr = self.controller.action(s["pos"], s["vel"], s.get("dir", (1.0, 0.0, 0.0)))
        return np.array([steer, thr], dtype=np.float32)

    def step(self, residual):
        residual = np.asarray(residual, dtype=np.float32).reshape(-1)
        ctrl = self._controller_action()
        cf = 0.0
        if self.corner_gated:
            # cf from the racing-line curvature at the car's current index (same signal the reward
            # gates on): 0 straight .. 1 corner. Bounds RAMP straight->corner per channel.
            cf = corner_factor(abs(float(_PROFILE_KAPPA[self.env.unwrapped._progress_idx])))
            ds = CG_STRAIGHT + cf * (CG_CORNER_STEER - CG_STRAIGHT)
            up = CG_STRAIGHT + cf * (CG_CORNER_THR_UP - CG_STRAIGHT)
            dn = CG_STRAIGHT + cf * (CG_CORNER_THR_DN - CG_STRAIGHT)
            low = np.array([-ds, -dn], dtype=np.float32)
            high = np.array([ds, up], dtype=np.float32)
            # relax the controller's BRAKING in corners (keep its steering full) so the RL's throttle
            # isn't fighting the grip-speed brake; at cf=1 the controller does no braking, RL owns throttle.
            if ctrl[1] < 0.0:
                ctrl = np.array([ctrl[0], ctrl[1] * (1.0 - cf)], dtype=np.float32)
        else:
            low, high = self.low, self.high
        clipped = np.clip(residual, low, high)                       # per-channel bound (ramped if corner-gated)
        applied = np.clip(ctrl + clipped, -1.0, 1.0)                 # controller (relaxed brake) + residual
        self.last_applied, self.last_residual = applied, clipped
        obs, reward, terminated, truncated, info = self.env.step(applied)
        info["residual_cf"] = float(cf)   # run31: corner-gate factor this step (0 straight .. 1 corner)
        # track mean |applied residual| per channel over the episode (how hard the RL pushes each)
        self._steer_sum += abs(float(clipped[0]))
        self._thr_sum += abs(float(clipped[1]))
        # run24: isolate the +throttle CAP from lift usage. residual_throttle_pos = mean of the
        # POSITIVE throttle residual (gas added); residual_throttle_satfrac = fraction of steps it
        # sits AT the +cap (raw wanted >= cap -> clipped == high[1]) = how often the cut is binding.
        self._thr_pos_sum += max(0.0, float(clipped[1]))
        self._thr_sat_n += int(clipped[1] >= high[1] - 1e-6)   # vs the ACTUAL (ramped) upper bound
        self._res_n += 1
        n = max(self._res_n, 1)
        info["residual_abs_steer"] = self._steer_sum / n
        info["residual_abs_throttle"] = self._thr_sum / n
        info["residual_abs"] = (self._steer_sum + self._thr_sum) / (2 * n)   # combined, for continuity
        info["residual_throttle_pos"] = self._thr_pos_sum / n
        info["residual_throttle_satfrac"] = self._thr_sat_n / n
        return obs, reward, terminated, truncated, info
