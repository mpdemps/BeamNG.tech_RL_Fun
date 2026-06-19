"""run21 STEP 1: the base controller for guided residual RL (training wheels). MIT-clean, rolled
ourselves -- no GPL/LGPL. Drives the existing racing line with two classic feedback loops:

  STEERING -- pure pursuit: aim at a lookahead point on the racing line (lookahead distance
    L_d = clip(k_v * speed, l_min, l_max)), compute the path curvature that arcs the car to
    that point (kappa = 2 sin(alpha) / L_d, alpha = angle from the car's heading to the point),
    and map curvature -> normalized steer command. Longer lookahead at speed -> gentler steer.

  THROTTLE/BRAKE -- P controller tracking the racing line's braking-aware v_target: thr =
    kp * (v_target - speed), positive = throttle, negative = brake (the env's action[1]
    convention). v_target already falls on the APPROACH to each corner (speed_profile's
    backward braking pass), so faithfully tracking it brakes EARLY to apex speed, then the
    forward pass lets it gas out. Optional small integral term kills steady-state tracking lag.

Outputs (steer, thr) in the env's [-1,1] x [-1,1] action convention, so the residual policy can
later add to it and the sum goes through the same env (steer-rate limit included). Self-contained
(own arc + nearest-point tracking) so it runs standalone for the must-lap gate AND inside training.
"""
import math
import numpy as np
from data.raceline_builtin import RACELINE
from envs.speed_profile import compute_speed_profile


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class BaseController:
    def __init__(self, line=RACELINE, k_v=0.6, l_min=6.0, l_max=25.0,
                 k_steer=22.0, kp_speed=0.25, ki_speed=0.0, steer_sign=-1.0,
                 speed_factor=1.0):
        self.P = np.asarray(line, float)[:, :2]
        self.n = len(self.P)
        seg = np.linalg.norm(np.roll(self.P, -1, axis=0) - self.P, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])      # n+1; cum[-1] = perimeter
        self.track = float(self.cum[-1])
        self.vt = np.asarray(compute_speed_profile(line)[0], float)
        self.k_v, self.l_min, self.l_max = k_v, l_min, l_max
        self.k_steer, self.kp_speed, self.ki_speed = k_steer, kp_speed, ki_speed
        self.steer_sign = steer_sign
        # speed_factor < 1 tracks a fraction of v_target -> grip margin everywhere (the base
        # controller is a clean must-lap "training wheels"; the RL residual recovers the speed).
        self.speed_factor = speed_factor
        self._idx = 0
        self._ispeed = 0.0

    def reset(self):
        self._idx = 0
        self._ispeed = 0.0

    def _nearest(self, pos):
        """Seam-safe windowed nearest-point search forward from the last index."""
        best, best_o = None, 0
        for o in range(-5, 30):
            c = (self._idx + o) % self.n
            d = (self.P[c, 0] - pos[0]) ** 2 + (self.P[c, 1] - pos[1]) ** 2
            if best is None or d < best:
                best, best_o = d, o
        self._idx = (self._idx + best_o) % self.n
        return self._idx

    def _point_at_arc(self, s):
        s = s % self.track
        j = int(np.searchsorted(self.cum, s) - 1)
        j = max(0, min(j, self.n - 1))
        seg = self.cum[j + 1] - self.cum[j]
        f = (s - self.cum[j]) / seg if seg > 1e-6 else 0.0
        a, b = self.P[j], self.P[(j + 1) % self.n]
        return a + f * (b - a)

    def action(self, pos, vel, dir_):
        """pos/vel/dir_ are world-frame (x,y,..) from agent_state. Returns (steer, thr)."""
        speed = math.hypot(vel[0], vel[1])
        i = self._nearest(pos)

        # --- pure-pursuit steering ---
        l_d = min(self.l_max, max(self.l_min, self.k_v * speed))
        tgt = self._point_at_arc(self.cum[i] + l_d)
        heading = math.atan2(dir_[1], dir_[0])
        alpha = _wrap(math.atan2(tgt[1] - pos[1], tgt[0] - pos[0]) - heading)
        kappa = 2.0 * math.sin(alpha) / max(l_d, 1e-3)          # curvature to reach the point
        steer = max(-1.0, min(1.0, self.steer_sign * self.k_steer * kappa))

        # --- speed-profile P(I) throttle/brake ---
        err = self.speed_factor * float(self.vt[i]) - speed
        self._ispeed = max(-5.0, min(5.0, self._ispeed + err))   # clamped anti-windup
        thr = self.kp_speed * err + self.ki_speed * self._ispeed
        thr = max(-1.0, min(1.0, thr))
        return steer, thr
