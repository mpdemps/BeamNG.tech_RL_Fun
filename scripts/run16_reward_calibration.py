"""run16 reward calibration gate (offline): the make-or-break of the plan. The over-speed
penalty max(0, v-(v_target-offset))^2 has ZERO gradient at v_target, so the reward-optimal
speed v* sits ABOVE v_target by W_PROG*dt/(2*W_OVER); the conservative 1.2g target must
absorb that. Two checks (Mike's gate):
  1. HELD-AT-TARGET out-rewards FLOOR-IT over the T1 approach (the literal failure we escape).
  2. The reward-optimal cruise speed lands at/below the grippable limit, not just near target.
Pure reward-form math over the speed profile -- no policy/BeamNG needed (a fresh policy
can't 'hold at target' to test this live; it's a property of the weights + profile)."""
import math, os, sys, bisect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.centerline_racetrack_builtin import CENTERLINE
from envs.speed_profile import compute_speed_profile
from envs.beamng_env import (W_PROG, W_OVER, W_SLIP, OVER_SPEED_OFFSET, BETA_SLIP_DEAD)

DT_STEP = 3.0 / 60.0          # raw_progress ~= v * DT_STEP (3 physics steps @ 60 Hz)
A_GRIP = 16.0                 # ~1.6 g measured cornering-grip ceiling (the real limit)
V_MAX = 33.0


def reward_step(v, v_target):
    progress = W_PROG * (v * DT_STEP) * 1.0          # align ~ 1.0 on-line forward
    over = max(0.0, v - (v_target - OVER_SPEED_OFFSET))
    return progress - W_OVER * over * over            # (slip term 0 when not sliding)


def main():
    v_t, R, cum, track, _k = compute_speed_profile(CENTERLINE)
    n = len(v_t)
    print(f"weights: W_PROG={W_PROG} W_OVER={W_OVER} OFFSET={OVER_SPEED_OFFSET} "
          f"W_SLIP={W_SLIP} BETA_DEAD={BETA_SLIP_DEAD}; DT_STEP={DT_STEP}\n")

    def idx_at(arc):
        return max(0, min(bisect.bisect_right(cum, arc % track) - 1, n - 1))

    # --- Check 1: HELD-AT-TARGET vs FLOOR-IT over the T1 approach (arc 280..345) ---
    seg = [idx_at(a) for a in range(280, 346, 2)]
    r_hold = sum(reward_step(v_t[i], v_t[i]) for i in seg)         # track the target
    r_floor = sum(reward_step(V_MAX, v_t[i]) for i in seg)         # floor it at V_MAX
    print("=== Check 1: T1 approach (arc 280-345), summed step reward ===")
    print(f"   HELD-AT-TARGET: {r_hold:8.3f}")
    print(f"   FLOOR-IT (33):  {r_floor:8.3f}")
    print(f"   -> {'PASS: holding target out-rewards flooring' if r_hold > r_floor else 'FAIL: flooring wins (raise W_OVER / add offset)'}\n")

    # --- Check 2: reward-optimal speed v* vs grippable limit, at each corner ---
    # v* solves d/dv[progress - penalty] = 0 -> W_PROG*DT_STEP = 2*W_OVER*(v*-(vt-off))
    margin = W_PROG * DT_STEP / (2 * W_OVER)
    print(f"=== Check 2: reward-optimal v* = (v_target - offset) + {margin:.2f} m/s vs grip limit ===")
    print(f"   {'corner':>6}{'apex':>6}{'R_loc':>6}{'v_target':>9}{'v*':>7}{'v_grip':>8}  verdict")
    CORNERS = [("T1", 338), ("T4", 1004), ("T5", 1274), ("T6", 1336), ("T8", 2990), ("T11", 3574)]
    all_ok = True
    for name, apex in CORNERS:
        i = idx_at(apex)
        vt = v_t[i]; Rl = R[i]
        v_star = (vt - OVER_SPEED_OFFSET) + margin
        v_grip = math.sqrt(A_GRIP * Rl)
        ok = v_star <= v_grip
        all_ok = all_ok and ok
        print(f"   {name:>6}{apex:>6}{Rl:>6.0f}{vt:>9.1f}{v_star:>7.1f}{v_grip:>8.1f}  {'PASS' if ok else 'HOT'}")
    print(f"\n   margin v*-v_target = {margin:.2f} m/s (the zero-gradient overshoot)")
    print(f"   -> {'PASS: reward-optimal speed is within grip at every corner' if all_ok else 'FAIL: runs hot somewhere -> raise W_OVER or set OVER_SPEED_OFFSET>0'}")
    print(f"\n   (to pin v* exactly AT v_target, set OVER_SPEED_OFFSET = {margin:.2f})")


if __name__ == "__main__":
    main()
