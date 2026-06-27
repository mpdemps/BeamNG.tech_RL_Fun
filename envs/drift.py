"""run27 Phase 2 DRIFT scaffolding (pure functions, no BeamNG, MIT-clean).

Phase 1 taught the GTS NOT to slide; Phase 2 teaches it to slide ON PURPOSE. The whole
objective is the slip angle beta (the angle between where the car POINTS and where it MOVES).
This module holds the drift-specific pieces the env switches on when drift_mode=True:

  beta_target(speed)  -- the slip angle we WANT at a given speed. run27 (Mikey) STARTS SMALL: a
                         modest ~20 deg slide at low speed tapering to ~12 deg at high speed,
                         scaled by a single knob DRIFT_SCALE so a later warm-start can raise the
                         whole band toward a big showy drift without any other change.

  drift_reward(beta, beta_target) -- a 0..1 bump that PEAKS when |beta| == beta_target. ASYMMETRIC
                         (Mikey, anti-timidity): below target it rises to the peak (gripping is just
                         mildly less rewarded); ABOVE target it falls GENTLY toward 0 only as beta
                         approaches the spin cutoff -- so bigger angles are never PUNISHED, the car
                         is free to explore out toward the spin. The over-drift side is anchored to
                         the spin (OVERDRIFT_ANCHOR), NOT to the small target, so shrinking the
                         target does not tighten the upper side into timidity.

  the drift terminator thresholds -- a controlled drift (slip in/near the band, still going down-
                         track) must NOT end the episode; only a true spin (slip past the band, or
                         nose backward) should. DECOUPLED from the target: stays at the big-showy
                         setting regardless of how small the starting target is.

Mikey owns the reward BALANCE (W_DRIFT vs the progress weight, in beamng_env.py) + the corner gate;
this file defines the shapes.
"""
import math

# --- the slip-angle band edges (degrees) ---
# SPIN_BETA_DEG is the band's hard upper edge (= the terminator cutoff): past this the car is spun,
# NOT drifting. Defined first because the reward's over-drift side is anchored to it.
SPIN_BETA_DEG = 60.0          # deg; |beta| past this = lost it (the terminator fires; probe measured
                              # a real spin pegging beta ~180 deg, vs a target band <= ~20 deg here)

# --- beta_target(speed): speed-scaled target slip angle (degrees) ---
# Base band at DRIFT_SCALE=1.0 is Mikey's small starting target; raising DRIFT_SCALE later (warm
# start) lifts the whole band toward big showy drifts.
BETA_LOW = 20.0      # deg, base target slide at low speed (modest start; was 35)
BETA_HIGH = 12.0     # deg, base target slide at high speed (was 17)
V_LOW = 8.0          # m/s; at/below this, ask for the full BETA_LOW (== the env's ESC floor)
V_HIGH = 30.0        # m/s; at/above this, ask for BETA_HIGH
DRIFT_SCALE = 1.0    # single scale knob on the whole band. 1.0 = the small start; a warm-start can
                     # raise it (e.g. ~1.75 -> a ~35/21 deg big-showy band) with nothing else changed.
BETA_TARGET_OBS_NORM = 40.0  # deg; obs normalization anchor for beta_target. Fixed at the BIG-regime
                     # max (NOT the small start) so that when DRIFT_SCALE is raised later, the obs
                     # value grows consistently instead of re-scaling under the policy.


def beta_target(speed_mps: float, scale: float = None) -> float:
    """Target slip angle (deg) for the current speed, scaled by DRIFT_SCALE (or `scale`)."""
    s = DRIFT_SCALE if scale is None else scale
    if speed_mps <= V_LOW:
        base = BETA_LOW
    elif speed_mps >= V_HIGH:
        base = BETA_HIGH
    else:
        base = BETA_LOW + (BETA_HIGH - BETA_LOW) * (speed_mps - V_LOW) / (V_HIGH - V_LOW)
    return s * base


# --- drift_reward: an ASYMMETRIC bump peaking at beta == beta_target ---
SIGMA_LOW = 12.0                  # deg; below-target falloff (gentle rise to the peak)
OVERDRIFT_ANCHOR = SPIN_BETA_DEG  # deg; above target the reward declines to 0 at THIS (the spin),
                                  # not at the target -> bigger angles cost only near the spin.


def drift_reward(beta_deg: float, target_deg: float) -> float:
    """0..1, peaks (==1) when |beta| matches the target slide angle.
    Below target: Gaussian rise. Above target: GENTLE linear decline to 0 at OVERDRIFT_ANCHOR (the
    spin cutoff) -- never negative, so exploring bigger angles is never punished (anti-timidity)."""
    b = abs(beta_deg)
    if b <= target_deg:
        return math.exp(-0.5 * ((b - target_deg) / SIGMA_LOW) ** 2)
    span = max(1e-6, OVERDRIFT_ANCHOR - target_deg)
    return max(0.0, 1.0 - (b - target_deg) / span)


# --- drift terminator (thresholds grounded by the drift probe; DECOUPLED from the target) ---
# Phase 1's loss-of-control terminator fires on sustained high YAW RATE (>150 deg/s for 8 ticks).
# That is a SPIN signal, but a drift INITIATION flick also spikes yaw while still under control --
# so in drift mode we do NOT terminate on yaw rate at all. A true spin instead shows up as the SLIP
# ANGLE blowing past the band (SPIN_BETA_DEG) or the nose pointing backward. These stay at the big-
# showy setting no matter how small the target is, so a shrunk target never makes the car timid.
DRIFT_SPIN_STEPS = 12         # consecutive over-the-band steps before we call it (~0.6 s at 20 Hz) --
                              # long enough to ride out a drift-initiation flick that briefly overshoots
DRIFT_NO_PROGRESS_STEPS = 60  # ~3 s of no forward progress = donut-in-place, end it (tighter than
                              # Phase 1's 200-step stuck so a contained donut doesn't run forever)


def is_drift_spin(beta_deg: float, heading_align: float) -> bool:
    """True on a given step if the car is SPUN this step: slip angle past the band, or the nose
    pointing backward. The env counts CONSECUTIVE True steps and ends the episode at DRIFT_SPIN_STEPS,
    so a brief initiation flick is ridden out but a sustained spin terminates -- regardless of whether
    the spinning car is still translating down-track. A controlled drift keeps |beta| in the band
    (<= SPIN_BETA_DEG) and the nose forward, so this stays False and the drift continues."""
    return abs(beta_deg) > SPIN_BETA_DEG or heading_align < 0.0
