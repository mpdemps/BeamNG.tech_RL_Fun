"""run27 Phase 2 DRIFT scaffolding (pure functions, no BeamNG, MIT-clean).

Phase 1 taught the GTS NOT to slide; Phase 2 teaches it to slide ON PURPOSE. The whole
objective is the slip angle beta (the angle between where the car POINTS and where it
MOVES). This module holds the drift-specific pieces the env switches on when drift_mode=True:

  beta_target(speed)  -- the slip angle we WANT at a given speed. Big and showy at low speed
                         (~35 deg), tapering to a safer ~17 deg at high speed (a 35 deg slide
                         at 50 m/s is a crash; at 10 m/s it's a tidy donut). Fed to the obs so
                         the policy knows the angle it is being asked to hold right now.

  drift_reward(beta, beta_target) -- a 0..1 bump that PEAKS when |beta| == beta_target and
                         falls off both ways: too little = just gripping (boring), too much =
                         spinning (out of control). The env multiplies this by the on-track +
                         forward-progress gates and the drift/progress weights.

  the drift terminator thresholds -- a controlled drift (high slip but still going down-track)
                         must NOT end the episode; only a true spin (pointing backward / past
                         the band, or stopped) should. Constants proposed from the drift probe.

Mikey owns the reward BALANCE (W_DRIFT vs the progress weight) and tunes it at review; this
file just defines the shapes.
"""
import math

# --- beta_target(speed): speed-scaled target slip angle (degrees) ---
# Piecewise-linear: hold BETA_LOW below V_LOW, taper linearly to BETA_HIGH at/above V_HIGH.
BETA_LOW = 35.0      # deg, target slide at low speed (showy donut)
BETA_HIGH = 17.0     # deg, target slide at high speed (a big angle up here is a crash)
V_LOW = 8.0          # m/s; at/below this, ask for the full BETA_LOW (== the env's ESC floor)
V_HIGH = 30.0        # m/s; at/above this, ask for BETA_HIGH


def beta_target(speed_mps: float) -> float:
    """Target slip angle (deg) for the current speed. ~35 deg slow -> ~17 deg fast."""
    if speed_mps <= V_LOW:
        return BETA_LOW
    if speed_mps >= V_HIGH:
        return BETA_HIGH
    f = (speed_mps - V_LOW) / (V_HIGH - V_LOW)
    return BETA_LOW + (BETA_HIGH - BETA_LOW) * f


# --- drift_reward: a bump peaking at beta == beta_target ---
# Gaussian so the gradient points toward the target from both sides: too little slip (gripping)
# AND too much (spinning) both score less. SIGMA sets how forgiving the band is.
DRIFT_SIGMA_DEG = 10.0   # deg; ~0.61 reward at +/-10 deg off target, ~0.14 at +/-20 deg


def drift_reward(beta_deg: float, target_deg: float) -> float:
    """0..1, peaks (==1) when |beta| matches the target slide angle."""
    d = abs(beta_deg) - target_deg
    return math.exp(-0.5 * (d / DRIFT_SIGMA_DEG) ** 2)


# --- drift terminator (thresholds grounded by the drift probe) ---
# Phase 1's loss-of-control terminator fires on sustained high YAW RATE (>150 deg/s for 8 ticks).
# That is a SPIN signal, but a drift INITIATION flick also spikes yaw briefly while the car is still
# under control -- so in drift mode we do NOT terminate on yaw rate at all. A true spin instead shows
# up as the SLIP ANGLE blowing past the band: the probe measured a controlled target of ~17-35 deg
# vs a real spin pegging beta at ~180 deg, so SPIN_BETA_DEG=60 sits cleanly between them. The no-
# forward-progress (donut-in-place) case is a SEPARATE condition, handled by the drift stuck timer
# (DRIFT_NO_PROGRESS_STEPS) in the env -- NOT and-gated with the slip test (a car spinning at 180 deg
# while still sliding down-track is lost regardless of translation, which is why C1 in the probe must
# terminate even though it kept progressing).
SPIN_BETA_DEG = 60.0          # deg; |beta| past this = lost it (well above the ~35 deg target band,
                              # well below the ~180 deg a real spin reaches -- probe-measured)
DRIFT_SPIN_STEPS = 12         # consecutive over-the-band steps before we call it (~0.6 s at 20 Hz) --
                              # long enough to ride out a drift-initiation flick that briefly overshoots
DRIFT_NO_PROGRESS_STEPS = 60  # ~3 s of no forward progress = donut-in-place, end it (tighter than
                              # Phase 1's 200-step stuck so a contained donut doesn't run forever)


def is_drift_spin(beta_deg: float, heading_align: float) -> bool:
    """True on a given step if the car is SPUN this step: slip angle past the drift band, or the nose
    pointing backward. The env counts CONSECUTIVE True steps and ends the episode at DRIFT_SPIN_STEPS,
    so a brief initiation flick is ridden out but a sustained spin terminates -- regardless of whether
    the spinning car is still translating down-track. A controlled drift keeps |beta| in the band
    (<= SPIN_BETA_DEG) and the nose forward, so this stays False and the drift continues."""
    return abs(beta_deg) > SPIN_BETA_DEG or heading_align < 0.0
