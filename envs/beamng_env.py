"""
BeamNG.tech RL environment as a plain Gymnasium env.

Plain-language overview, Mikey:
- This file is the rules of the game for our car-driving AI.
- It tells the AI what it can see (9 numbers about the car and the track).
- It tells the AI what buttons it can push (2 numbers: steering, gas/brake).
- It tells the AI when it wins (lap finished) or loses (crash, off-track, stuck).
- Each game-tick, we ask BeamNG to fast-forward 50 ms of car-time and then
  look at what changed.

We follow the methodology in docs/phase1_env_spec.md. The rtgym wrapper
(docs/references.md, TMRL precedent) is a v2 concern — see the TODO in
step().
"""

import bisect
import math
from typing import Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Damage, Electrics, State


# ---- Tunable constants (one place to tweak everything) ----
MAP_NAME = "west_coast_usa"
VEHICLE_MODEL = "scintilla"         # Civetta Scintilla; model codename verified
                                    # via bng.vehicles.get_available(). Phase 2
                                    # car, much faster than run1's etk800 sedan.
VEHICLE_PART_CONFIG = "vehicles/scintilla/race.pc"   # Race config (Mikey's pick,
                                    # not default gts). Exact .pc path verified
                                    # inside content/vehicles/scintilla.zip.
VEHICLE_ID = "ego"
PHYSICS_STEPS_PER_STEP = 3          # 3 steps at 60 Hz = 50 ms = 20 Hz env tick.
DETERMINISTIC_STEPS_PER_S = 60
MAX_SPEED_M_S = 70.0                # for obs normalization
# Lookahead (run6 rebuild): the old observation looked at CENTERLINE[idx+1/2/3],
# a ~13m horizon that SHRANK in corners (points cluster where the track curves).
# At 1.395g braking, 13m of vision caps blind-safe speed near ~19 m/s -- the car
# could never learn the straights at speed. New design: 6 points at FIXED ARC
# DISTANCES ahead of the car's true arc position, sized for the car's POTENTIAL
# (speed-profile peak ~77-78 m/s; worst-case braking ~215m) not historical
# policy speed. Geometric spacing: dense near (placement), sparse far
# (anticipation). 280m = worst-case braking sight + margin.
LOOKAHEAD_DISTANCES_M = [10.0, 20.0, 40.0, 80.0, 160.0, 280.0]
# Normalizer for the lookahead euclidean distances (euclidean <= arc <= 280, so
# /300 spans ~0..0.93 and never clips).
MAX_LOOKAHEAD_DIST_M = 300.0
CENTER_OFFSET_CLIP_M = 10.0
OFF_TRACK_THRESHOLD_M = 8.0         # run17: ~max road half-width (saved avg full width
                                    # 11.41m -> ~5.7m half) + margin, so the episode ends
                                    # the instant the car leaves the road (was 20m, which
                                    # let a cut corner wander deep into grass before ending)
STUCK_STEPS_THRESHOLD = 200
STOPPED_SPEED_M_S = 0.5             # below this, treat the car as stationary
                                    # for alignment purposes (no backward penalty)
FLIP_PENALTY = -10.0
OFF_TRACK_PENALTY = -10.0
STUCK_PENALTY = -5.0
LAP_BONUS = 100.0                   # jackpot for completing the lap (the goal)
# Curriculum: the centerline is split into N_CHECKPOINTS evenly-spaced (by
# distance) segments. The first time per episode the car's cumulative progress
# reaches an intermediate checkpoint, it earns CHECKPOINT_BONUS once -- a clear
# "got further" spike ON TOP of the per-step progress reward (~1-1.5/tick,
# ~273 accumulated per segment), small enough not to swamp smooth driving. The
# final checkpoint is the finish line, rewarded by LAP_BONUS instead.
N_CHECKPOINTS = 16
CHECKPOINT_BONUS = 10.0
# Anti-crawl speed reward: pays for going fast, but only when on the racing line
# and pointed forward. speed_reward = SPEED_WEIGHT * forward_speed *
# gated_alignment, so a crawling-for-safe-points car earns less than one that
# carries speed. run2 (0.0075) was big enough that the powerful race car learned
# to FLOOR-AND-SPIN: a donut still drifts forward across the evenly-spaced
# checkpoints, and at 0.0075 the speed reward outweighed careful driving. run3
# cut it ~75% to 0.002 so the spin's (low-align) speed reward is negligible vs
# the progress/checkpoint reward from actually getting around the track. SAME
# soft formula. run5 nudges it 0.002 -> 0.003 (small anti-timidity bias toward
# confident throttle now the heading gate forbids spinning for points); still a
# nudge, far below run2's disastrous 0.0075.
SPEED_WEIGHT = 0.003
# Anti-wobble smoothing penalty (run2): run1 steered bang-bang left-right
# because nothing penalized it. smoothness_penalty = -SMOOTH_WEIGHT *
# abs(steer - prev_steer) penalizes the CHANGE in steering, not steering itself,
# so smooth sustained cornering is free and only rapid flip-flopping costs.
# run6 raises 0.1 -> 0.2: 0.1 did not bite (run5 still oscillated). A real
# hairpin is a few large changes (affordable under this total-variation penalty)
# vs continuous oscillation (accumulates), so a higher weight suppresses the
# flutter without preventing genuine steering.
SMOOTH_WEIGHT = 0.2
# Anti-chatter throttle/brake smoothness (run6, PRIMARY fix): action[1] is the
# combined throttle(+)/brake(-) axis, and its CHANGE was completely uncosted, so
# run5 chattered gas<->brake (occasionally flooring into a corner -> spin). Same
# total-variation form as steering: penalize abs(action[1] - prev). One term
# covers throttle chatter, brake chatter, AND throttle<->brake flipping. Started
# LOWER than steering (0.05): the car needs DECISIVE throttle/brake (floor on
# exit, hard brake on entry) more than decisive steering, so over-penalizing this
# axis is the most direct route to timidity. Raise toward 0.08-0.1 only if
# chatter persists.
THROTTLE_SMOOTH_WEIGHT = 0.05
# Weave fix (run10, STRUCTURAL -- ends the reward-penalty era). Runs 6-9 tried four
# reward penalties; all failed. CC's traces proved the weave is a CONTROL
# INSTABILITY, not a rewarded proxy: heading_err aimed at a FIXED 10m point, so the
# preview time (10m/speed) collapsed from ~1.5s at low speed to ~0.32s at 115 kph
# -- the classic pure-pursuit hunting setup. Measured signature (run8 trace): the
# weave's half-period shrank 1.09s->0.28s (frequency ~4x) and amplitude grew as
# speed rose 58->109 kph. You cannot penalize away a control instability; fix the
# reference. run10 scales the steering-reference lookahead with speed so the preview
# time stays ~constant (PREVIEW_TIME) instead of collapsing:
#   L_d = clip(PREVIEW_TIME * speed_horizontal, L_MIN, L_MAX)
#   heading_err = bearing(car -> centerline point L_d ahead)  [interpolated on _cum_arc]
# L_MIN=10 == the old fixed point, so launch is unchanged; L_MAX=120 keeps ~1.5s
# preview to the car's ~77 m/s top speed on the 694m straight (a 50m cap would drop
# it to ~0.65s there and the weave could re-emerge). Geometry check: 120m only binds
# at high speed, which occurs on the long straights; corner-cutting at the long-
# straight ENDS is a known G14 watch-item (lower PREVIEW_TIME/L_MAX if it understeers).
# The run9 oscillation reward-penalty is DROPPED (the reference fix is the sole weave
# mechanism, for clean attribution). Pose-independent (centerline arc geometry).
PREVIEW_TIME = 1.5           # s; target preview time the speed-scaled lookahead holds
PREVIEW_L_MIN = 10.0         # m; floor (== old fixed point, keeps launch working)
PREVIEW_L_MAX = 120.0        # m; cap (~1.5s preview to top speed; only binds on long straights)
# Anti-wheelspin penalty (run4): punish burning the rear tyres off the line, the
# traction root-cause under run3's spin-out variance. slip = wheelspeed (wheel-
# rotation speed from Electrics) minus speed_horizontal (true ground speed); a
# spinning wheel reads far faster than the car moves. Penalize only slip beyond a
# deadzone. run11 raises SLIP_DEADZONE 2.0 -> 4.0 to match TC_SLIP_DEAD: CC's launch
# sweep showed optimal acceleration at slip ~3 (throttle 0.3 -> 21 m/s) and runaway
# at slip >=7.5 (accel collapses), so 2-4 is the PRODUCTIVE slip band the TC cap
# deliberately permits. With the deadzone at 4.0 the penalty no longer charges on
# that productive band (it would fight the cap) -- it only fires where TC is already
# cutting. spin_penalty = -SPIN_WEIGHT * max(0, slip - SLIP_DEADZONE).
SLIP_DEADZONE = 4.0
SPIN_WEIGHT = 0.05
# Traction control (run11, STRUCTURAL throttle-axis fix). Locked diagnosis (CC's spin
# probes): rear slip from a pinned/pulsing-high throttle LEADS the steering sign-flip
# 3-6 steps at both peak and maturity -> throttle-induced oversteer, the steering
# weave is reactive counter-steer (five steering-axis runs 6-10 never touched it). A
# reward penalty gets outvoted (run4 wheelspin penalty charged but the policy floored
# anyway), so the fix is structural: cap applied throttle on measured rear slip, like
# real TC. Below TC_SLIP_DEAD full power; DEAD..FULL scales down; >=FULL cut to
# TC_MIN_THR. Constants from the launch sweep (optimum slip ~3, runaway ~7.5): DEAD
# above the productive band, FULL at runaway onset, MIN a small anti-bog floor so the
# standing-start transient can't stall the launch. Gated on last-step slip (one 50ms
# step, real-TC sensor lag). Brake path untouched. See scripts/run11_traction_sweep.py.
TC_SLIP_DEAD = 4.0
TC_SLIP_FULL = 7.0
TC_MIN_THR = 0.1
# run14 ESC (electronic stability control): cut throttle on lateral SLIP-ANGLE
# beta -- the axis TC misses. TC watches longitudinal wheelspin; in all three
# run13 spins rear slip stayed <4 (TC never fired) while the car slid sideways at
# beta 17-46deg. Clean tracking sits at beta p90 ~5deg, so DEAD=9 leaves clean
# cornering uncut; FULL=22 is deep oversteer, below the 25-46deg full-spin band.
# esc_factor scales applied throttle from 1.0 (beta<=DEAD) down to the env's
# esc_min floor (beta>=FULL). Multiplicative with tc_factor; cut-throttle-only.
ESC_BETA_DEAD = 9.0
ESC_BETA_FULL = 22.0
# ESC speed floor: below this, slip-angle beta is just velocity-vector noise (a tiny
# lateral component at low speed reads as a large angle) and a low-speed slide is
# harmless -- while the standing start NEEDS full throttle. So beta is pinned to 0
# below the floor (ESC off). Every corner is >=17 m/s and every measured spin was
# 25-36 m/s, so an 8 m/s floor costs zero corner coverage. (run14 smoke caught ESC
# strangling the launch without this.)
ESC_MIN_SPEED_M_S = 8.0
# run15 speed-scaled steering RATE limit: the proactive complement to ESC. ESC cuts
# throttle reactively once the rear is already sliding (too late for a full-lock
# reversal at speed). The spin is a REVERSAL (a rate phenomenon), while a fast corner
# is SUSTAINED lock (low rate) -- so tightening the run13 steering-rate cap with speed
# blocks the violent reversal WITHOUT capping the lock the fast corners hold. This is
# why a rate limit beats a magnitude cap here: the fastest corners (T2 ~31, T7 ~34, T9
# ~35 m/s) live IN the 33-36 m/s spin band, so a magnitude cap could not separate
# corner (~0.6 lock) from slam (~0.9-1.0) without starving the corner; a rate limit
# separates them by HOW FAST the wheel moves instead. Effective rate = steer_rate at
# /below V_LO (full agility), linear taper to the env's steer_rate_hi by V_HI, flat
# above. steer_rate_hi<0 => OFF (flat steer_rate, default). Uses last-step speed.
STEER_RATE_V_LO = 27.0
STEER_RATE_V_HI = 31.0
# Heading kill-switch (run5): the reward was heading-BLIND -- it gated on
# velocity-vs-tangent but never on which way the nose points, so a spun-out car
# sliding/coasting down-track earned full progress + speed + checkpoint reward
# (the leak behind run4's spin-and-reverse). heading_align = dir.tangent (nose vs
# track tangent, parallel to the existing velocity_align). When the nose points
# more than ~90deg off down-track (heading_align < threshold) we ZERO the entire
# step reward (progress, speed, AND checkpoint/lap bonuses) -- a kill-switch, not
# a multiplier (a multiplier would tax legitimate corners where the nose is
# naturally 10-30deg off). -0.2 gives margin so an extreme-but-recoverable slide
# is not instantly zeroed. A genuine spin-out (sustained heading_align < -0.2)
# also terminates the episode after BACKWARD_TERM_STEPS.
HEADING_KILL_THRESHOLD = -0.2
BACKWARD_TERM_STEPS = 40    # ~2 s at the 20 Hz env tick; resets when nose recovers
# Price the backward exit (run7). run6 shipped this termination as a NEUTRAL
# terminal (0.0) while every other death cost something (off-track -10, stuck
# -5, flip -10) -- and the policy measurably converged onto the one free door:
# backward-endings CLIMBED 51% -> 88% over training, 86% of them at 50-150m on
# the opening straight (harvest ~100m of progress, then exit free). -25 makes
# it deliberately the WORST death on the menu, because it is the door the
# policy actually uses. Felt at spin onset as -25 * 0.99^40 ~ -16.7, decisively
# below even a crawler's continuation value (~+20); NOT larger than -25 because
# terminal fear generalizes to nearby fast-approach states (timidity is the
# other observed failure mode) and -25 stays within the critic's target scale.
BACKWARD_TERM_PENALTY = -25.0
# Monotonic progress tracker: each step we re-find the car's centerline index by
# searching only a LOCAL window around the last index, never globally. This
# keeps cumulative distance continuous across the start/finish seam (idx 0 and
# the last index are ~0.67 m apart; a global nearest-point search flips between
# them and makes distance jump by a full lap). FWD must exceed the farthest the
# car moves in one step (~1.5 m, far under 20 indices); BACK gives margin for a
# spun or backward-sliding car.
PROGRESS_WINDOW_BACK = 5
PROGRESS_WINDOW_FWD = 20
RANDOM_HEADING_DEG = 0.0            # intentionally 0 for early training:
                                    # deterministic spawn along the tangent
                                    # gives a sharper learning signal. Widen
                                    # to 5-10° once the policy has a baseline.
# Built-in centerline Z values are road-surface elevation. The ETK 800's
# spawn position needs to be vehicle center-of-mass elevation, ~0.5 m above
# the road; spawning at road Z makes wheels penetrate terrain, which the
# physics engine resolves by shoving the car sideways — which then *looks*
# like a yaw bug. Adding a small vertical offset drops the car onto the road
# cleanly. Better to fall a few cm than to clip into terrain.
SPAWN_Z_OFFSET_M = 1.0
# Reserved for future use if a systematic bias is identified between the
# computed centerline tangent and the visual road direction. Currently 0
# (no-op); applied inside _smoothed_forward_yaw so a single value change
# adjusts both the spawn orientation and the alignment-gate's tangent.
SPAWN_YAW_BIAS_DEG = 0.0
# Spawn-orientation frame correction (run2 fix). MEASURED in two stages by
# scripts/spawn_angle_test.py, which reads the car's SETTLED forward vector after
# each spawn (not the intended-heading math):
#   1. With no correction, the settled heading was a rock-solid 90.6 deg
#      CLOCKWISE of the centerline tangent (mean -90.61, std 0.02 over 15 trials).
#   2. Applying a naive +90 deg offset did NOT cancel it -- it overshot to ~180
#      deg (car pointed backwards). A pure offset would have predicted ~0; it
#      gave 180. That disproves "offset" and proves a HANDEDNESS INVERSION:
#      BeamNG's heading convention is the NEGATION of ours plus a 90 deg axis
#      offset (its vehicle local forward is +Y, ours is atan2's +X). Empirically
#      settled_dir ~= -sent_yaw - 90.
# So to make the car settle along a desired world heading theta, we send
#   sent_yaw = -theta - 90deg.
# This corrects the SENT quaternion only; it deliberately does NOT touch
# SPAWN_YAW_BIAS_DEG / _smoothed_forward_yaw, which feed the reward's alignment
# tangent (already correct -- velocity and tangent both live in world XY there,
# no frame issue). This is a spawn-encoding bug only, NOT a physics mirror: if
# steering were mirrored run1 could not have driven 87% of the track. The old
# belief that "the car settles to the correct heading within the first step" is
# disproven; the mis-encoded heading is what handicapped run1.
SPAWN_QUAT_YAW_CORRECTION_DEG = 90.0
RANDOM_SPEED_MAX_M_S = 30.0
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 25252


# ---- Centerline checkpoints ----
# The centerline comes from BeamNG's built-in DecalRoad geometry, extracted
# once by scripts/extract_centerline.py — the authoritative game data, no
# recording wobble or unclosed-loop artifacts. The old manual-recording
# import is kept here, commented, as a fallback reference: switch back if
# the built-in data ever proves wrong.
# from data.centerline_racetrack import CENTERLINE  # original manual recording
from data.centerline_racetrack_builtin import CENTERLINE
from envs.speed_profile import compute_speed_profile, V_MAX as PROFILE_V_MAX

# run16 paradigm reset: braking-aware target-speed profile, computed once over the
# centerline (index-aligned with CENTERLINE / _cum_arc). The reward rewards matching it
# (slow-in) and the obs exposes it; this is what makes BRAKING emerge instead of being
# scripted. See docs/mikey_run16_plan_learn_to_corner.md + envs/speed_profile.py.
V_TARGET_PROFILE, _PROFILE_R, _PROFILE_ARC, _PROFILE_TRACKLEN, _PROFILE_KAPPA = \
    compute_speed_profile(CENTERLINE)

# run16 reward weights (calibrated in the smoke per the W_OVER/W_PROG gate).
W_PROG = 1.0           # progress * alignment (cover the track, forward only)
W_OVER = 0.05          # over-speed penalty: -W_OVER * max(0, v - (v_target - offset))^2
W_SLIP = 0.15          # run19: 0.05->0.15 (3x). slip-angle penalty: -W_SLIP * max(0, |beta_deg| -
                       # BETA_SLIP_DEAD). run18 T1 trace: in the recoverable zone (beta 9-20deg) the
                       # old -0.08..-0.5/step was dwarfed by the throttle's ~+4/step (progress+match),
                       # so flooring at turn-in won 10x. 0.15 makes a building slide (beta 8->15) cost
                       # -0.15..-1.35, rivaling the match term so feathering beats flooring BEFORE the
                       # spin commits. Ladder: still spins -> 5-6x before the racing line; timid -> lower.
W_MATCH = 0.10         # run18 anti-timid nudge: +W_MATCH * min(v, v_target) * alignment.
                       # CAPPED at v_target (flat above -> adds NO incentive past target,
                       # so v* stays ~v_target) and alignment-gated (no reward for off-line
                       # speed). Rewards carrying speed UP TO target -> breaks the timid
                       # crawl the progress term alone couldn't (run17 crawled at 2.4 m/s).
OVER_SPEED_OFFSET = 0.0  # m/s; start the over-speed penalty this far BELOW v_target so the
                         # gradient is nonzero AT target (fixes the zero-gradient margin).
                         # 0 = off; raise in the smoke if reward-optimal speed runs hot.
BETA_SLIP_DEAD = 7.0   # run19: 9->7 deg. run18 T1 trace: the rear breaks loose at beta 5-8deg
                       # (throttle high, steering loaded) but the old 9deg deadband read EXACTLY 0
                       # through that recoverable window -- the penalty only engaged after the slide
                       # was committed. 7deg catches it at onset; clean tracking p90 ~5deg + clean-
                       # steering transients ~6-7deg stay ~free, so only the building slide pays.
CURV_PREVIEW_KAPPA_SCALE = 0.04  # 1/m; obs curvature-preview normalization (R~25m -> ~1.0)


# One BeamNG connection shared by both the training env and the eval env.
# They never run at the same time (EvalCallback pauses training while it
# evaluates), so sharing one game window and one vehicle is safe and means
# Mikey can watch the same car learn the whole time.
#
# The "initialized" flag is the idempotency guard for _connect(): once setup
# completes successfully, every subsequent reset() short-circuits past the
# launch + scenario + sensor-attach work. Sensors get attached exactly once
# per process; resets only teleport.
_shared: dict = {"bng": None, "vehicle": None, "initialized": False}


def _hf_energy_frac(stream) -> float:
    """run12 logging only: fraction of an action stream's spectral energy in the upper
    half of the frequency band (a high-frequency / chatter proxy). Trends DOWN as the
    Grad-CAPS regularizer removes zigzag; complements mean |Δaction|."""
    if len(stream) < 8:
        return 0.0
    x = np.asarray(stream, dtype=np.float64)
    x = x - x.mean()
    mag = np.abs(np.fft.rfft(x)) ** 2
    total = mag.sum()
    if total <= 0.0:
        return 0.0
    half = len(mag) // 2
    return float(mag[half:].sum() / total)


def _connect(home: Optional[str], host: str, port: int, launch: bool,
             headless: bool, nogpu: bool) -> None:
    """Open BeamNG, load our scenario, and put it in deterministic mode.

    Idempotent: once initialization succeeds, repeat calls do nothing.
    This is critical because train_env and eval_env each call _connect()
    on every reset, but we want the launch + scenario + sensor-attach
    work to happen exactly once per process. Re-attaching sensors raises
    BNGValueError ("duplicate sensor name").
    """
    if _shared["initialized"]:
        return
    # headless and nogpu both belong on the BeamNGpy constructor, not
    # bng.open(). headless appends "-headless" to the BeamNG.tech.x64
    # command line so no window appears; nogpu additionally appends
    # "-gfx null" to skip the rendering pipeline entirely (sensors that
    # require rendering — camera, lidar — go unavailable). BeamNGpy's own
    # constructor logic forces headless=True whenever nogpu=True. Both
    # options are only effective with launch=True.
    bng = BeamNGpy(host, port, home=home, headless=headless, nogpu=nogpu)
    bng.open(launch=launch)

    scenario = Scenario(MAP_NAME, "phase1_lap")
    vehicle = Vehicle(VEHICLE_ID, model=VEHICLE_MODEL,
                      part_config=VEHICLE_PART_CONFIG)
    # BeamNG already attaches a sensor named "state" by default on vehicle
    # spawn (undocumented). We use "agent_state" to avoid the collision.
    vehicle.sensors.attach("agent_state", State())
    vehicle.sensors.attach("electrics", Electrics())
    vehicle.sensors.attach("damage", Damage())
    initial_pos = (CENTERLINE[0][0], CENTERLINE[0][1],
                   CENTERLINE[0][2] + SPAWN_Z_OFFSET_M)
    scenario.add_vehicle(vehicle, pos=initial_pos, rot_quat=(0, 0, 0, 1))
    scenario.make(bng)

    bng.scenario.load(scenario)
    bng.scenario.start()

    # Deterministic stepping: BeamNG pauses between explicit `bng.step()`
    # calls. This is what lets us advance physics in 50 ms chunks without
    # racing the simulator's real-time clock.
    try:
        bng.settings.set_deterministic(DETERMINISTIC_STEPS_PER_S)
    except AttributeError:
        # Older BeamNGpy API — fall back to the legacy top-level call.
        bng.set_deterministic(DETERMINISTIC_STEPS_PER_S)

    _shared["bng"] = bng
    _shared["vehicle"] = vehicle
    _shared["initialized"] = True


class BeamNGRaceEnv(gymnasium.Env):
    """A plain Gymnasium env that drives an ETK 800 around West Coast USA."""

    metadata = {"render_modes": []}

    def __init__(self, random_spawn: bool, home: Optional[str] = None,
                 host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 launch: bool = False, headless: bool = False,
                 nogpu: bool = False, steer_rate: float = 0.0):
        super().__init__()
        # Connection params are stored here; we actually open BeamNG lazily
        # in reset(), so constructing the env never fails on its own.
        self.random_spawn = random_spawn
        self.home = home
        self.host = host
        self.port = port
        self.launch = launch
        self.headless = headless
        self.nogpu = nogpu
        # run13 steering slew-rate limit: symmetric cap on |Δsteer|/step (action[0]).
        # 0.0 = OFF (default), so the shared env file does not change behavior for any
        # run that does not explicitly enable it (protects the concurrently-running
        # run12 if it self-heals and re-imports this module). run13 sets 0.5.
        self.steer_rate = steer_rate

        # 18 = speed + heading_err + center_off + 6 lookahead x (dist,bearing) + run16's
        # [15] signed curvature preview, [16] v_target/V_MAX, [17] slip-angle (beta).
        # NOTE: shape change -> pre-run16 models unloadable against this env.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(18,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._last_centerline_dist = 0.0
        self._steps_since_progress = 0
        # Diagnostic fields surfaced via step()'s info dict. Initialized so
        # the first step's info has sane defaults before _compute_reward runs.
        self._last_raw_progress = 0.0
        self._last_raw_alignment = 1.0   # signed; 1.0 = "no data yet"
        self._last_final_reward = 0.0
        self._last_speed_reward = 0.0
        self._last_smoothness_penalty = 0.0
        self._last_slip = 0.0
        self._last_spin_penalty = 0.0
        self._last_heading_align = 1.0   # nose vs tangent (run5 heading gate)
        self._backward_steps = 0         # consecutive backward-facing steps
        self._last_throttle_smooth_penalty = 0.0
        self._center_off = 0.0           # set by _get_observation -> obs[2]
        # run7 instrumentation (pure logging, no reward-path effect): why the
        # episode ended, plus per-episode aggregates for diagnosing timidity
        # (mean speed), near-spins that were caught (recovered_count, min
        # heading), wheelspin frequency (max slip), and exact death location
        # (max arc) without re-deriving them from reward arithmetic.
        self._last_term_reason = "run"
        self._recovered_count = 0
        self._dip_steps = 0              # consecutive steps heading < -0.2
        self._ep_steps = 0
        self._speed_sum = 0.0
        self._max_arc = 0.0
        self._min_heading_align = 1.0
        self._max_slip = 0.0
        self._last_beta = 0.0        # run14/16: last-step slip angle (deg); feeds obs + slip penalty
        self._max_beta = 0.0         # telemetry: episode max slip angle
        self._beta_sum = 0.0         # telemetry: for episode-mean slip angle
        self._beta_list = []         # logging-only: per-step beta for episode p90
        self._progress_sum = 0.0     # logging-only: per-episode reward-term sums
        self._match_sum = 0.0
        self._over_pen_sum = 0.0
        self._slip_pen_sum = 0.0
        self._over_speed_sum = 0.0   # run16 telemetry: sum of max(0, v - v_target) per step
        self._over_speed_steps = 0   # run18 telemetry: count of steps over v_target
        self._steer_stream = []      # run12 logging: per-episode applied steer/throttle
        self._thr_stream = []
        self._rl_prev_steer = 0.0    # run13: last APPLIED steer (rate-limit ramps from here)
        self._steer_clip_steps = 0   # run13: episode count of steps the steer-rate clipped
        # Steering smoothness tracker (run2): _cur_steer is this tick's steer,
        # _prev_steer last tick's; their difference drives smoothness_penalty.
        # run6 adds the same for throttle/brake (action[1]).
        self._prev_steer = 0.0
        self._cur_steer = 0.0
        self._prev_throttle = 0.0
        self._cur_throttle = 0.0

        # Cumulative arc length from CENTERLINE[0] to each index, plus the
        # full-loop track length. The monotonic progress tracker uses these.
        n = len(CENTERLINE)
        self._cum_arc = [0.0] * n
        for i in range(1, n):
            self._cum_arc[i] = self._cum_arc[i - 1] + _dist(
                CENTERLINE[i - 1], CENTERLINE[i])
        self._track_length = self._cum_arc[n - 1] + _dist(
            CENTERLINE[n - 1], CENTERLINE[0])

        # Curriculum checkpoints: cumulative-distance thresholds at evenly
        # spaced fractions of the track (k=1..N_CHECKPOINTS-1 are intermediate;
        # the final fraction is the finish line / LAP_BONUS).
        self._checkpoint_distances = [
            k / N_CHECKPOINTS * self._track_length
            for k in range(1, N_CHECKPOINTS)
        ]
        self._checkpoints_hit: set = set()
        self._checkpoints_reached = 0

        # Monotonic progress-tracker state (re-initialized each reset()).
        self._progress_idx = 0    # car's current centerline index (windowed)
        self._laps = 0            # forward seam crossings; -1 = just behind start
        self._cur_centerline_dist = 0.0
        self._lap_done = False    # set True on the step a genuine lap completes

    def reset(self, seed=None, options=None):
        """Put the car at a spawn point and hand back the first observation.

        Spawn-selection contract:
        - If `options` is a dict containing "spawn_idx", that index is used
          verbatim (modulo centerline length for safety), with heading
          offset and start speed forced to 0 — fully deterministic spawn.
          This path is for eval/debug from a chosen position; it overrides
          both `random_spawn=True` and the deterministic idx=0 default.
        - Otherwise, behavior is the constructor-time default: random idx /
          heading / speed when `random_spawn=True`, else idx=0 from rest.
        """
        super().reset(seed=seed)
        _connect(self.home, self.host, self.port, self.launch,
                 self.headless, self.nogpu)

        forced_idx = (options.get("spawn_idx")
                      if isinstance(options, dict) else None)
        if forced_idx is not None:
            idx = int(forced_idx) % len(CENTERLINE)
            heading_offset = 0.0
            start_speed = 0.0
        elif self.random_spawn:
            idx = int(self.np_random.integers(0, len(CENTERLINE)))
            heading_offset = math.radians(float(self.np_random.uniform(
                -RANDOM_HEADING_DEG, RANDOM_HEADING_DEG)))
            # run17: cap the random start speed at the spawn point's braking-aware
            # target (never spawn ABOVE the corner's grip-limited speed -> not
            # pre-crashed). 0 .. v_target[idx].
            start_speed = float(self.np_random.uniform(0.0, V_TARGET_PROFILE[idx]))
        else:
            idx = 0
            heading_offset = 0.0
            start_speed = 0.0

        forward_yaw = self._smoothed_forward_yaw(idx)
        final_yaw = forward_yaw + heading_offset   # desired world heading
        # Invert BeamNG's heading convention so the car SETTLES at final_yaw:
        # measured settled_dir ~= -sent_yaw - 90, so sent_yaw = -final_yaw - 90.
        # (See SPAWN_QUAT_YAW_CORRECTION_DEG for the measurement.)
        sent_yaw = -final_yaw - math.radians(SPAWN_QUAT_YAW_CORRECTION_DEG)
        sent_quat = _yaw_to_quat(sent_yaw)

        # Hand-recompute the tangent so the print shows raw inputs (lets us
        # spot a unit-conversion or coordinate-frame mistake in the helper).
        prev_pt = CENTERLINE[(idx - 1) % len(CENTERLINE)]
        curr_pt = CENTERLINE[idx]
        next_pt = CENTERLINE[(idx + 1) % len(CENTERLINE)]
        hand_dx = next_pt[0] - prev_pt[0]
        hand_dy = next_pt[1] - prev_pt[1]
        hand_atan2 = math.atan2(hand_dy, hand_dx)

        self._teleport_to(idx, sent_quat, start_speed)
        _shared["vehicle"].sensors.poll()

        # Spawn diagnostic: the track tangent (intended down-track heading), the
        # frame correction we now apply (sent = -heading - 90), and the
        # resulting sent yaw. We
        # do NOT read orientation back here -- sensors.poll() fires before the
        # teleport settles, so a read-back is one frame stale. The settled pose
        # is instead measured out-of-band by scripts/spawn_angle_test.py, which
        # is what proved the 90-deg offset this correction cancels. Do not trust
        # an in-reset read-back for spawn validation; trust the offline test and
        # the G14 visual check.
        print(
            f"[reset] spawn idx={idx}, "
            f"desired heading={math.degrees(final_yaw):+.1f}° "
            f"-> sent yaw={math.degrees(sent_yaw):+.1f}° "
            f"(=-heading-{SPAWN_QUAT_YAW_CORRECTION_DEG:.0f}°, frame fix)\n"
            f"[reset]   hand tangent: "
            f"prev=({prev_pt[0]:.2f}, {prev_pt[1]:.2f}, {prev_pt[2]:.2f}) "
            f"curr=({curr_pt[0]:.2f}, {curr_pt[1]:.2f}, {curr_pt[2]:.2f}) "
            f"next=({next_pt[0]:.2f}, {next_pt[1]:.2f}, {next_pt[2]:.2f}), "
            f"dx={hand_dx:+.3f} dy={hand_dy:+.3f}, "
            f"atan2={math.degrees(hand_atan2):+.2f}°",
            flush=True,
        )

        # Initialize the progress tracker at the spawn index, zero laps. d starts
        # at the spawn's absolute arc position (0 for the fixed-start idx 0).
        self._progress_idx = idx
        self._laps = 0
        self._lap_done = False
        self._cur_centerline_dist = self._cum_arc[idx]
        self._last_centerline_dist = self._cur_centerline_dist
        self._steps_since_progress = 0
        self._checkpoints_hit = set()
        # run17: on a mid-track spawn, mark checkpoints already BEHIND the spawn arc as
        # hit so they don't all instant-award on step 0 (d>=thresh for everything passed).
        # Only checkpoints AHEAD of the spawn earn as the car actually drives forward.
        for _k, _thresh in enumerate(self._checkpoint_distances):
            if _thresh <= self._cur_centerline_dist:
                self._checkpoints_hit.add(_k)
        self._checkpoints_reached = 0
        # No prior action at spawn, so the first step has zero steering/throttle
        # change.
        self._prev_steer = 0.0
        self._cur_steer = 0.0
        self._prev_throttle = 0.0
        self._cur_throttle = 0.0
        # run5 heading gate: spawn faces down-track, no backward streak yet.
        self._last_heading_align = 1.0
        self._backward_steps = 0
        # run7 instrumentation: fresh episode aggregates.
        self._last_term_reason = "run"
        self._recovered_count = 0
        self._dip_steps = 0
        self._ep_steps = 0
        self._speed_sum = 0.0
        self._max_arc = 0.0
        self._min_heading_align = 1.0
        self._max_slip = 0.0
        self._last_beta = 0.0        # run14/16: last-step slip angle (deg); feeds obs + slip penalty
        self._max_beta = 0.0         # telemetry: episode max slip angle
        self._beta_sum = 0.0         # telemetry: for episode-mean slip angle
        self._beta_list = []         # logging-only: per-step beta for episode p90
        self._progress_sum = 0.0     # logging-only: per-episode reward-term sums
        self._match_sum = 0.0
        self._over_pen_sum = 0.0
        self._slip_pen_sum = 0.0
        self._over_speed_sum = 0.0   # run16 telemetry: sum of max(0, v - v_target) per step
        self._over_speed_steps = 0   # run18 telemetry: count of steps over v_target
        self._steer_stream = []      # run12 logging: per-episode applied steer/throttle
        self._thr_stream = []
        self._rl_prev_steer = 0.0    # run13: last APPLIED steer (rate-limit ramps from here)
        self._steer_clip_steps = 0   # run13: episode count of steps the steer-rate clipped
        return self._get_observation(), {}

    def step(self, action):
        """Push the AI's buttons, advance 50 ms of car-time, then look around."""
        # action[0] -> steering in [-1, 1]
        # action[1] -> throttle (>0) or brake (<0)
        requested_steer = float(np.clip(action[0], -1.0, 1.0))
        # run16: the ONE retained scripted constraint -- a FLAT steering slew-rate limit
        # (steer_rate=0.5). Caps |Δsteer|/step symmetrically so a full-lock slam ramps over
        # several steps. The run15 speed-scaling is gone (the reward now teaches the speed/
        # line that the speed-taper was patching). steer_rate=0 -> OFF.
        if self.steer_rate > 0.0:
            req_delta = requested_steer - self._rl_prev_steer
            delta = max(-self.steer_rate, min(self.steer_rate, req_delta))
            steer = self._rl_prev_steer + delta
            if abs(req_delta) > self.steer_rate:
                self._steer_clip_steps += 1
        else:
            steer = requested_steer
        self._rl_prev_steer = steer
        self._cur_steer = steer   # APPLIED steer; read by control
        thr = float(np.clip(action[1], -1.0, 1.0))
        self._cur_throttle = thr
        self._steer_stream.append(steer)   # fluctuation logging (APPLIED steer)
        self._thr_stream.append(thr)
        throttle = max(0.0, thr)
        brake = max(0.0, -thr)
        # run16: NO throttle scripting. run11 TC + run14 ESC are dropped -- the policy now
        # learns to brake/not-spin via the speed-target reward + slip-angle penalty, instead
        # of having the throttle capped (which masked the consequence it must learn from).
        _shared["vehicle"].control(steering=steer, throttle=throttle, brake=brake)

        # Advance 3 physics steps (~50 ms at 60 Hz).
        # TODO(v2 rtgym migration): replace this with rtgym's elastic
        # real-time clock once Tuple-obs handling is sorted out. See
        # docs/references.md (TMRL) for the precedent. Plain bng.step() is
        # fine for v1 — it runs faster than real-time, which means faster
        # training; we just lose the wall-clock alignment a human would have.
        _shared["bng"].step(PHYSICS_STEPS_PER_STEP)

        _shared["vehicle"].sensors.poll()
        # Advance the monotonic progress tracker FIRST so the observation, the
        # reward, and the done-check all read the same updated forward progress.
        self._advance_progress(_shared["vehicle"].sensors["agent_state"]["pos"])
        obs = self._get_observation()
        reward = self._compute_reward()
        terminated, term_bonus = self._check_done()
        truncated = False
        # Diagnostics: raw_progress is the unsigned centerline-distance
        # delta this tick; alignment is the SIGNED dot product (-1..+1, not
        # clamped) so reverse driving shows as negative; final_reward is
        # the post-gate reward BEFORE the terminal bonus is added.
        info = {
            "raw_progress": float(self._last_raw_progress),
            "alignment": float(self._last_raw_alignment),
            "final_reward": float(self._last_final_reward),
            "speed_reward": float(self._last_speed_reward),
            "smoothness_penalty": float(self._last_smoothness_penalty),
            "throttle_smooth_penalty": float(self._last_throttle_smooth_penalty),
            "slip": float(self._last_slip),
            "spin_penalty": float(self._last_spin_penalty),
            "heading_align": float(self._last_heading_align),
            "backward_steps": int(self._backward_steps),
            "checkpoints_reached": int(self._checkpoints_reached),
            "lap_completed": bool(self._lap_done),
            # run7 instrumentation (Monitor logs the final step's values).
            "termination_reason": str(self._last_term_reason),
            "recovered_count": int(self._recovered_count),
            "mean_speed": float(self._speed_sum / max(self._ep_steps, 1)),
            "max_arc": float(self._max_arc),
            "min_heading_align": float(self._min_heading_align),
            "max_slip": float(self._max_slip),
            "steer_clip_frac": float(self._steer_clip_steps / max(self._ep_steps, 1)),
            # run16 learn-to-corner telemetry: slip-angle (does it slide?), the mean
            # over-speed vs the target profile (does it brake?), and v_target at episode
            # end. The watch: over_speed_mean -> ~0 (braking emerged) and beta low.
            "beta_max": float(self._max_beta),
            "beta_mean": float(self._beta_sum / max(self._ep_steps, 1)),
            "beta_p90": float(np.percentile(self._beta_list, 90)) if self._beta_list else 0.0,
            # logging-only reward-term decomposition (per-episode sums):
            "r_progress": float(self._progress_sum),
            "r_match": float(self._match_sum),
            "r_overspeed": float(self._over_pen_sum),
            "r_slip": float(self._slip_pen_sum),
            "over_speed_mean": float(self._over_speed_sum / max(self._ep_steps, 1)),
            "over_speed_frac": float(self._over_speed_steps / max(self._ep_steps, 1)),
            "v_target_here": float(V_TARGET_PROFILE[self._progress_idx]),
        }
        return obs, reward + term_bonus, terminated, truncated, info

    def close(self):
        """Leave the shared BeamNG connection open — other envs may still need it."""
        # We intentionally don't tear down `_shared` here. The single BeamNG
        # process is shared by train_env and eval_env, and Python exit will
        # release the socket. Tearing down mid-run would kill the other env.
        pass

    # ---- Helpers ----

    def _teleport_to(self, idx: int, quat: tuple, speed: float):
        """Move the car onto a centerline point with the given orientation quat."""
        cx, cy, cz = CENTERLINE[idx]
        pos = (cx, cy, cz + SPAWN_Z_OFFSET_M)
        v = _shared["vehicle"]
        v.teleport(pos, rot_quat=quat, reset=True)
        if speed > 0:
            try:
                v.set_velocity(speed)
            except Exception:
                # set_velocity is best-effort; if BeamNGpy version doesn't
                # support it on a non-shifted vehicle, we just start at rest.
                pass

    def _smoothed_forward_yaw(self, idx: int) -> float:
        """Local tangent yaw at idx: direction from CENTERLINE[idx-1] to [idx+1].

        Using a forward chord (idx → idx+N) cuts across curves on a curved
        section, pointing the car off the road. The local tangent (prev →
        next) follows the track's actual direction at each point. Wraps
        around the closed loop at both ends.

        SPAWN_YAW_BIAS_DEG is added to the raw atan2 — defaults to 0°
        (no-op); reserved as a single tuning knob in case a systematic
        tangent-vs-visual bias is identified later.
        """
        prev_pt = CENTERLINE[(idx - 1) % len(CENTERLINE)]
        next_pt = CENTERLINE[(idx + 1) % len(CENTERLINE)]
        raw = math.atan2(next_pt[1] - prev_pt[1], next_pt[0] - prev_pt[0])
        return raw + math.radians(SPAWN_YAW_BIAS_DEG)

    def _point_at_arc(self, arc: float):
        """The (x, y, z) point at arc-position `arc` along the centerline loop.

        Binary-searches the precomputed cumulative-arc array, then LINEARLY
        INTERPOLATES between the two bracketing centerline points. Interpolation
        (not snap-to-nearest) matters: segment spacing reaches 27.8 m on the
        sparse straights, so snapping would put up to ~14 m of error on the 10 m
        near point. Wraps the start/finish seam via modulo (read-only on the
        cum-arc machinery; cannot disturb lap counting).
        """
        n = len(CENTERLINE)
        arc = arc % self._track_length
        if arc >= self._cum_arc[n - 1]:
            # Closing segment: last point -> first point across the seam.
            a, b = CENTERLINE[n - 1], CENTERLINE[0]
            seg = self._track_length - self._cum_arc[n - 1]
            frac = (arc - self._cum_arc[n - 1]) / seg if seg > 1e-9 else 0.0
        else:
            i = bisect.bisect_right(self._cum_arc, arc) - 1
            a, b = CENTERLINE[i], CENTERLINE[i + 1]
            seg = self._cum_arc[i + 1] - self._cum_arc[i]
            frac = (arc - self._cum_arc[i]) / seg if seg > 1e-9 else 0.0
        return (a[0] + frac * (b[0] - a[0]),
                a[1] + frac * (b[1] - a[1]),
                a[2] + frac * (b[2] - a[2]))

    def _get_observation(self) -> np.ndarray:
        """Pack the 15 numbers the AI sees this tick."""
        s = _shared["vehicle"].sensors["agent_state"]
        pos = s["pos"]
        vel = s["vel"]
        forward = s.get("dir", (1.0, 0.0, 0.0))
        speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)

        # Anchor at the car's TRUE arc position (cum_arc[idx] + projection along
        # the current segment), NOT cum_arc[idx] alone: segments reach 27.8 m on
        # the sparse straights, so anchoring at the index could resolve the 10 m
        # near point BEHIND the car. _cur_centerline_dist already carries the
        # exact value (laps absorbed by the modulo). Read-only on the seam-safe
        # cum-arc machinery -- consistent with the progress the car is rewarded
        # for, never a global nearest search.
        anchor = self._cur_centerline_dist % self._track_length
        look_pts = [self._point_at_arc(anchor + d_ahead)
                    for d_ahead in LOOKAHEAD_DISTANCES_M]

        # heading_err aims at the 10 m near point (pure-pursuit style aim).
        # center_off stays on the LOCAL idx->idx+1 segment: it is a lateral
        # placement signal at the car's position, and a 10 m chord would pick up
        # ~0.8 m of false offset (sagitta) in the R15 hairpin.
        idx = self._progress_idx
        c_curr = CENTERLINE[idx]
        c_next = CENTERLINE[(idx + 1) % len(CENTERLINE)]
        # run10 speed-scaled steering reference: aim L_d ahead, L_d growing with
        # speed so the preview time stays ~PREVIEW_TIME instead of collapsing at
        # speed (the fixed-10m point was the pure-pursuit hunting setup). Floors at
        # PREVIEW_L_MIN (== old 10m point, so launch is unchanged), caps at
        # PREVIEW_L_MAX. Interpolated on _cum_arc (pose-independent geometry).
        speed_h = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
        l_d = min(PREVIEW_L_MAX, max(PREVIEW_L_MIN, PREVIEW_TIME * speed_h))
        ref_point = self._point_at_arc(anchor + l_d)
        heading_err = _bearing_to(forward, pos, ref_point)
        center_off = _perp_distance(pos, c_curr, c_next)
        self._center_off = center_off   # lateral placement signal -> obs[2]

        vals = [
            speed / MAX_SPEED_M_S,
            heading_err / math.pi,
            np.clip(center_off / CENTER_OFFSET_CLIP_M, -1.0, 1.0),
        ]
        for p in look_pts:
            vals.append(_dist(pos, p) / MAX_LOOKAHEAD_DIST_M)
            vals.append(_bearing_to(forward, pos, p) / math.pi)
        # run16 additions (the learn-to-corner signals):
        # [15] signed curvature preview at the speed-scaled lookahead — how sharp / which way.
        prev_arc = (anchor + l_d) % self._track_length
        j = max(0, min(bisect.bisect_right(self._cum_arc, prev_arc) - 1, len(_PROFILE_KAPPA) - 1))
        vals.append(float(np.clip(_PROFILE_KAPPA[j] / CURV_PREVIEW_KAPPA_SCALE, -1.0, 1.0)))
        # [16] braking-aware target speed at the car — the number the reward rewards matching.
        vals.append(float(np.clip(V_TARGET_PROFILE[idx] / PROFILE_V_MAX, -1.0, 1.0)))
        # [17] slip-angle beta (last step, one-step lag) — "are you sliding".
        vals.append(float(np.clip(self._last_beta / 45.0, -1.0, 1.0)))
        obs = np.array(vals, dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _compute_reward(self) -> float:
        """Progress along the racing line, gated by whether the car faces forward.

        Raw progress is just centerline distance moved since last step. That
        accidentally rewards driving *backward* along the line, since the
        position projects onto the same arc. We multiply by an alignment
        factor — dot product of the car's horizontal velocity with the local
        smoothed forward direction — so:
          - moving forward at speed   -> alignment ~ 1.0, full reward
          - drifting/sliding sideways -> alignment < 1.0, reduced reward
          - moving backward           -> alignment <= 0, zero reward (clamped)
          - nearly stopped            -> alignment forced to 1.0 so a spun-out
                                         car isn't punished while recovering.
        """
        d = self._cur_centerline_dist
        raw_progress = d - self._last_centerline_dist
        self._last_centerline_dist = d

        s = _shared["vehicle"].sensors["agent_state"]
        vel = s["vel"]
        speed_horizontal = math.sqrt(vel[0] ** 2 + vel[1] ** 2)

        # Track tangent at the car's current centerline index, used by BOTH the
        # velocity alignment and the run5 heading gate.
        forward_yaw = self._smoothed_forward_yaw(self._progress_idx)
        fx, fy = math.cos(forward_yaw), math.sin(forward_yaw)

        if speed_horizontal < STOPPED_SPEED_M_S:
            # No directional signal when essentially stopped — pin alignment
            # at 1.0 so a spun-out car can recover without losing reward.
            raw_alignment = 1.0
        else:
            vx, vy = vel[0] / speed_horizontal, vel[1] / speed_horizontal
            raw_alignment = vx * fx + vy * fy   # signed, -1 to +1

        # run5 HEADING gate: where the NOSE points (State["dir"]), not where it
        # moves. heading_align = dir.tangent (parallel to velocity_align). dir is
        # valid at any speed (no stopped-pin), so a slow backward-facing car is
        # still caught -- closing the heading-blind leak (a spun car sliding
        # down-track used to earn full reward).
        fdir = s.get("dir", (fx, fy, 0.0))
        hn = math.sqrt(fdir[0] ** 2 + fdir[1] ** 2)
        heading_align = (fdir[0] * fx + fdir[1] * fy) / hn if hn > 1e-6 else 1.0
        self._last_heading_align = heading_align
        # run14 ESC: lateral slip angle beta (deg) = angle between the velocity
        # vector and the nose (dir). This is the axis TC never watched -- the rear
        # slides sideways while longitudinal wheelspin stays low. Computed here,
        # before the kill-switch return, so BOTH paths leave a fresh value for next
        # step's throttle gate (one-step lag, like TC). Pinned to 0 below the ESC
        # speed floor (noise at low speed; the launch needs throttle) and at standstill.
        if speed_horizontal < ESC_MIN_SPEED_M_S or hn < 1e-6:
            self._last_beta = 0.0
        else:
            cos_b = (vel[0] * fdir[0] + vel[1] * fdir[1]) / (speed_horizontal * hn)
            self._last_beta = math.degrees(math.acos(max(-1.0, min(1.0, cos_b))))
        self._max_beta = max(self._max_beta, self._last_beta)
        self._beta_sum += self._last_beta
        self._beta_list.append(self._last_beta)   # logging-only: for per-episode beta p90
        if heading_align < HEADING_KILL_THRESHOLD:
            self._backward_steps += 1
        else:
            self._backward_steps = 0

        # run7 instrumentation (logging only). A "recovery" = a genuine dip
        # (>= 3 consecutive steps below the kill threshold) that ends with the
        # nose back above ZERO -- the hysteresis stops boundary wobble at -0.2
        # from inflating the count. Plus per-episode aggregates: mean speed
        # (timidity trend), max arc (exact death location), min heading, and
        # max slip (wheelspin frequency), all readable from the monitor CSV.
        if heading_align < HEADING_KILL_THRESHOLD:
            self._dip_steps += 1
        elif heading_align >= 0.0:
            if self._dip_steps >= 3:
                self._recovered_count += 1
            self._dip_steps = 0
        self._ep_steps += 1
        self._speed_sum += speed_horizontal
        # Furthest forward arc distance reached (death-location proxy). Use the
        # raw cumulative distance, NOT % track_length: a backward drift across
        # the start seam makes _cur_centerline_dist slightly negative (laps=-1),
        # which the running max() ignores -- whereas the modulo would wrap it to
        # ~4361 and falsely read as a completed lap.
        self._max_arc = max(self._max_arc, self._cur_centerline_dist)
        self._min_heading_align = min(self._min_heading_align, heading_align)

        # KILL-SWITCH: nose more than ~90deg off down-track -> ZERO the entire
        # step reward (progress, speed, AND checkpoint/lap bonuses). The early
        # return also means the checkpoint loop below never runs while backward,
        # so the checkpoint bonus is heading-gated at the source.
        if heading_align < HEADING_KILL_THRESHOLD:
            self._prev_steer = self._cur_steer
            self._prev_throttle = self._cur_throttle
            wheelspeed = float(_shared["vehicle"].sensors["electrics"].get(
                "wheelspeed", speed_horizontal))
            self._last_raw_progress = raw_progress
            self._last_raw_alignment = raw_alignment
            self._last_final_reward = 0.0
            self._last_speed_reward = 0.0
            self._last_smoothness_penalty = 0.0
            self._last_throttle_smooth_penalty = 0.0
            self._last_slip = wheelspeed - speed_horizontal
            self._max_slip = max(self._max_slip, self._last_slip)
            self._last_spin_penalty = 0.0
            self._steps_since_progress += 1
            return 0.0

        # Gate: clamp negative to zero so backward driving earns no reward.
        gated_alignment = max(0.0, raw_alignment)
        # run16: progress is the cover-the-track core (kept). The speed-EVERYWHERE term,
        # the steering/throttle smoothness penalties, and the wheelspin spin_penalty are
        # all DROPPED -- braking/cornering are LEARNED via the two terms below.
        final_reward = W_PROG * raw_progress * gated_alignment

        # THE BRAKE SIGNAL: penalize exceeding the braking-aware target speed at the car's
        # position (squared: gentle near target, sharp when hot). OVER_SPEED_OFFSET shifts
        # the knee below v_target so the gradient is nonzero AT target (calibration knob).
        v_target = float(V_TARGET_PROFILE[self._progress_idx])
        over = max(0.0, speed_horizontal - (v_target - OVER_SPEED_OFFSET))
        over_speed_penalty = -W_OVER * over * over
        self._over_speed_sum += over
        if over > 0.01:
            self._over_speed_steps += 1

        # run18 anti-timid nudge: reward carrying speed UP TO the target (capped at
        # v_target so it adds no incentive to exceed it; alignment-gated so off-line
        # speed earns nothing). This is what breaks the timid crawl.
        match_reward = W_MATCH * min(speed_horizontal, v_target) * gated_alignment

        # Slip-angle penalty (replaces ESC + the wheelspin spin_penalty): a reward signal
        # to not slide, so the policy LEARNS grip instead of having throttle cut. beta deg.
        slip_penalty = -W_SLIP * max(0.0, self._last_beta - BETA_SLIP_DEAD)

        # logging-only: per-episode sums of each reward term so TB can show real progress
        # apart from the match term ("paid to crawl"). Behavior-neutral.
        self._progress_sum += final_reward
        self._match_sum += match_reward
        self._over_pen_sum += over_speed_penalty
        self._slip_pen_sum += slip_penalty

        # wheelspin 'slip' kept for max_slip telemetry only (no longer penalized; TC gone)
        wheelspeed = float(_shared["vehicle"].sensors["electrics"].get(
            "wheelspeed", speed_horizontal))
        slip = wheelspeed - speed_horizontal
        self._max_slip = max(self._max_slip, slip)
        self._prev_steer = self._cur_steer
        self._prev_throttle = self._cur_throttle

        self._last_raw_progress = raw_progress
        self._last_raw_alignment = raw_alignment
        self._last_final_reward = final_reward
        self._last_speed_reward = over_speed_penalty       # diag slot now = over-speed penalty
        self._last_smoothness_penalty = 0.0
        self._last_throttle_smooth_penalty = 0.0
        self._last_slip = slip
        self._last_spin_penalty = slip_penalty             # diag slot now = slip-angle penalty

        if final_reward > 0.01:
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

        # Checkpoint bonuses: the first time this episode cumulative distance d
        # crosses each evenly-spaced checkpoint, award CHECKPOINT_BONUS once.
        # _checkpoints_hit is cleared each reset, so wiggling back and forth
        # across a line can't farm it. This sits ON TOP of the progress reward;
        # the stuck counter above stays keyed on progress, not these spikes.
        checkpoint_bonus = 0.0
        for k, thresh in enumerate(self._checkpoint_distances):
            if k not in self._checkpoints_hit and d >= thresh:
                self._checkpoints_hit.add(k)
                self._checkpoints_reached += 1
                checkpoint_bonus += CHECKPOINT_BONUS
        # run16 total: progress + checkpoints − over-speed − slip-angle. The over-speed
        # term is the brake signal; progress rewards covering ground at/below v_target.
        return float(final_reward + checkpoint_bonus + over_speed_penalty + slip_penalty
                     + match_reward)

    def _check_done(self) -> Tuple[bool, float]:
        """Did the episode end? If so, what bonus or penalty applies?

        Also records the termination reason (run7 instrumentation) so the
        monitor CSV logs WHY each episode ended instead of leaving it to be
        inferred from final-step heading_align.
        """
        if self._is_flipped():
            self._last_term_reason = "flip"
            return True, FLIP_PENALTY
        if self._is_off_track():
            self._last_term_reason = "off_track"
            return True, OFF_TRACK_PENALTY
        # Lap bonus only counts when forward-facing (run5): _lap_done is position-
        # based (seam crossing) and heading-blind, so gate it so a backward slide
        # across the seam can never collect the +100.
        if self._lap_done and self._last_heading_align >= HEADING_KILL_THRESHOLD:
            self._last_term_reason = "lap"
            return True, LAP_BONUS
        # Backward-facing termination (run5 rule, run7 price): a sustained
        # spin-out (nose >~90deg off down-track for BACKWARD_TERM_STEPS in a
        # row) ends the episode at BACKWARD_TERM_PENALTY. run6 left this door
        # free (0.0) and the policy converged onto it; see the constant's
        # comment for the economics.
        if self._backward_steps >= BACKWARD_TERM_STEPS:
            self._last_term_reason = "backward"
            return True, BACKWARD_TERM_PENALTY
        if self._steps_since_progress > STUCK_STEPS_THRESHOLD:
            self._last_term_reason = "stuck"
            return True, STUCK_PENALTY
        self._last_term_reason = "run"
        return False, 0.0

    def _advance_progress(self, pos) -> None:
        """Update the monotonic progress tracker from the car's position.

        Re-find the car's centerline index by searching only a LOCAL window
        [-BACK, +FWD] around the last index -- never globally -- so the index
        cannot flip across the start/finish seam (idx 0 <-> last index, ~0.67 m
        apart) and make distance jump by a full lap. Cumulative distance is
        d = laps * track_length + cum_arc[idx] + projection; the laps term
        absorbs the seam wrap so d stays continuous (raw_progress is always
        ~the real per-step travel, never a +/-track-length spike or cliff).
        """
        n = len(CENTERLINE)
        best_o, best_d2 = 0, None
        for o in range(-PROGRESS_WINDOW_BACK, PROGRESS_WINDOW_FWD + 1):
            cand = (self._progress_idx + o) % n
            d2 = _dist(pos, CENTERLINE[cand])
            if best_d2 is None or d2 < best_d2:
                best_d2, best_o = d2, o

        raw_target = self._progress_idx + best_o
        crossed_forward = raw_target >= n
        if crossed_forward:
            self._laps += 1
        elif raw_target < 0:
            self._laps -= 1

        # A genuine lap = a forward seam crossing that leaves laps >= 1. The
        # -1 -> 0 case (undoing a backward drift across the start) is also a
        # forward crossing but leaves laps == 0, so it does NOT count -- this is
        # what stops start-line wiggling from farming LAP_BONUS. Reaching laps
        # >= 1 requires progress_idx near the last index with laps 0, which only
        # happens after a genuine full forward traversal (cum_arc ~ track_len).
        self._lap_done = crossed_forward and self._laps >= 1

        self._progress_idx = raw_target % n
        nxt = (self._progress_idx + 1) % n
        seg_len = _dist(CENTERLINE[self._progress_idx], CENTERLINE[nxt])
        proj = _project_along(pos, CENTERLINE[self._progress_idx],
                              CENTERLINE[nxt])
        proj = max(0.0, min(proj, seg_len))
        self._cur_centerline_dist = (self._laps * self._track_length
                                     + self._cum_arc[self._progress_idx] + proj)

    def _closest_checkpoint_idx(self, pos) -> int:
        return int(np.argmin([_dist(pos, c) for c in CENTERLINE]))

    def _is_flipped(self) -> bool:
        """The car is on its roof (or close to it)."""
        up = _shared["vehicle"].sensors["agent_state"].get("up", (0.0, 0.0, 1.0))
        return up[2] < 0.5

    def _is_off_track(self) -> bool:
        """The car wandered too far from the racing line."""
        pos = _shared["vehicle"].sensors["agent_state"]["pos"]
        idx = self._closest_checkpoint_idx(pos)
        return _dist(pos, CENTERLINE[idx]) > OFF_TRACK_THRESHOLD_M


# ---- Small math helpers ----

def _dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                     + (a[2] - b[2]) ** 2)


def _bearing_to(forward, pos, target) -> float:
    """Signed angle (radians) between forward heading and direction to target."""
    target_yaw = math.atan2(target[1] - pos[1], target[0] - pos[0])
    fwd_yaw = math.atan2(forward[1], forward[0])
    diff = target_yaw - fwd_yaw
    return (diff + math.pi) % (2 * math.pi) - math.pi


def _perp_distance(pos, a, b) -> float:
    """Signed perpendicular distance from pos to the segment a->b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    nrm = math.sqrt(dx * dx + dy * dy)
    if nrm < 1e-6:
        return 0.0
    px, py = pos[0] - a[0], pos[1] - a[1]
    return (px * dy - py * dx) / nrm


def _project_along(pos, a, b) -> float:
    """Projection of (pos - a) onto (b - a), as a scalar length."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    nrm = math.sqrt(dx * dx + dy * dy)
    if nrm < 1e-6:
        return 0.0
    px, py = pos[0] - a[0], pos[1] - a[1]
    return (px * dx + py * dy) / nrm


def _yaw_to_quat(yaw: float):
    """Rotation quaternion (x, y, z, w) for a pure yaw rotation, in radians.

    Matches BeamNGpy's own `angle_to_quat([0, 0, yaw_deg])` for the
    yaw-only case — same standard math convention, no sign flip.
    """
    half = yaw / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


# ---- Factory: kept for symmetry with the old API, now a trivial wrapper ----

def make_beamng_env(random_spawn: bool, home: Optional[str] = None,
                    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                    launch: bool = False, headless: bool = False,
                    nogpu: bool = False, steer_rate: float = 0.0):
    """Return a fresh BeamNGRaceEnv. No rtgym wrapping in v1."""
    return BeamNGRaceEnv(
        random_spawn=random_spawn, home=home, host=host, port=port,
        launch=launch, headless=headless, nogpu=nogpu, steer_rate=steer_rate,
    )
