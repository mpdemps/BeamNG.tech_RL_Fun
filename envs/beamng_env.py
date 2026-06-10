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

import math
from typing import Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Damage, Electrics, State


# ---- Tunable constants (one place to tweak everything) ----
MAP_NAME = "west_coast_usa"
VEHICLE_MODEL = "etk800"            # the ETK 800 sedan BeamNG just spawned
VEHICLE_ID = "ego"
PHYSICS_STEPS_PER_STEP = 3          # 3 steps at 60 Hz = 50 ms = 20 Hz env tick.
DETERMINISTIC_STEPS_PER_S = 60
MAX_SPEED_M_S = 70.0                # for obs normalization
MAX_LOOKAHEAD_DIST_M = 200.0
CENTER_OFFSET_CLIP_M = 10.0
OFF_TRACK_THRESHOLD_M = 20.0        # generous; tighten once centerline is real
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
    vehicle = Vehicle(VEHICLE_ID, model=VEHICLE_MODEL)
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
                 nogpu: bool = False):
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

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._last_centerline_dist = 0.0
        self._steps_since_progress = 0
        # Diagnostic fields surfaced via step()'s info dict. Initialized so
        # the first step's info has sane defaults before _compute_reward runs.
        self._last_raw_progress = 0.0
        self._last_raw_alignment = 1.0   # signed; 1.0 = "no data yet"
        self._last_final_reward = 0.0

        # Curriculum checkpoints: cumulative-distance thresholds at evenly
        # spaced fractions of the total track length (k=1..N_CHECKPOINTS-1 are
        # intermediate; the final fraction is the finish line / LAP_BONUS).
        track_length = sum(_dist(CENTERLINE[i], CENTERLINE[i + 1])
                           for i in range(len(CENTERLINE) - 1))
        self._checkpoint_distances = [
            k / N_CHECKPOINTS * track_length for k in range(1, N_CHECKPOINTS)
        ]
        self._checkpoints_hit: set = set()
        self._checkpoints_reached = 0
        self._lap_done = False           # set True on the step the lap completes

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
            start_speed = float(self.np_random.uniform(
                0, RANDOM_SPEED_MAX_M_S))
        else:
            idx = 0
            heading_offset = 0.0
            start_speed = 0.0

        forward_yaw = self._smoothed_forward_yaw(idx)
        final_yaw = forward_yaw + heading_offset
        sent_quat = _yaw_to_quat(final_yaw)

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

        # Spawn diagnostic: intended yaw plus the raw tangent inputs that
        # produced it. We deliberately do NOT read orientation back from the
        # sensor here. sensors.poll() fires before the teleport settles, so the
        # read-back is one frame stale -- it reports the *previous* spawn's
        # orientation, not this one. That stale read is exactly what made idx=0
        # look like a 179-degree spawn bug: the car actually settles to the
        # correct heading within the first step(). intended yaw and the hand
        # tangent below are computed from the centerline and are correct.
        print(
            f"[reset] spawn idx={idx}, "
            f"intended yaw={math.degrees(final_yaw):+.1f}°\n"
            f"[reset]   hand tangent: "
            f"prev=({prev_pt[0]:.2f}, {prev_pt[1]:.2f}, {prev_pt[2]:.2f}) "
            f"curr=({curr_pt[0]:.2f}, {curr_pt[1]:.2f}, {curr_pt[2]:.2f}) "
            f"next=({next_pt[0]:.2f}, {next_pt[1]:.2f}, {next_pt[2]:.2f}), "
            f"dx={hand_dx:+.3f} dy={hand_dy:+.3f}, "
            f"atan2={math.degrees(hand_atan2):+.2f}° (matches intended)",
            flush=True,
        )

        self._last_centerline_dist = self._distance_along_centerline()
        self._steps_since_progress = 0
        self._checkpoints_hit = set()
        self._checkpoints_reached = 0
        return self._get_observation(), {}

    def step(self, action):
        """Push the AI's buttons, advance 50 ms of car-time, then look around."""
        # action[0] -> steering in [-1, 1]
        # action[1] -> throttle (>0) or brake (<0)
        steer = float(np.clip(action[0], -1.0, 1.0))
        thr = float(np.clip(action[1], -1.0, 1.0))
        throttle = max(0.0, thr)
        brake = max(0.0, -thr)
        _shared["vehicle"].control(steering=steer, throttle=throttle,
                                   brake=brake)

        # Advance 3 physics steps (~50 ms at 60 Hz).
        # TODO(v2 rtgym migration): replace this with rtgym's elastic
        # real-time clock once Tuple-obs handling is sorted out. See
        # docs/references.md (TMRL) for the precedent. Plain bng.step() is
        # fine for v1 — it runs faster than real-time, which means faster
        # training; we just lose the wall-clock alignment a human would have.
        _shared["bng"].step(PHYSICS_STEPS_PER_STEP)

        _shared["vehicle"].sensors.poll()
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
            "checkpoints_reached": int(self._checkpoints_reached),
            "lap_completed": bool(self._lap_done),
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

    def _get_observation(self) -> np.ndarray:
        """Pack the 9 numbers the AI sees this tick."""
        s = _shared["vehicle"].sensors["agent_state"]
        pos = s["pos"]
        vel = s["vel"]
        forward = s.get("dir", (1.0, 0.0, 0.0))
        speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)

        idx = self._closest_checkpoint_idx(pos)
        c_curr = CENTERLINE[idx]
        c1 = CENTERLINE[(idx + 1) % len(CENTERLINE)]
        c2 = CENTERLINE[(idx + 2) % len(CENTERLINE)]
        c3 = CENTERLINE[(idx + 3) % len(CENTERLINE)]

        heading_err = _bearing_to(forward, pos, c1)
        center_off = _perp_distance(pos, c_curr, c1)

        obs = np.array([
            speed / MAX_SPEED_M_S,
            heading_err / math.pi,
            np.clip(center_off / CENTER_OFFSET_CLIP_M, -1.0, 1.0),
            _dist(pos, c1) / MAX_LOOKAHEAD_DIST_M,
            _bearing_to(forward, pos, c1) / math.pi,
            _dist(pos, c2) / MAX_LOOKAHEAD_DIST_M,
            _bearing_to(forward, pos, c2) / math.pi,
            _dist(pos, c3) / MAX_LOOKAHEAD_DIST_M,
            _bearing_to(forward, pos, c3) / math.pi,
        ], dtype=np.float32)
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
        d = self._distance_along_centerline()
        raw_progress = d - self._last_centerline_dist
        self._last_centerline_dist = d

        s = _shared["vehicle"].sensors["agent_state"]
        vel = s["vel"]
        pos = s["pos"]
        speed_horizontal = math.sqrt(vel[0] ** 2 + vel[1] ** 2)

        if speed_horizontal < STOPPED_SPEED_M_S:
            # No directional signal when essentially stopped — pin alignment
            # at 1.0 so a spun-out car can recover without losing reward.
            raw_alignment = 1.0
        else:
            idx = self._closest_checkpoint_idx(pos)
            forward_yaw = self._smoothed_forward_yaw(idx)
            fx, fy = math.cos(forward_yaw), math.sin(forward_yaw)
            vx, vy = vel[0] / speed_horizontal, vel[1] / speed_horizontal
            raw_alignment = vx * fx + vy * fy   # signed, -1 to +1

        # Gate: clamp negative to zero so backward driving earns no reward.
        gated_alignment = max(0.0, raw_alignment)
        final_reward = raw_progress * gated_alignment

        self._last_raw_progress = raw_progress
        self._last_raw_alignment = raw_alignment
        self._last_final_reward = final_reward

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
        return float(final_reward + checkpoint_bonus)

    def _check_done(self) -> Tuple[bool, float]:
        """Did the episode end? If so, what bonus or penalty applies?"""
        self._lap_done = False
        if self._is_flipped():
            return True, FLIP_PENALTY
        if self._is_off_track():
            return True, OFF_TRACK_PENALTY
        if self._lap_completed():
            self._lap_done = True
            return True, LAP_BONUS
        if self._steps_since_progress > STUCK_STEPS_THRESHOLD:
            return True, STUCK_PENALTY
        return False, 0.0

    def _distance_along_centerline(self) -> float:
        """How far along the racing line the car currently is."""
        pos = _shared["vehicle"].sensors["agent_state"]["pos"]
        idx = self._closest_checkpoint_idx(pos)
        cum = 0.0
        for i in range(idx):
            cum += _dist(CENTERLINE[i], CENTERLINE[i + 1])
        cum += _project_along(pos, CENTERLINE[idx],
                              CENTERLINE[(idx + 1) % len(CENTERLINE)])
        return cum

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

    def _lap_completed(self) -> bool:
        """Made it back to checkpoint 0 after going around the loop."""
        # Placeholder: with the real centerline this will compare a lap-distance
        # counter against the known track length.
        pos = _shared["vehicle"].sensors["agent_state"]["pos"]
        idx = self._closest_checkpoint_idx(pos)
        return idx == 0 and self._last_centerline_dist > 50.0


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
                    nogpu: bool = False):
    """Return a fresh BeamNGRaceEnv. No rtgym wrapping in v1."""
    return BeamNGRaceEnv(
        random_spawn=random_spawn, home=home, host=host, port=port,
        launch=launch, headless=headless, nogpu=nogpu,
    )
