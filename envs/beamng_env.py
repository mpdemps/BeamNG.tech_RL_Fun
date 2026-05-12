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
PHYSICS_STEPS_PER_STEP = 3          # 3 steps at 60 Hz = 50 ms = 20 Hz env tick
DETERMINISTIC_STEPS_PER_S = 60
MAX_SPEED_M_S = 70.0                # for obs normalization
MAX_LOOKAHEAD_DIST_M = 200.0
CENTER_OFFSET_CLIP_M = 10.0
OFF_TRACK_THRESHOLD_M = 20.0        # generous; tighten once centerline is real
STUCK_STEPS_THRESHOLD = 200
FLIP_PENALTY = -10.0
OFF_TRACK_PENALTY = -10.0
STUCK_PENALTY = -5.0
LAP_BONUS = 50.0
RANDOM_HEADING_DEG = 30.0
RANDOM_SPEED_MAX_M_S = 30.0
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 25252


# ---- Centerline checkpoints ----
# TODO(Mike+Mikey): replace with real centerline points from West Coast USA's
# racetrack. We'll record them with a one-off "drive-and-log" script after
# this env builds and connects. For now these are placeholder coords near the
# default racetrack starting straight so the env can at least load and we can
# verify the BeamNGpy connection end-to-end.
CENTERLINE: list[tuple[float, float, float]] = [
    (-717.0, 101.0, 118.0),
    (-707.0, 101.0, 118.0),
    (-697.0, 101.0, 118.0),
    (-687.0, 101.0, 118.0),
    (-677.0, 101.0, 118.0),
    (-667.0, 101.0, 118.0),
    (-657.0, 101.0, 118.0),
    (-647.0, 101.0, 118.0),
    (-637.0, 101.0, 118.0),
    (-627.0, 101.0, 118.0),
]


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
             headless: bool) -> None:
    """Open BeamNG, load our scenario, and put it in deterministic mode.

    Idempotent: once initialization succeeds, repeat calls do nothing.
    This is critical because train_env and eval_env each call _connect()
    on every reset, but we want the launch + scenario + sensor-attach
    work to happen exactly once per process. Re-attaching sensors raises
    BNGValueError ("duplicate sensor name").
    """
    if _shared["initialized"]:
        return
    bng = BeamNGpy(host, port, home=home)
    bng.open(launch=launch)

    scenario = Scenario(MAP_NAME, "phase1_lap")
    vehicle = Vehicle(VEHICLE_ID, model=VEHICLE_MODEL)
    # BeamNG already attaches a sensor named "state" by default on vehicle
    # spawn (undocumented). We use "agent_state" to avoid the collision.
    vehicle.sensors.attach("agent_state", State())
    vehicle.sensors.attach("electrics", Electrics())
    vehicle.sensors.attach("damage", Damage())
    scenario.add_vehicle(vehicle, pos=CENTERLINE[0], rot_quat=(0, 0, 0, 1))
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

    if headless:
        # TODO: confirm the right BeamNGpy 0.34+ call for hiding the window.
        try:
            bng.hide_hud()
        except Exception:
            pass

    _shared["bng"] = bng
    _shared["vehicle"] = vehicle
    _shared["initialized"] = True


class BeamNGRaceEnv(gymnasium.Env):
    """A plain Gymnasium env that drives an ETK 800 around West Coast USA."""

    metadata = {"render_modes": []}

    def __init__(self, random_spawn: bool, home: Optional[str] = None,
                 host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 launch: bool = False, headless: bool = False):
        super().__init__()
        # Connection params are stored here; we actually open BeamNG lazily
        # in reset(), so constructing the env never fails on its own.
        self.random_spawn = random_spawn
        self.home = home
        self.host = host
        self.port = port
        self.launch = launch
        self.headless = headless

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._last_centerline_dist = 0.0
        self._steps_since_progress = 0

    def reset(self, seed=None, options=None):
        """Put the car at a spawn point and hand back the first observation."""
        super().reset(seed=seed)
        _connect(self.home, self.host, self.port, self.launch, self.headless)

        if self.random_spawn:
            idx = int(self.np_random.integers(0, len(CENTERLINE)))
            heading_offset = math.radians(float(self.np_random.uniform(
                -RANDOM_HEADING_DEG, RANDOM_HEADING_DEG)))
            start_speed = float(self.np_random.uniform(
                0, RANDOM_SPEED_MAX_M_S))
        else:
            idx = 0
            heading_offset = 0.0
            start_speed = 0.0

        self._teleport_to(idx, heading_offset, start_speed)
        _shared["vehicle"].sensors.poll()
        self._last_centerline_dist = self._distance_along_centerline()
        self._steps_since_progress = 0
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
        return obs, reward + term_bonus, terminated, truncated, {}

    def close(self):
        """Leave the shared BeamNG connection open — other envs may still need it."""
        # We intentionally don't tear down `_shared` here. The single BeamNG
        # process is shared by train_env and eval_env, and Python exit will
        # release the socket. Tearing down mid-run would kill the other env.
        pass

    # ---- Helpers ----

    def _teleport_to(self, idx: int, heading_offset_rad: float, speed: float):
        """Move the car onto a centerline point, facing roughly forward."""
        pos = CENTERLINE[idx]
        nxt = CENTERLINE[(idx + 1) % len(CENTERLINE)]
        yaw = math.atan2(nxt[1] - pos[1], nxt[0] - pos[0]) + heading_offset_rad
        quat = _yaw_to_quat(yaw)
        v = _shared["vehicle"]
        v.teleport(pos, rot_quat=quat, reset=True)
        if speed > 0:
            try:
                v.set_velocity(speed)
            except Exception:
                # set_velocity is best-effort; if BeamNGpy version doesn't
                # support it on a non-shifted vehicle, we just start at rest.
                pass

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
        """The whole reward: distance moved along the racing line since last step."""
        d = self._distance_along_centerline()
        progress = d - self._last_centerline_dist
        self._last_centerline_dist = d
        if progress > 0.01:
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1
        return float(progress)

    def _check_done(self) -> Tuple[bool, float]:
        """Did the episode end? If so, what bonus or penalty applies?"""
        if self._is_flipped():
            return True, FLIP_PENALTY
        if self._is_off_track():
            return True, OFF_TRACK_PENALTY
        if self._lap_completed():
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
    """Rotation quaternion (x, y, z, w) from a yaw angle in radians."""
    half = yaw / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


# ---- Factory: kept for symmetry with the old API, now a trivial wrapper ----

def make_beamng_env(random_spawn: bool, home: Optional[str] = None,
                    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                    launch: bool = False, headless: bool = False):
    """Return a fresh BeamNGRaceEnv. No rtgym wrapping in v1."""
    return BeamNGRaceEnv(
        random_spawn=random_spawn, home=home, host=host, port=port,
        launch=launch, headless=headless,
    )
