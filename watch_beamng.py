"""Watch a trained PPO model drive in BeamNG.

Loads a saved policy, runs it deterministically in the visible (windowed) env,
and prints per-step telemetry plus per-episode summaries.

Usage:
    venv\\Scripts\\python watch_beamng.py
    venv\\Scripts\\python watch_beamng.py --model checkpoints/overnight_v1/rolling_75000_steps.zip
    venv\\Scripts\\python watch_beamng.py --episodes 3 --max-steps 3000
"""

import argparse
import os
import signal
import subprocess
import sys

# Match train_beamng.py behavior: kill any lingering BeamNG before launching.
for proc in ("BeamNG.tech.exe", "support.exe"):
    subprocess.run(["taskkill", "/F", "/IM", proc], capture_output=True)

from stable_baselines3 import PPO, SAC
from envs.beamng_env import (
    make_beamng_env, _shared, MAX_SPEED_M_S, CENTER_OFFSET_CLIP_M,
    MAX_LOOKAHEAD_DIST_M, LOOKAHEAD_DISTANCES_M,
    N_CHECKPOINTS, CHECKPOINT_BONUS, LAP_BONUS)
from envs.residual_hybrid import ResidualHybrid
from envs.drift import BETA_TARGET_OBS_NORM

DEFAULT_MODEL = "checkpoints/overnight_v1/best_model/best_model.zip"
BEAMNG_HOME = r"C:\BeamNG\BeamNG.tech.v0.38.5.0"


# ---- Live readout for Mikey: what the AI SEES and what it DOES, every step ----
# The policy's observation (envs/beamng_env.py _get_observation) is 9 numbers,
# all scaled to roughly -1..1: [0] speed, [1] how well it's lined up with the
# next track point, [2] how far it has drifted off the center line, and
# [3/5/7]+[4/6/8] the distance and turn direction to the next three points.
# The action is 2 numbers: steer (-1 hard left .. +1 hard right) and
# throttle (>0 = gas, <0 = brake). We draw little block-bars so a kid can
# glance and instantly read "turning left, medium gas, well lined up".

BLOCK = "█"   # full block
SHADE = "░"   # light shade


def _enable_ansi():
    """Make the live panel render reliably: force UTF-8 output for the block and
    arrow glyphs, and (on Windows) turn on virtual-terminal processing so the
    in-place redraw escapes work. Both are best-effort and safe to call always."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # avoid UnicodeEncodeError on cp1252 consoles
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # ENABLE_PROCESSED_OUTPUT|WRAP_AT_EOL|VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


def _fill_bar(frac, width=8):
    """A 0..1 value as a left-filled bar, e.g. gas pedal."""
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return BLOCK * n + SHADE * (width - n)


def _center_bar(val, width=7):
    """A -1..1 value as a bar that grows LEFT of center for negative and RIGHT
    for positive; the '|' in the middle is zero (e.g. steering)."""
    val = max(-1.0, min(1.0, val))
    half = width // 2
    n = int(round(abs(val) * half))
    left, right = [" "] * half, [" "] * half
    if val < 0:
        for i in range(n):
            left[half - 1 - i] = BLOCK
    else:
        for i in range(n):
            right[i] = BLOCK
    return "".join(left) + "|" + "".join(right)


def _turn_arrow(bearing):
    """Normalized bearing (+ = next point is to the left) -> glanceable arrow."""
    if bearing > 0.06:
        return "←"   # left
    if bearing < -0.06:
        return "→"   # right
    return "↑"       # straight ahead


def _readout(obs, action):
    """Return the 2-3 line live panel (what it sees + what it does).

    Matches the run6 15-element obs: [speed, heading_err, center_off] + 6
    lookahead points x (dist, bearing) at LOOKAHEAD_DISTANCES_M ahead. The ROAD
    line shows exactly the points the policy sees (the old dashboard showed
    different points than the observation, which misled us once). NOTE: pre-run6
    models (9-element obs) cannot load against this env; to watch them, check
    out the pre-run6 commit.
    """
    speed, aligned, offset = float(obs[0]), float(obs[1]), float(obs[2])
    legs = [(float(obs[3 + 2 * i]), float(obs[4 + 2 * i]))
            for i in range(len(LOOKAHEAD_DISTANCES_M))]
    steer, thr = float(action[0]), float(action[1])

    kmh = speed * MAX_SPEED_M_S * 3.6
    align_word = "good" if abs(aligned) < 0.08 else ("left" if aligned > 0 else "right")
    track_word = "centered" if abs(offset) < 0.08 else ("left" if offset < 0 else "right")

    def leg(d, b):
        return f"{_turn_arrow(b)}{d * MAX_LOOKAHEAD_DIST_M:3.0f}"

    sees = (f" SEES  speed[{_fill_bar(speed)}]{kmh:3.0f}km/h"
            f"  aligned[{_center_bar(aligned)}]{align_word:5s}"
            f"  on-track[{_center_bar(offset)}]{track_word}")
    road = (" ROAD  ahead(m): "
            + "  ".join(leg(d, b) for d, b in legs))
    steer_word = "straight" if abs(steer) < 0.08 else ("LEFT" if steer < 0 else "RIGHT")
    if thr >= 0:
        pedal = f"gas  [{_fill_bar(thr)}]{thr * 100:3.0f}%"
    else:
        pedal = f"BRAKE[{_fill_bar(-thr)}]{-thr * 100:3.0f}%"
    does = f" DOES  steer[{_center_bar(steer)}]{steer_word:8s}   {pedal}"
    lines = [sees, road, does]
    # run27 drift: with the 20-dim drift obs, show the LIVE slide angle vs the target it is being
    # asked to hold right now. obs[17]=beta/45 (the actual slide), obs[19]=beta_target/norm (the goal).
    if len(obs) >= 20:
        slide = float(obs[17]) * 45.0
        target = float(obs[19]) * BETA_TARGET_OBS_NORM
        gap = slide - target
        lines.append(f"DRIFT  slide[{_fill_bar(slide / 45.0)}]{slide:3.0f}°"
                     f"   target {target:3.0f}°   {'+' if gap >= 0 else ''}{gap:.0f}° "
                     f"{'OVER' if gap > 3 else ('under' if gap < -3 else 'on target')}")
    return lines


def _reward_lines(progress, cp_bonus, speed, smooth, step_total, score, cp, banner):
    """The reward dashboard, so Mikey can watch his reward design work live:
    the per-step breakdown by term (progress / checkpoint / speed / smooth) plus
    the step total, then the running episode SCORE and checkpoints reached. When
    a checkpoint lands, `banner` carries a celebration that the caller holds on
    screen for ~1 second so the +10 jump is impossible to miss."""
    total_cp = N_CHECKPOINTS - 1   # intermediate checkpoints; the last lap is LAP_BONUS
    step = (f" STEP  progress {progress:+.2f}  checkpt {cp_bonus:+.0f}"
            f"  speed {speed:+.2f}  smooth {smooth:+.2f}  total {step_total:+.2f}")
    middle = f"   {banner}   " if banner else "          "
    score_line = f" SCORE ▶ {score:8.1f} pts{middle}checkpoints {cp}/{total_cp}"
    return [step, score_line]


# Redraw the panel in place each step (cursor up N lines, overwrite) so it reads
# like a live dashboard instead of scrolling 20 lines a second.
_panel = {"drawn": False, "n": 0}


def _draw(lines):
    out = sys.stdout
    if _panel["drawn"]:
        out.write(f"\x1b[{_panel['n']}A")   # move cursor up to panel top
    for ln in lines:
        out.write("\r\x1b[2K" + ln + "\n")   # clear line, write, next line
    out.flush()
    _panel["drawn"], _panel["n"] = True, len(lines)


def _report_active_config():
    """Read back which vehicle config ACTUALLY spawned (not assumed) and print it,
    so the watcher itself confirms the race config vs a silent gts fallback. The
    distinguishing slot is scintilla_coilover_F: _race vs _adaptive (gts)."""
    try:
        cfg = _shared["vehicle"].get_part_config()
        chosen = {}

        def walk(n):
            if isinstance(n, dict):
                if n.get("id") and n.get("chosenPartName"):
                    chosen[n["id"]] = n["chosenPartName"]
                for v in n.values():
                    walk(v)
            elif isinstance(n, list):
                for v in n:
                    walk(v)

        walk(cfg)
        coil = chosen.get("scintilla_coilover_F", "?")
        dash = chosen.get("scintilla_dash", "?")
        tag = ("RACE" if "race" in coil
               else "GTS (default/fallback!)" if "adaptive" in coil
               else "UNKNOWN")
        print(f"  ACTIVE CAR CONFIG: {tag}"
              f"   (coilover_F={coil}, dash={dash})")
    except Exception as e:
        print(f"  (could not read active car config: {e!r})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Path to .zip policy file (default: %(default)s)")
    parser.add_argument("--episodes", type=int, default=2,
                        help="How many episodes to run before quitting")
    parser.add_argument("--max-steps", type=int, default=4000,
                        help="Hard cap on steps per episode")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use mean action instead of sampled (default: True)")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false",
                        help="Sample from policy instead of taking the mean")
    parser.add_argument("--spawn-idx", type=int, default=None,
                        help="Force spawn at specific centerline index "
                             "(default: env's normal spawn behavior)")
    parser.add_argument("--residual", action="store_true",
                        help="run22 additive bounded residual hybrid: wrap the env so the watched "
                             "car drives the FULL hybrid (base_controller + clip(policy, +/-delta)), "
                             "the same as training. The model is the plain-SAC residual policy.")
    parser.add_argument("--residual-delta", type=float, default=0.12,
                        help="run22: residual authority bound (must match the trained run; 0.12).")
    parser.add_argument("--residual-throttle-up", type=float, default=None,
                        help="run24: cap the positive throttle residual (e.g. 0.05); must match the "
                             "trained run. None = symmetric (run22/23).")
    parser.add_argument("--drift", action="store_true",
                        help="run27 Phase 2: watch a DRIFT policy -- builds the GTS drift car + the "
                             "20-dim drift obs so the saved drift model loads and drives. Required for "
                             "run27 checkpoints (else the 19-dim env shape-mismatches the policy).")
    args = parser.parse_args()
    _enable_ansi()   # so the live panel can redraw in place (Windows-safe)

    if not os.path.isfile(args.model):
        print(f"ERROR: model not found at {args.model}")
        sys.exit(1)

    print("=" * 64)
    print(f"Model:        {args.model}")
    print(f"Episodes:     {args.episodes}")
    print(f"Max steps:    {args.max_steps}")
    print(f"Action mode:  {'deterministic' if args.deterministic else 'stochastic'}")
    print(f"BeamNG home:  {BEAMNG_HOME}")
    print(f"Mode:         headed (visible window, GPU rendering)")
    print("=" * 64)
    print("Loading model and launching BeamNG (silent for ~60s)...")

    # Visible window. No headless, no nogpu. We want to watch.
    env = make_beamng_env(home=BEAMNG_HOME, launch=True, headless=False, nogpu=False,
                          random_spawn=False, drift_mode=args.drift)
    if args.drift:
        print("DRIFT MODE: GTS drift car (drift mode + LSD diff), 20-dim drift obs.")
    # run22: wrap so the watched car drives the FULL hybrid (controller + clipped residual),
    # exactly as training does -- the model is just the residual policy. Without this the bare
    # policy (a small +/-delta trim) would drive alone and go nowhere.
    if args.residual:
        env = ResidualHybrid(env, delta=args.residual_delta, throttle_up=args.residual_throttle_up)
        print(f"RESIDUAL HYBRID: base controller + clip(policy, +/-{args.residual_delta}"
              f"{'' if args.residual_throttle_up is None else f', throttle +{args.residual_throttle_up}'}); "
              f"the DOES panel shows the APPLIED (controller+residual) action.")

    # Load the policy without caring which algorithm trained it: run1/run2 are
    # PPO, run3+ are SAC. Try each and use whichever loads cleanly, so this one
    # watcher works for every run (the decisive G14 spin-check needs to load any
    # snapshot). A mismatched algorithm raises on the policy structure, so the
    # one that loads is the right one.
    model, algo_name, _errs = None, None, {}
    for _algo in (SAC, PPO):
        try:
            model = _algo.load(args.model, device="cpu")
            algo_name = _algo.__name__
            break
        except Exception as e:
            _errs[_algo.__name__] = repr(e)
    if model is None:
        print(f"ERROR: could not load {args.model} as SAC or PPO: {_errs}")
        sys.exit(1)
    print(f"Algorithm:    {algo_name} (auto-detected)")

    print("Ready. Ctrl+C to stop early.")
    print()

    try:
        reset_kwargs = ({"options": {"spawn_idx": args.spawn_idx}}
                        if args.spawn_idx is not None else {})
        for ep in range(args.episodes):
            obs, info = env.reset(**reset_kwargs)
            if ep == 0:
                _report_active_config()   # confirm race vs gts by read-back, once
            ep_reward = 0.0
            ep_steps = 0
            prev_cp = 0           # checkpoints reached as of last step
            celebrate = 0         # steps left to hold a checkpoint/lap banner
            celebrate_msg = ""
            _panel["drawn"] = False   # fresh panel below this episode's spawn log

            for step in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs_seen = obs    # what the policy was given to decide this action
                obs, reward, terminated, truncated, info = env.step(action)
                # run22: the panel should show what the CAR did (controller+residual), not the
                # bare policy trim. The wrapper stashes the applied action each step.
                shown_action = getattr(env, "last_applied", None)
                if shown_action is None:
                    shown_action = action
                ep_reward += float(reward)
                ep_steps += 1

                # Per-step checkpoint bonus = how many checkpoints the env counted
                # this step times CHECKPOINT_BONUS. A jump triggers a held banner.
                cp = int(info.get("checkpoints_reached", 0))
                cp_bonus = (cp - prev_cp) * CHECKPOINT_BONUS
                if cp > prev_cp:
                    celebrate, celebrate_msg = 20, f"★ +{cp_bonus:.0f} CHECKPOINT! ★"
                if info.get("lap_completed"):
                    celebrate, celebrate_msg = 40, f"★★★ LAP! +{int(LAP_BONUS)} ★★★"
                prev_cp = cp

                lines = _readout(obs_seen, shown_action) + _reward_lines(
                    progress=float(info.get("final_reward", 0.0)),
                    cp_bonus=cp_bonus,
                    speed=float(info.get("speed_reward", 0.0)),
                    smooth=float(info.get("smoothness_penalty", 0.0)),
                    step_total=float(reward), score=ep_reward, cp=cp,
                    banner=(celebrate_msg if celebrate > 0 else ""))
                _draw(lines)
                if celebrate > 0:
                    celebrate -= 1

                if terminated or truncated:
                    break

            reason = "terminated" if terminated else ("truncated" if truncated else "max_steps")
            print(f"[ep {ep}] DONE: reward={ep_reward:+.2f} steps={ep_steps} reason={reason}")
            print()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"env.close() raised {e!r} (ignoring)")


if __name__ == "__main__":
    main()