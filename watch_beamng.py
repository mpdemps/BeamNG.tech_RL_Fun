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
import time

# Match train_beamng.py behavior: kill any lingering BeamNG before launching.
for proc in ("BeamNG.tech.exe", "support.exe"):
    subprocess.run(["taskkill", "/F", "/IM", proc], capture_output=True)

from stable_baselines3 import PPO
from envs.beamng_env import make_beamng_env

DEFAULT_MODEL = "checkpoints/overnight_v1/best_model/best_model.zip"
BEAMNG_HOME = r"C:\BeamNG\BeamNG.tech.v0.38.5.0"


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
    args = parser.parse_args()

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
    env = make_beamng_env(home=BEAMNG_HOME, launch=True, headless=False, nogpu=False, random_spawn=False)
    model = PPO.load(args.model, device="cpu")

    print("Ready. Ctrl+C to stop early.")
    print()

    try:
        reset_kwargs = ({"options": {"spawn_idx": args.spawn_idx}}
                        if args.spawn_idx is not None else {})
        for ep in range(args.episodes):
            obs, info = env.reset(**reset_kwargs)
            ep_reward = 0.0
            ep_steps = 0
            last_log = time.time()

            for step in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_steps += 1

                # Log every ~1 second of wall-clock to avoid spam at 20Hz.
                if time.time() - last_log > 1.0:
                    raw_prog = info.get("raw_progress", float("nan"))
                    align = info.get("alignment", float("nan"))
                    print(f"  ep{ep} step={step:4d} | "
                          f"r={float(reward):+6.2f} | "
                          f"progress={raw_prog:+5.2f} | "
                          f"align={align:+.2f} | "
                          f"steer={float(action[0]):+.2f} throttle={float(action[1]):+.2f}")
                    last_log = time.time()

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