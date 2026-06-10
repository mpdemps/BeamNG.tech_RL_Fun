"""
Train a PPO agent to drive a lap on West Coast USA in BeamNG.tech.

Plain-language overview, Mikey:
- The script opens BeamNG by itself, then puts the car in it.
- The car tries to drive; the stable-baselines3 PPO algorithm learns from
  trial and error.
- The best version of the car-brain gets saved automatically — not just
  the last one.
- Everything is logged so we can stare at training curves in TensorBoard.
- After training, a row is written in RUNS.md so future-us remembers what
  we tried.

By default this launches a fresh BeamNG instance — a manually-launched
BeamNG.tech.x64 doesn't open the TechCom port for Python, so attaching to
an existing instance (--no-launch) is rarely useful and exists only for
debugging.

Troubleshooting:
- If a previous run crashed partway through, a BeamNG.tech.x64 process may
  still be lingering. Kill it with `pkill -f BeamNG` before re-running,
  otherwise the new BeamNG launch will fight the zombie one for the
  BeamNGpy port and hang.
"""

import argparse
import datetime
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback)
from stable_baselines3.common.monitor import Monitor

from envs.beamng_env import make_beamng_env, _shared


class MilestoneSnapshotCallback(BaseCallback):
    """Save a labeled model the first time the run hits a new milestone.

    Tier 2 -- furthest checkpoint: each time `checkpoints_reached` (from the env
    info dict) sets a new run-wide high-water mark, save
    `milestone_cp{NN}_step{steps}.zip` ("first model to reach checkpoint NN").
    Tier 3 -- lap trophy: the first time the env reports `lap_completed`, save
    `milestone_lap_step{steps}.zip`. Together with Tier 1's rolling_* time
    snapshots, these give Mikey the full learning arc, watchable in order.
    """

    def __init__(self, save_dir, verbose=1):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.furthest = 0       # run-wide max checkpoints_reached so far
        self.lap_saved = False  # the lap trophy is saved once, on the first lap

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            cp = int(info.get("checkpoints_reached", 0))
            if cp > self.furthest:
                self.furthest = cp
                path = self.save_dir / f"milestone_cp{cp:02d}_step{self.num_timesteps}.zip"
                self.model.save(str(path))
                if self.verbose:
                    print(f"[milestone] first to reach checkpoint {cp} "
                          f"at step {self.num_timesteps} -> {path.name}", flush=True)
            if info.get("lap_completed") and not self.lap_saved:
                self.lap_saved = True
                path = self.save_dir / f"milestone_lap_step{self.num_timesteps}.zip"
                self.model.save(str(path))
                if self.verbose:
                    print(f"[milestone] LAP COMPLETED at step {self.num_timesteps} "
                          f"-> {path.name}  (the trophy!)", flush=True)
        return True


# PPO entropy coefficient. The dead-policy 250k G14 run used 0.01 (confirmed
# from its saved overnight_v1 model), so 0.01 is the CONTROL BASELINE here, not
# a known fix -- entropy already failed once at this value on the old reward.
# Held constant so the curriculum (fixed-start + checkpoint rewards) is the only
# intervention, letting any learning be attributed to the curriculum rather than
# a hyperparameter change. If the policy still collapses at 0.01 with the
# curriculum, THAT is the evidence to revisit this and other causes (learning
# rate, alignment-gate reward clamping, observation scaling).
ENT_COEF = 0.01


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=None,
                   help="Descriptive run name (used for TensorBoard and "
                        "checkpoint paths). Defaults to a timestamp.")
    p.add_argument("--timesteps", type=int, default=500_000,
                   help="Total training timesteps.")
    p.add_argument("--eval-freq", type=int, default=10_000,
                   help="How often EvalCallback runs (in env steps).")
    p.add_argument("--checkpoint-freq", type=int, default=5_000,
                   help="Tier 1: how often to save a rolling time-based "
                        "checkpoint. 5000 steps ~= one snapshot every ~11 min, "
                        "capturing the early flailing too. Files are ~150 KB.")
    p.add_argument("--home", default=os.environ.get("BEAMNG_HOME"),
                   help="BeamNG.tech install directory. Defaults to the "
                        "BEAMNG_HOME env var.")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=25252)
    p.add_argument("--no-launch", dest="launch", action="store_false",
                   help="Attach to an already-running BeamNG instead of "
                        "launching a new one. Rare — manually-launched "
                        "BeamNG doesn't open the TechCom port for Python, "
                        "so this is only useful for special debugging.")
    p.set_defaults(launch=True)
    p.add_argument("--headless", action="store_true",
                   help="Hide the BeamNG window (faster, no watching).")
    p.add_argument("--nogpu", action="store_true",
                   help="Skip the BeamNG rendering pipeline entirely "
                        "(implies --headless). Big speedup for physics-only "
                        "training. Breaks any sensor that needs rendering "
                        "(camera, lidar) — safe here since we only use "
                        "State / Electrics / Damage.")
    p.add_argument("--no-journal", action="store_true",
                   help="Skip writing an entry to RUNS.md (useful for smoke "
                        "tests).")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.home:
        raise SystemExit("Set BEAMNG_HOME or pass --home (path to the BeamNG.tech install).")
    run_name = (args.run_name
                or f"phase1_{datetime.datetime.now():%Y%m%d_%H%M%S}")

    log_dir = Path("logs") / run_name
    ckpt_dir = Path("checkpoints") / run_name
    best_dir = ckpt_dir / "best_model"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Startup banner — gives the user something to look at while BeamNG
    # silently loads (which can take 30-60 seconds).
    print("=" * 64)
    print(f"Run name:         {run_name}")
    print(f"Timesteps:        {args.timesteps:,}")
    if args.nogpu:
        mode_str = "nogpu (no rendering, implies headless)"
    elif args.headless:
        mode_str = "headless"
    else:
        mode_str = "headed (windowed)"
    print(f"Mode:             {mode_str}")
    print(f"BeamNG home:      {args.home}")
    print(f"BeamNG launch:    "
          f"{'NEW (launching fresh)' if args.launch else 'NO (attaching to existing)'}")
    print(f"TensorBoard log:  {log_dir}")
    print("=" * 64)
    print("Loading BeamNG (this can take 30-60 seconds — silent until done)...")
    print(flush=True)

    # Monitor logs ep-end values of the listed info-dict keys to a CSV
    # alongside the standard r/l/t columns. NOTE: this is per-EPISODE-END,
    # not per-step — for true per-step CSV we'd need a custom callback.
    monitor_info_keys = ("raw_progress", "alignment", "final_reward",
                         "speed_reward", "smoothness_penalty",
                         "checkpoints_reached")
    train_env = Monitor(
        make_beamng_env(
            # Curriculum: fixed start at the start/finish line (idx=0) every
            # episode, so the car learns the track as a sequence and gets
            # further each time. (random_spawn=False -> idx=0, heading 0, rest.)
            random_spawn=False, home=args.home, host=args.host, port=args.port,
            launch=args.launch, headless=args.headless, nogpu=args.nogpu,
        ),
        filename=str(log_dir / "train"),
        info_keywords=monitor_info_keys,
    )
    # Eval env shares the same BeamNG connection (singleton in beamng_env.py)
    # and never runs at the same time as training, so launch=False here
    # always — we don't want a second BeamNG window.
    eval_env = make_beamng_env(
        random_spawn=False, home=args.home, host=args.host, port=args.port,
        launch=False, headless=args.headless, nogpu=args.nogpu,
    )

    # v1: PPO with the hyperparameters from docs/phase1_env_spec.md.
    # TODO(v2 SAC migration): swap the next block for SAC for better sample
    # efficiency on continuous control. Something like:
    #   from stable_baselines3 import SAC
    #   model = SAC("MlpPolicy", train_env,
    #               learning_rate=3e-4, buffer_size=1_000_000,
    #               learning_starts=1000, batch_size=256, tau=0.005,
    #               gamma=0.99, train_freq=1, gradient_steps=1,
    #               ent_coef="auto", tensorboard_log=str(log_dir),
    #               device="cuda", verbose=1)
    # Callbacks, learn(), and journaling below stay identical.
    hyperparams = dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ENT_COEF,
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(log_dir),
        # PPO with MlpPolicy is faster on CPU than GPU; GPU only helps for
        # CNN policies. SB3 itself warns about this if you set device="cuda".
        device="cpu",
        **hyperparams,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )
    ckpt_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(ckpt_dir),
        name_prefix="rolling",
    )
    milestone_callback = MilestoneSnapshotCallback(save_dir=ckpt_dir, verbose=1)

    started = datetime.datetime.now()
    completed_timesteps = 0
    interrupted = False
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_callback, ckpt_callback, milestone_callback],
            tb_log_name="ppo",
        )
        completed_timesteps = model.num_timesteps
    except KeyboardInterrupt:
        interrupted = True
        completed_timesteps = model.num_timesteps
        print("\nInterrupted by user. Saving final model and journaling...")
    finally:
        model.save(str(ckpt_dir / "final"))
        if not args.no_journal:
            _append_run_journal(
                run_name=run_name,
                started=started,
                hyperparams=hyperparams,
                target_timesteps=args.timesteps,
                completed_timesteps=completed_timesteps,
                interrupted=interrupted,
                eval_callback=eval_callback,
                log_dir=log_dir,
                ckpt_dir=ckpt_dir,
            )
        train_env.close()
        eval_env.close()
        # The env close()s above are intentional no-ops (the shared BeamNG
        # connection is kept open mid-run so train_env and eval_env don't
        # tear each other down). The run is genuinely done now, so terminate
        # the launched BeamNG process — otherwise it lingers holding the
        # TechCom port and the next launch fights the zombie for it.
        bng = _shared.get("bng")
        if bng is not None:
            try:
                bng.close()
            except Exception as e:
                print(f"BeamNG close() raised {e!r} (ignoring)")


def _append_run_journal(*, run_name, started, hyperparams, target_timesteps,
                        completed_timesteps, interrupted, eval_callback,
                        log_dir, ckpt_dir):
    """Append one entry to RUNS.md so future-us remembers this run."""
    runs_path = Path("RUNS.md")
    peak = getattr(eval_callback, "best_mean_reward", None)
    final = getattr(eval_callback, "last_mean_reward", None)
    lines = [
        "",
        f"## {run_name}",
        "",
        "### Date",
        "",
        started.strftime("%Y-%m-%d"),
        "",
        "### Hyperparameters",
        "",
        *[f"- {k}: {v}" for k, v in hyperparams.items()],
        f"- Target timesteps: {target_timesteps}",
        f"- Completed timesteps: {completed_timesteps}"
        + (" (interrupted)" if interrupted else ""),
        "",
        "### Peak Eval Reward",
        "",
        f"{peak if peak is not None else '(no eval ran)'}",
        "",
        "### Final Eval Reward",
        "",
        f"{final if final is not None else '(no eval ran)'}",
        "",
        "### Watching Notes",
        "",
        "(fill in after watching with watch_beamng.py)",
        "",
        "### Mikey's Hypothesis for Next Run",
        "",
        "(fill in after Mikey has watched)",
        "",
        "### Artifacts",
        "",
        f"- TensorBoard log dir: `{log_dir}`",
        f"- Best model: `{ckpt_dir / 'best_model'}`",
        f"- Final model: `{ckpt_dir / 'final.zip'}`",
        "",
    ]
    with runs_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
