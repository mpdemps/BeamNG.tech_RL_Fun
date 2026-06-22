"""
Train a SAC agent to drive a lap on West Coast USA in BeamNG.tech.

Plain-language overview, Mikey:
- The script opens BeamNG by itself, then puts the car in it.
- The car tries to drive; the stable-baselines3 SAC algorithm learns from
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
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC  # run16: plain SAC (Grad-CAPS dropped in the paradigm reset)
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback)
from stable_baselines3.common.monitor import Monitor

from envs.beamng_env import make_beamng_env, _shared
from envs.base_controller import BaseController
from envs.blend_sac import BlendSAC
from envs.residual_hybrid import ResidualHybrid


class TBEvalCallback(EvalCallback):
    """EvalCallback + trustworthy TB scalars from the EVAL (start-line, deterministic)
    episodes, so the clean reads sit next to eval/mean_reward and we stop doing manual
    start-line replays. Behavior-neutral: only reads the env info dict at episode end.

    Logs (over the n_eval_episodes): eval/mean_speed, eval/max_arc(+_best),
    eval/over_speed_frac, eval/beta_mean, eval/beta_p90, eval/checkpoints_mean(+_max),
    the reward-term decomposition (eval/r_progress|r_match|r_overspeed|r_slip), and the
    termination-reason fractions (eval/term_<reason>)."""
    _TERMS = ("off_track", "flip", "stuck", "backward", "lap", "run", "loss_of_control")
    _MEAN_KEYS = ("mean_speed", "max_arc", "over_speed_frac", "beta_mean", "beta_p90",
                  "checkpoints_reached", "r_progress", "r_match", "r_overspeed", "r_slip",
                  "residual_abs", "residual_abs_steer", "residual_abs_throttle")  # run22/24: mean
                  # |applied residual| at eval, split per channel (0/absent -> plain run)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_infos = []

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        if locals_.get("done"):
            info = locals_["info"]
            self._eval_infos.append({k: info.get(k) for k in
                                     self._MEAN_KEYS + ("termination_reason",)})

    def _on_step(self):
        result = super()._on_step()
        if self._eval_infos:                      # an eval just completed this step
            infos, n = self._eval_infos, len(self._eval_infos)
            for k in self._MEAN_KEYS:
                vals = [i[k] for i in infos if i[k] is not None]
                if vals:
                    self.logger.record(f"eval/{k}", float(np.mean(vals)))
            self.logger.record("eval/max_arc_best", float(max(i["max_arc"] for i in infos)))
            self.logger.record("eval/checkpoints_max",
                               float(max(i["checkpoints_reached"] for i in infos)))
            cnt = Counter(i["termination_reason"] for i in infos)
            for reason in self._TERMS:
                self.logger.record(f"eval/term_{reason}", cnt.get(reason, 0) / n)
            self.logger.dump(self.num_timesteps)  # write at the eval step
            self._eval_infos = []
        return result


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


# run3 switches PPO -> SAC. SAC tunes its entropy coefficient automatically
# (ent_coef="auto", set in the hyperparams below), so there is no fixed entropy
# constant to pin here as there was for PPO. SAC's entropy objective plus
# off-policy replay tend toward controlled solutions rather than the degenerate
# spin run2's PPO converged to.


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
    p.add_argument("--warm-start", default=None,
                   help="Path to a saved SAC .zip to warm-start from "
                        "(SAC.load) instead of fresh init. Use --learning-rate "
                        "and --learning-starts to control gentle continuation.")
    p.add_argument("--learning-rate", type=float, default=3e-4,
                   help="SAC learning rate. Lower (e.g. 1e-4) for warm-start "
                        "fine-tuning so the loaded policy is nudged, not "
                        "overwritten.")
    p.add_argument("--learning-starts", type=int, default=1000,
                   help="Random-action steps before learning. Set 0 for "
                        "warm-start so the LOADED policy drives from step 0 "
                        "(SAC takes random actions during these steps).")
    p.add_argument("--steer-rate", type=float, default=0.0,
                   help="run13/16 steering slew-rate limit: flat symmetric cap on "
                        "|Δsteer|/step (action[0]). 0.0 = OFF; run16 uses 0.5 (the one "
                        "retained scripted constraint).")
    p.add_argument("--random-spawn", action="store_true",
                   help="run17 spawn curriculum: distribute TRAIN episode starts around "
                        "the whole track (eval env always starts at the line). Off by default.")
    p.add_argument("--blend-fade", action="store_true",
                   help="run21 guided residual RL: blend the base controller into the applied "
                        "action during data collection (beta*controller + (1-beta)*policy), "
                        "fading beta 1->0. Eval is always beta=0 (standalone policy). Off by default.")
    p.add_argument("--beta-warmup", type=int, default=100_000,
                   help="run21: hold beta=1 (pure controller) for this many steps, filling the "
                        "buffer with clean controller laps before the anneal.")
    p.add_argument("--beta-anneal-end", type=int, default=400_000,
                   help="run21: beta reaches 0 at this step (linear anneal from --beta-warmup); "
                        "beta=0 held from here to the end (policy stands alone).")
    p.add_argument("--beta-offset", type=int, default=0,
                   help="run21: global steps already completed before this segment. The wrapper "
                        "passes this on a warm-restart so beta tracks GLOBAL progress (num_timesteps "
                        "resets to 0 each segment). 0 for a fresh first segment.")
    p.add_argument("--residual", action="store_true",
                   help="run22 additive bounded residual hybrid: applied = clip(base_controller + "
                        "clip(policy, +/-delta), -1, 1). Controller at FULL; the policy only trims. "
                        "Eval runs the full hybrid (it's an env wrapper). Off by default.")
    p.add_argument("--residual-delta", type=float, default=0.12,
                   help="run22: residual authority bound (normalized action units). The policy can "
                        "shift each control by at most this much on top of the controller.")
    p.add_argument("--residual-throttle-up", type=float, default=None,
                   help="run24 throttle-authority cut: cap the POSITIVE throttle residual at this "
                        "(e.g. 0.05) while steer stays +/-delta and throttle-down stays -delta. "
                        "Limits over-throttle-into-spin; keeps full lift/brake. None = symmetric.")
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
    # run16: 'speed_reward' diag slot now carries the over-speed penalty, 'spin_penalty'
    # the slip-angle penalty (see env _compute_reward). over_speed_mean + beta are the
    # learn-to-corner watch signals; tc/esc/gradcaps/fluct telemetry retired with them.
    monitor_info_keys = ("raw_progress", "alignment", "final_reward",
                         "speed_reward", "spin_penalty",
                         "slip", "heading_align",
                         "checkpoints_reached",
                         "termination_reason", "recovered_count",
                         "mean_speed", "max_arc", "min_heading_align",
                         "max_slip", "steer_clip_frac",
                         "beta_max", "beta_mean",
                         "over_speed_mean", "over_speed_frac", "v_target_here")
    # run22: additive bounded residual hybrid. The residual lives in an env WRAPPER (not a
    # sampler hook) so eval (predict) also runs the FULL hybrid: applied = clip(controller +
    # clip(residual, +/-delta), -1, 1). residual_abs is logged. Each env gets its OWN controller
    # (shared controller state would cross-contaminate train<->eval).
    if args.residual:
        monitor_info_keys = monitor_info_keys + ("residual_abs", "residual_abs_steer",
                                                 "residual_abs_throttle")
    _train_core = make_beamng_env(
        # run17 spawn curriculum: random_spawn distributes episode starts around the
        # whole track (random idx + per-idx heading + start-speed capped at v_target),
        # so every corner's line gets practiced instead of only T1-from-the-start-line.
        random_spawn=args.random_spawn, home=args.home, host=args.host, port=args.port,
        launch=args.launch, headless=args.headless, nogpu=args.nogpu,
        steer_rate=args.steer_rate,
    )
    if args.residual:
        _train_core = ResidualHybrid(_train_core, delta=args.residual_delta,
                                     throttle_up=args.residual_throttle_up)
    train_env = Monitor(
        _train_core,
        filename=str(log_dir / "train"),
        info_keywords=monitor_info_keys,
    )
    # Eval env shares the same BeamNG connection (singleton in beamng_env.py)
    # and never runs at the same time as training, so launch=False here
    # always — we don't want a second BeamNG window.
    eval_env = make_beamng_env(
        random_spawn=False, home=args.home, host=args.host, port=args.port,
        launch=False, headless=args.headless, nogpu=args.nogpu,
        steer_rate=args.steer_rate,
    )
    if args.residual:
        eval_env = ResidualHybrid(eval_env, delta=args.residual_delta,
                                  throttle_up=args.residual_throttle_up)

    # SAC with SB3 sensible defaults for continuous control. buffer_size 1M holds
    # the whole 500k run; ent_coef "auto" self-tunes exploration. learning_rate
    # and learning_starts come from args so a warm-start can lower the lr and set
    # learning_starts=0. MlpPolicy SAC runs fine on CPU (the ~9-11 fps BeamNG env
    # is the bottleneck, not the gradient step).
    hyperparams = dict(
        learning_rate=args.learning_rate,
        buffer_size=1_000_000,
        learning_starts=args.learning_starts,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
    )
    # run21: BlendSAC (controller fade-out) when --blend-fade, else plain SAC. BlendSAC is a
    # drop-in SAC subclass; with the controller/schedule unset it behaves as plain SAC, so
    # load/construct are identical and only the post-construction attribute set differs.
    ModelCls = BlendSAC if args.blend_fade else SAC
    if args.warm_start:
        # run4 warm-start: load a prior SAC policy and continue training it,
        # rather than fresh init. custom_objects overrides BOTH learning_rate and
        # lr_schedule (the schedule is what the optimizer reads each step, so the
        # bare attribute is not enough); learning_starts and tensorboard_log pass
        # as kwargs (SB3 load does model.__dict__.update(kwargs)). With
        # learning_starts=0 the LOADED policy drives from step 0 instead of SAC's
        # usual random-action warmup -- the proof the warm-start took.
        if not os.path.isfile(args.warm_start):
            raise SystemExit(f"--warm-start file not found: {args.warm_start}")
        print(f"WARM-START: {ModelCls.__name__}.load({args.warm_start})  "
              f"lr={args.learning_rate}  learning_starts={args.learning_starts}", flush=True)
        model = ModelCls.load(
            args.warm_start,
            env=train_env,
            device="cpu",
            custom_objects={
                "learning_rate": args.learning_rate,
                "lr_schedule": lambda _: args.learning_rate,
            },
            learning_starts=args.learning_starts,
            tensorboard_log=str(log_dir),
        )
    else:
        model = ModelCls(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=str(log_dir),
            device="cpu",
            **hyperparams,
        )

    if args.blend_fade:
        # run21: attach the base controller + fade schedule. Set AFTER construction so
        # load()/SAC.__init__ are untouched. Eval (policy.predict) never blends -> beta=0.
        model.controller = BaseController()
        model.beta_warmup = args.beta_warmup
        model.beta_anneal_end = args.beta_anneal_end
        model.beta_offset = args.beta_offset
        print(f"BLEND-FADE: base controller ON; beta=1 -> 0 over "
              f"[{args.beta_warmup:,}, {args.beta_anneal_end:,}] global steps, beta=0 after; "
              f"offset={args.beta_offset:,}; eval always beta=0 (standalone policy).", flush=True)

    eval_callback = TBEvalCallback(
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
            tb_log_name="sac",
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
