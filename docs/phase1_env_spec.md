# Phase 1 Environment Spec: `beamng_env.py`

> Source of methodology: Yosh's Trackmania RL approach, adapted to BeamNG.tech's richer physics and sensor suite. See `docs/references.md` for the original video transcript.

## Purpose of this document

When we're ready to start Phase 1, paste the prompt at the end of this file into Claude Code. CC will use it (alongside `CLAUDE.md`) to build `beamng_env.py` correctly the first time, instead of evolving toward correctness through five rewrites.

This doc also serves as a record of *why* the env is built the way it is. Mikey can read it when he's older and see the engineering reasoning behind the choices.

## Design principles, ranked

1. **Engineered inputs, not raw pixels.** Small number of meaningful numbers beats high-dimensional images.
2. **Simplest possible reward.** Progress along centerline. Add nothing else until v1 has proven what it can and can't learn.
3. **Random spawn always.** Overfitting to the start of the track is the #1 failure mode in this kind of RL. Defend against it from day one.
4. **Separate training from eval.** Training is noisy and full of exploration. Eval is clean and deterministic. They serve different purposes and should not be conflated.
5. **Save best, not last.** Models can get worse over time (we learned this with CarRacing v1). Always evaluate and keep the best.
6. **MlpPolicy, not CnnPolicy.** Image-based learning is overkill for what we're doing and burns 10x the training time.

## Observation space (9 floats, MlpPolicy)

| # | Name | Range | Source |
|---|------|-------|--------|
| 0 | speed_m_s | 0 to ~70 | `state.vel` magnitude |
| 1 | heading_error_rad | -π to +π | angle between vehicle forward and direction to next checkpoint |
| 2 | center_offset_m | -10 to +10 | signed perpendicular distance from racing line; negative = left, positive = right |
| 3 | lookahead_1_dist | 0 to ~200 | distance to checkpoint +1 |
| 4 | lookahead_1_angle | -π to +π | bearing to checkpoint +1 |
| 5 | lookahead_2_dist | 0 to ~200 | distance to checkpoint +2 |
| 6 | lookahead_2_angle | -π to +π | bearing to checkpoint +2 |
| 7 | lookahead_3_dist | 0 to ~200 | distance to checkpoint +3 |
| 8 | lookahead_3_angle | -π to +π | bearing to checkpoint +3 |

All observations are normalized to roughly -1 to +1 range before being returned. SB3 doesn't strictly require this but it helps training stability.

**Why 3 lookahead checkpoints?** Yosh used a similar concept ("further ahead" inputs). Three is enough for the agent to anticipate compound corners without overwhelming the policy network. We can add more in v2 if Mikey finds the AI is too reactive.

## Action space (2 continuous floats)

| # | Name | Range | Meaning |
|---|------|-------|---------|
| 0 | steering | -1.0 to +1.0 | full left to full right |
| 1 | throttle | -1.0 to +1.0 | full brake to full throttle |

Continuous, not discrete. Yosh used 6 discrete actions (DQN constraint). PPO handles continuous natively and gives smoother driving. Map throttle: positive → `vehicle.control(throttle=x)`, negative → `vehicle.control(brake=-x)`.

## Reward (v1: Yosh-simple)

```python
def _compute_reward(self):
    # Distance moved along the racing line since the last step.
    # This is the ENTIRE reward function. Resist adding more terms until v1 plateaus.
    current_centerline_dist = self._distance_along_centerline()
    progress = current_centerline_dist - self._last_centerline_dist
    self._last_centerline_dist = current_centerline_dist
    return progress
```

That's it. No penalty for being off-center (the lookahead inputs let the AI figure out cornering itself). No penalty for jerky steering (let's see if it's even a problem first). No bonus for completing laps (the cumulative reward handles that naturally).

**When Mikey asks "but what about \[X\]?"** good. Write his proposed term in `IDEAS.md` with a date. After v1 trains and we watch it, we'll see whether his proposed term is needed.

## Termination conditions

```python
def _check_done(self):
    if self._is_flipped():
        return True, -10.0  # done, penalty
    if self._is_off_track(threshold_m=5.0):
        return True, -10.0
    if self._lap_completed():
        return True, +50.0  # done, big bonus
    if self._steps_since_progress > 200:
        return True, -5.0  # stuck
    return False, 0.0
```

The penalty/bonus on termination is *added to* the per-step reward, not replacing it. Yosh's "zero reward forever" approach (just stop the episode) works too; we add explicit penalties because they accelerate learning slightly with PPO.

## Reset randomization

Every `reset()`:
- Pick a random point along the track centerline (any of N checkpoints).
- Spawn vehicle there with heading set to centerline direction ± random offset of up to 30°.
- Set initial speed to a random value between 0 and 30 m/s.
- Set initial steering to 0, throttle to 0.

This is the single most important anti-overfitting measure. Don't skip it. Yosh discovered this through painful trial and error; we get to start with it.

## Training/eval split

Two `gym.Env` instances, identical except:
- **Training env**: random spawn (as above), wrapped in `Monitor` for logging.
- **Eval env**: fixed spawn at checkpoint 0, deterministic, used by `EvalCallback`.

```python
from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/eval/",
    eval_freq=10_000,         # every 10k steps
    n_eval_episodes=3,
    deterministic=True,
    render=False,
)

model.learn(total_timesteps=500_000, callback=eval_callback)
```

This gives us:
- A clean reward curve (eval) alongside the noisy training curve.
- The best model auto-saved, not the final model.
- A natural stopping point: when eval reward plateaus across many checkpoints.

## Hyperparameters (PPO, v1 starting point)

```python
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,           # SB3 default, fine to start
    n_steps=2048,                 # rollout length before update
    batch_size=64,
    n_epochs=10,
    gamma=0.99,                   # discount factor; matches Yosh's "long-term reward"
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,                # entropy bonus = exploration
    verbose=1,
    tensorboard_log="./logs/",
    device="cuda",
)
```

We can tune later. These are standard PPO defaults that work for most continuous control tasks.

## Action frequency

20 Hz. Each `step()` advances BeamNG by 50 ms of sim time (typically 3 sim steps at 60 Hz). Yosh used 10 Hz for training and 30 Hz for eval; 20 Hz is a good compromise for BeamNG's higher-fidelity physics, where the simulation needs time to "settle" between agent decisions.

## File structure

```
beamng-mikey/
├── envs/
│   └── beamng_env.py          # the env class
├── train_beamng.py             # entry point for training
├── watch_beamng.py             # load a model, drive in BeamNG, watch
├── checkpoints/                # gitignored; saved models
│   └── best_model/             # auto-saved by EvalCallback
├── logs/                       # gitignored; TensorBoard logs
├── docs/
│   ├── phase1_env_spec.md      # this file
│   └── references.md           # Yosh transcript, papers, etc.
└── RUNS.md                     # journal: one entry per training run
```

## What we're explicitly *not* doing in v1

- No camera observations. (Could add later for a vision-based experiment.)
- No multi-vehicle training. (Could add later for population-based methods.)
- No curriculum (easy → hard maps). (Random spawn does most of the work.)
- No reward shaping beyond progress. (Earn the right to complicate.)
- No checkpoint detection from BeamNG's built-in waypoints. (Use our own checkpoint list, simpler.)

If Mikey wants any of these, write them in `IDEAS.md` for v2+.

---

## The CC prompt

When ready to build, paste this into Claude Code:

> Read `docs/phase1_env_spec.md` and `CLAUDE.md` carefully before writing anything.
>
> Build `envs/beamng_env.py` according to the spec. It should be a single Gymnasium-compatible environment class called `BeamNGRaceEnv` with:
>
> - `__init__` that takes a `random_spawn: bool` flag (True for training, False for eval) and a `home: str` path to the BeamNG.tech install.
> - 9-dim Box observation space and 2-dim Box action space as specified.
> - `reset()` that handles random spawn when the flag is True, deterministic spawn at checkpoint 0 when False.
> - `step()` that maps action[0] to steering and action[1] to throttle/brake, advances the sim 3 physics steps (~50ms at 60Hz), computes observation, reward, and termination.
> - Helper methods: `_get_observation`, `_compute_reward`, `_check_done`, `_distance_along_centerline`, `_is_flipped`, `_is_off_track`, `_lap_completed`.
> - Hardcode a list of ~20 checkpoint positions for the first test track. We'll pick the actual map and extract real centerline points later; use placeholder positions for now and add a TODO comment.
> - All numeric thresholds (off-track distance, stuck threshold, etc.) should be class constants at the top of the file for easy tuning.
>
> Also build `train_beamng.py`:
>
> - Creates train_env (random_spawn=True) and eval_env (random_spawn=False).
> - Wraps train_env in Monitor.
> - Configures PPO with the hyperparameters from the spec.
> - Sets up EvalCallback saving best model to `./checkpoints/best_model/`.
> - Trains for 500,000 timesteps by default, configurable via `--timesteps` CLI arg.
> - Takes `--run-name` CLI arg for TensorBoard logging.
> - On exit (clean or interrupted), appends an entry to `RUNS.md` with the run name, hyperparameters, timesteps completed, peak eval reward, final eval reward, and a `# Notes` section for me to fill in after watching.
>
> And `watch_beamng.py`:
>
> - Takes `--model-path` CLI arg (defaults to `./checkpoints/best_model/best_model.zip`).
> - Loads the model, creates an eval env, runs N episodes (default 3) printing total reward per episode.
> - Uses `deterministic=True` by default; add `--stochastic` flag to use `deterministic=False`.
>
> Don't run anything yet. Just write the files. After writing, show me a `git status` and a high-level diff summary. I'll review before we commit.
>
> Constraints from `CLAUDE.md`:
> - Mikey is 9 and should be able to follow what each function does. Add a short comment above every method explaining it in plain language.
> - Minimum code, no speculative abstractions.
> - `tech.key` must stay untracked. Verify after writing.

---

## Followup tasks (after env is built and tested)

1. Pick the actual first map with Mikey. Extract centerline checkpoints via BeamNGpy (one-time script).
2. Replace the placeholder checkpoints in `beamng_env.py`.
3. Run a sanity-check training (50k steps, ~10 min) to confirm reward goes up at all.
4. If reward goes up, kick off the real 500k-step training overnight.
5. Watch with `watch_beamng.py`. Mikey describes what he sees.
6. Update `RUNS.md`. Decide what v2 changes.
