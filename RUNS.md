# Training Runs Journal

One entry per training run, in chronological order. **Append, never overwrite.** This file is the record of what we tried, what happened, and what Mikey wants to do next. When we look back in six months wondering "why did we change the entropy coefficient?", the answer lives here.

Update this file *after* every training session, *before* starting the next one. If we skip a run's entry, the journal stops being useful.

Use descriptive run names that hint at what's different: `lr3e4_300k_v2`, not `PPO_5`.

---

## Run 0: example_template_run

(This is a template entry showing the format. Delete or ignore when reading; real runs start at Run 1.)

### Date

2026-05-11

### Hyperparameters

- learning_rate: 3e-4
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Changes since last run: N/A (first run)

### Timesteps

300,000

### Peak Eval Reward

(filled in after training: highest mean reward across all `EvalCallback` evaluations)

### Final Eval Reward

(filled in after training: mean reward at the last evaluation)

### Watching Notes

Plain-language description of what the agent looks like when we run `watch_*.py`. Does it drive straight? Does it brake into corners? Does it spin out in the same spot every lap? Mikey writes most of this.

### Mikey's Hypothesis for Next Run

What Mikey wants to change and why. Example: "It keeps under-steering on the second left. Maybe give it more turning power by making the steering action stronger." This becomes the change-set for Run N+1.

## real_10k_watch

### Date

2026-05-12

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 10000
- Completed timesteps: 0

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\real_10k_watch`
- Best model: `checkpoints\real_10k_watch\best_model`
- Final model: `checkpoints\real_10k_watch\final.zip`

## real_10k_watch

### Date

2026-05-12

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 10000
- Completed timesteps: 322 (interrupted)

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\real_10k_watch`
- Best model: `checkpoints\real_10k_watch\best_model`
- Final model: `checkpoints\real_10k_watch\final.zip`

## real_10k_watch

### Date

2026-05-12

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 10000
- Completed timesteps: 283 (interrupted)

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\real_10k_watch`
- Best model: `checkpoints\real_10k_watch\best_model`
- Final model: `checkpoints\real_10k_watch\final.zip`

## real_10k_watch

### Date

2026-05-12

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 10000
- Completed timesteps: 10240

### Peak Eval Reward

-530.5219072728158

### Final Eval Reward

-530.5219072728158

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\real_10k_watch`
- Best model: `checkpoints\real_10k_watch\best_model`
- Final model: `checkpoints\real_10k_watch\final.zip`

## real_10k_v2

### Date

2026-05-12

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 10000
- Completed timesteps: 10240

### Peak Eval Reward

-348.50212588835467

### Final Eval Reward

-348.50212588835467

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\real_10k_v2`
- Best model: `checkpoints\real_10k_v2\best_model`
- Final model: `checkpoints\real_10k_v2\final.zip`

## yaw_smoke

### Date

2026-05-13

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 2000
- Completed timesteps: 2048

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\yaw_smoke`
- Best model: `checkpoints\yaw_smoke\best_model`
- Final model: `checkpoints\yaw_smoke\final.zip`

## yaw_smoke_v2

### Date

2026-05-13

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 2000
- Completed timesteps: 2048

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\yaw_smoke_v2`
- Best model: `checkpoints\yaw_smoke_v2\best_model`
- Final model: `checkpoints\yaw_smoke_v2\final.zip`

## builtin_smoke

### Date

2026-05-13

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 2000
- Completed timesteps: 2048

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\builtin_smoke`
- Best model: `checkpoints\builtin_smoke\best_model`
- Final model: `checkpoints\builtin_smoke\final.zip`

## zero_offset_smoke

### Date

2026-05-13

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 2000
- Completed timesteps: 2048

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\zero_offset_smoke`
- Best model: `checkpoints\zero_offset_smoke\best_model`
- Final model: `checkpoints\zero_offset_smoke\final.zip`

## z_offset_smoke

### Date

2026-05-13

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 2000
- Completed timesteps: 2048

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\z_offset_smoke`
- Best model: `checkpoints\z_offset_smoke\best_model`
- Final model: `checkpoints\z_offset_smoke\final.zip`

## zero_offset_v2

### Date

2026-05-13

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 2000
- Completed timesteps: 2048

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\zero_offset_v2`
- Best model: `checkpoints\zero_offset_v2\best_model`
- Final model: `checkpoints\zero_offset_v2\final.zip`

## yaw_bias_smoke

### Date

2026-05-13

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 2000
- Completed timesteps: 2048

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\yaw_bias_smoke`
- Best model: `checkpoints\yaw_bias_smoke\best_model`
- Final model: `checkpoints\yaw_bias_smoke\final.zip`

---

> **Backfill note (2026-06-13):** Entries run4 through run8 below were
> reconstructed from `SESSION_HANDOFF_2026-06-13_run8.md` after the journal lapsed
> following the PPO smoke runs above. Hyperparameters and rewards are from the
> handoff record (some pre-date per-episode logging and are approximate, marked
> `~`). Watching notes summarize the G14 watches. "Mikey's Hypothesis" reflects the
> next-run direction actually taken; it is not Mikey's verbatim wording.

---

## run4_wheelspin (SAC, warm-started from run3, archived)

### Date

(mid-May 2026, exact date not recorded)

### Hyperparameters

- Algorithm: SAC (warm-started from run3 weights)
- learning_rate: 1e-4 (set here; 1e-4 from run4 onward)
- SPEED_WEIGHT: 0.002
- Changes since last run: added the wheelspin penalty
  (`-SPIN_WEIGHT * max(0, slip - SLIP_DEADZONE)`, sensor-validated)

### Timesteps

(~not recorded)

### Peak Eval Reward

~226 (training reward; the eval/behavior was bad despite this)

### Final Eval Reward

(not recorded)

### Watching Notes

Reward climbed to ~226 but the car stayed stuck at cp01. Watching exposed the real
behavior: it still floored and spun, AND it drove in REVERSE down-track. This
surfaced the HEADING-BLIND REWARD LEAK: progress/speed were gated on
velocity-vs-tangent but never on nose direction, so a backward-sliding car farmed
full reward.

### Mikey's Hypothesis for Next Run

Stop paying the car when its nose points the wrong way. Add a heading kill-switch
and start fresh, because warm-starting a policy whose main trick you are about to
penalize just entrenches the trick (the lesson run4 taught).

### Artifacts

- TensorBoard log dir: `logs\mikey_run4`
- Best model: `checkpoints\mikey_run4\best_model`
- Final model: `checkpoints\mikey_run4\final.zip`

---

## run5_heading_fix (SAC, fresh)

### Date

(mid-May 2026, exact date not recorded)

### Hyperparameters

- Algorithm: SAC (fresh)
- learning_rate: 1e-4
- SPEED_WEIGHT: 0.002 -> 0.003
- Changes since last run: heading kill-switch (zero the entire step reward when
  heading_align < -0.2, i.e. nose >90deg off track) + backward termination after
  ~40 consecutive backward steps; SPEED_WEIGHT bump

### Timesteps

~290,000

### Peak Eval Reward

~peaked around 80k steps (best_model captured the peak)

### Final Eval Reward

collapsed (peak-then-decay over ~190k)

### Watching Notes

Drove FORWARD now (the leak was fixed) and reached cp07 by ~80k, then collapsed
over the next ~190k. Watching best_model: forward driving but BANG-BANG control,
steering oscillating and throttle chattering.

### Mikey's Hypothesis for Next Run

Two problems to fix: the car can barely see ahead, and its controls are jerky. Give
it a much longer look down the road and penalize jerky steering/throttle so it
drives smoothly.

### Artifacts

- TensorBoard log dir: `logs\mikey_run5`
- Best model: `checkpoints\mikey_run5\best_model`
- Final model: `checkpoints\mikey_run5\final.zip`

---

## run6_lookahead_smoothness (SAC, fresh)

### Date

(late May 2026, exact date not recorded)

### Hyperparameters

- Algorithm: SAC (fresh)
- learning_rate: 1e-4
- Changes since last run (bundled, signatures non-overlapping):
  (A) 6-point fixed-arc lookahead at [10, 20, 40, 80, 160, 280] m replacing the old
  3-point idx+1/+2/+3 (~13m) horizon; (B) action-smoothness penalties: throttle
  smoothness 0.05 (new), steering smoothness 0.1 -> 0.2

### Timesteps

500,000

### Peak Eval Reward

~110-130 (flat; plateaued from ~40k to 500k)

### Final Eval Reward

~110-130 (no milestone past cp01 @ ~2300 reward)

### Watching Notes

Plateaued the whole run. CC's data: backward-ending episodes climbed 51% -> 88% and
locked in; 86% of spins at 50-150m on the OPENING STRAIGHT (before the 300m
corner). G14 watch of best_model: no bang-bang (the smoothness penalty worked), no
burnout, but a timid ~14 KPH launch; never spun deterministically; drove to the
first corner and into the wall. DIAGNOSIS: the backward termination was a FREE EXIT
(0.0 cost), so the optimal policy was to harvest ~100m of cheap progress then take
the zero-cost door out. An EXPERIENCE failure: episodes died ~150m before the
corner, so the policy never learned the corner.

### Mikey's Hypothesis for Next Run

The car is quitting on purpose because quitting is free. Make quitting cost
something (a penalty for ending up backward) so it has to keep driving and actually
reach the corner.

### Artifacts

- TensorBoard log dir: `logs\mikey_run6`
- Best model: `checkpoints\mikey_run6\best_model`
- Final model: `checkpoints\mikey_run6\final.zip`

---

## run7_price_the_exit (SAC, fresh)

### Date

(early June 2026, exact date not recorded)

### Hyperparameters

- Algorithm: SAC (fresh)
- learning_rate: 1e-4
- Changes since last run (ONE change + logging): BACKWARD_TERM_PENALTY 0 -> -25
  (price the free exit); added termination_reason + recovered_count + per-episode
  stats (mean_speed, max_arc, min_heading_align, max_slip) to the monitor CSV
- Commit: 9a9a449

### Timesteps

(reached ~136k+ at analysis; target 500k)

### Peak Eval Reward

(climbing; not the limiting factor)

### Final Eval Reward

(see watching notes; reward was not the problem)

### Watching Notes

The -25 worked as a survival lever: ep_len jumped to ~585 early, episodes reached
200-376m (past run6's 50-150m), mean_speed 11-16 m/s (real pace, not the crawl).
BUT backward terminations were still ~63% at 136k, now happening AT THE CORNER
(200-335m) rather than mid-straight. G14 watch of rolling_135000 (3x): the BANG-BANG
STEERING WAS BACK, ~0.75 Hz weave on the straight, and when it throttled at the
extreme of a swing it spun. CRITICAL REFRAME (Mike): the car never reaches the
corner; it OSCILLATES ON THE STRAIGHT and spins out there. So run6's "clean"
smoothness was a LUCKY SEED, not a reliable result. CC's math: abs(steer -
prev_steer) is total variation, 4A per cycle regardless of frequency, so a slow
0.75 Hz weave pays almost nothing; the FORM is wrong, not the magnitude.

### Mikey's Hypothesis for Next Run

The jerky-steering penalty can be dodged by weaving slowly. Switch to a penalty
that punishes holding the wheel over when the road ahead is straight, so frequency
can't dodge it.

### Artifacts

- TensorBoard log dir: `logs\mikey_run7`
- Best model: `checkpoints\mikey_run7\best_model`
- Final model: `checkpoints\mikey_run7\final.zip`

---

## run8_weave_spatial (SAC, fresh, then warm-restarted from rolling_65000)

### Date

2026-06-13

### Hyperparameters

- Algorithm: SAC (fresh launch; warm-restarted from rolling_65000 after a sim
  freeze, learning_starts=0, run namespaced `mikey_run8_resume`)
- learning_rate: 1e-4
- Changes since last run (ONE change): CAPS-style SPATIAL-SMOOTHNESS weave penalty,
  frequency- and pose-independent:
  `weave_penalty = -WEAVE_WEIGHT * max(0, |steer| - STEER_DEAD) * straightness * on_line`
  WEAVE_WEIGHT=0.6, STEER_DEAD=0.1; straightness from centerline tangent bend over
  the next 80m (BEND_DEAD 3deg, BEND_FULL 15deg); on_line from |center_off|
  (OFF_DEAD 1.0m, OFF_FULL 2.5m). Both gates pose-independent. The -25 backward
  penalty kept. Probe + smoke passed pre-launch.

### Timesteps

Fresh run froze at ~67.8k (mechanical BeamNG hang, working hypothesis: a violent
corner-rollover hangs the sim). Warm-restarted from the clean rolling_65000;
reached ~115-118k resume steps (~180k training age) at the watch.

### Peak Eval Reward

~280 (eval), rollout reward ~272; new highs, still climbing, no plateau

### Final Eval Reward

~275 (eval) at ~110k resume steps

### Watching Notes

G14 watch of rolling_115000 (3x, deterministic), consistent across all three:
launch fixed (brief floor, no loss of control), throttle modulated, real pace to
~115 kph. BUT the weave is NOT gone: a left-right oscillation starts and GROWS with
speed from ~70 to ~115 kph until the car spins, on the opening straight before the
corner. VERDICT: run8 did not kill the weave. Critical contradiction: weave_penalty
(CSV col 10) read ~0.0 throughout despite the visible weave, so the penalty is not
biting the dangerous weave. Therefore raising WEAVE_WEIGHT (the reserve-ladder
reflex) is the wrong move until we know why it reads zero (1.5 x ~0 is still ~0).

### Mikey's Hypothesis for Next Run

Before changing any dial, find out why the weave-penalty meter reads zero while the
car is clearly weaving. Likely the "are we on the line" or "is the road straight"
gate switches the penalty off exactly when the car is going fast. Measure it on a
real spin, then fix that gate. See `docs\mikey_run8_postwatch_weave_diagnostic.md`.

### Artifacts

- TensorBoard log dir: `logs\mikey_run8` (fresh, pre-freeze) and
  `logs\mikey_run8_resume` (warm-restart)
- Clean checkpoint used for the watch: `checkpoints\mikey_run8_resume\rolling_115000_steps.zip`
- Warm-restart load source: `checkpoints\mikey_run8\rolling_65000_steps.zip`
- Diagnostic brief: `docs\mikey_run8_postwatch_weave_diagnostic.md`
