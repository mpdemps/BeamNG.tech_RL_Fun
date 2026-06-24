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

---

> **Journal gap, runs 9-17.** These were tracked in `docs\mikey_run*_plan_*.md`
> rather than here. In brief: the constraint-stack saga, runs 10-15 bolted
> hand-tuned action constraints onto the policy to stop a T1 spin (run11 traction
> cap, run12 Grad-CAPS action-smoothness, run13 steering-rate limit, run14 ESC
> slip-angle throttle cut, run15 authority cap). Deep research
> (`docs\research_learn_to_corner_2026-06-15.md`) concluded the constraint-stacking
> was the anti-pattern and braking/cornering must be LEARNED via reward +
> observation + curriculum. run16 reset to a braking-aware speed-target reward
> (`envs\speed_profile.py`, the v_target profile) + an 18-dim obs with a
> speed-scaled curvature preview + slip-angle, retiring the constraint stack
> (kept ONE steering-rate limit). run17 added the spawn curriculum (random starts
> around the track) + a clean 8m off-track termination. run17's verdict: the
> timid local optimum, it crawled to ~100m at ~2.4 m/s from the start line and
> never reached T1.

---

## run18_anti_timid_nudge (SAC, fresh, 500k)

### Date

2026-06-16 launch to 2026-06-18 complete; G14 watch 2026-06-19

### Hyperparameters

- Algorithm: SAC (plain), fresh launch, full 500k (restarted once at step 0 to
  add the eval/* TensorBoard instrumentation, behavior-neutral logging only)
- learning_rate: 1e-4, gamma: 0.99
- `--steer-rate 0.5`, `--random-spawn` (run17 spawn curriculum kept)
- Changes since last run (ONE behavioral change): the anti-timid nudge, a new
  reward term `+ W_MATCH * min(v, v_target) * gated_alignment`, W_MATCH=0.10,
  alignment-gated so it only pays for speed going forward on-line. The `min`
  cap is load-bearing, flat above v_target, so it adds zero incentive to exceed
  the grip-limited target (calibration confirmed reward peak stays at
  v_target). Directly attacks run17's crawl local optimum: carrying speed now
  pays, so "crawl" stops being reward-optimal. Plan:
  `docs\mikey_run18_plan_anti_timid_nudge.md`.
- Also added (instrumentation, not behavior): 20 eval/* TB cards, mean_speed,
  max_arc, over_speed_frac, beta mean/p90, per-termination fractions, reward-term
  decomposition (r_progress / r_match / r_overspeed / r_slip), checkpoints. So we
  read the trustworthy metrics live instead of manual start-line replays.

### Timesteps

500,000 (complete)

### Peak Eval Reward

~680 (eval mean_reward, mid-run). NOTE: the reward is partly the match term
paying for any forward speed, so it is NOT the trustworthy signal this run. The
honest metrics: eval mean_speed plateaued ~20 m/s (from run17's 2.4 crawl),
eval r_progress ~300 (real ground covered), term_stuck pinned at 0.

### Final Eval Reward

~615 (eval mean_reward). Decisive honest numbers: mean_speed ~20 m/s, but
max_arc plateaued ~316 (best-ever 347) and checkpoints stuck at 1, the car
reaches T1's apex (338m) but never the exit (394m). End-of-run eval showed
beta_p90 spiking to ~53 and an over-speed blip, the throttle-oversteer signature
(see watch).

### Watching Notes

G14 watch of best_model (3x deterministic, identical every time): (1) good
controlled acceleration to ~90 kph; (2) slight left-right oscillation but in
control; (3) approaching T1, BRAKES correctly, slows to ~40 kph; (4) at TURN-IN
(entry, nowhere near the apex/exit), FLOORS the throttle and the rear steps out;
(5) continued full throttle, SPINS and slides off the OUTSIDE (right side of the
left corner) and off the track, before reaching the apex. VERDICT: the run18
thesis is PROVEN, the timidity is broken, the car now drives at real pace AND
brakes for the corner. The remaining failure is NOT braking and NOT the line, it
is THROTTLE DISCIPLINE on corner ENTRY: at turn-in it mashes the gas on a 748hp
RWD car, the rear breaks loose (power oversteer) and it spins off the outside
before even getting through the corner. NOTE on the reward: at turn-in v_target
is at the corner's low point, so flooring it trips BOTH the over-speed penalty
AND the slip penalty, yet the policy does it anyway, so those penalty signals are
too weak versus the throttle's progress reward (or the policy is simply
unconverged at a corner it rarely gets through). The slip-angle penalty meant to
teach throttle control is the most direct lever.

### Mikey's Hypothesis for Next Run

(Proposed, pending Mikey's approval.) The car drives great and even brakes for
the turn, but the instant it starts turning into the corner it hits the gas too
hard and the back wheels spin out, like flooring a powerful car on a wet road.
Make spinning the tires cost more points so it learns to squeeze the gas gently
as it turns in instead of flooring it: raise the tire-slip penalty (W_SLIP) and
likely lower BETA_DEAD from 9 degrees so it notices the slide sooner. This is the reward teaching the skill
(the GT Sophy tire-slip lesson), NOT re-adding scripted traction control. Likely
WARM-start from run18's best_model, the policy already brakes and turns, it just
needs to refine the throttle, not relearn driving.

### Artifacts

- TensorBoard log dir: `logs\mikey_run18`
- Best model: `checkpoints\mikey_run18\best_model\best_model.zip`
- Plan: `docs\mikey_run18_plan_anti_timid_nudge.md`

---

## run19_slip_penalty (SAC, warm from run18, STOPPED ~220k of 500k)

### Date

2026-06-19

### Hyperparameters

- Algorithm: SAC, warm-start from run18's best_model
  (`--warm-start checkpoints/mikey_run18/best_model/best_model.zip`,
  learning-starts 5000)
- learning_rate: 1e-4, gamma: 0.99, `--steer-rate 0.5`, `--random-spawn`
- Changes since last run (ONE term, two knobs): strengthen the slip-angle penalty
  to teach throttle discipline at T1 turn-in. BETA_SLIP_DEAD 9.0 -> 7.0 (fire at
  the slide onset, not after), W_SLIP 0.05 -> 0.15 (3x, so the slide outcosts the
  throttle in the recoverable zone). Calibrated against a deterministic spin trace
  of run18 (penalty was reading 0 through beta 5-8 deg and only -0.08 once on).
  Everything else stays run18. Plan: `docs\mikey_run19_plan_slip_penalty.md`.

### Timesteps

STOPPED at ~220k of 500k. The mid-run G14 watch gave a decisive (negative)
verdict; no reason to finish.

### Peak Eval Reward

~550 (eval mean_reward, mid-run). As always the reward is not the trustworthy
signal. The honest cards at ~220k: mean_speed on a clean DOWNWARD trend
(~20 -> ~13.6, the timidity guard breaching), max_arc stuck ~287 (best ~340,
never the 394 exit), r_slip still deeply negative (~-100, not climbing toward 0),
beta_p90 oscillating 10-45 (still sliding intermittently).

### Final Eval Reward

n/a (stopped). The state at stop: slower AND still spinning AND stuck at the T1
wall, with the two signals pointing opposite ways (speed down = lower W_SLIP;
still-spinning = raise W_SLIP), i.e. no good single weight.

### Watching Notes

Mid-run G14 watch of best_model @ ~220k (3x deterministic, identical): (1) floors
it for a second at launch, then backs off and VERY SLOWLY accelerates ~30 -> ~100
kph by T1 (the timidity, confirmed); (2) all three run wide to the RIGHT at T1
entry and then FLOOR it; (3) no attempt to turn, one terminated off-track, two
drove straight into the T1 wall on the right. VERDICT: the policy has stopped
trying to corner. Root cause: at T1, "cornering hard at the limit" and "the rear
starting to spin" produce the SAME slip-angle (beta), so the strengthened penalty
taxed the aggressive cornering, not just the spin. The policy's safest response to
"turning hard costs me" is to stop turning, going straight and flooring it
generates almost no lateral slip, so it dodges the penalty and crashes instead.
This is why the cards conflicted and why no W_SLIP value works: the slip penalty
cannot distinguish good cornering slip from bad spin slip. It is the wrong lever
for T1.

### Mikey's Hypothesis for Next Run

The "don't slide" rule backfired, the car learned that turning hard is what makes
it slide, so it stopped turning and drives straight off the road. We cannot fix
that by tuning the rule, because hard turning and spinning look the same to it. So
give the car a RACING LINE (an explicit drawn path through the corner to follow)
plus TRACK EDGES in its eyes (so it can see the wall coming instead of only being
terminated by it). The line encodes "after you enter wide-right, turn left to the
apex," the exact move it is not making; the edges give boundary awareness for the
wall it keeps hitting. See `docs\mikey_run20_plan_racing_line.md`.

### Artifacts

- TensorBoard log dir: `logs\mikey_run19`
- Best model: `checkpoints\mikey_run19\best_model\best_model.zip`
- Plan: `docs\mikey_run19_plan_slip_penalty.md`

---

## run20_racing_line (SAC, warm from run18, 500k)

### Date

2026-06-19

### Hyperparameters

- Algorithm: SAC, warm from run18's best_model. plain SAC, gamma 0.99,
  `--steer-rate 0.5`, `--random-spawn`.
- Changes since last run (ONE coherent change): RETARGET the obs/reward reference
  from the centerline to an offline MINIMUM-CURVATURE RACING LINE
  (`envs\racing_line.py` + `data\raceline_builtin.py`, rolled ourselves, MIT-clean,
  no LGPL TUM dep). Pure retarget: obs SHAPE unchanged (18 dims, just computed
  relative to the line), no new reward terms, no edge-distance dims (we have no
  authoritative edges). Off-track still measured vs the real ROAD centerline (8m),
  not the line. Slip penalty REVERTED to run18's gentle W_SLIP=0.05/BETA_DEAD=9
  (run19's 0.15/7.0 caused corner-avoidance). Plan:
  `docs\mikey_run20_plan_racing_line.md`.

### Timesteps

500,000

### Peak Eval Reward

~680 (eval mean_reward, mid-run ~150-180k). max_arc peaked ~270 smoothed (best
~347), mean_speed ~20-24 m/s.

### Final Eval Reward

Degraded past the peak (the run20 instability): max_arc fell back toward ~250,
reward off its peak. Never cleared T1 cleanly / no completed lap.

### Watching Notes

G14 watch of the peak best_model (3x): the BEST driving of the project so far. It
follows the racing line and ENTERS T1 well (2 of 3 runs). Two failures remain:
(1) the old left-right WEAVE on the straights, (2) OVER-BRAKING while turning, so
the rear overtakes the front (trail-brake oversteer) because it arrives ~100 kph
and brakes late/hard mid-corner. And pure RL on the line PLATEAUED then degraded
(peaked ~150-180k, declined after), never punching through. The line concept is
sound (it fixed run19's corner-avoidance); pure RL just hit the compute ceiling.

### Mikey's Hypothesis for Next Run

The line works, but the AI can't smoothly execute it on its own at our compute
(one machine). Give it training wheels: a hand-coded autopilot that follows the
line and brakes at the right spots, and let the AI ride on top making small
corrections (residual RL). The autopilot does the smooth driving the AI keeps
getting slightly wrong. See `docs\mikey_run21_plan_residual_fadeout.md`.

### Artifacts

- TensorBoard log dir: `logs\mikey_run20`
- Best model: `checkpoints\mikey_run20\best_model\best_model.zip`
- Plan: `docs\mikey_run20_plan_racing_line.md`

---

## run21_residual_fadeout (SAC/BlendSAC, warm from run18, 600k)

### Date

2026-06-19

### Hyperparameters

- Algorithm: BlendSAC (guided residual with CONTROLLER FADE-OUT). Warm from run18.
  600k total: beta=1 warmup (0-100k), linear anneal 1->0 (100k-400k), beta=0 hold
  (400k-600k). EVAL ALWAYS at beta=0 (measures the standalone policy).
- Built `envs\base_controller.py`: pure-pursuit steering + speed-profile P
  throttle/brake. The controller ALONE LAPS the full track (first thing in the
  project to lap 4326 m), MIT-clean. Also fixed a latent off-track bug
  (`_is_off_track` was point-to-VERTEX on the sparse centerline; changed to
  point-to-SEGMENT), which had been silently false-killing back-half spawn
  episodes since run17.
- Goal: fade the controller out so the END PRODUCT is a PURE learned policy
  (training wheels removed). Plan: `docs\mikey_run21_plan_residual_fadeout.md`.

### Timesteps

600,000

### Peak Eval Reward

The beta=0 (solo) policy came alive in the back half of the anneal (~330k): max_arc
climbed to ~162 (best ~234), beta_p90 fell from ~160 to ~30, reward climbed. That
was the peak.

### Final Eval Reward

Degraded in the beta=0 hold. Once the controller was fully gone, the solo policy
fell apart (max_arc back to ~70-90). The peak (~234) was still SHORT of T1's entry
(294 m).

### Watching Notes

G14 watch of the peak best_model (the solo beta=0 policy): it is a WEAVING CRAWLER.
Floors it briefly, backs off, accelerates very slowly to ~30 kph weaving, slows to
a stop, then reverses. VERDICT: full controller removal FAILED at our compute. The
policy leaned entirely on the controller and could not drive solo. The eval reward
that climbed to ~300 at the peak was flattered by the match term paying for the
crawl, the cards-lying trap again, your eyes were the truth.

### Mikey's Hypothesis for Next Run

A fully self-driving AI is beyond this one machine (the famous result needed 1000+
machines). So keep the controller permanently and let the AI add small polish on
top, a controller-led hybrid. The car laps for sure (the controller does it), and
the AI improves it where it can. See `docs\mikey_run22_plan_residual_hybrid.md`.

### Artifacts

- TensorBoard log dirs: `logs\mikey_run21`, `logs\mikey_run21_resume`
- Best model: `checkpoints\mikey_run21\best_model\best_model.zip`
- New code: `envs\base_controller.py` (the lapping controller), `envs\blend_sac.py`
- Plan: `docs\mikey_run21_plan_residual_fadeout.md`

---

## run22_residual_hybrid (SAC, FRESH, 500k)

### Date

2026-06-19

### Hyperparameters

- Algorithm: plain SAC, FRESH. ADDITIVE BOUNDED RESIDUAL:
  `applied = controller(obs) + clip(policy(obs), +/-0.12)` (`envs\residual_hybrid.py`).
  Controller at FULL (laps by construction); the RL adds a small bounded
  correction it can never use to collapse the car. Eval = the FULL hybrid.
  Controller-alone lap logged as the fixed baseline. Reward unchanged.
- Changes since last run: switched from the convex fade-out blend to the additive
  bounded residual (controller-led, not policy-led). Plan:
  `docs\mikey_run22_plan_residual_hybrid.md`.

### Timesteps

500,000

### Peak / Final Eval Reward

The hybrid laps controller-led; the residual reached far past T1 from the warm
foundation. Numbers not the trustworthy signal here (see watch).

### Watching Notes

G14 watch of best_model: the controller takes the racing line through T1 cleanly,
but the RESIDUAL over-throttles on the T1 EXIT and SPINS, then does full-throttle
DONUTS that NEVER TERMINATE, even out in the grass. VERDICT: a contained in-place
donut stays within the 8m off-track radius, and no terminator catches it (off_track
is position-based; backward/stuck timers reset every revolution). The
non-terminating donut corrupts the buffer AND flatters the eval (long episodes that
never end), so the residual never gets a clean penalty for over-throttling.

### Mikey's Hypothesis for Next Run

The car doesn't know it's off the road on grass, so it keeps flooring it, and a
spun-out car never stops to learn its mistake. Add a spin detector that ends the
episode (a hard stop when it loses control) and give the car a sense of when it's
off-track on low grip, so it learns to ease off. See
`docs\mikey_run23_plan_grip_awareness.md`.

### Artifacts

- TensorBoard log dir: `logs\mikey_run22`
- Best model: `checkpoints\mikey_run22\best_model\best_model.zip`
- New code: `envs\residual_hybrid.py`
- Plan: `docs\mikey_run22_plan_residual_hybrid.md`

---

## run23_grip_awareness (SAC, FRESH, 500k; power-outage resume)

### Date

2026-06-19 to 2026-06-20

### Hyperparameters

- Algorithm: plain SAC, FRESH. Kept the run22 residual hybrid (controller +
  clip(policy, +/-0.12)). Three-part grip-awareness bundle:
  1. LOSS-OF-CONTROL terminator: yaw rate > 150 deg/s sustained 8 ticks (~0.4s) ->
     end episode + crash penalty (-10). Catches the contained donut regardless of
     position. (Yaw rate chosen over slip-angle beta: beta pins to 0 below the
     8 m/s floor, so a slow donut would evade it; yaw is speed-independent.)
  2. GRIP obs[18] = normalized dist-to-road (0 on road -> 1 at edge). obs 18->19.
  3. Off_track investigation: confirmed working; the donut was a spin-detector gap,
     not an off_track bug.
- Plan: `docs\mikey_run23_plan_grip_awareness.md`.

### Timesteps

500,000 (power outage killed the mini at ~380k; resumed cleanly from
rolling_380000 to 500k, namespace `mikey_run23_resume`).

### Peak Eval Reward

~7,776 (pre-outage peak, ~370-380k). max_arc reached ~3,500 m (best ~3,594), about
81% of the 4326 m lap, mean_speed ~28 m/s. CORNERING PARALYSIS SOLVED (far past the
old T1 wall).

### Final Eval Reward

The resume was high-variance and degraded (swung 127-5339, ended low). Never
completed a lap (term_lap = 0 throughout). Still spinning at ~3,500 m
(term_loss_of_control ~0.33-1.0 on the far evals).

### Watching Notes

G14 watch of the pre-outage peak best_model: drives WELL, reaches turn 12/13 (= T11
in the reference) before spinning. T11 is a sharp right-hander that is OFF-CAMBER
and DOWNHILL. Also: does not take the optimal line on a few fast/gentle corners
(the min-curvature line stays near center on gentle bends), and the back straight
is capped at ~114 kph (the artificial V_MAX=33 cap; diagnostic showed 75% of the
lap capped there, and a measured top speed of 62.8+ m/s shows huge headroom).

### Mikey's Hypothesis for Next Run

The donut/buffer fix worked and the car nearly laps. The remaining spin is at T11,
and the throttle is the suspected lever (the residual stomping gas mid-corner). Cut
the residual's ability to ADD throttle but keep its ability to LIFT (asymmetric
cap), so it can save a slide but can't floor itself into a spin. See
`docs\mikey_run24_plan_throttle_authority_cut.md`.

### Artifacts

- TensorBoard log dirs: `logs\mikey_run23`, `logs\mikey_run23_resume`
- Best model: `checkpoints\mikey_run23\best_model\best_model.zip` (pre-outage peak)
- Plan: `docs\mikey_run23_plan_grip_awareness.md`

---

## run24_throttle_authority_cut (SAC, warm from run23, 500k)

### Date

2026-06-20 to 2026-06-21

### Hyperparameters

- Algorithm: plain SAC, WARM from run23 MAIN best_model. ASYMMETRIC per-channel
  residual bound (`envs\residual_hybrid.py`): steering +/-0.12 (full), throttle
  [-0.12, +0.05] (full LIFT, ADD-throttle cut hard). Sign verified: positive =
  throttle, negative = brake/lift. Added SIGNED throttle metrics
  (residual_throttle_pos, residual_throttle_satfrac) to isolate the +cap binding
  from lift usage. Reward unchanged. Plan:
  `docs\mikey_run24_plan_throttle_authority_cut.md`.

### Timesteps

500,000

### Peak Eval Reward

~7,800 (matching run23). BEST PROGRESS of any run: max_arc ~3,500 m (~81%),
mean_speed ~28 m/s (up from the ~24 norm).

### Final Eval Reward

No completed lap (term_lap = 0 throughout). Still spinning at ~3,500 m (T11).
Critically, residual_throttle_satfrac stayed PINNED at ~0.70 the entire run: the
policy is jammed against the +0.05 throttle cap (wants more gas) AND still spins.
So the throttle residual is NOT the remaining spin cause, the throttle lever is
EXHAUSTED. The donut corruption is gone (episodes terminate cleanly throughout).

### Watching Notes

G14 watch of best_model: 114 kph max (the V_MAX=33 cap), mild left-right weave on
straights, loses it at T11 every time (sharp right, off-camber, downhill). Top-speed
probe measured 62.8 m/s (226 kph) in just the 300m opening straight, still climbing,
so V_MAX=33 is throttling the car to under half its capability. Slope diagnostic
later confirmed T11 is a -23% grade at turn-in, uniquely the steepest corner on the
track (next worst -3%), which is why its rear is the lightest and it is the spin wall.

### Mikey's Hypothesis for Next Run

The throttle lever is used up, and the spin is one specific corner: T11, which is
steeply downhill and off-camber so the rear is light there. The speed profile
assumes the same grip everywhere, so it asks too much speed at T11. Make the profile
GRIP-AWARE: open up the straights (raise V_MAX from 33 toward 55+), and slow down
the off-camber/downhill corners like T11 to give the grip margin back. Go fast AND
finish the lap. See `docs\mikey_run25_plan_grip_aware_profile.md` (and the
`run24_raceline_diagnostic` / top-speed findings).

### Artifacts

- TensorBoard log dir: `logs\mikey_run24`
- Best model: `checkpoints\mikey_run24\best_model\best_model.zip`
- Plan: `docs\mikey_run24_plan_throttle_authority_cut.md`

---

## run25_grip_aware_profile (SAC, FRESH, 500k)

### Date

2026-06-21 to 2026-06-24

### Hyperparameters

- Algorithm: plain SAC, FRESH, lr 3e-4. Residual hybrid kept (controller +
  clip(policy, steer +/-0.12, throttle -0.12/+0.05)). GRIP-AWARE speed profile
  (`envs\speed_profile.py`): V_MAX 33 -> 55 (straights open to ~187 kph), and
  per-corner A_LAT trimmed by downhill slope + an explicit off-camber cap A_LAT=8
  over T11 (v_target there -18%). Flat corners unchanged at 12.
- Slope measurement (the key finding): T11 is a -23% grade at turn-in, UNIQUELY
  the steepest (next worst -3%), which is why its rear is the lightest and it is
  the spin wall. Controller-alone gate: laps the new fast profile cleanly,
  survives T11. Plan: `docs\mikey_run25_plan_grip_aware_profile.md`.

### Timesteps

500,000 (single clean segment, no freeze-restarts).

### Peak / Final Eval Reward

REGRESSION. Peak max_arc ~2,494 m (~58%), WORSE than run24's ~3,500 m. No lap
(term_lap = 0 throughout). The policy never converged, it oscillated between
decent runs (~2,500 m) and T11 spin-outs (~85-120 m) the whole way, ending ugly.

### Watching Notes

G14 watch of the peak (~330k) best_model: same T11 spin as run24, plus the mild
straight weave. The decisive cards: residual_abs_steer pinned at the full 0.12
EVERY eval (the policy wants more authority than the +/-0.12 box allows), and
term_loss_of_control swinging to 1.0 on the craters, correlated with high
residual_throttle_satfrac. VERDICT: three flavors of bounded residual now
(run22 additive, run24 throttle-cut, run25 grip-profile) ALL plateau the same
way, a controller that laps alone + a saturated small residual = an unstable
hybrid that spins, never a lap. The bounded residual is the wrong lever; the
policy fights the controller at its cap and destabilizes it. Bigger picture: at
our one-machine compute, RL in EVERY form tried (pure run20/21, residual
22/24/25) either cannot drive the track or destabilizes the controller. The only
thing that laps the whole track is the hand-coded controller.

### Mikey's Hypothesis for Next Run

The real villain across all 25 runs is the CAR: a 748hp rear-drive race car that
spins at the slightest provocation. The RL has been failing to tame an unforgiving
supercar, not failing to drive. Switch to a car that can BOTH lap and drift: the
GTS (road) config of the same Scintilla, same RWD V10 so it still drifts, but
detuned + street tires + softer suspension so its limit is gentle and learnable.
See the "Car switch" note below.

### Artifacts

- TensorBoard log dir: `logs\mikey_run25`
- Best model: `checkpoints\mikey_run25\best_model\best_model.zip`
- Plan: `docs\mikey_run25_plan_grip_aware_profile.md`

---

## Car switch: RACE -> GTS (the "do-both" platform)

### Date

2026-06-24

### Why

25 runs proved the bottleneck is the car, not the pipeline. The Scintilla RACE
config (748hp RWD, race tires/coilovers/wing/LSD) has a razor-edge limit that
punishes any imprecision, so pure RL plateaus and the residual destabilizes. Mike
asked for ONE car that can both LAP and DRIFT. The answer is RWD with enough power
to drift but a forgiving, progressive limit so it is learnable, which is exactly
the GTS (road) config of the same car.

### GTS vs RACE (measured)

- Same 5.0 V10 + RWD (still drifts; has a factory drift mode), but DETUNED: stock
  ECU (8600 rpm) vs race ECU (9000-10500), sport intake. Top speed 76 m/s (274
  kph) vs race 87 m/s (314 kph).
- Grip ~0.8x race (~1.3g true vs ~1.6g), street sport tires (narrower,
  progressive) vs race compound, softer adaptive suspension, open diff (race has
  LSD), no wing. Gentler breakaway (170 vs 243 deg/s step-steer).
- The proof of the softer limit: the race controller laps its profile, but the
  GTS spins out at T6 on that exact setup, it cannot hold race speeds, so it needs
  its own gentler profile.

### Recalibration (architecture UNCHANGED)

Only car-specific constants moved (`envs\speed_profile.py` GTS variant):
A_LAT_MAX 12 -> 10, A_BRAKE 9 -> 7, A_ACCEL 6 -> 5, V_MAX 55 -> 50, T11 cap
8 -> 6.5. Controller steering (pure-pursuit) gains transferred as-is; the data did
not call for speed/brake gain changes, so they were left untouched. Obs (19-dim),
reward, racing line, and the residual-hybrid architecture all carry over.

### The gate (controller alone, no policy)

LAP @ 83.1s, max_arc 4329 m, term=lap, peak 47.3 m/s (170 kph). Every corner OK,
INCLUDING both T11 segments (the off-camber downhill corner that wrecked runs
18-25). The forgiving do-both platform is proven. Map: `docs\gts_gate_map.png`.

---

## run26_purerl_gts (SAC, FRESH, 500k) -- IN PROGRESS

### Date

2026-06-24 (launched)

### Hyperparameters

- Algorithm: plain SAC, FRESH. PURE RL: NO controller, NO residual, the policy
  outputs the full steering + throttle and learns to drive from scratch. Keeps
  the racing-line reward, the 19-dim grip-aware obs, the loss-of-control spin
  terminator, the GTS grip-aware profile, --steer-rate 0.5, --random-spawn.
- The test: can the RL learn to lap a car that COOPERATES (the GTS), where it
  plateaued on the unforgiving race car (run20)? This is the project's core
  question. Pre-flight probe + 7k smoke passed (plumbing clean, terminators fire,
  no NaN; the fresh policy only crawls at ~3.7 m/s, as expected). Smoke note: 10
  of 40 episodes ended in a FLIP, the taller/softer GTS rolls where the race car
  spun; watch eval/term_flip.

### Verdict

TBD (running). The read: eval/max_arc climbing toward 4326 as a LEARNED lap, with
term_off_track / term_flip / term_loss_of_control falling and mean_speed climbing.

### Artifacts

- TensorBoard log dir: `logs\mikey_run26`
- Launcher: `scripts\run_mikey26.sh`
