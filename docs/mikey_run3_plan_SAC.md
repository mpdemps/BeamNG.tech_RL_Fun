# mikey_run3 Plan: SAC + Reduced Speed Reward (Fresh)

Hand to Claude Code AFTER mikey_run2 is stopped and archived. This REPLACES the
earlier warm-start plan (mikey_run3_plan warm-start version is dead, see below).

## Why the warm-start plan is dead

Watching run2's best_model AND milestone_cp05 on the G14 revealed the truth the
reward curves hid: the car never learned to drive. It learned to FLOOR IT AND DO
DONUTS, spinning continuously while drifting forward enough to cross the
evenly-spaced checkpoints by luck. The climbing reward curve (~270 peak, eval
~470, milestones cp01-cp05) was reward hacking: the speed reward (0.0075) on the
powerful race car made spinning-for-speed out-earn careful driving. Both
best_model and the cp05 milestone are donut-spinners, so there is NO good policy
in run2 to warm-start from. Lesson (the recurring one, now decisive): the reward
curve looked like success; the car was spinning. Watch the car, not just the
metric.

Decisive comparison: run1 (NO speed reward) drove properly and reached cp14.
run2 (speed reward 0.0075) learned to spin. The speed reward is the culprit, and
it made things much worse, not better.

## run3 approach: fresh, SAC, much smaller speed reward

Two changes from run2, both targeting the two most likely causes (algorithm
stability + speed-reward magnitude):

1. SWITCH PPO -> SAC. SAC is more sample-efficient (fewer of the slow ~9fps
   BeamNG steps) and more stable for continuous fine-motor control, which is the
   field standard for this kind of problem and what we planned for drift anyway.
   run2's policy was unstable (spiky eval, peaked then regressed into spinning);
   SAC's entropy objective and off-policy replay tend to find controlled
   solutions rather than degenerate ones. Runs on CPU, no GPU needed. SB3 makes
   it a drop-in: `from stable_baselines3 import SAC`.

2. SPEED_WEIGHT 0.0075 -> 0.002 (a ~75% cut). Keep the SAME soft formula
   speed_reward = SPEED_WEIGHT * forward_speed * align (do NOT add a hard align
   gate, we are changing one thing at a time; the soft multiplier already shrinks
   a spinning car's speed reward via its low align, the problem was just that
   0.0075 was big enough to win anyway). At 0.002 the spin's speed reward should
   be negligible vs the progress/checkpoint reward from actually getting around
   the track. If 0.002 still gets gamed, the NEXT lever is a hard align gate
   (align > 0.8), but not yet.

## Keep unchanged from run2

- Race Scintilla (race.pc) + spawn fix (formula C) + SMOOTH_WEIGHT=0.1 +
  seam-fixed progress tracker + fixed-start curriculum + checkpoint/lap bonuses.
- Only two things change: PPO->SAC, and SPEED_WEIGHT 0.0075->0.002.

## Fresh start, not warm-start

No warm-start. Both run2 policies are donut-spinners; warm-starting carries the
spinning forward. Learn clean. (Also, SAC has its own replay buffer; starting
fresh is the clean path.)

## Before launching

1. STOP run2 gracefully (SIGINT in its tmux, let it save), then ARCHIVE it:
   `cp -r ~/projects/beamng-mikey/checkpoints/mikey_run2 ~/projects/beamng-mikey/checkpoints/mikey_run2_ARCHIVE`
   (Preserve run2 as the documented reward-hacking example; do not clobber.)

2. SAC hyperparameters: use SB3 SAC sensible defaults for continuous control to
   start (learning_rate ~3e-4, buffer_size large enough for the run, batch_size
   256, tau 0.005, gamma 0.99, train_freq 1, gradient_steps 1, ent_coef "auto").
   Report the config you pick before launching. Do not over-tune; defaults first.

3. SMOKE TEST HARD (~5-10k steps): SAC is a bigger change than a weight tweak, so
   verify carefully. Confirm: (a) SAC loads and runs, (b) reward is bounded/sane,
   (c) speed_reward stays a SMALL fraction of total (report the ranges, like we
   did for run2's smoke), (d) the car is NOT immediately spinning. NOTE: SAC's
   very early behavior looks random/exploratory (it fills a replay buffer first),
   so "not good yet" is expected; we are checking "runs cleanly + reward sane,"
   not "drives well." Show me the smoke reward ranges before the full run.

## Run config

- Name: mikey_run3.
- Algorithm: SAC (fresh).
- SPEED_WEIGHT=0.002, soft formula speed*align unchanged. SMOOTH_WEIGHT=0.1.
- Race Scintilla, spawn fix, all else per run2.
- 500k timesteps (or note: SAC may need fewer to reach the same point given
  sample efficiency; 500k is the cap, watch the curves).
- tmux, no systemd timer, pkill BeamNG first, tee to logs/mikey_run3.console.log.

## What to watch (and the key trap to avoid)

DO NOT trust the reward curve alone this time. The decisive check is WATCHING the
car (load a snapshot on the G14) to confirm it DRIVES rather than spins. Plan to
watch an early milestone once one appears, if the car is doing donuts again, the
metric is lying again. Specifically:
- Watch rollout reward AND eval, but treat them as necessary-not-sufficient.
- The real gate: does a milestone model drive the racing line (controlled,
  brakes for corners) or spin? Watch it.

## Expected payoff / fallbacks

If SAC + 0.002 speed works: the car drives controlled (no spin), the small speed
reward prevents crawling, and it should progress through the track properly like
run1 did, but on the race car and (via SAC) more sample-efficiently.

Fallbacks if it STILL spins (run4 levers, introduce ONE at a time for clean
attribution, do NOT stack):
- Add a yaw-rate spin penalty: spin_penalty = -SPIN_WEIGHT * max(0,
  abs(yaw_rate) - YAW_THRESHOLD). Threshold set HIGH so normal cornering (incl.
  the 80km/h left) is free and only genuine spinning (sustained high rotation) is
  penalized. This directly targets the donut's defining feature. Preferred first
  fallback if SAC+0.002 still spins. (yaw_rate available from vehicle state/IMU.)
- Alternative: slip-angle penalty (penalize heading-vs-velocity divergence, i.e.
  car sideways). Note this is the OPPOSITE of the drift-phase reward; here we want
  the nose pointed where it's going.
- Strengthen the off-track penalty (already exists as a terminal ~-10; could
  raise it, or add a graded dist-from-center penalty so the car learns to stay
  centered continuously rather than only at the boundary cliff).
- Add the hard align gate (speed only when align > 0.8) to deny the donut its
  speed reward directly.
- Strengthen SMOOTH_WEIGHT (penalize the donut's wild steering harder).
- Accept the race car is spin-happy and better suited to the DRIFT phase (where
  breaking traction is a feature, not a bug) than to careful racing; consider
  run1's slower car for the lap goal.

Note: off-track is ALREADY penalized (terminal ~-10). The donut problem is not
that off-track is unpunished; it is that the spinner earned enough speed +
progress + checkpoint reward BEFORE going off-track to outweigh it. SAC + the
0.002 speed cut attack that root cause (remove the spin's reward source). The
yaw-rate penalty is the targeted backup if the root-cause fix is not enough.
