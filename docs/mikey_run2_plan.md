# mikey_run2 Plan: Spawn Fix + Speed + Smooth Steering

Hand this to Claude Code AFTER mikey_run1 finishes (or is stopped). Do NOT touch
envs/beamng_env.py while mikey_run1 is training (shared file).

Context: mikey_run1 (PPO, fixed-start curriculum, seam-fixed progress tracker)
learned to drive ~93% of the track (cp14 of 15), eval reward ~1,564, rollout
~1,100. It did NOT produce a milestone_lap file and watching best_model on the
G14 did not close the full loop.

Working hypothesis (high confidence): the training car spawns ~90 degrees
clockwise wrong EVERY episode, same as the G14 watcher. Evidence: the policy is
visibly, consistently expert at recovering from a 90-degree spawn, a skill it
could only have acquired by practicing it thousands of times in training. The
[reset] log saying "atan2=-179.76 (matches intended)" only checks the INTENDED
heading math (hand tangent), NOT the car's actual settled orientation, so it
never caught this. The wasted per-episode recovery is the suspected reason the
car tops out at cp14 and cannot close the formal lap.

## Task 1 (FIRST): Confirm the training spawn is actually 90 degrees wrong

Before changing anything, turn the inference into a measured fact.

- Spawn the car headless at idx=0 (a standalone test; if mikey_run1 is still
  running, use a separate port and separate userfolder to avoid contention, or
  run only after it stops).
- After spawning and stepping ~10 times with zero throttle, read the car's
  ACTUAL world-space rotation quaternion from vehicle state, convert to a forward
  vector, and compute the angle between that forward vector and the centerline
  tangent at idx=0.
- Do NOT rely on the hand-tangent / intended-heading check; that only validates
  the math, not the outcome. Measure the settled pose directly, the way
  watch_beamng.py does.
- Repeat 10-20 times and log the per-spawn error.

Report: the measured forward-vs-tangent angle distribution.
- If consistently ~90 degrees clockwise: hypothesis confirmed, proceed to the
  fixed-offset fix in Task 2.
- If it varies (e.g. 90 one time, 180 another): it is a settle/timing issue, not
  a fixed offset. Do NOT apply a blind offset; instead step the sim to settle
  rotation before proceeding, and report back before guessing.

## Task 2: Fix the spawn (shared env code, fixes BOTH training and watcher)

Mike observed a consistent ~90-degree clockwise error on the G14 (car points at
the right edge of the track, drops in, corrects left, drives). A clean constant
offset points at an axis/sign/coordinate-frame convention mismatch in the heading
quaternion construction in _teleport_to (e.g. +X-forward vs +Y-forward between our
code and BeamNG's vehicle frame), not random failure.

- If Task 1 confirms constant ~90 CW: correct the spawn quaternion by 90 degrees
  counter-clockwise in _teleport_to.
- Also check whether the teleport/spawn `cling=True` ground-snapping is
  interacting with the rotation (candidate culprit; documented BeamNGpy behavior
  snaps z to ground level and may affect orientation handling).
- This is shared code, so it fixes both training and the watcher.
- Validate on the G14: watch a spawn and confirm the car points straight down the
  track from the start (no 90-degree recovery).

### Validation gate (necessary AND sufficient only together)

The headless re-measurement on the Mini PC is NECESSARY but NOT SUFFICIENT. We
were misled all session by a Mini PC number that checked intent, not outcome.
After the fix, Mike validates VISUALLY on the G14:

- Load a model in watch_beamng.py and watch the spawn.
- Confirm the car points straight down-track from the FIRST frame (no 90-degree
  rotation, no drop-and-turn-left recovery).
- Judge SPAWN ORIENTATION ONLY, not the subsequent driving. If validating with a
  run1 model, the car will spawn straight but may then steer left out of trained
  habit (it learned to fight the old broken spawn). That is EXPECTED and actually
  confirms the fix.
- The spawn fix is car-agnostic (shared _teleport_to), so this works the same
  whether watching the etk800 or the Scintilla.

HARD GATE: do NOT launch the full mikey_run2 until Mike confirms the G14 visual
spawn check passes.

Note: set_velocity is a known silent no-op in this BeamNG version; rotation is a
separate call and IS sent (sent quat is correct), the problem is consistent
mis-application, treat as its own issue, not the same as the velocity no-op.

## Task 3: Add speed reward (anti-crawl)

run1 slowed to ~32 kph through the hard middle section because progress reward
accrues regardless of speed, so crawling is "safe points."

- speed_reward = SPEED_WEIGHT * forward_speed * align
- align is the existing alignment term (0..1), so speed only pays when on the
  racing line, pointed forward. (Mike: reward speed only when on-track/aligned.)
- Start SPEED_WEIGHT small as a named constant. The existing
  progress/alignment/checkpoint rewards must stay dominant; this is a gentle
  nudge. Tune up if the car still crawls.
- VEHICLE: phase 2 uses the Civetta Scintilla (model codename "scintilla",
  verified via bng.vehicles.get_available(); default gts config) instead of
  run1's etk800. Because the Scintilla is much faster, the original 0.02 weight
  would swamp the progress/checkpoint terms and reward reckless flooring, so
  start SPEED_WEIGHT at ~0.005-0.01 (set to 0.0075). The Scintilla is livelier
  and easier to spin/overshoot, so early training may look messier and the
  smoothness reward matters more, not less. That is expected, not a problem.

## Task 4: Add smooth-steering reward (anti-wobble)

run1 used bang-bang left-right steering because nothing penalized it.

- Track previous steering in the env: self._prev_steer, init in reset().
- smoothness_penalty = -SMOOTH_WEIGHT * abs(steer_now - prev_steer)
- Penalizes CHANGE in steering (the alternation), NOT steering itself, so smooth
  sustained corner turns are fine; only rapid flip-flopping costs.
- Start SMOOTH_WEIGHT small (~0.1) as a named constant. Tune up if wobble
  persists.

## Reward assembly

```
reward = progress_reward + alignment_gate + checkpoint_bonus + lap_bonus  # existing, dominant
reward += speed_reward + smoothness_penalty                               # new, small nudges
```

## Also investigate: missing milestone_lap

run1 reached cp14 but never wrote milestone_cp15 or milestone_lap despite eval
reward ~1,564. Check the milestone-saver logic for the lap-completion / cp15
trigger, likely a separate code path or fencepost that does not fire on
loop-closure. Confirm whether the car ever formally completes a lap (crosses
start/finish) or only drives the track length without closing. This affects how
we interpret run1 and whether the spawn fix alone unlocks the lap.

## Launch and validation

- Name the run mikey_run2. Reward numbers will NOT be comparable to run1 (reward
  function changed). run1's cp14 / best_model stay as the baseline trophy.
- FRESH start, do NOT warm-start from run1 weights. run1's weights carry the
  90-degree-recovery reflex and the bang-bang wobble we are trying to remove;
  inheriting them means un-learning bad habits. Learn clean from scratch.
- All three changes stacked deliberately (Mike's call): optimizing for the full
  lap, accepting we cannot attribute which fix helped most.
- Keep unchanged: ent_coef, the seam-fixed progress tracker, fixed-start
  curriculum, checkpoint/lap bonuses. Only Tasks 2-4 change.
- Smoke-test first: short run (~5-10k steps) confirming reward is bounded/sane
  (no spikes), spawn is straight (no recovery turn), steering smoother, no crashes
  from the new terms. THEN launch the full 500k.

## Expected payoff

If the hypothesis holds, a correctly-spawning car drives the track from step one
(no wasted recovery) and should have the room to close the full lap, the trophy
run1 could not reach. The speed and smoothness rewards further improve the driving
quality.
