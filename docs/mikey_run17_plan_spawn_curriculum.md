# run17 plan: spawn curriculum + clean off-track — teach the line

Date: 2026-06-16
Status: DRAFT for CC + Mike review. Chat drafts; CC measures + refines against the real
code; Mike approves; CC smokes; Mike approves the full run. Do not implement from this
draft.

Baseline: run16 (the learn-to-corner reward/obs reset), KEPT as-is. run17 adds the
spawn curriculum and tightens off-track termination.

---

## 0. The corrected diagnosis (read this first — the cliff was wrong)

The run16 "T1 is a 10m cliff" finding was a MEASUREMENT ARTIFACT (the arc-wrap
z-vs-centerline bug, the same class CC flagged earlier). It is REFUTED by the G14 watch
(Mike, 3x): there is NO cliff and NO drop. What actually happens at T1:

- The car drives slow (~40-45 kph / ~11 m/s — the timid local optimum), and
- at T1 (a LEFT corner) it turns in too EARLY and too HARD, cutting the INSIDE (left),
  and leaves the track into FLAT GRASS at road level, then wanders into a wall.

So: not a cliff, not the outside, not a fall. The obs was verified sane (curvature
preview ~0 on the straight, ramps only at the true T1 entry, magnitude matches the real
curvature), so it is not an obs bug either. And it is not entry speed (it is slow). It
is an UNLEARNED LINE: the policy has never gotten through T1, so it never learned the
correct (later, gentler) turn-in, and it cuts the corner.

Two things keep it from learning that line:
1. It only ever approaches T1 from the start line, blind, over and over (the chicken-egg).
2. When it cuts the corner, the 20m off-track margin lets it plow deep into the grass and
   crash, so its "I left the road" experience is a messy wander, not a clean signal.

run17 fixes both.

## 1. Change A — spawn curriculum (the behavioral change)

Use the already-wired `random_spawn` (random idx + per-idx heading fix + start-speed) so
episodes start distributed around the whole track, not only the start/finish line. The
agent then practices EVERY corner's line from many approaches, instead of being forced to
master the one corner it cannot get past before it sees any other.

This is the standard fix for exactly our failure: RSI/DeepMimic (fixed start -> a hop;
distributed start -> the full skill) and GT Sophy (spawns spread around the whole circuit,
on-track, speed 0-104 km/h). It directly breaks the chicken-egg, T1's line gets learned
from dozens of approaches per run, and the downstream corners (T2+) finally get practiced
at all.

Design (CC to set against the code):
- Spawn idx: uniform random over the 985 centerline points (every corner + straight).
- Start speed: random but SANE for the spawn point, roughly 0 .. v_target[idx], so it
  never spawns above the corner's grip-limited speed (do not spawn it pre-crashed).
- Heading: the existing per-idx down-track heading fix.
- Schedule: uniform random from the start is the baseline. (Reserve: if uniform proves
  too hard, a reverse/expanding curriculum, start near the start line and widen, per
  Florensa et al.)

Checkpoints/lap bonuses and progress are all relative to arc, so they work from any
spawn; CC confirms the checkpoint/lap logic and the backward/kill-switch behave correctly
under random spawn.

## 2. Change B — tighten OFF_TRACK_THRESHOLD (clean signal, not fall-prevention)

`OFF_TRACK_THRESHOLD_M = 20.0` lets the car travel 20m off the road before the episode
ends, so a cut corner becomes a long grass-and-wall wander that corrupts the learning
signal. Tighten it to about the road half-width + a small margin, so the episode ends
with a crisp off_track termination the INSTANT the car leaves the road.

- Set from the measured width (`extract_centerline.py` computes left/right edges +
  width). OFF_TRACK_THRESHOLD ~= max(road half-width along the lap) + ~1-2m margin, so the
  car can use the full road but a real departure ends it immediately. Do NOT set it so
  tight it false-terminates on the widest legal part of the road.
- Keep the off_track terminal penalty (the "you left the road" signal). The point is
  TIMING: end on departure, not 20m later.
- This is a clean-signal fix (the line error ends crisply), NOT to prevent a fall — there
  is no fall.

## 3. Keep everything from run16

The run16 braking-aware speed-target reward (W_PROG*progress*align - W_OVER*over_speed^2
- W_SLIP*slip), the 18-dim obs (verified sane), plain SAC, gamma 0.99, the ONE flat
steering-rate limit (0.5), the heading kill-switch, checkpoint/lap bonuses. run17 changes
only the start-state distribution and the off-track threshold.

## 4. Fresh, not warm

Fresh. The run16 policy is the timid local optimum (crawl, bank cp1, cut T1) — exactly
the behavior we are escaping, so warm-starting it would carry the timidity. A fresh policy
under random spawn learns cornering broadly from step 0 (random spawn breaks the
chicken-egg immediately). Obs shape is unchanged so warm is technically possible, but
fresh is the right call. Keep run16 artifacts.

## 5. Anti-timid nudge — RESERVED, not in run17

The car is genuinely timid (slow), but the timidity is plausibly a learned response to
T1 being un-passable (any speed -> a crash it can't recover from). With random spawn (it
learns corners are makeable) + clean off-track (a line error becomes a mild reset, not a
crash), the downside of carrying speed drops, and the progress term should pull speed up
on its own. So hold the under-speed nudge (+W_MATCH*min(v, v_target)) in reserve; add it
in run18 only if the watch shows it still crawling after it can corner.

## 6. Validation plan

1. Spawn probe (no training): random_spawn produces sane starts across the track —
   on-track, down-track heading, start speed <= v_target at the spawn point. And confirm
   the tightened OFF_TRACK terminates at the road edge (terminate ~1-2m off the road, but
   NOT on the widest legal road section). Print a few spawns + the off-track distance.
2. Smoke ~7k fresh: random spawn active (episodes start at varied arc), off_track fires
   at the new threshold, reward/obs/mechanics intact, no NaN, plain SAC.
3. Fresh 500k on Mike's approval. Wrapper (buffer fix) + temp logger armed.

## 7. G14 watch question (the verdict)

After maturity, watch 3+ times:
1. Does it learn correct corner LINES — at T1, does it stop cutting the inside and take a
   sensible turn-in, and does it hold the line through other corners it now practices?
2. Does it finally get THROUGH T1 (past 394m) and onto the lap, the wall no run has ever
   cleared?
3. Is it carrying speed (tracking v_target) or still crawling (-> the reserved nudge)?

## 8. Reserve ladder

- Still timid after it can corner -> the under-speed nudge (run18).
- Uniform random spawn too hard (fails everywhere) -> reverse/expanding curriculum.
- OFF_TRACK too tight (false-terminates on the legal road) -> loosen to the true edge.
- Lap-time / apex-cutting optimal line -> the width-based racing line (later).

## 9. Constraints reminder for CC

- run17 = spawn curriculum (behavioral) + the off-track threshold fix (a signal-quality
  fix that rides with it). Everything else stays run16.
- GROUND IT IN THE WATCH, not the refuted probe: no cliff, the car cuts the INSIDE of T1
  into flat grass, it is an unlearned line. Do not design from the "outside drop-off."
- Structural fix is the start-state distribution + clean termination, not new action
  scripts or reward terms.
- Pattern: CC measures the road width + verifies random_spawn/checkpoint behavior,
  proposes the final OFF_TRACK value + spawn-speed range, Mike approves, CC smokes, Mike
  approves the full run.
