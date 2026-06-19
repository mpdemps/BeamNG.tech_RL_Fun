# run20 plan: the racing line + track edges — give the car a path to follow

Date: 2026-06-19
Status: DRAFT for CC + Mikey review. Chat drafts; CC builds the offline line + measures
against the real code; Mikey approves the direction; CC smokes; Mike approves the full run.
Do not implement from this draft.

Baseline: run18 (the anti-timid nudge policy that carries speed and brakes for T1), the last
GOOD policy. run20 is a coherent BUNDLE (not a single dial), the planned Phase-1 racing-line
feature, replacing the centerline reference with an optimal line and giving the car boundary
awareness. This is the research's primary recommendation
(`docs\research_learn_to_corner_2026-06-15.md`).

---

## 0. Why a bundle, and why now (the diagnosis chain)

- run18: carries ~20 m/s, BRAKES for T1, but at turn-in floors the 748hp RWD car -> power
  oversteer -> spins off the outside. A throttle-discipline failure.
- run19: strengthened the slip penalty to teach throttle discipline. It BACKFIRED into
  corner-AVOIDANCE: the watch showed it run wide-right at T1 and drive straight into the
  wall with no turn attempt. Root cause: at T1, "cornering hard at the limit" and "the rear
  starting to spin" produce the SAME slip-angle, so the penalty taxed the cornering itself.
  No W_SLIP value works, the slip penalty cannot tell good cornering slip from bad spin slip.
  It is the wrong lever for T1.

The research's conclusion, reached independently: braking/cornering must be referenced to a
RACING LINE (an optimal path + speed profile), TAL-style, not coaxed out of penalties. The
line gives the car the one thing it lacks, a concrete answer to "where do I go through this
corner," instead of leaving it to discover cornering against a penalty that punishes the very
thing it needs to do. The two pieces (path + boundaries) are complementary for the exact
failure (runs wide, hits the wall, does not turn), so we do them together, like run16's reset.

## 1. The change (coherent bundle, three parts)

### 1a. Offline racing line (computed once, read-only)

A minimum-curvature path that stays inside the track edges, plus its own minimum-time SPEED
PROFILE (the apex speeds + brake points). For a LEFT corner like T1 the line enters wide
(right), cuts to the apex (left/inside), exits wide (right), exactly the turn-left-to-apex
move the car is not making.

- Inputs we already have: the 985-pt centerline (`CENTERLINE`), and the left/right edges +
  width from `extract_centerline.py` (the same data that set the 8m off-track threshold). No
  new measurement.
- Method: minimum-curvature optimization over the centerline within the width (Heilmeier
  VSD 2019). Then a forward-backward speed-profile pass on the LINE's curvature, the same
  machinery as `envs\speed_profile.py` (which already does this on the centerline), just fed
  the racing-line radii instead of centerline radii.
- LICENSE (hard constraint, repo is public + MIT-only, no GPL/LGPL deps): do NOT pull in the
  TUM `global_racetrajectory_optimization` package (LGPL). Roll our own min-curvature line, it
  is a tractable QP over the centerline points within the width band, well within our scope,
  and keeps the repo clean. CC confirms the license call before writing any dependency.

### 1b. Observation, reference the line (NO new dims)

- Switch heading_err and the lateral offset to be relative to the RACING LINE instead of the
  centerline (the aim point and `center_off` track the line, not the middle of the road).
- Keep the speed-scaled curvature preview, but on the racing line's curvature.
- v_target [16] becomes the racing line's speed target.
- NO new obs dims: the obs SHAPE stays IDENTICAL to run18 (18 dims), just computed relative to
  the line. This is a pure retarget, which is what makes warm-from-run18 clean (confirmed: the
  retarget obs probe passed lap-wide, and run18's policy warm-loaded with no shape error).
- Edge-distance dims are DROPPED. We have no authoritative per-node edges (get_road_network
  only returns regenerating fragments), so a constant-width "edge distance" adds nothing the
  line does not already encode; and following an on-road line keeps the car on-road. Revisit
  real edge dims only if we later capture true edges AND the line-follow still wanders to a
  wall. NOTE: off-track termination still measures vs the real ROAD centerline (8m), NOT the
  line, so the car is never terminated for correctly driving the line's apex (~4.2m off road
  center, well inside 8m).

### 1c. Reward, TAL-style line reference (and back OFF the slip penalty)

- Reward matching the racing line's SPEED and position (TAL: reward shrinks with
  |v - v_line| and lateral offset from the line). Slow-in/fast-out falls out of the line's
  speed profile. Keep the anti-timid match term (W_MATCH) referenced to the line's v_target.
- RETURN the slip penalty to a GENTLE setting (run18's W_SLIP=0.05, BETA_DEAD=9, or softer).
  The line now handles "where to go and how fast"; the slip penalty is back to a light
  backstop against genuine spins, NOT the primary lever. Do NOT keep run19's 0.15/7.0, that
  is what taught corner-avoidance.

## 2. Why this fixes the T1 failure

The car runs wide-right and does not turn because nothing tells it where the corner's path
goes and the penalty punished the turn. The racing line gives an explicit left-hand path to
track through T1 (enter right, apex left, exit right) and rewards following it; the edge
distances let it see the right wall coming. Together: a path to follow + boundaries to
respect, which is exactly the spatial information "no attempt to make the turn" is missing.

## 3. Warm vs fresh — WARM from run18 (decided)

Because run20 is a pure retarget with the obs SHAPE unchanged (18 dims), warm-start is clean
and is the right call: warm from run18's best_model, the corner-ATTEMPTING policy. run18
already knows "drive the offset/heading to zero, brake, carry speed"; now those references
point at the line, so it begins tracking the line immediately and just adapts the corner.
NEVER warm from run19 (the corner-AVOIDANT policy). Confirmed: run18's 18-dim policy
warm-loaded against the line-relative env with no shape error, and at the first eval (step
3500, the warm policy under the line) it already reached max_arc ~375m, ~20m further than
run18 under the centerline.

## 4. Keep from prior runs

Spawn curriculum (random starts), 8m off-track termination, the anti-timid match term
(W_MATCH, now line-referenced), plain SAC, gamma 0.99, the one flat steering-rate limit (0.5),
the 20 eval/* TB cards. run20 changes the reference frame (centerline -> racing line), adds
edge awareness, and softens the slip penalty back to a backstop.

## 5. Validation plan

1. Racing-line probe (no training): the computed line stays strictly inside the edges, apexes
   the corners sanely (wide-in/apex/wide-out at T1 and the other corners), and its speed
   profile is within grip everywhere (reuse the run16 calibration check on the line).
2. Obs probe: line-relative heading/offset and the two edge distances are sane across the lap
   (offset ~0 when on the line, edge distances shrink approaching a wall), bounded.
3. Smoke ~7k: line reference live in obs + reward, edge dims present, gentle slip penalty,
   spawn curriculum + off-track intact, no NaN, the 20 eval/* cards intact (add an
   eval/offset-from-line card if cheap).
4. Fresh 500k on Mike's approval. Wrapper (buffer fix) + temp logger armed.

## 6. G14 watch question (the verdict)

After maturity, watch 3+ times:
1. Does it FOLLOW the line through T1, entering wide-right, turning left to the apex, exiting,
   instead of running straight into the wall?
2. Does it finally get THROUGH T1, past 394 m, onto the lap?
3. Does it carry the line's speed (not crawl), and does it hold the line through the
   downstream corners it now reaches?

## 7. Reserve ladder

- Pure RL on the line struggles at our compute (one CPU mini, ~11 fps) -> RESIDUAL RL: a
  fixed pure-pursuit controller tracks the offline line (so it brakes/turns by construction)
  and RL learns only a small correction on top. The research's scale-friendly fallback
  (RLPP/Trumpp); far fewer steps needed. This is the strong Plan B if from-scratch RL on the
  line is too slow.
- Still spins on power despite the line -> nudge the slip penalty up modestly (but gently,
  the line should make this rare).
- Line geometry wrong (cuts a wall, bad apex) -> fix the offline optimizer, not the policy.

## 8. Constraints reminder for CC

- This is a coherent BUNDLE (offline line + line-referenced obs/reward + edge-distance obs +
  softened slip penalty), justified as the planned Phase-1 feature and the fix for run19's
  corner-avoidance. Not a single-dial change.
- LICENSE: MIT-only, public repo. Roll our own min-curvature line; do NOT add the LGPL TUM
  package. Confirm before writing any dependency.
- Ground it in the run19 watch: the car has a throttle/where-to-go problem at T1, not a brake
  problem. The line answers where-to-go; the edges answer where-the-wall-is.
- Soften the slip penalty back to a backstop (run18 level), do NOT carry run19's 0.15/7.0.
- Clock: BeamNG license expires 2026-08-06. This is the run that should get us through T1 and
  toward Phase 1 (lap-time). If pure-RL-on-the-line is slow, go residual RL early rather than
  grinding.
- Pattern: CC builds + probes the offline line, sets the obs layout, proposes the reward
  weights + fresh/warm, Mikey approves the direction, Mike approves the full run.
