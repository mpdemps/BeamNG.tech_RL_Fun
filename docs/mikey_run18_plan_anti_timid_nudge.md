# run18 plan: anti-timid nudge — pay for carrying speed toward v_target

Date: 2026-06-16
Status: DRAFT for CC + Mike review. Chat drafts; CC measures + calibrates the weight
against the real code; Mike approves; CC smokes; Mike approves the full run. Do not
implement from this draft.

Baseline: run17 (spawn curriculum + clean off-track), KEPT as-is. run18 adds ONE reward
term: the reserved under-speed nudge. This is the run17 plan's §5 reserve, now triggered.

---

## 0. The verdict that triggered this (numbers, not the curve)

run17 at 150k. Two deterministic replays of best_model from the START LINE (the exact
eval condition):

- EP0: len 3048, stuck, max_arc 96m, mean_speed 2.4 m/s, 0 checkpoints.
- EP1: len 3682, stuck, max_arc 101m, mean_speed 2.5 m/s, 0 checkpoints.

From the start it crawls to ~100m at ~2.4 m/s and gets stuck. It does not reach T1's
entry (294m), let alone the 394m exit. The climbing eval reward (242) and ep_length
(3650) were the "creep forward then putter for 3000+ steps dodging the stuck timer,
trickling progress-reward" case. The reward flattered us again. The spawn curriculum is
working mechanically (whole track practiced, off-track fires clean), but it did NOT break
the timidity on its own. This is the §5 reserve trigger.

## 1. Why the timidity got DEEPER under the curriculum (the diagnosis)

run16 (same reward, no spawn curriculum) reached max_arc 346. run17 (added curriculum)
collapsed to ~100m. The curriculum likely DEEPENED the timidity: when episodes start
distributed all over the track, the single policy that is safe from EVERY spawn is
"crawl" — crawling never leaves the road and never eats the off-track penalty anywhere.
So the curriculum handed the policy more reason to be slow, with no incentive to carry
speed. The progress term alone cannot escape this: at crawl speed progress*alignment
still trickles positive, and the over-speed penalty only ever pushes speed DOWN, so there
is no gradient pulling speed UP. The reward has a reward-positive crawl basin and nothing
to climb out of it.

## 2. The change (one change): the under-speed reward term

Add the reserved nudge to the run16/17 reward:

```
reward = W_PROG*progress*gated_align
       + W_MATCH * min(v, v_target)        # NEW: pay for speed, capped at the target
       - W_OVER * max(0, v - v_target)^2    # unchanged
       - W_SLIP * max(0, |beta| - BETA_DEAD) # unchanged
```

Why `min(v, v_target)`, not raw `v`:
- It rewards carrying speed UP to the corner's grip-limited target, then FLATTENS. Above
  v_target the match term is constant (no further gain) while the unchanged over-speed
  penalty bites. So the reward peak sits AT v_target, not above it.
- This is the TAL "match the line's speed" idea (research §3.2), the capped-speed reward,
  NOT the progress*speed trap we started in. The cap is what keeps it from re-becoming
  the over-speed-into-T1 spin.

This directly fills the hole from §1: a dense, always-on gradient pulling speed toward
v_target, present from every spawn, so "crawl" stops being reward-optimal.

## 3. The calibration CC must do (the binding constraint)

The whole safety of this term is the balance between W_MATCH and W_OVER. CC sets W_MATCH
so that:

1. PEAK AT TARGET: the reward as a function of v (holding line/progress fixed) peaks at
   v ~= v_target. Concretely, the marginal match-gain just below v_target must be
   meaningful, and just ABOVE v_target the over-speed penalty's slope must already exceed
   the (now-zero) match slope, so v* stays at/just under v_target. Re-run the run16
   speed-profile calibration (held-at-target vs floor-it) WITH the new term and confirm
   v* is still ~v_target, not pushed high.
2. STRONG ENOUGH TO MOVE IT: the match-gain from crawl (2.4 m/s) up toward v_target must
   dominate the progress-trickle the crawl currently banks, or the basin won't break.
   First guess: size W_MATCH so that at half-target speed the match term is comparable to
   the typical per-step progress term, then tune in the smoke. CC proposes the number.
3. NOT SO STRONG IT OVER-SPEEDS: if W_MATCH is too large relative to W_OVER, the optimum
   creeps above v_target and we are back to over-speed spins. The smoke must confirm
   over_speed_frac stays low and beta stays clean once speed comes up.

Report the chosen W_MATCH + the recomputed v* from the calibration before the smoke.

## 4. Keep everything from run17

The braking-aware speed-target reward (W_PROG, W_OVER, W_SLIP), the spawn curriculum
(random idx + heading fix + sane start speed), OFF_TRACK 8m clean termination, the
checkpoint pre-mark fix, the 18-dim obs (verified sane), plain SAC, gamma 0.99, the ONE
flat steering-rate limit (0.5), the heading kill-switch, checkpoint/lap bonuses. run18
adds ONLY the +W_MATCH*min(v, v_target) term.

## 5. Fresh, not warm

Fresh. The run17 policy is the DEEP timid crawl — the exact thing we are escaping —
so warm-starting carries the collapse. A fresh policy under the curriculum + the new
speed incentive learns "carry speed AND practice every corner" from step 0. Obs shape is
unchanged so warm is technically possible, but fresh is the right call. Keep run17
artifacts.

## 6. Validation plan

1. Reward probe (no training): with the new term, the reward-vs-speed curve peaks at
   ~v_target at several track points (a slow corner, a fast corner, the straight); above
   v_target the over-speed penalty dominates. Confirm v* ~= v_target everywhere (the §3.1
   recomputed calibration).
2. Smoke ~7k fresh: the match term is live, mean_speed climbs off the crawl vs a no-nudge
   reference, over_speed_frac stays low, beta clean, no NaN, spawn curriculum + off-track
   + all run17 mechanics intact. Log mean_speed, over_speed_frac, and the match-term
   contribution so we can read that speed is coming up WITHOUT over-speeding.
3. Fresh 500k on Mike's approval. Wrapper (buffer fix) + temp logger armed.

## 7. G14 watch questions (the verdict)

After maturity, watch 3+ times:
1. SPEED: is the car finally carrying speed (tracking v_target down the straight, not the
   ~2.4 m/s crawl)? mean_speed and max_arc from the start line are the disambiguators, NOT
   the eval reward.
2. T1: does carrying real speed now get it to T1's entry (294m) and THROUGH the 394m exit
   the wall no run has cleared — and when it gets there, does it take a sensible line
   (not the cut-the-inside-into-grass) or over-speed back into a spin?
3. BALANCE: did the nudge over-correct? Watch for over-speed spins returning at T1 (->
   lower W_MATCH or raise W_OVER) vs still-too-timid (-> raise W_MATCH).

## 8. Reserve ladder

- Still timid after the nudge (crawl persists) -> raise W_MATCH (basin not broken).
- Over-speeds back into T1 spins -> lower W_MATCH / raise W_OVER (peak pushed past target).
- Carries speed but still cuts T1's line into the grass (a LINE error, not a speed error)
  -> the width-based apex-cutting racing line (the deferred frontier).
- Pure-RL still can't find the line at our compute -> residual RL on a pure-pursuit base
  controller tracking the offline racing line (research §4 fallback).

## 9. Constraints reminder for CC

- ONE change: the +W_MATCH*min(v, v_target) reward term. Everything else stays run17.
- The cap (min with v_target) is load-bearing: it is what keeps this from being the
  progress*speed over-speed trap. Do not drop the cap.
- Calibrate the weight against the real speed_profile + reward (the §3 peak-at-target and
  strong-enough-to-move-it constraints) BEFORE the smoke; report W_MATCH + recomputed v*.
- GROUND IT IN THE NUMBERS: the verdict is max_arc 96-101m / mean_speed 2.4 from the start
  line. The target is mean_speed up and max_arc past 394m, read from the eval/start-line
  replay, NOT the eval reward curve.
- Pattern: CC measures + calibrates W_MATCH against the code, proposes the number + the v*
  check, Mike approves, CC smokes, Mike approves the full run.
