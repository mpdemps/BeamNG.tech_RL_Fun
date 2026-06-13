# run9 plan: weave penalty redesign (oscillation signature, not position)

Date: 2026-06-13
Status: DRAFT for CC review (poke holes against real env code + the run8 trace).
Do not implement yet. Chat drafts, CC redesigns/validates, Mike approves, CC smokes,
Mike approves the run.

Depends on: `docs/mikey_run8_postwatch_weave_diagnostic.md` and CC's per-step trace
`logs/run8_weave_trace.csv`.

---

## 1. Why run8 failed (the one-line cause)

The run8 weave penalty was position-gated (`on_line` from |center_off|). A weave is
an oscillation, and the oscillation moves the car through exactly the off-line
positions that open the `on_line` gate. CC's trace: the car rides a steady ~2.2m
offset and swings to 3.0-3.4m, so `on_line` sits near zero (mean 0.34) and the
penalty never bites (mean -0.003 across the whole 80->115 kph weave-into-spin).
`straightness` finishes the kill past ~235m where the 80m horizon pulls R40 into
view. Two leaks, same root: we gated the penalty on the car's position, and the
weave lives in the positions that disable it.

Watched at resume-180k (~245k training age): still oscillating into a slide. The
policy is not self-correcting; it reaches 484m by bulling through, not by driving
straight. So this is a reward-form problem, confirmed twice (115k and 180k watches).

## 2. The core invariant the redesign must satisfy

Distinguish a WEAVE from a legitimate RECOVERY by their shape, never by absolute
position:

- A weave is OSCILLATING lateral motion: center_off snakes back and forth, lateral
  velocity keeps reversing sign. This is what we penalize.
- A recovery (drift-back) is MONOTONIC: the car steers one way and |center_off|
  decreases steadily toward the line. This must stay unpenalized. This is the case
  `on_line` was trying to protect, and the reason we cannot just delete the gate and
  penalize raw |steer|.
- A corner is also (mostly) monotonic lateral motion plus sustained held steer, and
  is handled by the straightness gate (penalty off when the road bends).

So: penalize lateral oscillation, gated by straightness only. No position gate.

## 3. Proposed form (primary candidate)

Penalize the reversal of lateral motion, scaled by its magnitude, on straights.

```
lateral_vel   = (center_off - prev_center_off)        # per step (dt constant at 20Hz)
reversed      = sign(lateral_vel) != sign(prev_lateral_vel)
swing         = max(0, |lateral_vel| - LAT_DEAD)      # deadband kills noise/micro
weave_penalty = -WEAVE_WEIGHT * swing * reversed * straightness
```

Why this is robust:
- Pose-independent: center_off is centerline geometry, not nose-contaminated.
- Outcome-based: it penalizes the snaking PATH, not a control input, so it cannot
  be dodged by any steering trick. To avoid the penalty the car must stop snaking.
- Frequency-robust (run7's failure): a slow weave still reverses; each reversal is
  charged by amplitude, so it cannot hide in low frequency.
- Position-robust (run8's failure): no `on_line` term, so going off-line does not
  disable it. The steady 2.2m offset produces ~0 lateral velocity, so it is not
  penalized (we do not care that the car is off-center, only that it snakes).
- Recovery-safe: a monotonic drift-back has no sign reversals, so `reversed` = 0 and
  it is unpenalized, replacing what `on_line` was meant to do.

Alternate candidate (for CC to weigh against the trace): penalize steering-direction
reversals (sign flips of steer beyond a deadband) instead of lateral-velocity
reversals. Steering is the cause, center_off is the effect; the effect is the safer
target because it is what we actually want gone and it ignores corrective steering
that does not produce snaking. CC: pick the one that fires cleanly across the trace
weave and stays ~0 on a held-line straight and a real corner.

## 4. The straightness-gate fix (H2, do not inherit it)

The current straightness gate uses an 80m horizon, which releases the penalty from
~235m onward. The spins cluster at 180-280m, so the gate is OFF exactly where the
weave is worst. Whatever signal we choose, if it is gated by straightness we must
fix this.

Tension to resolve (CC's geometry call): the gate must release early enough to allow
legitimate corner setup (a good driver turns in before the geometric corner) but
stay ON through the back of the straight where the weave spins the car. R40 is at
~300m; the legit turn-in zone is roughly the last 30-50m. A shorter straightness
horizon (start point ~30-40m vs 80m) would keep the penalty live to ~260-270m and
release just before turn-in. CC: validate the chosen horizon against ALL corner
types (the R252 kink and faster corners may need earlier release than R40), using
the measured geometry, so we do not start penalizing legitimate high-speed setup.

## 5. The logging fix (H3, so we are not blind again)

CSV column 10 logs the terminal-step weave_penalty, which the heading kill-switch
forces to 0 on every spin, so the CSV read ~0 and hid the failure for the whole run.
Change the per-episode weave_penalty logging to an episode MEAN (or sum) over steps,
captured before the kill-switch zeroing, so next time the CSV actually tells us
whether the penalty is biting without needing a full instrumented rollout.

## 6. Validation plan (same rigor as run8)

1. Re-run CC's instrumentation on `run8_weave_trace.csv` (or a fresh deterministic
   rollout) with the NEW term computed offline. It must show a strongly negative
   penalty across the 120-280m weave (not the current -0.003), i.e. it would have
   bitten the behavior we watched.
2. Probe (synthetic): the new term fires on an oscillating path; reads ~0 on a
   straight held line at a steady offset; reads ~0 on a monotonic drift-back
   recovery; reads ~0 through a real corner (straightness gate). Confirm no
   intermittent unlock and bounded magnitude.
3. Smoke (~7k fresh): term fires live, all run7/run8 mechanics intact (-25 backward,
   kill-switch, wheelspin, lookahead), bounded, no NaN, logging shows the new
   episode-mean weave column.
4. One change only: this replaces the old position-gated weave term. Nothing else
   moves.

## 7. Risks to watch on the G14 (after it trains)

- Steering timidity / understeer: if the deadband is too tight or the weight too
  high, the car may stop making legitimate micro-corrections. Watch for a car that
  refuses to adjust and washes wide.
- Over-suppression into a new escape: confirm the policy does not learn a new proxy
  (e.g. crawling so lateral_vel stays under the deadband). Speed on the straight
  must stay real pace.
- The corner is the next problem regardless. A weave-free car will finally expose
  the cornering / entry-speed issue (the flips at corner distance), which is the
  run10+ racing-line work.

## 8. Fresh vs keep-running (Mike's call at launch)

Recommendation: FRESH. The weave is the entrenched main strategy at 245k training
age, and a penalty that finally bites it will be fighting that entrenched policy.
The run4 lesson is explicit: warm-starting a policy whose dominant behavior you then
penalize ENTRENCHES it. run8_resume's 484m corner progress is real but it belongs to
a weaving policy we do not want to carry forward. Going fresh costs compute, not a
banked peak (EvalCallback saves best_model). Keep run8_resume training until the
moment we launch run9 so we lose nothing if Mike wants to ride it longer first.

## 9. Constraints reminder for CC

- One change at a time, measure don't assume, smoke before any full run.
- Pose-independent gating only: centerline geometry + center_off, never observation
  bearings (the earlier trap).
- The weave penalty is pure reward arithmetic, zero sim load. The sim freeze at
  ~67.8k is a separate mechanical issue (corner-rollover hypothesis); keep the
  freeze-detector armed on whatever is training.
- Pattern: CC pokes holes / refines this against the real env code and the trace,
  proposes the final form, Mike approves, CC smokes, Mike approves the full run.
