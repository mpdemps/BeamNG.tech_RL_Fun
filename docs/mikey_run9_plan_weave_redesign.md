# run9 plan: weave penalty redesign (oscillation signature, not position)

Date: 2026-06-13 (updated after CC's offline validation overturned the first form)
Status: APPROVED design, one offline refinement pending before build (CC to lock N
and the trigger), then build -> probe -> smoke -> fresh 500k on Mike's approval.

Depends on: `docs/mikey_run8_postwatch_weave_diagnostic.md`, CC's per-step trace
`logs/run8_weave_trace.csv`, and CC's offline study `scripts/run9_weave_redesign_offline.py`.

Scope note: run9 is the STRAIGHT-ONLY weave fix. It suppresses the snake on the
straight and stays cost-free in corners (the failure we have watched twice is a
straight weave that spins the car before it ever reaches the corner). Positive
good-line shaping through corners is the RACING LINE work, deferred to run10 (see
section 8), which will subsume the geometry gate entirely.

---

## 1. Why run8 failed (the one-line cause)

The run8 weave penalty was position-gated (`on_line` from |center_off|). A weave is
an oscillation, and the oscillation moves the car through exactly the off-line
positions that open the `on_line` gate. CC's trace: the car rides a steady ~2.2m
offset and swings to 3.0-3.4m, so `on_line` sits near zero (mean 0.34) and the
penalty never bites (mean -0.003 across the whole 80->115 kph weave-into-spin).
`straightness` (80m horizon) finished the kill past ~235m where R40 enters view.
Confirmed by two G14 watches (rolling_115000 and rolling_180000, ~245k training
age): still oscillating into a slide; the policy is not self-correcting.

## 2. The core invariant the redesign must satisfy

Distinguish a WEAVE from a legitimate RECOVERY by their shape, never by position:

- A weave is OSCILLATING lateral motion: center_off snakes, lateral velocity keeps
  reversing sign. This is what we penalize.
- A recovery (drift-back) is MONOTONIC: one-way, |center_off| decreasing toward the
  line. Must stay unpenalized. This is what `on_line` was trying to protect, and the
  reason we cannot just delete the gate and penalize raw |steer|.
- A corner is handled by the straightness gate (penalty off when the road bends).

So: penalize lateral oscillation, gated by straightness only. No position gate.

## 3. The form (CC's V4, after the first form failed)

The first proposed form multiplied the penalty by `reversed` (fire only on the
sign-flip step). CC's trace proved it dead (-0.002, 3 firings in 69 steps): the
sign-flip is the turning point of the swing, where |lateral_vel| is at its MINIMUM,
so the deadband ate nearly every firing. Charging at the reversal instant samples
exactly the wrong moment. Fix: charge the swing while oscillation is ACTIVE, not at
the reversal.

Locked form (one offline refinement pending, see section 6 step 0):

```
lateral_vel     = center_off - prev_center_off          # per 20Hz step; center_off is pose-independent geometry
reversal        = (lateral_vel and prev_lateral_vel nonzero) and sign(lateral_vel) != sign(prev_lateral_vel)
osc_active      = (>= 2 reversals in the last WEAVE_OSC_WINDOW steps)   # genuine oscillation, not a single correction
swing           = max(0, |lateral_vel| - WEAVE_LAT_DEAD)
weave_penalty   = -WEAVE_WEIGHT * swing * osc_active * straightness
# straightness from centerline bend over WEAVE_LOOK_M (BEND_DEAD 3deg / BEND_FULL 15deg, unchanged)
```

Why this is robust:
- Pose-independent: center_off is centerline geometry, not nose-contaminated.
- Outcome-based: penalizes the snaking PATH, not a control input. To avoid it the
  car must stop snaking; no steering trick dodges it.
- Frequency-robust (run7's failure): a slow weave still reverses; charged by
  amplitude per step while active, so it cannot hide in low frequency.
- Position-robust (run8's failure): no `on_line`. Going off-line does not disable it.
  The steady 2.2m offset has ~0 lateral velocity, so it is free (we do not care that
  the car is off-center, only that it snakes).
- Recovery-safe AND timidity-safe: a monotonic drift-back has zero reversals, and a
  single isolated correction has exactly ONE reversal, so neither reaches the "2+
  reversals" trigger. Only genuine oscillation (>=2 reversals per cycle) fires. This
  replaces what `on_line` was for, without the leak, and without taxing legitimate
  single corrections.

## 4. The straightness-gate fix (H2), validated

`WEAVE_LOOK_M` 80 -> 40. CC validated across all 18 detected corners: at 40m the gate
stays LIVE through the entire 180-270m spin zone and releases at ~271m, 24m before
R40 entry (apex 335m), leaving turn-in room. At 80m it released at 231m, deep in the
spin zone (the measured leak). Across all 18 corners the 40m apex-lead is >=34m
(mostly 40-160m), erring early (generous) for sharp/fast corners, which is the safe
direction (does not tax legitimate setup).

## 5. The logging fix (H3)

CSV column 10 logged the terminal-step weave_penalty, which the heading kill-switch
forces to 0 on every spin, so the CSV read ~0 and hid the failure all run.
Accumulate `_weave_sum += weave_penalty` each step and log
`info["weave_penalty"] = _weave_sum / max(_ep_steps, 1)` (episode mean). Monitor
already logs that key, so the CSV column auto-switches to the mean bite. Next run the
CSV alone tells us whether it is biting, no instrumented rollout needed.

## 6. Validation plan

0. (refinement, offline, BEFORE build) Measure the inter-reversal spacing of the
   weave in the trace; set WEAVE_OSC_WINDOW = N to span ~one weave cycle (so 2
   reversals fall inside). Switch the trigger to ">=2 reversals in last N." Confirm
   the new form still bites the trace weave hard (vs V4's -0.085) and that all
   control cases below stay clean.
1. Offline over the trace: penalty strongly negative across the 120-280m weave (not
   -0.003); it would have bitten what we watched.
2. Control cases, all must read ~0: held line at a steady 2.2m offset; monotonic
   drift-back 3->0m; a real corner (straightness 0); AND a single isolated lateral
   correction (one reversal, then settle) -> this is the timidity check.
   The oscillating weave (the disease) must fire hard.
3. Probe (synthetic) + ~7k fresh smoke: term fires live, episode-mean weave column
   populated, all run7/run8 mechanics intact (-25 backward, kill-switch, wheelspin,
   lookahead), bounded, no NaN.
4. One change only: replaces the old position-gated weave term. Nothing else moves.

## 7. Risks to watch on the G14 (after it trains)

- Steering timidity / understeer: addressed by construction via the 2-reversal
  trigger, but still watch for a car that refuses to adjust and washes wide.
- New escape proxy: confirm the policy does not learn to crawl so |lateral_vel|
  stays under LAT_DEAD; straight speed must stay real pace.
- WEAVE_WEIGHT magnitude: the -0.085 mean bite is ~6% of per-step progress, in the
  same ballpark as the old "nearly free" 5% TV penalty. Treat 3.0 as a FLOOR; watch
  whether behavior actually changes and be ready to raise.
- The corner is the next problem regardless. A weave-free car will finally expose
  the cornering/entry-speed issue (the flips at corner distance) -> run10 racing line.

## 8. Fresh vs keep, and why the racing line is run10 not run9

Fresh: confirmed by chat and CC. The weave is the entrenched main strategy at ~245k
training age; a penalty that finally bites it would fight that policy, the run4
entrenchment mistake a third time. run8_resume's 484m corner progress belongs to a
weaving policy we do not want to carry. Cost is compute, not a banked peak. Keep
run8_resume training until the moment we launch run9 so nothing is lost.

Racing line deferred to run10 (Mike's design point): the right way to make a good
corner line "more valuable than the oscillation cost" is to make the reference path
the racing line itself, so progress rewards following the good line and oscillation
is penalized around it everywhere, with no corner gate. That is the root fix for
corner shaping, but it needs track-edge extraction + an offline minimum-curvature
optimizer, and the car cannot yet reliably reach corners. So fix the observed straight
weave first (run9), then let the racing line shape corners and subsume the geometry
gate (run10).

## 9. Constraints reminder for CC

- One change at a time, measure don't assume, smoke before any full run.
- Pose-independent gating only: centerline geometry + center_off, never observation
  bearings.
- The weave penalty is pure reward arithmetic, zero sim load. The sim freeze at
  ~67.8k is a separate mechanical issue (corner-rollover hypothesis); keep the
  freeze-detector armed on whatever is training.
- Pattern: CC locks N + the trigger and builds, Mike approves, CC smokes, Mike
  approves the full run.
