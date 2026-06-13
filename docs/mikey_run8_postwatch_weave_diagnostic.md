# run8 post-watch: weave persists, penalty reads ~0 (diagnostic brief for CC)

Date: 2026-06-13
Model watched: `checkpoints/mikey_run8_resume/rolling_115000_steps.zip`
(resume-115k, ~180k training age: rolling_65000 warm-start + 115k resume steps)
Watch: G14, deterministic, 3 runs, consistent behavior.

This is the working-pattern poke-holes step. Chat (analysis) hands this to CC.
CC validates or refutes the hypotheses below against the REAL env code and a REAL
instrumented episode, reports what it finds, then we design the fix. Do NOT
implement a fix from this doc. Measure first.

---

## 1. The watch verdict

Priority-1 question (is the weave gone on the straights): FAIL.

What the car does, same across all 3 runs:
- Launch is fixed. Floors it for ~1s at the start, does NOT lose control.
- Throttle is modulated, ramps smoothly to ~70 kph.
- Left-right steering oscillation starts and then GROWS with speed.
- From ~70 to ~115 kph the oscillation increases until the car spins.

This is the run7 weave, now at real pace. It happens on the opening straight,
before the 300m R40 corner, which matches the CSV terminations clustered at
180-280m. The spatial-smoothness proxy did not suppress the weave at racing speed.

Good news worth keeping: launch, throttle modulation, and straight-line speed
(115 kph ~ 32 m/s, real pace, not a crawl) all work. The lookahead (run6) and
survival pressure (run7) are intact. The straight has exactly one problem left,
and it is the weave.

---

## 2. The contradiction to resolve FIRST

The car visibly weaves and spins, but `weave_penalty` (monitor CSV column 10) has
read ~0.0 across nearly the whole run.

Both cannot be innocent. If the car weaves and the logged penalty is ~0, the
penalty is NOT biting the dangerous weave.

Consequence for the reserve ladder: "still weaving -> raise WEAVE_WEIGHT 0.6 -> 0.9"
is the WRONG next move here. That rung assumed the term fires and is merely too
weak. 1.5 x ~0 is still ~0. Raising the weight does nothing until we know why the
term reads zero. Do not touch WEAVE_WEIGHT until the contradiction is explained.

---

## 3. Hypotheses (to MEASURE, not assume)

H1. on_line gate releases at speed.
The probe measured the weave excursion at 0.04m, ~25x inside OFF_DEAD (1.0m). But
that was a low-speed or synthetic weave. At 20-32 m/s the SAME steering
oscillation throws the car much farther laterally per swing. If center_off crosses
1.0m the on_line factor starts dropping; past OFF_FULL (2.5m) it is fully zero. So
the penalty may switch off exactly when speed makes the weave dangerous, while
still firing on the harmless low-speed weave the probe tested.

H2. straightness gate releases in the back half of the straight.
Spins log at 180-280m. The straightness factor uses bend over the next 80m. At
~220m, arc+80m reaches the 300m R40 corner, so bend_ahead rises and straightness
falls toward 0, releasing the gate right where the worst weave/spin happens.

H3. logging artifact.
Confirm what CSV column 10 actually aggregates: per-step value at termination,
episode mean, or episode sum. A final-step value captured during the spin (car
off-line, nose off) would log ~0 by construction even if the penalty fired hard
mid-episode. If so, the term may be biting and our verdict-by-CSV was misreading
a bad metric.

H4. steer deadzone / sampling.
Confirm |steer| at the logged sample points during the real weave actually exceeds
STEER_DEAD (0.1). If the aggregation samples points where |steer| is near zero
(swing crossings), the max(0, |steer|-0.1) term is ~0 there.

These are not mutually exclusive. H1 and H2 together would fully explain a visible
weave with a ~0 logged penalty.

---

## 4. What to instrument (one real weave-to-spin episode)

Run a scripted/deterministic rollout of rolling_115000 and log, PER STEP, through
the 70 -> 115 kph weave-into-spin:

- arc position (`self._cur_centerline_dist`)
- speed_horizontal
- center_off  (H1: does it cross 1.0m / 2.5m during the weave?)
- straightness and bend_ahead  (H2: does straightness fall in the 180-280m band?)
- |steer|, and max(0, |steer| - STEER_DEAD)  (H4)
- the four multiplied factors of weave_penalty AND their product
  (where, and due to which factor, does the product collapse to ~0?)
- and separately, confirm the column-10 aggregation method  (H3)

Deliverable from CC: a per-step table or quick plot over the weave-into-spin, plus
a one-line statement of which factor zeroes the product and at what
speed/arc-position. That single finding decides the fix.

---

## 5. Then (do NOT pre-commit; depends on what we measure)

- If on_line is the leak: the gate is the bug. Its purpose was to spare a genuine
  drift-back from penalty, but a high-speed weave is not a drift-back. Options to
  weigh: widen the on_line band, gate on rate-of-lateral-change instead of absolute
  offset, or drop on_line and lean on straightness alone.
- If straightness is the leak: the 80m horizon may pull the corner into the gate
  too early; shorten it, or the spin location is genuinely corner-entry, which
  reframes the problem as cornering (run9 racing line) rather than weave.
- If logging is the leak: fix the metric. The term may already be biting and we
  need a cleaner verdict.

Whatever we pick is ONE change, smoked first. Fresh-vs-warm gets decided per the
entrenchment rule (fresh whenever the change penalizes the prior policy's main
strategy; the weave IS the prior policy's main strategy, so a real strengthening
of the penalty likely wants fresh).

---

## 6. Reminders for CC

- One change at a time. Measure, do not assume. Smoke before any full run.
- The weave penalty is pure reward arithmetic, zero added sim load. The sim freeze
  at ~67.8k was a separate mechanical issue (working hypothesis: a violent
  corner-rollover hangs BeamNG). Keep the freeze-detector armed; if it trips,
  capture the pre-event episodes to test the rollover trigger.
- Pose-independence still holds: gate on centerline geometry + center_off, never on
  nose-contaminated observation bearings (that was the earlier trap).
- Pattern: CC pokes holes / measures against the real code and data, proposes the
  redesign, Mike approves, CC smokes, Mike approves the full run.
