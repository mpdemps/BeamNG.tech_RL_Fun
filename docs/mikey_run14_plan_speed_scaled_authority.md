# run14 plan: speed-scaled steering authority (kill full-lock-at-speed)

Date: 2026-06-15
Status: DRAFT for CC review. Chat drafts, CC measures + pokes holes against the real
code/traces, proposes the final curve, Mike approves, CC smokes, Mike approves the full
run. Do not implement from this draft.

Depends on: the run13 deterministic spin trace (`logs/run13_g14_spin_trace.csv`),
`docs/track_reference.md`.

Scope note: run14 completes the steering axis, the magnitude analog of the traction
cap. Racing line shifts to run15+.

---

## 1. Why (the run13 deterministic trace)

run13 killed the fast slam (HF energy collapsed), but the deterministic trace of
rolling_200000 shows the residual is steering MAGNITUDE: at every grip onset the
throttle is already near-full and STEADY (Δthrottle ~0, no spike, TC cuts the slip
correctly), and the trigger is the policy driving to ±0.97 FULL LOCK at 100-130 kph.
Full lock at that speed rotates the car regardless of throttle. The held throttle keeps
the rear at the edge (background enabler), but the trigger is steering magnitude.

run13's rate limit caps the SLEW, not the DESTINATION: the policy ramps to full lock
over 2-3 steps, each under the 0.5/step cap, and full lock at 120 kph still arrives. So
the rate limit was necessary but not sufficient. The fix the rate limit cannot reach is
a cap on the steering MAGNITUDE that tightens with speed.

This is the steering analog of the traction cap (TC caps throttle magnitude by slip;
this caps steering magnitude by speed) and the run14 reserve we flagged at run13.

## 2. The change (one change): speed-scaled steering authority cap

In step(), after the run13 rate limit, clip the steering to a speed-dependent cap:

```
requested = clip(action[0], -1, 1)
# run13 rate limit (slew):
rate_limited = prev_steer + clip(requested - prev_steer, -STEER_RATE, +STEER_RATE)
# run14 authority cap (magnitude vs speed):
cap = steer_authority(speed)                     # in [CAP_HIGH_SPEED, 1.0]
applied_steer = clip(rate_limited, -cap, +cap)
prev_steer = applied_steer                        # store; reset 0 each reset()
```

`steer_authority(speed)` = full lock at low speed (for the slow tight corners), tapering
to a low value at high speed (where full lock spins and no corner exists). Proposed
placeholder shape (CC sets the real breakpoints from the sweep, section 4):

- speed <= 15 m/s (~54 kph): cap = 1.0 (full lock available; tightest corners are slow)
- speed >= 30 m/s (~108 kph): cap = CAP_HIGH (low, e.g. ~0.25, enough to correct a line
  drift on the straight, not enough to rotate the car)
- linear interpolation between.

The governing principle: the cap should track the GRIP-LIMIT steering, the most steering
the car can hold at each speed without the rear breaking loose. Below it the tire grips;
above it (e.g. full lock at 25 m/s, tighter than any grippable radius) it slides. The
spin is the policy exceeding that limit; the cap enforces it.

## 3. Keep everything else (the whole stack)

Keep run13 steering-rate limit (STEER_RATE=0.5), run12 Grad-CAPS (λ_T=1.0), run11 TC
(DEAD=4/FULL=7/MIN=0.1, wheelspin deadzone 4.0), run10 speed-scaled reference, -25,
kill-switch, 6-point lookahead, and the wrapper buffer fix (WARM_LEARNING_STARTS=5000).
Obs 15-dim. One new action transform (the authority cap), composed after the rate limit.
The rate limit stays, it caps slew; this caps magnitude; together they bound the
steering on both, with Grad-CAPS on the chatter and the speed-scaled reference on the aim.

## 4. Measure first (sets the authority curve)

The clean-tracker drive failed before, so measure from existing data, not a fresh
controller:
1. From the run13 trace + corner-passing episodes, find for each speed band the MAX
   |steer| that was sustained WITHOUT grip loss (slip controlled, no spin, heading
   holding). That empirical "grip-limit steering vs speed" curve is the cap target,
   set the cap at or just above it.
2. Cross-check against what the corners NEED at their real entry speeds (use
   docs/track_reference.md), especially the fast T8 (R68, entry ~90+ kph / ~25 m/s) and
   the slow tight T1/T5/T6 (taken at low speed, need high lock). The cap must allow each
   corner's required steer at its entry speed or it will understeer that corner.
3. Confirm the gap: full lock (±0.97) at the 19-32 m/s spin band must be above the cap
   (so it's blocked), while corner needs at their speeds are below the cap (so they pass).
Report the curve (breakpoints + CAP_HIGH) from the data, not the placeholder.

Key risk to size against: CAP_HIGH too low -> can't correct a line drift or turn at
speed (understeer/wash wide); too high -> still spins. The T8 entry-steer at its entry
speed is the binding constraint on the high-speed end.

## 5. Validation plan

1. Probe: the cap clips |steer| to steer_authority(speed); full lock passes at low
   speed, limited at high; composes with the rate limit + Grad-CAPS + TC; T8's required
   steer at its entry speed still passes (not strangled).
2. Smoke ~7k fresh: cap active (steer clipped at high speed), no NaN, mechanics intact
   (rate limit, Grad-CAPS, TC all live), obs unchanged. Log steer_cap_frac (fraction of
   steps the authority cap binds) + applied-steer-vs-speed so we can read it.
3. Fresh 500k on Mike's approval. Wrapper (with buffer fix) + temp logger armed.

## 6. Fresh, not warm

Fresh, consistent with the project. (Warm-from-run13 is now lower-risk with the buffer
fix in place and could save ~12h, but we are constraining the entrenched over-steering,
so default fresh. Mike's call at launch.) Keep run13 artifacts.

## 7. Risks and the G14 watch questions (the verdict)

After maturity, watch 3+ times:
1. THE STRAIGHT AT SPEED: is the car finally stable, holding the line at 100+ kph with
   no full-lock rotate-out, and does it actually reach and get THROUGH T1 (the wall it's
   been stuck at all of runs 10-13)?
2. THE CORNERS (the key risk): does it still take the corners, especially the fast T8
   and the tight T1/hairpins, or does the authority cap make it understeer / wash wide?
   Wash-wide at a corner = cap too tight at that speed -> raise it there.
3. Does it get further than ~300m for the first time, onto the rest of the lap?

## 8. Reserve ladder

- Understeer / wash wide at a corner -> raise the cap at that corner's speed band.
- Still spins at speed -> lower CAP_HIGH (or the spin has another cause; re-measure).
- Can't correct a line drift at speed (cap floor too low) -> raise CAP_HIGH a little.
- If the stack now holds straights AND corners -> the stability work is DONE and run15
  is the racing line (optimal path, lap time), the frontier we've been deferring.

## 9. Constraints reminder for CC

- One behavioral change: the speed-scaled authority cap. Keep the entire run10-13 stack
  and the wrapper buffer fix.
- Measure the authority curve from the run13 traces + corner geometry before building
  (the T8 high-speed-corner constraint is the binding one). Smoke before the full run.
- Structural constraint, not a reward penalty.
- Pattern: CC measures + refines against the real code and the run13 trace, proposes the
  final curve (breakpoints + CAP_HIGH, + fresh/warm), Mike approves, CC smokes, Mike
  approves the full run.
