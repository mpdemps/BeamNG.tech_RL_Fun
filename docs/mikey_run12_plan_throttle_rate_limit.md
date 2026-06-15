# run12 plan: throttle-rate limit (kill the bang-bang spikes)

Date: 2026-06-15
Status: DRAFT for CC review. Chat drafts, CC measures + pokes holes against the real
code, proposes the final form + constants, Mike approves, CC smokes, Mike approves the
full run. Do not implement from this draft.

Depends on: run11 (traction control) and its G14 watch, plus the run10/run11 spin
instrumentation.

Scope note: run12 is the throttle-rate limit, the last piece of the throttle-control
stack. Racing line shifts to run13+.

---

## 1. Why (the run11 watch result)

run11's traction-control cap held steady traction on launch and acceleration, which
confirmed the throttle-root diagnosis: with throttle applied steadily, the rear stays
planted. But on the straight the policy BANG-BANGS the throttle (flooring pulses), and
each pulse spikes the slip faster than TC's one-step reactive cap can react, so the rear
still steps out and spins (with the counter-steer weave riding along). Mike's watch:
"holds good traction on rollout and acceleration, then bang-bang and throttle control
lost, flooring and tire spin forcing a spin," and the spin starts ON THE STRAIGHT (not
the corner, so this is not the friction-circle / cornering case).

TC is a reactive MAGNITUDE cap; it does not stop the policy from issuing a bang-bang
COMMAND, and the transient spike beats its lag. The fix is to stop the command from
spiking: limit how fast throttle can change per step so it must ramp, not jump.

This is the same structural move the whole project has converged on: remove the ability
to misbehave rather than penalize it. SAC gravitates to extremal/bang-bang actions
(the bang-bang RL result), so a reward penalty would be outvoted as always; a hard rate
limit makes the spike impossible.

## 2. The change (one change): throttle slew-rate limit

In step(), rate-limit the throttle REQUEST, then apply run11's TC on top:

```
requested      = max(0, action[1])
delta          = requested - prev_thr
delta          = clip(delta, -THR_RATE_DOWN, +THR_RATE_UP)   # per 20Hz step
rate_limited   = prev_thr + delta
prev_thr       = rate_limited                                # store for next step
# run11 traction control on the rate-limited request:
tc_factor      = clip(1 - (slip - 4.0)/(7.0 - 4.0), 0.1, 1.0)
applied_throttle = rate_limited * tc_factor                  # brake (action[1]<0) untouched
```

Rationale for the order: rate-limit the intent (kills the flooring spike), then TC cuts
for slip (holds the grip limit). The applied throttle is both ramped and slip-capped.

Parameters (THR_RATE_UP set from the sweep in section 4, not guessed):
- THR_RATE_UP = the key one: max throttle INCREASE per step. This is what stops the
  flooring pulse. Set just below the Δthrottle/step that spikes slip into runaway.
- THR_RATE_DOWN = generous / effectively unlimited. Lifting off has no traction risk
  and we must keep fast throttle-cut for corner entry, so do NOT restrict decreases.
- prev_thr resets each reset() (start from 0 throttle).

Note: when TC releases (slip drops, tc_factor 0.1 -> 1.0) the applied throttle can rise,
but only because grip returned, so no spike. The rate limit governs the request ramp;
TC governs the slip response. They compose cleanly.

## 3. Keep everything else

run11 TC (DEAD=4/FULL=7/MIN=0.1), the wheelspin deadzone at 4.0, run10's speed-scaled
steering reference, the -25 backward penalty, kill-switch, 6-point lookahead. Obs stays
15-dim unchanged (this is a throttle action-transform, it never touches the obs).

## 4. Measure first (sets THR_RATE_UP from data)

Two measurements before any build:

1. Confirm the chain on a real run11 straight-line spin: instrument it and show the
   order throttle-delta spike -> slip spike -> rear steps out -> spin, with TC cutting
   but lagging. This validates that the bang-bang COMMAND (not steady throttle) is the
   trigger, so a rate limit is the right lever.
2. Throttle-step sweep (the THR_RATE_UP number): from steady speed at a few speeds
   (~10, ~20, ~30 m/s), apply a one-step throttle STEP of increasing size and record
   the resulting slip spike. Find the largest Δthrottle/step that keeps the slip spike
   below the ~7 runaway threshold. THR_RATE_UP = below that. Report the speed-dependence;
   if it varies a lot, use the most restrictive value for a single constant (start
   simple), or propose a speed-scaled rate if the spread demands it.

## 5. Validation plan (same rigor as run11)

1. Probe: throttle increases are capped at THR_RATE_UP/step, decreases pass freely, a
   flooring request ramps smoothly over several steps, composes correctly with TC,
   bounded, brake path untouched.
2. Smoke ~7k fresh: rate-limit fires live (a flooring request shows up ramped), TC
   still cuts when slip is high, no NaN, mechanics intact, obs unchanged. Log a
   rate-limit-active metric alongside tc_cut so we can read both from the CSV.
3. Fresh 500k on Mike's approval. Auto-restart wrapper + temp logger armed.

## 6. Fresh vs warm

Default: fresh, consistent with every run since run5. One nuance worth CC weighing
against the Aug-6 clock: unlike the run4 entrenchment case, a rate limit TRANSFORMS the
action rather than PENALIZING the dominant strategy, so warm-starting from run11's
350k policy (which already has good steady-traction behavior) carries less entrenchment
risk than run4 did and could save ~12h. If CC is confident the warm policy adapts
cleanly to the ramped command, warm-from-run11-rolling_350000 is defensible. Otherwise
fresh. Mike's call at launch. Keep run11 until run12 launches so nothing is lost.

## 7. Risks and the G14 watch questions (the verdict)

The watch decides. After maturity:
1. THE STRAIGHT: is the weave finally gone? With no throttle spikes there are no slip
   spikes, so the rear should never break loose and there is nothing to counter-steer.
   If the straight is smooth AND fast, the throttle-control stack is complete and the
   weave that has run through this entire project is beaten.
2. ACCELERATION: is it still quick, or does the ramp make it sluggish off the line and
   out of slow corners? Main risk of the rate limit. If sluggish, raise THR_RATE_UP.
3. THE CORNER: now that the straight should be clean, what happens at the corner. If it
   spins applying power MID-CORNER (lateral grip in use), that is the friction circle ->
   run13. Otherwise we are into racing-line territory.

## 8. Reserve ladder

- Smooth-throttle car still spins MID-CORNER on power -> friction circle (couple the
  throttle cap to |steer|). run13.
- Rate limit too slow / sluggish -> raise THR_RATE_UP or make it speed-scaled.
- Policy still finds an instability the rate limit cannot smooth, or training is
  unstable -> CAPS actor-loss action-smoothness (the research's policy-level fix).
- Do NOT add a throttle reward penalty; it gets outvoted (proven repeatedly).

## 9. Constraints reminder for CC

- One behavioral change: the throttle slew-rate limit. Everything else stays as run11.
- Measure THR_RATE_UP from the throttle-step sweep before building. Smoke before the
  full run.
- Structural constraint, not a reward penalty.
- Pattern: CC measures + refines against the real code and the run11 spin, proposes the
  final form + THR_RATE_UP (+ THR_RATE_DOWN, fresh-vs-warm), Mike approves, CC smokes,
  Mike approves the full run.
