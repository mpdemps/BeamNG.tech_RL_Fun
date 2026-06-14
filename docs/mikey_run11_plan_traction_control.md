# run11 plan: traction-control throttle gate (structural)

Date: 2026-06-14
Status: DRAFT for CC review. Chat drafts, CC pokes holes against the real code + the
run10 spin traces and proposes the final form, Mike approves, CC smokes, Mike approves
the full run. Do not implement from this draft.

Depends on: the run10 spin-causality probes (rolling_60000 and rolling_165000) and
`docs/research_weave_oscillation_fix_2026-06-14.md`.

Scope note: run11 is the THROTTLE/TRACTION fix. The racing line shifts to run12+.

---

## 1. Why the fix moved to the throttle axis (the locked diagnosis)

CC instrumented run10 spins at both the peak (rolling_60000) and maturity
(rolling_165000). In every spin, rising rear slip from a pinned/pulsing-high throttle
LEADS the steering sign-flip by 3-6 steps. The mechanism: throttle floored -> rear
slip climbs past the 2.0 deadzone (slip = wheelspeed - ground speed) -> rear breaks
loose -> the car counter-steers to catch it and overcorrects -> the left-right weave
-> spin. At maturity this happens at on-line center_off (2-5m), which is the exact
subtle weave watched since run7, just a finer expression of the same throttle root.
There is no separate steering-led mechanism.

That is why five runs of steering-axis work (runs 6-10: lookahead, three smoothness
penalties, the speed-scaled reference) never touched it. The steering is the symptom;
the throttle is the cause.

Why a structural fix, not a reward change: the run4 wheelspin penalty IS charging
(slip 4-11, far over its 2.0 deadzone), and the policy floors the throttle anyway
because speed and progress outvote it. This is the same lesson the weave taught three
times: a reward penalty gets outvoted. We considered and reject rebalancing the
wheelspin penalty / speed reward for this reason. The robust fix constrains the action
structurally so the car physically cannot dump spin-inducing power.

## 2. The change (one change): slip-gated throttle cap (traction control)

In the env step(), before the throttle command is sent to BeamNG, gate the applied
throttle on measured rear slip, like real traction control: when the rear is slipping,
cut power proportionally until grip returns.

```
slip = wheelspeed(Electrics) - speed_horizontal           # the existing slip metric
requested = max(0, action[1])                              # policy's throttle request
tc_factor = clip(1 - (slip - TC_SLIP_DEAD)/(TC_SLIP_FULL - TC_SLIP_DEAD), TC_MIN_THR, 1.0)
applied_throttle = requested * tc_factor                   # brake (action[1] < 0) untouched
```

Behavior: below TC_SLIP_DEAD the throttle passes through unchanged (grippy, full power
allowed); between TC_SLIP_DEAD and TC_SLIP_FULL it scales down; at/above TC_SLIP_FULL
it is cut to TC_MIN_THR. The loop settles where slip sits near the deadzone, i.e. at
the grip limit, exactly how TC holds a tire near its optimal slip.

Proposed starting constants (CC validates/tunes against the traces, see section 4):
- TC_SLIP_DEAD = 2.0 m/s (matches the wheelspin deadzone; full throttle below it)
- TC_SLIP_FULL = 6.0 m/s (the spins ran slip to 4-11; cut hard before runaway)
- TC_MIN_THR = 0.0 (full cut at TC_SLIP_FULL; CC may set a small floor ~0.1 if the
  car bogs)

Notes:
- This is an action transform (request -> applied), like a steering-rate limit on the
  steering axis. The policy may keep flooring; the car just can't spin from it.
- It reacts to current-step slip (one 50ms-step lag, like real TC sensor lag). Fine.
- Keep everything else from run10 unchanged: the speed-scaled steering reference (it
  is sound), the -25 backward penalty, the heading kill-switch, the 6-point lookahead,
  the wheelspin penalty (leave it in; once TC holds slip near the deadzone the penalty
  reads ~0 and goes quiet, so removing it is a second change we do not need). Obs is
  unchanged (15-dim).

## 3. What we are NOT doing yet (reserve ladder, one lever at a time)

- If a slip-controlled car still spins applying power MID-CORNER (power-on oversteer
  where lateral grip is in use) -> escalate to the full friction circle: also reduce
  the throttle cap as |steer| rises, so cornering and acceleration share the grip
  budget. run12.
- If TC makes the car timid / slow to accelerate -> raise TC_SLIP_DEAD or soften the
  cap (lower the cut rate / add a TC_MIN_THR floor).
- If the policy chatters (floors, gets cut, floors) in a way that hurts -> consider a
  throttle-rate limit. Reserve.
- Do NOT raise the wheelspin penalty or add a throttle reward term. It gets outvoted;
  that is the whole reason we went structural.

## 4. Measure first (before any build)

1. Confirm in code where throttle is applied and that the slip metric (Electrics
   wheelspeed - speed_horizontal) is the right signal to gate on; insert the cap
   cleanly before the BeamNG throttle command.
2. From the run10 traces, characterize slip during LEGITIMATE hard straight-line
   acceleration (a gripped launch has a nonzero optimal slip) versus the spin-runaway
   slip. TC_SLIP_DEAD must sit above the legitimate-accel slip and below the runaway,
   or TC will throttle real acceleration and make the car slow. This is the key
   number that sets the constants and the main risk control.

## 5. Validation plan (same rigor as run9/run10)

1. Probe: at low slip the throttle passes through unchanged; at high slip it is cut
   proportionally; the cut is smooth, bounded, no discontinuity; brake path untouched.
2. Smoke ~7k fresh: TC fires live (applied throttle < requested when slip is high),
   no NaN, obs shape unchanged, all run7/run10 mechanics intact (-25, kill-switch,
   wheelspin, lookahead, speed-scaled reference), episode logging works. Add a logged
   metric for how often/how much TC is cutting, so we can read it from the CSV.
3. Fresh 500k on Mike's approval. Auto-restart wrapper + temp logger armed (run10 did
   not freeze, but keep them).

## 6. Fresh, not warm

Fresh. We are constraining the policy's dominant strategy (flooring the throttle), and
the run4 lesson is explicit: warm-starting a policy whose main behavior you then
constrain entrenches it. The obs is unchanged so warm is technically possible, but
fresh is the safe call. Keep run10 training until run11 launches so nothing is lost.

## 7. Risks and the G14 watch questions (the verdict)

The watch decides. After maturity, watch on the G14 3+ times:
1. THE STRAIGHT: is the weave gone now? With slip held near the grip limit, the rear
   should not break loose, so there should be no slide to counter-steer and no
   left-right. This is the payoff and the test of the entire throttle-root diagnosis.
2. SPEED / TIMIDITY: is the car still fast, real pace, getting power down on the
   straights? The main risk of TC is an over-cut, timid, slow car. If it crawls or
   will not accelerate, TC_SLIP_DEAD is too low.
3. CORNERS: does it still spin applying power mid-corner (-> run12 friction circle), or
   does it finally take the corner? This is the next problem if run11 lands.

## 8. Constraints reminder for CC

- One behavioral change: the slip-gated throttle cap. Everything else stays as run10.
- Measure before building (the legitimate-accel-slip number especially). Smoke before
  the full run.
- Structural constraint, not a reward penalty (penalties get outvoted, proven).
- Pose-independent geometry unchanged; this is a throttle-axis change only.
- Pattern: CC validates/refines against the real code and the run10 traces, proposes
  the final cap form + constants and whether to keep or drop the now-redundant
  wheelspin penalty, Mike approves, CC smokes, Mike approves the full run.
