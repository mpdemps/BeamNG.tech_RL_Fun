# run10 plan: speed-scaled steering reference (structural weave fix)

Date: 2026-06-14
Status: DRAFT for CC review. Chat drafts, CC pokes holes against the real code +
the run9 data and proposes the final form, Mike approves, CC smokes, Mike approves
the full run. Do not implement from this draft.

Depends on: `docs/research_weave_oscillation_fix_2026-06-14.md` (the evidence) and
`docs/mikey_run8_postwatch_weave_diagnostic.md` (the trace tooling).

Scope note: run10 is the STRUCTURAL weave fix. It ends the reward-penalty era
(runs 6-9, three failed penalty forms). The racing line, previously slated as
run10, shifts to run11+, since a weaving car cannot benefit from a racing line.

---

## 1. Why we are abandoning reward shaping for the weave

Three reward penalties failed: total-variation smoothness (run6/7, frequency-dodged),
position-gated spatial (run8, never engaged), oscillation-signature (run9, bit hard
at -50/episode but the car weaved through it and paid). Two G14 watches confirmed the
weave persists. The deep-research report explains why: this is a control instability,
not a rewarded proxy, and reward-penalty smoothing has a documented no-win window
(too small does nothing, too large collapses into timidity). You cannot penalize away
a control instability. Fix the controller.

## 2. The diagnosis (control instability, speed-dependent)

Three documented mechanisms compound, and all match our symptom (weave grows 70->115
kph, ends in a spin):

1. Vehicle yaw damping is inversely proportional to speed (Stanley/DARPA), so a
   steering behavior stable at low speed goes underdamped at high speed with no change
   in the policy. This is why the weave grows with speed.
2. Our heading_err is computed from a FIXED 10m lookahead point. A short fixed
   steering reference is the classic pure-pursuit hunting setup; preview time (10m /
   speed) shrinks as speed rises, losing phase margin. Lookahead should scale with
   speed (L_d = k * v).
3. SAC + tanh structurally gravitate to bang-bang / full-lock actions, which is the
   weave's command, and it only spins at speed because the steering-to-yaw plant gain
   rises with speed.

Ruled out: this is not exploration jitter; we watched the deterministic policy and the
weave is in the mean.

## 3. The change (one change): speed-scaled steering reference

Replace the fixed-10m heading reference with a speed-scaled one.

```
L_d   = clip(PREVIEW_TIME * speed_horizontal, L_MIN, L_MAX)
ref_point = centerline point at arc-distance L_d ahead of the car (interpolate on self._cum_arc)
heading_err = bearing(car -> ref_point), encoded as the existing heading_err is encoded
```

Proposed starting constants (CC to validate/tune against geometry + the run9 trace):
- PREVIEW_TIME ~ 1.5 s (so at 30 m/s, L_d ~ 45m; at launch speed it floors at L_MIN)
- L_MIN = 10 m (current value; keeps a near reference at low speed so launch still works)
- L_MAX = 50 m (cap so high speed does not understeer / cut corners)

Notes / why this is minimal:
- The observation ALREADY carries the 6-point fixed-arc preview [10,20,40,80,160,280]m
  from run6 (the speed-perception lookahead). run10 does NOT add points. The only
  change is which arc-distance feeds heading_err, the explicit steering reference.
- This is an OBSERVATION change (heading_err is part of the obs vector), so the policy
  must be retrained FRESH (the obs distribution moves). That also suits the
  entrenchment rule.
- Companion cleanup (recommended, CC's call): drop the run9 reward weave-penalty at the
  same time, since the research says it is the wrong layer and run9 proved it does not
  change behavior. Removing a behaviorally-inert term gives clean attribution (run10's
  result is purely the reference fix). If CC prefers strict isolation it may keep the
  penalty at its current weight, but carrying a known-wrong term is clutter.

## 4. What we are NOT doing yet (reserve ladder, one lever at a time)

- If the weave persists after the speed-scaled reference -> add a STEERING-RATE LIMIT
  in the env (action sets a steering target that slews at a capped rate, ~real-wheel
  rates; or clamp per-step delta-steer). Makes bang-bang impossible by construction.
  CRITICAL: must be trained IN-LOOP, never bolted onto a trained policy (CAPS Fig. 5:
  post-hoc filtering can cause catastrophic loss of control). Reserve for run11.
- If structure alone is not enough -> CAPS-style temporal smoothness in the ACTOR LOSS
  (not the reward): L_T = ||pi(s_t) - pi(s_{t+1})||, needs subclassing SB3 SAC.train().
  The literature's strongest-endorsed smoothness method. Reserve for run11/12.
- Do NOT raise WEAVE_WEIGHT or add a fourth reward penalty. The no-win window is
  documented and we have three failures.

## 5. Measure first (before any build)

Per the standing rule, confirm the diagnosis on existing data, do not assume:
1. Read the code: confirm heading_err is derived from the fixed 10m point and how the
   policy consumes it relative to the 6-point preview.
2. From the run9 trace (or a deterministic rollout), confirm the weave frequency and
   amplitude scale with speed the way pursuit instability predicts (worse preview time
   at higher speed). If the speed-scaling is NOT present in the data, stop and rethink,
   because the whole fix rests on that mechanism.

## 6. Validation plan (same rigor as run8/run9)

1. Offline / probe: heading_err now uses a speed-scaled arc distance; sane,
   monotonic, no discontinuities across the full speed range; L_d floors and caps
   correctly; corner geometry still resolves (no reference pointing past a corner at
   high speed in a way that cuts it).
2. Smoke ~7k fresh: learning starts, no NaN, obs shape correct, all run7/run9
   mechanics intact (-25 backward, kill-switch, wheelspin, the 6-point preview),
   episode-mean logging still works.
3. Fresh 500k on Mike's approval.

## 7. Fresh, and the freeze companion (do this or we babysit again)

Fresh: forced by the obs change and consistent with the entrenchment rule.

Freeze reality: both run8 (67.8k) and run9 (~78k) froze on the BeamNG corner-rollover
hang as the policy reached the corner. run10 will reach the corner too, so it WILL
likely freeze around the same point. Before/with run10, strongly recommend CC build an
EVENT-DRIVEN auto-warm-restart: when the freeze-detector fires (max_arc < 1e-3 /
mean_speed ~0 sustained), automatically relaunch from the last clean rolling
checkpoint with learning_starts=0. This is infrastructure, not a behavioral change, so
it does not violate one-change-at-a-time. The run8 handoff rejected a PERIODIC restart
wrapper as too heavy; an event-driven one is light and now justified by n=2. Without
it we hand-feed warm-restarts at every freeze, which burns time against the Aug 6
license.

## 8. Risks and G14 watch questions (the verdict, as always)

The watch decides, not the curves. After maturity (~120-150k, with warm-restarts past
the freeze), watch on the G14, 3+ times:
1. THE STRAIGHT AT SPEED: is the weave gone, especially at 100+ kph? This is the whole
   point of run10. The specific test is high-speed smoothness, since that is where the
   instability lived.
2. Understeer / corner-cutting: a speed-scaled reference that looks too far ahead can
   understeer or cut corners. Watch turn-in at R40 and the hairpins. If it understeers,
   lower PREVIEW_TIME or L_MAX.
3. Launch: L_MIN must keep the car able to get going from a stop.

## 9. Constraints reminder for CC

- One behavioral change (the steering reference). The auto-restart is infra, allowed
  alongside. Measure before building. Smoke before any full run.
- Pose-independent geometry only (centerline + arc), never nose-contaminated obs
  bearings for gating.
- The fix is in the controller/reference, not the reward. Resist adding penalties.
- Pattern: CC validates/refines against the real code and the run9 data, proposes the
  final form (constants, exact heading_err implementation, penalty keep-or-drop),
  Mike approves, CC smokes, Mike approves the full run.
