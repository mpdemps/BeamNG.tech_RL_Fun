# run22 plan: controller-led hybrid (additive bounded residual RL)

Date: 2026-06-20
Status: DRAFT for CC + Mikey review. Chat drafts; CC builds + probes the residual wrapper,
measures against the real code; Mikey approves the direction; CC smokes; Mike approves the
full run. Do not implement from this draft.

The decision (Mike + Mikey): keep a controller-LED hybrid. The hand-coded controller drives
(it laps the whole track cleanly), and the RL contributes a small polish on top. This is the
deliverable: a car that laps, robustly, with RL improving it where it can. Full controller
removal was tested in run21 and failed at our compute (the solo policy collapsed to a weaving
crawl), so we keep the controller permanently and let RL ride on top.

---

## 0. The architecture (and why NOT the convex-blend floor)

The run21 fade used a CONVEX blend: action = beta*controller + (1-beta)*policy. Holding that
at a small beta-floor would be MOSTLY policy, which is backwards: the policy is a crawler, so
letting it dominate wrecks the lap. Even a HIGH-beta floor is awkward, because the policy
outputs a full action weighted in, so its crawl/weave leaks through and it has to relearn to
match the controller.

The right architecture for "controller drives, RL polishes" is an ADDITIVE BOUNDED RESIDUAL:

```
applied_action = controller(obs) + clip(policy(obs), -delta, +delta)
```

- The controller ALWAYS outputs its full lapping action. The car laps by construction.
- The policy outputs a small CORRECTION, hard-clipped to +/- delta (small). It can nudge the
  controller's action but never replace or collapse it.
- The policy learns a correction DIRECTLY (not a full action), which is the right and easy
  thing to learn: "where can I shave a little, carry a little more speed, tighten the line."

This cannot degenerate the way run21 did: a bounded nudge on a lapping controller stays a
lapping car. Worst case the RL adds harmless small noise; best case it makes the lap faster
and smoother.

## 1. The change

One change: wrap the action as controller + clipped residual (above). Reuse CC's existing
controller (envs/base_controller.py, which laps) and the BlendSAC plumbing, switched from
convex blend to additive-bounded residual with the controller held at full.

## 2. The correction bound (delta)

delta sets how much the RL is allowed to change the controller's action. Start SMALL so the
car laps from the first step and the RL can only gently polish (CC proposes the value, e.g.
+/- 0.10 to 0.15 on the normalized action). If the RL proves it improves the lap without
destabilizing, delta can be raised later to give it more authority. Smaller is safer; this is
the conservative-first knob.

## 3. Reward (unchanged) + the polish target

Reward unchanged from run20/21 (follow the racing line, match its speed, gentle slip backstop,
anti-timid match term). The natural polish the RL can earn: the controller drives at
speed_factor 0.95 (a 5% grip margin), so there is real headroom for the RL to carry MORE speed
toward the line's full v_target where grip allows, and to tighten lines, earning more
progress/match reward. That is genuine improvement over the controller alone, not just noise.

## 4. Fresh, not warm

Fresh policy. run21's policy outputs full (crawler) actions; as a residual we want it to output
small corrections near zero. A fresh policy starts with tiny corrections (the car laps,
controller-led, from step 1) and learns useful nudges from there. run21's crawler does not
transfer to a residual role.

## 5. Eval = the FULL hybrid (controller + residual)

Unlike run21 (which eval'd the solo policy at beta=0 to test removal), here we DEPLOY the
hybrid, so EVAL runs the full controller + residual. The cards measure what we ship. Critically,
log the controller-ALONE lap as a fixed baseline (it laps; record its lap progress / mean_speed
/ lap time), so every eval answers the real question: does controller + residual BEAT the
controller alone, or just match it? The win is the hybrid lapping AND improving on the
controller (more speed, cleaner, eventually a faster lap).

## 6. Robustness (the point of this run)

A bounded correction on a lapping controller cannot collapse. So run22 is the SAFE outcome: it
will produce a car that laps, period. The open question is only how much the RL improves it.
That is the honest deal Mike and Mikey chose: a guaranteed lapping car, with RL polish that may
be modest.

## 7. Validation plan

1. Residual probe: applied_action = controller + clip(policy, +/-delta) computes correctly and
   stays bounded; with policy forced to 0 the action equals the controller (so it still laps);
   with policy at +/-delta the action stays sane.
2. Controller-baseline log: record the controller-alone lap (progress, mean_speed, lap proxy)
   as the fixed comparison line.
3. Smoke ~7k fresh: the hybrid laps (controller-led), residual is bounded, no NaN, the 20
   eval/* cards intact, eval runs the full hybrid. Log mean of |residual| so we can see how
   hard the RL is pushing.
4. Full run on Mike's approval. Wrapper (buffer fix) + temp logger armed.

## 8. G14 watch question (the verdict)

After maturity, watch the full hybrid (controller + residual) 3+ times:
1. Does it lap cleanly (expected, controller-led)?
2. Does the RL visibly improve it over the controller alone, more speed on the straights,
   tighter lines, smoother through T1, or is it indistinguishable (RL adding nothing)?
3. Anywhere the residual makes it WORSE (a wobble the controller did not have) -> lower delta.

## 9. Reserve ladder

- RL adds nothing visible -> raise delta (give it more authority), or accept the controller
  alone as the driver (the RL polish was the bet; if it does not pay, the controller still
  laps).
- RL destabilizes -> lower delta.
- RL clearly helps and wants more room -> raise delta gradually and re-evaluate.

## 10. Constraints reminder for CC

- Architecture: ADDITIVE bounded residual (controller at FULL + clipped policy correction), NOT
  the convex-blend floor. The controller's lapping action is always preserved.
- Reward unchanged from run20/21. Fresh policy. delta small first.
- Eval = full hybrid; log the controller-alone baseline so we can tell improvement from parity.
- MIT-clean (controller is ours). No secrets.
- Clock: BeamNG license expires 2026-08-06. This run is the safe, shippable result, a lapping
  car. Keep it simple.
- Pattern: CC wires the residual + baseline log, probes, proposes delta, Mikey approves, CC
  smokes, Mike approves the full run.
