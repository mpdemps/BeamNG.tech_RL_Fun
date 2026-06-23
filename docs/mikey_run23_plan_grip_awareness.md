# run23 plan: teach the hybrid to handle losing grip

Date: 2026-06-20
Status: DRAFT for CC + Mikey review. Chat drafts; CC builds + probes against the real code;
Mikey approves the direction; CC smokes; Mike approves the full run. Do not implement from this
draft.

Baseline: run22 (the controller-led additive bounded residual). run23 keeps that architecture
and fixes the two things the run22 watch exposed: a spun car never terminates (it does endless
donuts), and the car cannot see when it is off-track on low-grip surface.

---

## 0. The run22 watch (what we saw)

Major improvement: the controller takes the proper racing LINE through T1, the hybrid working
as designed. But the RESIDUAL over-throttles on the T1 EXIT and spins, and then the car does
full-throttle donuts and NEVER TERMINATES, even out in the grass. It has no idea it is off the
road on a low-grip surface.

## 1. The diagnosis (why the donut is the root cause)

The non-terminating donut is not just ugly, it is probably WHY the residual has not learned to
behave:
- Normally, over-throttle -> spin -> off-track termination + penalty teaches the residual "lift
  on the exit." But if the spin never terminates, that sharp penalty signal never arrives, so
  the residual never learns to lift.
- Worse, long non-terminating donut episodes likely FLATTER the eval reward (the episode keeps
  accumulating instead of ending), so this "best_model" may score well precisely because it
  does not terminate, the cards-lying trap again, caused by a broken termination.
- And the car cannot SEE that it is off-track on grass (no grip/surface signal in the obs), so
  it floods the throttle blindly instead of lifting.

So fixing termination + adding grip awareness should both stop the garbage data AND unblock the
residual learning to lift on the exit.

## 2. The changes (coherent bundle: "handle losing grip")

a. LOSS-OF-CONTROL termination. A sustained very-high slip angle (the car is spinning, well
   past any controlled slide) terminates the episode with the crash/off-track penalty,
   REGARDLESS of position, so a donut anywhere ends immediately. CC sets the threshold from the
   spin telemetry (e.g. |beta| > ~60-75 deg sustained for a few steps, clearly a spin, not a
   transient). NOTE for Phase 2: a future drift agent will WANT high beta, so this threshold is
   a lap-phase choice we will revisit when we design drift; for now a spun car must terminate.

b. FIX/verify the off-track termination during the donut. The 8m off-track did not fire while
   the car was in the grass, investigate why (likely the donut stays within 8m near the track
   edge, or a spin-position edge case) and make it fire reliably. The loss-of-control
   termination (a) is the catch-all, but off-track should also work.

c. OFF-TRACK / GRIP signal in the observation. Add a signal telling the policy when it is off
   the road / on a low-grip surface, so it can learn to lift and recover instead of flooring it
   blindly. Cheapest: the off-track distance we already compute for termination, normalized
   into the obs. Better if available: BeamNG's surface material (tarmac vs grass) if it is
   exposed. CC picks the cleanest; this adds an obs dim (shape change).

## 3. Keep from run22

The additive bounded residual (applied = controller + clip(policy, +/-delta), delta=0.12,
controller at FULL so it laps), reward unchanged, plain SAC, spawn curriculum, steer-rate 0.5,
eval = the full hybrid, the residual_abs + 20 eval/* cards. run23 changes only the termination
logic and adds the grip obs signal.

RESERVE (not in run23): if the residual STILL over-throttles on the exit after the termination
fix + grip awareness, dial down the residual's THROTTLE authority (a smaller throttle delta).
That is a hybrid tuning, not a traction-control script. Try the learning-honest fixes first.

## 4. Fresh, not warm

Fresh. The obs shape changes (the new grip dim), and run22's residual was learning under the
corrupted (non-terminating donut) signal, so it does not transfer cleanly. A fresh residual
starts near zero (the car laps, controller-led, from step 1) and learns clean under the fixed
termination + grip awareness.

## 5. Eval and the de-flattering effect

Eval stays the full hybrid (controller + residual). Importantly, with the termination fixed,
the eval reward will no longer be inflated by non-terminating donuts, so eval/mean_reward
becomes trustworthy again, AND watch eval/max_arc vs the 4326m controller baseline as before.
Expect the de-flattered eval to read LOWER at first than run22's flattered numbers, that is
correct, not a regression.

## 6. Validation plan

1. Termination probe: force a spin (huge beta) -> the loss-of-control termination fires with the
   penalty; force an off-track position -> the 8m off-track fires. No false-fire in clean
   driving or normal cornering (beta well below the spin threshold).
2. Grip-obs probe: the new signal reads on-track on the road, off-track in the grass, bounded;
   obs shape is the new dim count.
3. Smoke ~7k fresh: the hybrid laps (controller-led), residual bounded, spins/donuts now
   TERMINATE quickly (check episode lengths dropped vs run22's long donut episodes), grip signal
   live, no NaN, cards intact.
4. Full run on Mike's approval. Wrapper + temp logger armed.

## 7. G14 watch question (the verdict)

After maturity, watch the full hybrid 3+ times:
1. Does it take T1 cleanly (controller, expected) and does the residual now LIFT on the exit
   instead of over-throttling into a spin?
2. When it DOES lose grip, does it recover (lift, gather it up) or at least TERMINATE cleanly
   instead of donutting forever?
3. Does it lap, and beat the controller baseline?

## 8. Reserve ladder

- Residual still over-throttles the exit -> reduce the throttle delta (smaller residual throttle
  authority).
- Loss-of-control threshold too tight (terminates a recoverable slide) -> raise it.
- Grip signal not enough on its own -> combine with the throttle-delta reduction.

## 9. Constraints reminder for CC

- Coherent bundle: loss-of-control termination + off-track-fire fix + grip obs signal. All
  RL-consistent (correct termination + better observation), NOT throttle scripting.
- Keep the run22 residual hybrid (delta=0.12, controller at full, reward unchanged). Fresh.
- Investigate WHY the 8m off-track did not fire during the donut before adding the catch-all, so
  we fix the real cause, not just paper over it.
- The loss-of-control threshold is a LAP-phase choice; flag it for revisit at Phase 2 (drift
  wants high beta).
- MIT-clean, no secrets.
- Pattern: CC builds + probes the termination + grip obs, proposes the thresholds + the grip
  signal source, Mikey approves, CC smokes, Mike approves the full run.
