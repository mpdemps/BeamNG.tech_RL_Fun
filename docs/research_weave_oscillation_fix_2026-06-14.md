# Deep research: how to kill the steering weave (2026-06-14)

Question researched: the most effective, evidence-backed way to eliminate the
steering weave (a left-right limit-cycle that grows with speed and ends in a spin)
in our SAC sim-racing agent, given that three reward-shaping penalties (runs 6-9)
have failed.

Method: five parallel literature searches (root causes, CAPS/loss regularization,
action-space constraints, SAC-specific causes, racing-RL lookahead practice),
sources cross-checked. Load-bearing claims rest on peer-reviewed work, not the few
very recent preprints, which were down-weighted.

Bottom line up front: the literature strongly supports our reframe. The weave is a
CONTROL INSTABILITY, not a rewarded proxy, which is exactly why reward shaping keeps
bouncing off it. The fix is structural. The single best-fit first change for our
specific symptom is a speed-scaled steering reference, backed by a steering-rate
limit, with CAPS-style actor-loss regularization as the policy-level fallback.

---

## 1. Diagnosis: why it weaves, and why it gets worse with speed

Three mechanisms compound, and all three are documented.

**(a) Vehicle dynamics: yaw damping is inversely proportional to speed.** This is the
central reason the weave grows from 70 to 115 kph. The Stanley/DARPA paper derives
that the tires act as yaw dampers whose stabilizing reaction "is inversely
proportional to speed. As speed increases, the damping effect diminishes, creating a
need for active damping." So a steering behavior that is stable slow becomes
underdamped (oscillatory) fast, with no change in the policy. Our symptom is the
textbook signature.
Source: Hoffmann, Tomlin, Montemerlo, Thrun, "Autonomous Automobile Trajectory
Tracking for Off-Road Driving" (Stanley), 2007 —
https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf

**(b) The steering reference is too near for the speed.** Our heading_err comes from
a FIXED 10m lookahead point. A point-chasing controller with a short fixed lookahead
is the classic recipe for hunting: "if the distance is low, it can lead to
oscillations around the reference path." Lookahead must scale with speed (L_d = k*v),
because a fixed distance at higher speed means shorter PREVIEW TIME and less phase
margin. At 30 m/s, 10m is a third of a second of preview, which is twitchy. Pushing a
constant near lookahead at all speeds "led to undesirable behaviors including
oscillations, drifting and general loss of path tracking."
Sources: Sukhil & Behl, "Adaptive Lookahead Pure-Pursuit for Autonomous Racing,"
2021 — https://arxiv.org/abs/2111.08873 ; MathWorks Pure Pursuit Controller —
https://www.mathworks.com/help/nav/ug/pure-pursuit-controller.html ; Snider,
"Automatic Steering Methods for Autonomous Automobile Path Tracking," CMU 2009 —
https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf

**(c) SAC structurally gravitates to bang-bang / extremal actions.** Maximum-entropy
RL with a tanh-squashed Gaussian piles probability mass at the action limits, and
with no action cost the optimal control of a control-affine system is provably
bang-bang. A full-lock reversal is exactly the weave's command, and the bang-bang
only spins the car at speed because the steering-to-yaw plant gain rises with speed
(the low-pass smoothing that hides it slow fails fast).
Source: Seyde et al., "Is Bang-Bang Control All You Need?", NeurIPS 2021 —
https://arxiv.org/abs/2111.02552

We can rule out one alternative: this is not just SAC exploration noise. We watched
the deterministic rolling checkpoint and the weave is in the mean policy, not the
sampling jitter.

## 2. Why all three reward penalties failed (now explained, not just guessed)

The literature says reward shaping is the wrong layer for this, and it predicts our
exact three failure modes.

- It is a control instability. You cannot penalize away a dynamics/representation
  instability; you fix the controller (Stanley, CAPS framing).
- Reward-penalty smoothing has a documented narrow, brittle window: too small does
  nothing, too large collapses performance. LipsNet's head-to-head on vehicle
  trajectory tracking is the clearest evidence: a reward penalty at low weight left
  the control rough (return 825, fluctuation 0.27), and at high weight it killed
  return (825 -> 13) to get smoothness. There is almost no good middle. That is runs
  6/7/8 (too small, no effect) and the timidity risk we feared from just raising
  WEAVE_WEIGHT.
- And because SAC is structurally drawn to extremal actions and our progress+speed
  reward genuinely favors them, run9's penalty could bite hard (-50/episode,
  measured) and the policy still kept weaving and paid, because the task reward
  outweighed it. A safety/smoothness penalty fixed in absolute terms is eventually
  overwhelmed by the entropy/return pressure.
Sources: Song et al., "LipsNet," ICML 2023 —
https://proceedings.mlr.press/v202/song23b.html ; Mysore et al., CAPS, ICRA 2021 —
https://arxiv.org/abs/2012.06644 ; Seyde et al., NeurIPS 2021 (above).

## 3. The fixes, ranked by evidence and fit to our symptom

### Tier 1 — structural, attack the root, strongest evidence

**Fix A. Speed-scaled (and ideally multi-point) steering reference.**
Replace the fixed 10m heading reference with a lookahead that scales with speed
(L_d = k*v), or feed the policy a speed-scaled curvature preview rather than a single
near point. This is the most targeted fix for OUR symptom (fixed-near-reference +
oscillation-grows-with-speed). It is also how the best racing RL agent does it: GT
Sophy observes ~60 equally-spaced course points whose spacing scales with speed,
covering ~6 seconds ahead, not an instantaneous heading error.
Evidence: strong and directly on-point. Tradeoff: too-long a lookahead understeers /
cuts corners, so it must scale, not just lengthen.
Sources: Sukhil & Behl 2021; Wurman et al., "Outracing champion Gran Turismo drivers
with deep RL," Nature 2022 — https://www.nature.com/articles/s41586-021-04357-7

**Fix B. Steering-rate limit / slew-rate constraint in the env.**
Make the action a steering TARGET that the actual wheel slews toward at a capped rate
(or clamp per-step delta-steer, or reparameterize the action as steering-rate). This
makes bang-bang physically impossible regardless of what the policy wants. Real human
drivers rarely exceed ~90 deg/s at the wheel even at 0.6g, so a realistic cap costs
little legitimate capability. Stanley added an explicit steering-rate lead term for
exactly this reason.
Evidence: strong. CRITICAL caveat: this must be in the TRAINING loop, not bolted onto
a trained policy. CAPS Fig. 5 showed that filtering an already-trained RL policy
caused overshoot and, with an FIR filter, "total and catastrophic loss of control."
Tradeoff: too-tight a cap delays turn-in / understeers; tune toward real steering
rates.
Sources: Stanley 2007; Ford US9421973 (human steering-rate data); CAPS 2012.06644
(Fig. 5 warning).

**Fix C. CAPS-style action-smoothness regularization in the ACTOR LOSS (not the
reward).** Add a temporal smoothness term to the actor objective, L_T =
||pi(s_t) - pi(s_{t+1})||, so nearby-in-time states map to nearby steering. This is
the move from "penalize in the reward" (failed 3x) to "regularize the policy
mapping," which the literature repeatedly shows succeeds where reward penalties fail.
It is proven on exactly our kind of task: the image-based CAPS descendant (I-RAS)
raised a miniature racing success rate from 59% to 95% and won the 2022 AWS DeepRacer
championship; Grad-CAPS gave both higher return and smoother steering on Gym
CarRacing.
Evidence: strong, racing-specific. Tradeoff: over-smoothing -> sluggish/understeering
control; start temporal weight ~1.0, or use Grad-CAPS (penalizes zigzag while
allowing fast clean turns) to avoid over-smoothing. Implementation cost: requires
subclassing SB3 SAC.train() to add the term (the run8 reserve ladder already
anticipated this as the "full CAPS" escalation).
Sources: Mysore et al., CAPS, ICRA 2021; Hsu et al., I-RAS, 2022/2023 —
https://arxiv.org/abs/2307.08230 ; Lee et al., Grad-CAPS, 2024 —
https://arxiv.org/abs/2407.04315

### Tier 2 — cheap and complementary

- **Add previous action + a short observation history to the obs.** A state-only
  policy has no signal tying consecutive actions together, so smoothness is not even
  learnable. Including the last action (and stacking a few frames for velocity
  context) makes temporal coherence representable. Cheap, complements A/B/C.
  Source: CAPS framing; standard frame-stacking practice.
- **Do not over-reward center-line proximity.** AWS documents the DeepRacer "zigzag"
  as a reward-induced veer-then-overcorrect cycle from hugging the center line. Our
  reward gates on center_off; worth checking that we are not partly shaping the weave.
  Source: AWS DeepRacer docs —
  https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-how-it-works-action-space.html
- **Lower control frequency / action repeat.** Letting the network actuate faster
  than the car's dynamics settle invites chatter. Our tick is already 20 Hz; GT Sophy
  runs at 10 Hz. A modest reduction or action-repeat is a cheap lever, but too low
  loses corner reactivity. Treat as a knob, not a primary fix.
  Source: CAPS discussion; GT Sophy 10 Hz (Nature 2022).

### Ruled out
- More reward penalty (4th form, or just raising WEAVE_WEIGHT): the LipsNet result
  and three project failures say this is a no-win window.
- A deployment-time filter on the trained policy: CAPS Fig. 5 says this can be
  catastrophic; any filtering must be trained in-loop (that is Fix B done right).

## 4. Recommendation (prioritized, one change at a time)

Our specific evidence (oscillation grows with speed, reference is a fixed 10m point)
points most sharply at the steering reference. So:

1. **First, cheaply confirm the diagnosis** (measure, do not assume): from the run9
   trace, check that the weave frequency/amplitude scale with speed the way pursuit
   instability predicts, and confirm heading_err is the fixed 10m point. This is an
   offline read CC can do on existing data.
2. **First structural change: Fix A, speed-scaled steering reference.** Most targeted
   to our symptom, and a smaller change than re-architecting the loss. Likely we also
   remove the run9 reward weave-penalty at the same time, since the research says it
   is the wrong layer (keep it one clean change: drop the penalty, fix the
   reference).
3. **Strong backstop: Fix B, steering-rate limit in the env.** If A alone leaves a
   residual weave (the policy still wants to correct hard), the rate limit makes the
   high-frequency reversal impossible by construction. Could be combined with A as the
   structural package, or sequenced after it.
4. **Fallback if A+B do not fully resolve it: Fix C, CAPS temporal regularization in
   the actor loss.** The most evidence-backed smoothness method, and the right way to
   regularize if structure alone is not enough.

This ends the reward-penalty era (runs 6-9) for the weave. The next run should be a
structural change. Numbering and whether the racing line (previously run10) shifts
back is Mike's call; the weave fix should come first since a weaving car cannot
benefit from a racing line.

Separate track, not covered here: the BeamNG sim freeze on corner-rollovers (n=2) is
a mechanical issue, not addressed by any of the above. It still needs the
freeze-detector / auto-restart handling discussed earlier.

## 5. Key sources
- Stanley controller, yaw damping vs speed + steering lead term — https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf
- Adaptive lookahead pure-pursuit (L_d = k*v) — https://arxiv.org/abs/2111.08873
- Pure Pursuit lookahead tuning (MathWorks) — https://www.mathworks.com/help/nav/ug/pure-pursuit-controller.html
- Snider, automatic steering methods (CMU 2009) — https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf
- Is Bang-Bang Control All You Need? (SAC extremal actions), NeurIPS 2021 — https://arxiv.org/abs/2111.02552
- CAPS action-smoothness regularization, ICRA 2021 — https://arxiv.org/abs/2012.06644 (project: http://ai.bu.edu/caps/)
- I-RAS image-based CAPS, miniature racing (59%->95%, DeepRacer champ) — https://arxiv.org/abs/2307.08230
- Grad-CAPS (smoothness without over-smoothing), 2024 — https://arxiv.org/abs/2407.04315
- LipsNet (reward-penalty no-win window; network Lipschitz), ICML 2023 — https://proceedings.mlr.press/v202/song23b.html
- L2C2 locally-Lipschitz smoothness, IROS 2022 — https://arxiv.org/abs/2202.07152
- GT Sophy, speed-scaled multi-point preview, Nature 2022 — https://www.nature.com/articles/s41586-021-04357-7
- AWS DeepRacer zigzag (don't over-reward center hugging) — https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-how-it-works-action-space.html
