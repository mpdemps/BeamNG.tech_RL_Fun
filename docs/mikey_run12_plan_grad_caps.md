# run12 plan: Grad-CAPS action smoothness (policy-level, both axes)

Date: 2026-06-15
Status: DRAFT for CC review. Chat drafts, CC pokes holes against the real SB3/SAC code,
proposes the final implementation + constants, Mike approves, CC smokes, Mike approves
the full run. Do not implement from this draft.

SUPERSEDES `docs/mikey_run12_plan_throttle_rate_limit.md`: the deep-research deploy
decision (`docs/research_smoothness_deploy_decision_2026-06-15.md`) says the throttle-
rate limit is a single-axis damper of a coupled cycle. run12 is now Grad-CAPS. Racing
line shifts to run13+.

---

## 1. Why (the locked picture)

run11 confirmed the oscillation is a COUPLED steer-throttle limit cycle (a fishtail):
TC holds steady traction, but the policy bang-bangs the throttle on the straight, the
spikes beat the reactive cap, the rear steps out, and the steering counter-steers, the
two axes feeding each other. Six runs of per-axis fixes (three reward penalties, the
speed-scaled reference, the traction cap, and the proposed throttle-rate limit) each
addressed one facet of the same coupled cycle.

The research deploy decision: attack the GENERATOR, not a facet. The generator is the
policy (SAC's tanh-Gaussian bang-bang/extremal-action tendency). The fix is a
policy-level action-smoothness regularizer added to the actor loss, which (a) lives in
the loss, not the reward, so it cannot be outvoted (every reward penalty was), and (b)
is a single norm over the FULL action vector, so it damps both axes at once. Use
Grad-CAPS rather than plain CAPS so a fast car keeps its corner agility (plain CAPS
penalizes the magnitude of action change and rounds off hard corners; Grad-CAPS
penalizes the jerk, so big clean swings are free and only zigzag is charged).

## 2. The change (one change): Grad-CAPS in the SAC actor loss

Add a temporal action-smoothness regularizer to the actor objective, on the
DETERMINISTIC mean action mu(s):

```
# standard SAC actor loss (unchanged):
actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

# Grad-CAPS temporal term (penalize the change-in-change of the mean action,
# displacement-normalized so large CLEAN swings are nearly free and zigzag is charged):
a_prev = mu(obs_{t-1});  a_t = mu(obs_t);  a_next = mu(obs_{t+1})
d1   = a_t - a_prev
d2   = a_next - a_t
disp = a_next - a_prev
L_temp = || d2 - d1 ||_2  *  tanh( 1 / (||disp||_2 + eps) )      # over the full (steer, throttle) vector
actor_loss = actor_loss + LAMBDA_T * L_temp.mean()

# spatial term OFF initially (sim-only): LAMBDA_S = 0, so no noised forward pass.
# when we go sim-to-real later, add: L_spat = || mu(s) - mu(s + N(0, sigma)) ||_2
```

Key implementation points (CC to validate against SB3 SAC source):
- Regularize the deterministic mean mu(s) (e.g. via the actor's dist params), NOT the
  sampled action. The mean is the policy's actual behavioral output.
- Grad-CAPS needs a THREE-action window (a_prev, a_t, a_next). An off-policy batch gives
  only (obs_t, obs_{t+1}); the third needs obs_{t-1}. CC picks the cleanest source:
  store prev_obs / prev_action in the replay buffer, or sample sequential triplets.
  This 3-window plumbing is the main implementation cost; if it proves too fiddly,
  fall back to plain CAPS (2-window, free from obs/next_obs) and tune LAMBDA_T down to
  manage over-smoothing (see section 6).
- eps small (e.g. 1e-6) in the displacement normalization; the tanh keeps it bounded.
  Numerical stability matters: run11 ended on a late-run NaN, so the new term must not
  introduce a div-by-zero or unbounded gradient. Confirm bounded over the smoke.

## 3. Keep everything else (it composes)

Keep run11's traction cap (DEAD=4/FULL=7/MIN=0.1), the wheelspin deadzone at 4.0,
run10's speed-scaled steering reference, the -25 backward penalty, kill-switch, 6-point
lookahead. These are in-loop structural pieces the policy trains within, and the
research says they compose with a smoothness regularizer rather than conflicting. Obs
stays 15-dim. The throttle-rate limit is NOT added (Grad-CAPS smooths the throttle axis
at the policy level, making a separate single-axis slew limit redundant).

## 4. Constants (start values; CC may tune)

- LAMBDA_T = 1.0 (the racing-validated CAPS/Grad-CAPS temporal weight; Grad-CAPS is far
  less sensitive to this than plain CAPS)
- LAMBDA_S = 0.0 for now (sim-only; the spatial term buys sim-to-real robustness we do
  not need yet, and skipping it avoids the extra noised forward pass)
- sigma = observation noise/scale (only relevant once LAMBDA_S > 0, i.e. before any
  sim-to-real)
- eps = 1e-6 (displacement-normalization guard)

## 5. Fresh vs warm

Default: fresh. The bang-bang is the entrenched dominant strategy; learning smooth from
scratch is the clean, consistent call (and it has been right every run since run5). One
nuance CC may weigh against the Aug-6 clock: Grad-CAPS is a continuous in-loop gradient
pressure (not a one-shot constraint), so warm-starting from run11's competent-traction
350k policy and letting the regularizer pull it smooth is defensible and could save
~12h, at the risk of fighting entrenched bang-bang weights. Recommend fresh; warm is a
considered fallback. Keep run11 artifacts; nothing to lose.

## 6. Validation plan

1. Plumb + unit-check (measure-first): confirm the 3-action window source, the mean-
   action extraction in SB3 SAC.train(), and that L_temp computes correctly, near-zero
   on a clean steady ramp of actions, and large on an alternating zigzag. Confirm
   bounded / no NaN path.
2. Smoke ~7k fresh: training runs, no NaN, the Grad-CAPS term is active and the logged
   action-fluctuation is trending DOWN vs an unregularized baseline, TC + speed-scaled
   reference intact, obs unchanged. Log per-eval: mean ||a_t - a_{t-1}|| (both channels)
   and the FFT high-frequency energy of the steering and throttle streams.
3. Fresh 500k on Mike's approval. Auto-restart wrapper + temp logger armed.

## 7. Over-smoothing watch (the key risk, baked in)

Grad-CAPS is chosen specifically to resist over-smoothing, but watch for it anyway,
because an over-smoothed car is the failure mode for a racer.
- Online metrics (logged from step 1 above): action-fluctuation falling toward zero
  WHILE return stalls/falls is the over-smoothing signature; FFT high-freq energy
  should drop (good) but total action energy in corners should NOT collapse.
- G14 watch questions at maturity, 3+ times:
  1. THE STRAIGHT: is the weave finally gone on BOTH axes, smooth held steering and
     smooth throttle, no zigzag, no fishtail? This is the verdict on the whole arc.
  2. THE CORNER (the over-smoothing test): does the car still take fast, decisive,
     clean lines, or has it gone timid, rounding off / understeering / late to turn in?
     Rounded, sluggish corners with smooth straights = over-smoothing -> lower LAMBDA_T.
  3. Pace: still fast, getting power down (TC + smoothness should make it faster, not
     slower, per the slip sweep).

## 8. Reserve ladder

- Over-smoothed (sluggish corners) -> lower LAMBDA_T.
- Residual oscillation remains -> raise LAMBDA_T, or add the spatial term (LAMBDA_S>0).
- 3-action-window plumbing too fiddly -> plain CAPS (2-window) with LAMBDA_T tuned down,
  accepting more over-smoothing risk.
- Policy-level loss smoothness still insufficient -> LipsNet-L (network-level adaptive
  Lipschitz actor) as the deeper alternative.
- The reserved throttle-rate limit and the friction circle remain available if a
  specific residual symptom (transient throttle spike; mid-corner power-on oversteer)
  survives, but the research says Grad-CAPS should subsume the need.

## 9. Constraints reminder for CC

- One behavioral change: the Grad-CAPS actor-loss term. Keep TC + speed-scaled
  reference. Do not add the throttle-rate limit.
- This touches the TRAINING LOOP (subclass SAC, override train()), so it carries more
  implementation risk than an action transform: validate numerics (no NaN), confirm the
  term is actually applied to the mean action and actually reduces fluctuation in the
  smoke before the 500k.
- Loss regularization, not a reward penalty (penalties get outvoted, proven).
- Pattern: CC implements + validates against the real SB3 SAC code, proposes the final
  form (3-window source, LAMBDA_T, fresh-vs-warm), Mike approves, CC smokes, Mike
  approves the full run.
