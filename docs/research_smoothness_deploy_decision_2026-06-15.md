# Deep research: which smoothness fix to deploy for the coupled steer-throttle cycle (2026-06-15)

Question: given a confirmed COUPLED steering-throttle limit cycle (a fishtail) in our
SAC agent, where reward penalties were outvoted, the speed-scaled reference and the
traction cap each helped but were insufficient, and a throttle-rate limit would damp
only one axis, which single fix does the evidence say to deploy?

Method: five parallel literature searches (multi-axis regularization; method head-to-
head; loss-reg vs action constraints; fishtail control theory; SB3 implementation),
cross-checked. Load-bearing claims are peer-reviewed; recent preprints down-weighted.

## Bottom line (the deploy call)

Deploy GRAD-CAPS: policy-level action-smoothness regularization added to the SAC
actor loss, operating on the full (steering, throttle) action vector. Keep the run11
traction cap and the run10 speed-scaled reference (they compose). Do NOT ship the
throttle-rate limit (run12 as drafted) as the fix; it is a single-axis damper of a
coupled cycle. Drop reward smoothness penalties entirely (already done).

Grad-CAPS over plain CAPS specifically because our agent must stay fast: plain CAPS
over-smooths and rounds off hard corners; Grad-CAPS is built to keep large clean
steering swings while killing the zigzag.

## 1. Why a policy-level, both-axes fix (not another single-axis structural lever)

- The oscillation generator is the policy. SAC's tanh-Gaussian gravitates to extremal
  bang-bang actions as a learned property, along each action dimension, so a coupled
  steer+throttle cycle is a policy-level multi-dimensional phenomenon. A fix in the
  policy/loss attacks the cause; a per-axis actuator constraint shapes one output
  channel. (Seyde et al., "Is Bang-Bang Control All You Need?", NeurIPS 2021 —
  https://arxiv.org/abs/2111.02552)
- CAPS-family regularizers compute their smoothness loss as a single Euclidean norm
  over the entire action vector, so they regularize all axes jointly. They minimize
  the policy's temporal and spatial Lipschitz constants, which are properties of the
  whole state->action map, dimension-agnostic. This is exactly matched to a coupled
  cycle. (Mysore et al., CAPS, ICRA 2021 — https://arxiv.org/abs/2012.06644)
- Single-channel / post-hoc fixes are documented to be insufficient or dangerous:
  bolting a filter onto a trained NN policy caused overshoot and "total and
  catastrophic loss of control," and CAPS's own ablation shows a partial (single-term)
  constraint leaves residual oscillation. By analogy a one-axis rate limit leaves the
  cross-axis coupled cycle intact. (CAPS, Fig. 5 + Table II — same URL)
- Control theory agrees the target is coupled: a power-on fishtail is a genuine
  coupled lateral-longitudinal limit cycle (a Hopf bifurcation with drive torque as
  the parameter; friction-circle coupling means throttle directly modulates the
  lateral/yaw stability boundary). Production and racing remedies are both-axes by
  construction: ESC cuts torque AND brakes for yaw; integrated/combined-slip MPC and
  TUM's friction-limited racing MPC solve both axes under one friction budget. No
  source advocates a single-axis damper as a complete fix for a coupled yaw cycle.
  (Steindl et al., Nonlinear Dynamics 2019 —
  https://link.springer.com/article/10.1007/s11071-019-05081-8 ; powerslide combined-
  slip, Vehicle System Dynamics 2025 —
  https://www.tandfonline.com/doi/full/10.1080/00423114.2025.2471346 ; TUM autonomous
  motorsport — https://arxiv.org/pdf/2205.15979)

## 2. Which method: Grad-CAPS, for a car that must stay fast

The deployment risk we care about is over-smoothing into a sluggish, understeering
car. The methods differ exactly on this.

- Plain CAPS penalizes the MAGNITUDE of action change, so it cannot tell a clean fast
  ramp (a real corner) from a zigzag and punishes both. It over-smooths corners and is
  fragile to its weight: too low lets the zigzag back, too high kills responsiveness.
  (Mysore et al., ICRA 2021; over-smoothing quantified in Grad-CAPS below)
- Grad-CAPS penalizes the SECOND difference of actions (the change-in-change / jerk)
  with displacement normalization, so large steady swings are nearly free and only
  zigzag is charged. It is the only method with direct racing numbers that wins both
  axes: on Gym CarRacing with SAC, return 942 vs CAPS 932 vs vanilla 917, AND action
  fluctuation 0.08 vs CAPS 0.15 vs vanilla 0.35 (x10^-2, lower = smoother). It is also
  far less sensitive to its weight than CAPS. (Lee et al., Grad-CAPS, 2024 —
  https://arxiv.org/abs/2407.04315)
- LipsNet-L (network-level adaptive local Lipschitz) is the strong alternative if we
  want smoothness baked into the actor network with no loss-weight tuning; its
  noise-robustness margin over a plain MLP is large. Reward-penalty smoothing is a
  documented no-win (LipsNet Table 16: penalty too small does nothing, too large
  collapses return), and spectral norm over-constrains (global Lipschitz). (Song et
  al., LipsNet, ICML 2023 — https://proceedings.mlr.press/v202/song23b.html)

Recommendation: Grad-CAPS as primary. Plain CAPS is the peer-reviewed fallback if the
Grad-CAPS 3-action-window bookkeeping proves fiddly, accepting its over-smoothing risk
and tuning λ_T down. LipsNet-L is the alternative if we'd rather put smoothness in the
network than the loss.

## 3. Keep or replace the structural pieces

Keep, do not replace:
- The run11 traction cap and run10 speed-scaled reference are in-loop, baked into the
  dynamics/observation the policy trains within, so they compose with a smoothness
  regularizer rather than conflicting. In-loop action-space shaping (delta/rate-limited
  inputs, traction limits) is documented as complementary and as aiding sim-to-real.
  (CAPS Markov discussion; "Learning from Simulation, Racing in Reality" —
  https://arxiv.org/pdf/2011.13332)
- The throttle-rate limit (run12 as drafted) becomes unnecessary as the fix: Grad-CAPS
  smooths the throttle axis (and the steering axis) at the policy level, so a separate
  single-axis slew limit is redundant with it. Do not stack it as the headline change.
Drop:
- Reward smoothness penalties as a mechanism (outvoted, proven three times + LipsNet
  Table 16). Already dropped.
- Any plan to bolt a filter onto a trained policy (catastrophic per CAPS Fig. 5).

## 4. Deployment in SB3 SAC

- Subclass `SAC`, override `train()`, add `-λ_T·L_temp - λ_S·L_spat` to the actor loss
  before `actor.optimizer.step()`. Regularize the DETERMINISTIC mean action μ(s) (e.g.
  via `actor.get_action_dist_params`), not the sampled action.
- The temporal pair is free from the off-policy batch: `L_temp` uses
  `replay_data.observations` and `next_observations`. The spatial term needs one extra
  actor forward pass on `obs + N(0, σ)`.
- Grad-CAPS form: temporal term = ||(a_t - a_{t-1}) - (a_{t+1} - a_t)|| with
  displacement normalization; it needs a 3-action window, so store prev_action (or
  sample sequential transitions). (Lee et al., Eqs. 11, 15-18)
- Starting constants: λ_T = 1.0 (racing-validated), σ ≈ observation noise/scale, λ_S
  small or 0 while we are sim-only (raise λ_S only before any sim-to-real; the spatial
  term is what buys transfer robustness). (Mysore et al.; Grad-CAPS)
- Detect over-smoothing early: log mean ||a_t - a_{t-1}|| and the FFT high-frequency
  energy of the steering/throttle stream each eval. The over-smoothing signature is
  fluctuation falling while return stalls AND corners get rounded/sluggish (the
  square-wave failure: smooth straights, can't track the sharp corner). (Grad-CAPS
  Tables; CAPS FFT smoothness metric; I-RAS — https://arxiv.org/abs/2307.08230)

## 5. So run12 changes

run12 should be Grad-CAPS (actor-loss action smoothness), not the throttle-rate limit.
Keep the traction cap and speed-scaled reference. This is a bigger change than the
drafted rate limit (it touches the training loop, not just an action transform), so it
gets the same measure/probe/smoke rigor, and the watch specifically checks corner
agility for over-smoothing.

## 6. Caveats on the evidence

- The Grad-CAPS CarRacing numbers are single-map, SAC-only, and from a non-archival
  preprint (arXiv 2407.04315). Strong and directly on-point, but treat as indicative.
  CAPS (ICRA 2021), LipsNet (ICML 2023), L2C2 (IROS 2022), the bang-bang result
  (NeurIPS 2021), and the fishtail bifurcation analysis (Nonlinear Dynamics 2019) are
  peer-reviewed and carry the core argument.
- No canonical ready-made SB3-CAPS package surfaced; the subclass-`train()` pattern is
  the established community approach and matches SB3's architecture, so CC builds it.

## 7. Sources
- CAPS (action-smoothness, full-vector, filter warning) — https://arxiv.org/abs/2012.06644
- Grad-CAPS (2nd-difference, CarRacing win, anti-over-smoothing) — https://arxiv.org/abs/2407.04315
- LipsNet (reward-penalty no-win; adaptive Lipschitz) — https://proceedings.mlr.press/v202/song23b.html
- L2C2 (local Lipschitz, smoothness without expressiveness loss) — https://arxiv.org/abs/2202.07152
- Is Bang-Bang Control All You Need? (policy is the generator) — https://arxiv.org/abs/2111.02552
- Limit cycles at oversteer vehicle (fishtail = Hopf bifurcation, coupled) — https://link.springer.com/article/10.1007/s11071-019-05081-8
- Post-critical powerslide / combined-slip coupling — https://www.tandfonline.com/doi/full/10.1080/00423114.2025.2471346
- Image-based CAPS / I-RAS, racing (steer+throttle, DeepRacer champ) — https://arxiv.org/abs/2307.08230
- GT Sophy (joint policy + slip/steer penalties), Nature 2022 — https://www.nature.com/articles/s41586-021-04357-7
- Maramotti et al. (per-channel smoothness on both steer + accel) — https://arxiv.org/pdf/2207.02162
- High-Speed Autonomous Drifting with DRL (coupled steer+throttle needed action smoothing) — https://www.labsun.org/pub/RAL2020_high.pdf
