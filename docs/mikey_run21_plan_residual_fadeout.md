# run21 plan: guided residual RL with controller fade-out (training wheels)

Date: 2026-06-19
Status: DRAFT for CC + Mikey review. Chat drafts; CC builds + probes the base controller and
the blend, measures against the real code; Mikey approves the direction; CC smokes; Mike
approves the full run. Do not implement from this draft.

Goal stated up front (this is the point of the run): use a hand-coded controller as TRAINING
WHEELS that are faded out, so the END PRODUCT is a PURE learned policy that drives the track
with the controller REMOVED. Not permanent residual RL. The controller is a scaffold, not a
crutch.

Baseline: run20's peak policy (the racing-line policy that follows the line and enters T1 but
weaves and over-brakes). Pure RL on the line plateaued at our compute; the fade-out gives the
policy a through-T1 head start it never had, aiming at a clean standalone policy.

---

## 0. Where we are (the run20 watch)

run20 peak was the best driving yet: it accelerates to ~100 kph, FOLLOWS THE LINE, and two of
three runs ENTERED T1 well. Two failures remain, both execution-smoothness, not
where-to-go: (1) the left-right weave (one run lost control from it + throttle), and (2)
over-braking WHILE turning, so the rear overtakes the front (trail-brake oversteer) because it
arrives ~100 kph and brakes late/hard mid-corner. The line is right; the car just cannot
execute it smoothly, and pure RL plateaued then degraded trying to.

A pure-pursuit + speed-profile controller does exactly the smooth execution the policy lacks
(steer along the line, brake EARLY to the line's speed before turn-in). So we let the
controller drive while the policy watches and learns, then fade the controller out.

## 1. The approach: convex action-blend with a fade schedule

Applied action is a blend of the controller and the policy:

```
applied_action = beta * base_controller(obs) + (1 - beta) * policy(obs)
```

- beta = 1: the controller drives, the policy's action is masked but the policy STILL LEARNS
  (SAC is off-policy; the replay buffer fills with the controller's good through-T1
  transitions, and the critic pulls the policy toward those high-value actions, an implicit
  imitation phase).
- beta anneals 1 -> 0 over training. As beta shrinks, the policy supplies more of the full
  action and must learn to drive the track itself.
- beta = 0: the controller contributes NOTHING. The policy drives alone. This is the end state
  and the whole goal.

Why the convex blend (not an additive residual): an additive residual (base + small
correction) leaves the controller permanent, which is NOT what we want. The convex blend makes
the policy output the FULL action throughout (masked early, dominant late), so at beta=0 there
is simply no controller term left to remove, the policy already is the driver. Honest
tradeoff: this asks the policy to learn the full skill, so it is closer to pure RL in
difficulty than permanent residual RL; the controller's value here is the head start (good
exploration data + a safety net during the fade), not doing the job forever.

## 2. EVAL IS ALWAYS AT beta = 0 (the removal metric)

Training rolls out with annealing beta, but EVERY eval runs the POLICY ALONE (beta = 0,
controller off). So the eval/* cards always measure what we actually care about: can the
policy drive the track WITHOUT the controller. eval/max_arc clearing 394 at beta = 0 is the
win condition, the policy learned the track and the controller is gone. Early in training the
beta=0 eval will fail (the policy hasn't taken over yet); the signal is that beta=0 eval
improves toward clearing T1 as training proceeds.

## 3. The base controller (the training wheels)

Two hand-coded, deterministic pieces, MIT-clean (rolled ourselves):
- Pure-pursuit STEERING: aim at a lookahead point on the racing line (lookahead distance
  scaled to speed), steer by the standard pure-pursuit geometry. Smooth by construction (no
  weave).
- SPEED-PROFILE throttle/brake: a P (or PI) controller that tracks the racing line's v_target
  (the existing speed profile). Brakes EARLY to reach the apex speed before turn-in (fixes the
  over-braking-mid-corner), gas out.

THE KEY PROBE (do this before anything else): the controller ALONE must complete T1 and ideally
a lap. If the controller cannot lap, beta=1 gives the policy bad data and the whole scheme
fails, so fix the controller (lookahead distance, speed-controller gains, the line's speed
profile) until it laps cleanly. A controller that laps is the foundation the fade stands on.

## 4. The fade schedule + reward (unchanged) + warm start

- Schedule: beta = 1 for a short warmup (buffer fills with clean controller laps), then a
  linear anneal to 0 over the middle of training, then beta = 0 held for the final stretch (the
  policy must drive the whole track solo while we still have steps to refine it). CC proposes
  the exact breakpoints; the principle is fade slowly enough that the policy is keeping up at
  each beta before it drops further. (Reserve: performance-gated fade, only drop beta when the
  beta=0 eval is healthy.)
- Reward: UNCHANGED from run20 (follow the racing line, match its speed, gentle slip backstop
  W_SLIP=0.05/BETA_DEAD=9, anti-timid match term on the line's v_target). The fade is a
  training/action-space scheme, not a reward change. One coherent change.
- Warm from run20's PEAK best_model (run20_peak_best_model.zip): it already follows the line
  and enters T1, so the policy starts decent and the controller cleans up its weave/over-brake
  by example during the high-beta phase. NEVER run19. Fresh is the fallback if warm fights the
  blend.

## 5. Validation plan

1. Controller-alone probe (THE gate): the pure-pursuit + speed-profile controller drives T1
   and a lap on its own, no spin, no wall, brakes early to the apex speed. Fix the controller
   until it laps before training.
2. Blend probe: applied_action = beta*base + (1-beta)*policy computes correctly and stays
   bounded at beta = 1, 0.5, 0; eval forces beta = 0.
3. Smoke ~7k: the fade schedule is live (beta annealing in rollouts, beta=0 in eval), no NaN,
   the 20 eval/* cards intact, mean_speed sane, warm-load clean. Add an eval card or log line
   for the current beta so we can see the fade.
4. Full run on Mike's approval (length TBD; the fade needs enough steps for the policy to take
   over after beta hits 0). Wrapper (buffer fix) + temp logger armed.

## 6. G14 watch question (the verdict)

After maturity, watch the beta = 0 policy (controller OFF) 3+ times:
1. Does the policy, ON ITS OWN, follow the line through T1, brake early (no mid-corner
   over-brake), and NOT weave?
2. Does it get THROUGH T1 (past 394) and onto the lap, with the controller removed?
3. Does it hold the line through the downstream corners it now reaches, solo?
If yes, the car learned the track and the training wheels are off. That is the project goal.

## 7. Reserve ladder (and the honest fallback)

- The beta=0 hand-off fails (the policy cannot go fully solo at our compute) -> hold beta at a
  small nonzero FLOOR (e.g. 0.1-0.2). This degrades gracefully to light permanent residual RL:
  a thin controller stays as a safety net, the policy does almost everything. We still get a
  car that laps; we just did not fully remove the scaffold. This guarantees a result either
  way.
- The controller itself cannot lap -> fix the controller first (it is the foundation).
- The fade is unstable -> DISTILLATION alternative: let controller+policy generate good laps,
  train a FRESH pure policy to imitate them (behavioral cloning), then RL-fine-tune. Same end
  (a pure policy), different route.
- Policy degrades after beta=0 like run20 did -> lower LR / entropy tuning for stability in the
  solo phase.

## 8. The honest risk (say it plainly to Mikey)

This run AIMS to remove the controller, which is the ambitious, faithful-to-the-spirit goal:
the AI learns to drive the track itself. But removing the controller means the policy must
learn the genuinely hard part (pushing the car to its grip limit through T1) that pure RL
plateaued on. The fade gives it a much better starting point (it drives T1 correctly the whole
time it is learning, vs run20 which only ever crashed there), so it may break through. If the
final solo hand-off is too hard at our one-machine compute, the beta-floor fallback keeps a
thin controller and we still get a lapping car. So: best case, a pure learned driver through
T1; worst case, a near-pure driver with a thin safety net. Either way we move past the wall.

## 9. Constraints reminder for CC

- The GOAL is controller removal (beta -> 0). Build for the fade from the start; eval at beta=0
  always, so the cards measure the standalone policy.
- MIT-clean: the controller is rolled ourselves (pure-pursuit + a P/PI speed controller), no
  GPL/LGPL deps.
- Ground it in the run20 watch: the car FOLLOWS THE LINE and ENTERS T1, the failures are weave
  + over-braking (execution smoothness). The controller supplies smooth execution; the policy
  learns to reproduce it solo.
- This is NOT the run10-15 anti-pattern: the controller does not OVERRIDE the policy (no filter
  fighting the learner); it is BLENDED and FADED OUT, leaving the policy as the driver.
- Reward unchanged from run20. Warm from run20 peak. One coherent change: the guided fade-out
  training scheme.
- Clock: BeamNG license expires 2026-08-06. This is the run aimed at finally clearing T1 and
  getting onto the lap. The beta-floor fallback guarantees a lapping car if full removal is too
  hard at our compute.
- Pattern: CC builds + probes the controller (must lap), wires the blend + fade + beta=0 eval,
  proposes the schedule + warm/fresh, Mikey approves the direction, Mike approves the full run.
