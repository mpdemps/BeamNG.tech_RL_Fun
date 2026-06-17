# Deep research: stop scripting, make the agent LEARN to corner (2026-06-15)

Question: we've been bolting hand-tuned action constraints (TC, ESC, steering rate +
authority, now a curvature speed cap) onto a SAC policy to stop it spinning, which is
building a scripted controller, not RL. The agent still can't brake for T1 (carries 33
m/s into an R55 needing 22). How does the autonomous-racing-RL literature get braking
and cornering LEARNED instead of scripted?

Method: five parallel literature searches (reward, observation, curriculum, action
constraints, method/residual-RL), cross-checked. Load-bearing claims are peer-reviewed
(Nature, RA-L, SIGGRAPH, ICRA, ICML, JFR).

## Bottom line

The constraint-stacking is the documented anti-pattern, and Mike's instinct is correct.
In every serious racing-RL system braking and corner-entry speed are EMERGENT from the
reward, never scripted; the agent only ever outputs steering + throttle/brake. Our
failure to learn braking traces to three mis-designed pieces, in order of importance:

1. REWARD: progress x speed is the documented trap; it has to be referenced to a racing
   line's speed profile (or carry speed-scaled penalties + a slip penalty + a high
   discount) so slowing-for-corners pays.
2. OBSERVATION: the policy must SEE upcoming curvature far enough ahead, scaled to
   speed, plus its own slip angle, to anticipate braking. Our fixed near lookahead can't.
3. CURRICULUM: a fixed start line starves the agent of experience at every corner past
   T1; random start-state distribution (spawn variation) is how all corners get learned.

The fix is to change those three and RETIRE the constraint stack, organized around the
racing line, which is the planned feature and the literature's standard. A residual-RL-
on-a-base-controller path is the pragmatic, scale-friendly alternative.

## 1. Braking is emergent everywhere; our constraints are the anti-pattern

- GT Sophy's reward is four terms (course progress, speed-scaled off-course penalty,
  speed-scaled wall penalty, tire-slip penalty); braking/lines are never coded, they
  emerge. Sony explicitly calls the tire-slip penalty a learning ACCELERANT on top of an
  already-sufficient progress+boundary reward, i.e. hand-coded traction control is
  unnecessary when the reward carries a slip term. (Wurman et al., Nature 2022;
  https://www.nature.com/articles/s41586-021-04357-7 ; Sony AI blog,
  https://ai.sony/blog/training-the-worlds-fastest-gran-turismo-racer)
- CAPS states the principle directly: a behavior that is a function of the actions
  (smoothness, here also "don't over-apply") should be regularized in the loss, not
  forced via reward-hacking or external filters; and bolting a hand-tuned filter onto a
  trained net caused "total and catastrophic loss of control." Proliferating constraints
  to force a behavior the policy should learn is the smell that the reward/obs is wrong.
  (Mysore et al., ICRA 2021; https://arxiv.org/abs/2012.06644)
- Safe-RL: a filter/shield that always intervenes makes the agent over-conservative and
  it never learns the skill the filter handles; action projection causes "action
  aliasing" that destroys policy-gradient information. That is exactly our TC/ESC/caps
  doing the cornering for the policy so it never learns to. (Alshiekh et al., AAAI 2018;
  Markgraf et al. 2025, https://arxiv.org/abs/2509.12833)
- Legitimate exception: ONE physical rate limit is fine and standard, GT Sophy clamps
  delta-steering to +/-3 deg per 10 Hz step. One such limit, not a stack of five.

## 2. Why our agent never learns to brake

REWARD. The F1TENTH controlled study is our exact failure, reproduced: a baseline reward
of velocity x cos(heading error) - cross-track drives near-max speed everywhere, slip
angle 15-30deg (drifting/exploiting the sim), and lap completion collapses from 100% at
4 m/s to 0% at 8 m/s. Their fix (Trajectory-Aided Learning) rewards matching a
min-curvature racing line's SPEED and STEERING (r = 1 - |v_agent - v_line| - |delta_agent
- delta_line|); the agent then "slows in the corners and speeds up on the straights,"
slip drops to ~10deg, completion recovers to 75%+. Our progress x heading-alignment +
speed reward is the broken baseline. (Evans et al. 2023, https://arxiv.org/abs/2306.07003 ;
Evans et al. 2021, https://arxiv.org/abs/2103.10098)
Also: a low/short-horizon-discounted progress reward actively SUPPRESSES braking (Fuchs
et al.: "without [a kinetic-energy wall penalty] the policies did not brake and simply
grinded along the walls"; they used gamma ~0.98-0.982). The right shaping is GT Sophy's:
make mistakes cost more at speed (speed-scaled penalties), which converts "go fast" into
"go fast within grip." (Fuchs et al., RA-L 2021, https://arxiv.org/abs/2008.07971)

OBSERVATION. To brake for a corner it hasn't reached, the policy must see the upcoming
curvature, far enough ahead, scaled to speed. GT Sophy: 60 course points, speed-scaled,
~6 s ahead, plus per-tire slip angle and load. Its predecessor (Fuchs et al.): 10
inverse-radius curvature taps over 1.0-2.8 s, speed-scaled, and the paper directly
credits this for braking ~100 m before the hairpin. The TAL baseline (heading error +
cross-track, no anticipatory speed preview) never learns the corner speed profile. Our
6-point fixed-arc lookahead with heading-error from a fixed 10 m point, and no slip-angle
in the obs, is on the wrong side of this line. (Wurman 2022; Fuchs 2021; Evans 2023)

CURRICULUM. The chicken-egg (never reaches T2+ so never learns it) is the canonical
fixed-start failure. DeepMimic: with a fixed start the backflip is never learned ("a
small backwards hop"); Reference State Initialization (sampling episode starts along the
motion) fixes it. Theory: the start-state distribution bounds what can be learned
(Kakade & Langford). And racing does exactly this, GT Sophy randomizes the start
position across the whole track (on- and off-track, speed 0-104 km/h), ~200 cars spread
around the circuit; F1TENTH and DeepRacer spawn at random waypoints. (Peng et al.,
SIGGRAPH 2018, https://arxiv.org/abs/1804.02717 ; Kakade & Langford ICML 2002 ; Wurman
2022 ; Florensa et al. CoRL 2017, https://arxiv.org/abs/1707.05300)

## 3. The fix (keeps it RL), organized around the racing line

The racing line was always our planned next feature; the literature makes it the
organizing piece of the whole fix:

1. Compute the racing line OFFLINE: minimum-curvature path + a minimum-time SPEED PROFILE
   (the brake points and corner speeds). This is standard and open-source (TUM
   global_racetrajectory_optimization; Heilmeier VSD 2019). It directly encodes the
   corner-entry speeds the car cannot currently learn. Prerequisite: extract track
   edges/width (read-only, doable anytime).
2. REWARD: reference the racing line, TAL-style, reward matching the line's speed and
   steering (slow-in/fast-out falls out). Add a slip-angle penalty and keep mistakes
   speed-scaled. Raise gamma toward 0.98+ so braking isn't discounted away.
3. OBSERVATION: add a speed-scaled curvature preview (multi-point, ~several seconds
   ahead) and the car's slip angle. So it can see the corner coming and feel the rear.
4. CURRICULUM: spawn variation / RSI, start episodes distributed around the whole track
   so every corner is practiced.
5. RETIRE the scripted stack (TC, ESC, steering authority, the speed cap). Keep at most
   ONE steering-rate limit (GT Sophy style). Let the reward+obs+curriculum produce the
   smooth, braking, gripping behavior; the slip penalty replaces the TC/ESC's job.

## 4. The pragmatic alternative: residual RL on a base controller

The honest scale caveat: GT Sophy (the pure-RL success) needed 1000+ PS4s and ~45,000
driving-hours over ~10 days. We are on one CPU mini at ~10 fps (a 500k run is ~12 h).
Expecting pure RL to discover GT-Sophy-grade cornering at our compute is a stretch.

Two scale-friendly options the literature strongly supports:
- The racing-line-referenced reward (TAL) is itself the scale fix for pure RL, it gives a
  dense, correct objective so the agent doesn't have to discover the line from scratch,
  and it was demonstrated on small F1TENTH compute, not GT-Sophy scale. This keeps the
  project's RL spirit and is the recommended primary path.
- Residual RL: a fixed pure-pursuit base controller TRACKS the offline racing line
  (so it brakes for corners by construction), and RL learns only a small correction on
  top. This is the principled version of "a controller handles the basics, RL learns the
  rest", the legitimate form of what we stumbled into accidentally. Proven on racing:
  RL-residual-on-pure-pursuit cut lap times 4.55%, and RLPP transferred zero-shot to real
  hardware with an 8x smaller sim-to-real gap than from-scratch RL. Far less experience
  needed. This is the pragmatic fallback if the pure-RL TAL path struggles at our scale.
  (Johannink et al., ICRA 2019, https://arxiv.org/abs/1812.03201 ; Trumpp et al. 2023,
  https://arxiv.org/abs/2302.07035 ; RLPP 2025, https://arxiv.org/abs/2501.17311)

Note: the full-scale "drives the whole track at 270 km/h" results (TUM/IAC) are pure
optimization + MPC, no RL, the racing line + speed profile computed offline and tracked.
That confirms the brake points are normally SOLVED offline, not learned.

## 5. Recommendation

Stop adding constraints. Pivot to the racing line as the foundation and fix the three
learning pieces:
- Primary (keeps it pure RL, scale-appropriate): offline racing line -> TAL-style
  racing-line-referenced reward + slip penalty + higher gamma -> speed-scaled curvature
  preview + slip-angle in the observation -> spawn/RSI curriculum -> retire the
  constraint stack (keep one steering-rate limit).
- Fallback (most robust, fewest runs): residual RL, pure-pursuit tracking the offline
  racing line + an RL residual.

Either way the racing line is the next thing to build, and the scripted constraints come
out. The last six runs were not wasted, they broke T1, validated the infra and the
self-healing wrapper, and mapped the failure modes, but the durable path is to make the
agent LEARN to corner, which means reward + observation + curriculum, not more scripting.

## 6. Sources
- GT Sophy, Nature 2022 — https://www.nature.com/articles/s41586-021-04357-7
- Sony AI GT Sophy reward/start-state blog — https://ai.sony/blog/training-the-worlds-fastest-gran-turismo-racer
- Super-Human GTS (Fuchs et al.), RA-L 2021 — https://arxiv.org/abs/2008.07971
- Trajectory-Aided Learning (Evans et al.), 2023 — https://arxiv.org/abs/2306.07003
- Reward Signal Design for Autonomous Racing (Evans et al.), 2021 — https://arxiv.org/abs/2103.10098
- DeepMimic / RSI (Peng et al.), SIGGRAPH 2018 — https://arxiv.org/abs/1804.02717
- Approximately Optimal Approximate RL (Kakade & Langford), ICML 2002
- Reverse Curriculum Generation (Florensa et al.), CoRL 2017 — https://arxiv.org/abs/1707.05300
- CAPS (Mysore et al.), ICRA 2021 — https://arxiv.org/abs/2012.06644
- Safe RL via Shielding (Alshiekh et al.), AAAI 2018 — https://arxiv.org/abs/1708.08611
- Action Projection / aliasing (Markgraf et al.), 2025 — https://arxiv.org/abs/2509.12833
- Residual RL for Robot Control (Johannink et al.), ICRA 2019 — https://arxiv.org/abs/1812.03201
- Residual Policy Learning for Vehicle Control (Trumpp et al.), 2023 — https://arxiv.org/abs/2302.07035
- RLPP zero-shot real racing (ForzaETH), 2025 — https://arxiv.org/abs/2501.17311
- TUM Autonomous Motorsport (Betz et al.), JFR 2023 — https://onlinelibrary.wiley.com/doi/full/10.1002/rob.22153
- TUM global_racetrajectory_optimization — https://github.com/TUMFTM/global_racetrajectory_optimization
- High-speed Drifting RL (slip-angle in reward) (Cai et al.), RA-L 2020 — https://arxiv.org/abs/2001.01377
