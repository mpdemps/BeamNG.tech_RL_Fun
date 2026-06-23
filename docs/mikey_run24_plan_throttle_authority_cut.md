# run24 plan: stop the corner-exit spin — cut the residual's throttle authority

Date: 2026-06-21
Status: DRAFT for CC + Mikey review. Chat drafts; CC builds + probes the per-channel residual
bound against the real code; Mikey approves the direction; CC smokes; Mike approves the full
run. Do not implement from this draft.

Baseline: run23 (the controller-led additive residual + grip-awareness bundle: loss-of-control
termination, grip obs[18], fixed off_track). run24 keeps that whole architecture and changes ONE
thing — how much THROTTLE the residual is allowed to add — to kill the corner-exit spin that now
dominates, and to push for the first completed lap.

In plain words (for Mikey): the autopilot drives the racing line fine. The AI rides on top and is
allowed to nudge the steering and the gas a little. Right now, in the corners, the AI keeps
nudging the GAS up — and on our 748hp rear-wheel-drive car, a little extra gas mid-corner spins
the back end out, exactly the old enemy. So we're going to let the AI keep its full nudge on the
steering and keep its full ability to LIFT off the gas (to save a slide), but only let it add a
tiny bit of EXTRA gas. It can lift all it wants; it just can't floor it out of the corner anymore.
Nothing else changes, so we shouldn't lose the speed it just gained.

---

## 0. run23 results (what we saw, and what it proves)

From Mike's live TensorBoard (logs are gitignored, numbers recorded here):

- Two series: `mikey_run23/sac_1` (main, ~390k steps) and `mikey_run23_resume/sac_1` (a
  warm-restart, ~100-150k, still climbing out of the usual resume dip).
- HUGE jump over run20 (run20 rollout `ep_rew_mean` ~218): run23 rollout `ep_rew_mean` ~687
  (smoothed ~703), `ep_len_mean` ~158-170.
- `eval/mean_reward` ~6,800-7,776; `eval/mean_speed` ~28 m/s; `eval/over_speed_frac` ~0.02 (low —
  the car is NOT generally speeding past v_target).
- `eval/max_arc` ~2,557 (best ~3,467). The car now gets THOUSANDS of meters down the track, far
  past the old Turn-1 wall at ~394 m. **Cornering paralysis is solved.**
- Termination breakdown (eval): `term_stuck` = 0 (the old crawl-loophole failure is gone),
  `term_backward` = 0, `term_flip` ~0.02 (negligible), but **`term_loss_of_control` ~0.33-0.50**
  and **`term_off_track` ~0.35-0.57** now dominate, and **`term_lap` = 0** (no full lap yet).

What it proves:
1. run23's grip-awareness bundle worked. The loss-of-control terminator fires on real spins
   (the spin probe showed clean cornering peaks 99 deg/s, well under the 150 deg/s threshold), so
   `term_loss_of_control` ~0.33-0.50 is honest spins ending fast, not false fires — the buffer is
   no longer corrupted by non-terminating donuts (run22's bug).
2. The hybrid is fast and goes far (28 m/s, ~3.4 km best arc). The line-following + speed is
   genuinely good now. We must NOT regress this.
3. The remaining wall is two failure modes in the corners it now reaches: it **loses control
   (spins)** or **runs off-track (wide)**, and it has **not closed a lap**. Everything in run24
   targets exactly those two, while protecting the arc/speed gains.

## 1. The diagnosis (best-supported hypothesis)

The dominant remaining failure is **throttle indiscipline in the corners, expressed through the
residual's throttle channel** — the project's recurring villain (748hp RWD power oversteer), back
in a new costume.

The chain, grounded in the code and the run22/23 watch:

- The architecture is `applied = clip(controller + clip(residual, ±0.12), -1, 1)`
  (`envs/residual_hybrid.py`). The residual bound is **symmetric ±0.12 on BOTH channels**, and
  `action[1]` is throttle where **positive = throttle, negative = brake/lift**
  (`envs/base_controller.py` docstring). So the residual can add **up to +0.12 throttle** on top
  of the controller, anywhere — including mid-corner and on the corner exit.
- The controller tracks the grip line at `speed_factor=0.95` (a 5% grip margin,
  `base_controller.py`). That margin is a STEADY-STATE lateral-grip margin. On a RWD car the
  friction circle means any added longitudinal (throttle) force eats lateral grip: at the apex
  the car is already near `A_LAT_MAX` (12.0 m/s², ~1.22 g; `envs/speed_profile.py`), so a +0.12
  throttle residual pushes it past the circle and the rear lets go → spin (**loss_of_control**) or
  the car pushes wide off the line → **off_track** (>8 m from road center).
- The run22 watch saw this directly: *"the RESIDUAL over-throttles on the T1 EXIT and spins."*
  run23 made that spin TERMINATE (so the penalty signal now arrives) and gave the policy eyes for
  running wide (grip obs[18]) — but at ~390k steps the residual **has not yet LEARNED to lift on
  the exit**. The fix run23 itself pre-registered for exactly this outcome is to **reduce the
  residual's throttle authority** (run23 plan §RESERVE and Reserve ladder #1).
- `eval/over_speed_frac` ~0.02 does not exonerate this: over-speed is averaged over a whole
  episode, and corners are a small fraction of steps. A brief +throttle spike on ONE corner exit
  is a tiny fraction of steps yet ends the episode. The spike, not the average, is the killer.

Why throttle (not the other candidates) is the lead lever:

- **Steering authority/rate at speed** is a plausible *secondary* contributor (a ±0.12 steering
  residual at 28 m/s could snap the rear), but the controller's pure-pursuit steer is smooth and
  the residual is bounded; the watch attributed the spin to throttle, and steering tapers risk
  understeer at entry. → reserve, not primary.
- **Racing-line speed profile too hot at apexes** (lower `A_LAT_MAX`) would add margin but
  **regresses the speed/arc gains** we just earned, and the friction-circle story says the apex
  speed is fine for a neutral car — it's the added throttle that breaks it. → reserve, not
  primary.
- **Tightening off_track (<8 m)** would make terminations fire EARLIER and cut episodes shorter,
  punishing recoverable wide moments and starving the policy of recovery experience. → do NOT do
  this.

So: cut the residual's ability to ADD throttle, keep its ability to LIFT and to steer. This is a
residual-bound change, not throttle scripting and not a reward change, so it cannot re-introduce
run19's corner-avoidance, and it leaves the controller (which drives the line at 0.95 v_target)
untouched — the arc/speed gains are protected by construction.

## 2. The change (ONE coherent change: asymmetric per-channel residual bound)

Replace the single symmetric `delta` in `envs/residual_hybrid.py` with **per-channel, asymmetric
low/high bounds**:

- **Steering residual: unchanged, ±0.12.** The residual keeps full authority to trim the line.
- **Throttle residual: asymmetric.** Lift/brake side stays at full **−0.12** (the policy can lift
  ALL it wants to save a slide or settle the corner). Add-throttle side is cut hard to
  **≈ +0.04 to +0.05** (CC proposes the exact value from the residual_abs telemetry and the spin
  probe). The policy can no longer floor the controller out of a corner; it can only feather a
  touch of extra throttle where grip clearly allows.

Concretely, the wrapper's clip becomes per-channel, e.g.:

```
low  = np.array([-0.12, -0.12], dtype=np.float32)   # [steer, throttle] lift side full
high = np.array([+0.12, +0.05], dtype=np.float32)   # throttle ADD capped tight
clipped = np.clip(residual, low, high)
```

This is the only behavioral change. It directly attacks **loss_of_control** (no more
over-throttle to break the rear loose) and **off_track** (less corner-exit push-wide), and it
explicitly preserves the lift authority the policy needs to learn the save — which is the very
behavior run23 was trying to unblock.

CC should expose the bounds as flags (e.g. `--residual-steer-delta 0.12`,
`--residual-throttle-up 0.05`, `--residual-throttle-down 0.12`) so this stays one tunable knob,
and log per-channel mean |residual| (split `residual_abs` into steer/throttle) so we can SEE the
throttle-up cap binding.

### What we explicitly do NOT change (protect run23's gains)

- Reward UNCHANGED (W_PROG/W_OVER/W_SLIP/W_MATCH, BETA_SLIP_DEAD=9, the racing-line reference).
  No reward edits near corners — that is how run19 backfired.
- Speed profile UNCHANGED (`A_LAT_MAX=12.0`, `V_MAX=33`). We keep the apex speeds; we remove the
  over-throttle, not the target.
- Loss-of-control terminator UNCHANGED (150 deg/s × 8 ticks). It is working (honest spins, no
  false fires); leave it. Cutting throttle authority should make it fire LESS, which is the
  whole point.
- off_track UNCHANGED at 8 m. Do not tighten.
- Grip obs[18] UNCHANGED (19-dim obs). It is the policy's eyes for running wide; warm-continuing
  lets it keep learning to act on it.
- steer_rate 0.5, random-spawn curriculum, plain SAC, gamma 0.99, the 20 eval/* cards — all kept.

## 3. Warm vs fresh — WARM from run23's best_model (decided)

Warm. The obs shape is UNCHANGED (still 19 dims), the reward is UNCHANGED, and only a downstream
action CLAMP tightens — so run23's policy transfers cleanly. run23's best_model already follows
the line and reaches ~3,467 m at de-flattered eval (run23 fixed the termination that used to
inflate eval, so its best_model is a genuine line-follower, not a non-terminating-donut artifact).
We want to KEEP that competence and just retrain the throttle behavior under the new cap.

- Warm-start: `checkpoints/mikey_run23/best_model/best_model.zip` (the MAIN series, ~390k —
  NOT the `mikey_run23_resume` series, which is still climbing out of its resume dip).
- Use `--learning-rate 3e-4` and a modest `--learning-starts` (e.g. 5000) for a gentle
  continuation, as the run19→20 warm restarts did.
- Fallback: FRESH if the warm policy fights the new throttle cap in the smoke (e.g. it pins the
  throttle-up bound everywhere and can't settle). Fresh still laps from step 1 (controller-led),
  so it is a safe fallback; warm is the gain-preserving first choice.

## 4. Smoke test + full run

- **Probe (no training):** the per-channel clip computes correctly — steering residual reaches
  ±0.12, throttle residual reaches −0.12 but is capped at +~0.05; with residual forced to 0 the
  applied action equals the controller (so it still laps); the loss-of-control terminator still
  fires on a forced spin and NOT in clean cornering.
- **Smoke ~7k (warm):** warm-load is clean (no shape error, no random-action warmup), the hybrid
  laps controller-led, the throttle-up cap binds (per-channel residual telemetry shows it),
  spins still TERMINATE quickly, no NaN, the 20 eval/* cards intact.
- **Full run: 500k** on Mike's approval (eval-freq 10k, checkpoint-freq 5k). `CheckpointCallback`
  ARMED — save checkpoints every 5k, not just the final model (CLAUDE.md: first run lost its peak
  to policy collapse; never again).

## 5. Success criteria (the verdict)

Headline (the Phase-1 goal): **`term_lap` > 0 for the first time** — the hybrid closes at least
one full lap.

Failure-mode targets (vs run23):
- `eval/term_loss_of_control`: ~0.33-0.50 → **< ~0.15** (cut by at least half).
- `eval/term_off_track`: ~0.35-0.57 → **< ~0.25**.

Gains we must NOT regress (guard against over-correcting into timidity):
- `eval/max_arc`: hold or beat run23 — **mean ≥ ~2,557, best ≥ ~3,467**.
- `eval/mean_speed`: hold **~26-28 m/s**. A small dip from less over-throttle is fine; a collapse
  toward the old crawl (run17's ~2.4 m/s) means the throttle cap is too tight — back it off.
- `eval/over_speed_frac`: stays low (~0.02 or lower).

Diagnostic: the new per-channel `residual_abs` shows the throttle-up residual pinned at the cap
(the policy WANTS more throttle but can't have it) — confirms the bound is doing the work.

## 6. G14 watch question (after maturity, 3+ deterministic runs)

1. Through the corners it reaches, does it now **hold the line and stay on track** instead of
   spinning out / running wide on the exit?
2. When it DOES start to slide, does the residual **LIFT to save it** (the behavior we freed up)
   rather than over-throttling into a spin?
3. Does it **complete a lap** — past the start seam, `term_lap` firing — and does it carry the
   line's speed (not a re-timid crawl)?

## 7. Reserve ladder (pre-registered; do NOT do now)

- Still spinning after the throttle cut → modestly lower `A_LAT_MAX` in the speed profile (more
  conservative apex speeds, bigger friction-circle margin). **FLAG: regresses speed/arc — only if
  the throttle cut alone is insufficient.**
- Running wide (off_track) but NOT spinning → a GENTLE edge-proximity shaping cost on grip
  obs[18] (small, to pull the car off the edge before the 8 m cliff). **FLAG: a reward change
  near corners — watch closely for run19-style corner-avoidance.**
- Residual STEERING causing snap oversteer at speed → speed-scaled steering-rate taper on the
  applied action (the `steer_rate_hi` knob already sketched in `beamng_env.py`). **FLAG: risks
  understeer at corner entry.**
- Throttle-up cap so tight the car goes timid → raise it (e.g. +0.05 → +0.08) — the symmetric
  inverse of this run.
- Structural (Phase-1 completion) → sectioned authority: more residual room on straights, tighter
  in corners; or simply a longer warm continuation past 500k if the lap is close.

## 8. Constraints reminder for CC

- ONE coherent change: asymmetric per-channel residual bound (steer ±0.12; throttle −0.12 / +~0.05).
  NOT throttle scripting (the controller and reward are untouched), NOT a reward change (so it
  cannot re-create run19's corner-avoidance), NOT a speed-profile change (so the arc/speed gains
  are protected).
- Keep the whole run23 architecture (controller-led residual, grip obs[18], loss-of-control term,
  8 m off_track, random-spawn, plain SAC, gamma 0.99, steer_rate 0.5). Warm from run23's MAIN
  best_model; fresh only as the smoke fallback.
- Expose the bounds as flags and split `residual_abs` per channel so the cap is observable.
- `CheckpointCallback` armed (checkpoints every 5k, not just final). TensorBoard on **port 8765**.
- MIT-clean (the controller and wrapper are ours; no new deps). `tech.key` and `.env` must NEVER
  be committed — if either shows in `git status`, STOP and tell Mike.
- **Guardrails for launch: commit the plan doc AND the code change BEFORE launching. Smoke, then
  STOP for review — do not auto-run the 500k (it burns 20+ min of real GPU; CLAUDE.md). No
  destructive git ops. Co-author the commit with Mikey.**
- Pattern: CC wires the per-channel bound + per-channel residual logging, probes it against the
  real code, proposes the exact +throttle cap from the residual telemetry / spin probe, Mikey
  approves the direction, CC smokes, Mike approves the full run.
- After the run: append a `run24` entry to `RUNS.md` (note: RUNS.md is currently behind — its last
  entry is run19; runs 20-23 are owed a backfill from the live TB numbers).
