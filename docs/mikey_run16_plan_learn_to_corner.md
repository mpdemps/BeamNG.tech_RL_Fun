# run16 plan: learn to corner — the paradigm reset (reward + observation, drop the scripts)

Date: 2026-06-16
Status: DRAFT for Chat + Mike review. CC drafts; Chat + Mike poke holes; CC measures +
refines against the real code, proposes final weights/constants; Mike approves; CC
smokes; Mike approves the full run. **Do not implement from this draft.**

Basis: `docs/research_learn_to_corner_2026-06-15.md` (the deep-research writeup) + CC's
own run15 T1/T2 diagnosis, which independently found the root cause *before* the
literature framing: the car enters T1 at ~33 m/s into R55 (a ~2g demand it cannot meet),
washes wide, then snaps. It does not brake — because nothing in the reward or observation
ever tells it to.

---

## 0. Why this is ONE coherent change, not the usual one-knob run

Runs 10–15 layered six scripted action/loss constraints (TC, ESC, Grad-CAPS, steering
rate, speed-scaled rate, planned authority cap). Each shaved the *symptom* (the spin) and
the wall did not move: max_arc ~460 m across run13/14/15. That is the documented
anti-pattern — constraints fight the policy instead of teaching it. The failure is a
**reward + observation + curriculum** problem: our reward (`raw_progress·alignment +
SPEED_WEIGHT·speed·alignment`) pays for speed *everywhere* and never for slowing, and our
obs has no speed-relative curvature signal, so the policy is structurally unable to learn
to brake.

The fix is a bundle whose pieces are inseparable: a braking-aware speed-target reward
needs the obs to expose upcoming curvature, the obs change forces a fresh run anyway, and
two of the dropped constraints (TC, ESC) actively *corrupt* the braking signal by hiding
the consequence of entering too hot. So run16 is **one new baseline**. The
one-behavioral-change-per-run rule **resumes after run16** (run17, run18 below).

---

## 1. REWARD — curvature-aware, braking-distance-aware speed target

### 1a. The speed-target profile (computed ONCE, offline, over the 985-pt centerline)
Standard forward–backward racing-line speed pass, on the **centerline** (no width/racing
line yet):

1. **Curvature** at each centerline point i: `kappa[i] = |dtheta|/ds` from the smoothed
   tangent (`_smoothed_forward_yaw`) over a small arc window; `R[i] = 1/kappa[i]`
   (clip R to a large value on straights).
2. **Pointwise corner limit:** `v_curve[i] = min(V_MAX, sqrt(A_LAT_MAX * R[i]))`.
3. **Backward pass (braking), upstream i = N-1 .. 0:**
   `v[i] = min(v_curve[i], sqrt(v[i+1]^2 + 2*A_BRAKE*ds[i]))`
   — this makes the target ramp **down on the approach**, over the braking distance,
   instead of a cliff at the apex.
4. **Forward pass (accel), i = 0 .. N-1:**
   `v[i] = min(v[i], sqrt(v[i-1]^2 + 2*A_ACCEL*ds[i]))`
   — fast-out but feasible (no instant-acceleration demand leaving a corner).

Result: a precomputed `V_TARGET[i]` array aligned with `_cum_arc`. At runtime, `v_target`
at the car's arc is interpolated (closed loop, wraps).

**Constants (first guess, from the car's measured grip; tune in the smoke):**
- `A_LAT_MAX = 12.0 m/s^2` (~1.22 g). The measured grip ceiling is ~1.6 g (run13 EP2 held
  0.97 steer at 27.5 m/s while *tracking*; the spins were beyond that). Target a
  comfortably-makeable ~1.2 g so the profile is achievable without sliding; raise later
  for lap time (reserved knob).
- `A_BRAKE = 9.0 m/s^2` (~0.9 g) — conservative so braking into corners is comfortable.
- `A_ACCEL = 6.0 m/s^2` — traction-limited corner exit.
- `V_MAX = 33.0 m/s` (the straight-line speed the car actually reaches; caps the profile).

Sanity (validation §5): T1 R55 -> sqrt(12·55)=25.7 m/s; T5/T6 R34 -> 20.2 m/s; T8 R68 ->
28.6 m/s; T7 R133 -> 40 -> capped 33. Braking 33->26 needs (33^2-26^2)/(2·9)=23 m, so the
T1 target should start falling by ~arc 315 (apex 338), not at the apex.

### 1b. The reward (replaces `_compute_reward`'s speed/spin terms)
Keep the run5 heading kill-switch (zero reward when nose >~90° off down-track) and the
progress core. New step reward (before terminal/checkpoint bonuses):

```
r = W_PROG  * raw_progress * gated_alignment              # cover track, forward only (keep)
    - W_OVER * max(0, v - v_target)^2                     # THE BRAKE SIGNAL: faster than the
                                                          #   braking-aware target -> penalized
    - W_SLIP * max(0, |beta| - BETA_DEAD)                 # slip-angle penalty (replaces ESC +
                                                          #   the wheelspin spin_penalty)
```

- The **over-speed penalty is the new causal signal.** Because the backward pass makes
  `v_target` fall *before* a corner, exceeding it on the approach is exactly "you should
  be braking now," and the squared form gives a smooth gradient toward braking (gentle
  near target, sharp when 10 m/s hot). Calibrate `W_OVER` in the smoke so the
  reward-maximizing speed is ≈ `v_target` (the marginal over-speed penalty must exceed the
  marginal progress gain from going faster — progress ∝ speed, so this needs checking,
  not guessing). First guess `W_OVER ≈ 0.05`.
- **No under-speed penalty.** The progress term already rewards carrying speed where
  `v_target` is high (fast-out emerges); we don't punish caution, we only punish
  over-speed. (Reserve: a tiny `+W_MATCH·min(v, v_target)` if it crawls.)
- **Slip-angle penalty** replaces the wheelspin `spin_penalty` and the ESC throttle-cut:
  `BETA_DEAD = 9°` (clean tracking sits at β p90 ~5°; slides are 17–46°). First guess
  `W_SLIP ≈ 0.05` per degree (β=20° -> ~0.55/step). This is a *reward* signal that lets
  the policy learn not to slide, instead of a hard throttle cut that hides the slide.
- `W_PROG = 1.0` (raw_progress is meters/step·alignment).

All weights are first-guess and **calibrated in the smoke** (§5) against the principle
"reward-optimal speed ≈ v_target, sustained slide is clearly net-negative."

---

## 2. OBSERVATION — expose what braking needs (15 -> 18, fresh)

Current 15: `[0]` speed, `[1]` heading_err, `[2]` center_off, `[3..14]` 6×(dist,bearing)
lookahead. Add three:

- `[15]` **signed curvature preview** at a speed-scaled lookahead (`PREVIEW_TIME·speed`
  ahead, same scaling as the run10 reference), normalized by a κ scale — "how sharp, and
  which way, is what's coming."
- `[16]` **v_target at the car's position**, normalized by V_MAX — the braking-aware
  target speed. Paired with `[0]` current speed, the policy directly sees the gap it must
  close. (CC's add beyond the research list: this turns "infer the profile from curvature"
  into "track this given number" — the single biggest learnability lever. Flag for review;
  trimmable if Chat prefers curvature-only.)
- `[17]` **slip-angle beta**, normalized/clipped — "are you sliding."

`beta` is already computed (it fed ESC); curvature and v_target reuse existing centerline
geometry (`_point_at_arc`, `_smoothed_forward_yaw`, the new V_TARGET array). Obs box stays
[-1, 1]. (Reserve: a second curvature horizon at ~3 s if one preview proves too myopic ->
obs 19.)

---

## 3. DROP the scripted stack — exact list

**Remove:**
- `SPEED_WEIGHT` / `speed_reward` — the "reward speed everywhere" anti-pattern term.
- `SPIN_WEIGHT` / `spin_penalty` / `SLIP_DEADZONE` — wheelspin reward penalty (replaced by
  the slip-angle penalty).
- **TC:** `TC_SLIP_DEAD/FULL/MIN_THR`, the `tc_factor` block in `step()` (applied_throttle
  becomes just `throttle`), `_tc_cut_sum/_tc_cut_steps`, info `tc_cut_frac/tc_cut_mean`.
- **ESC:** `ESC_BETA_DEAD/FULL`, `ESC_MIN_SPEED_M_S`, the `esc_min` param + `esc_factor`
  block, `_esc_cut_steps`, info `esc_cut_frac`. **(Keep the `beta` computation — it now
  feeds obs `[17]` + the slip penalty; keep `beta_max/beta_mean` telemetry.)**
- **Grad-CAPS:** `train_beamng` uses plain SB3 `SAC`, not `GradCapsSAC`; `--lambda-t`
  retired. (`gradcaps_sac.py` left in the repo, unused.)
- **Speed-scaled steering rate:** `steer_rate_hi` param, `STEER_RATE_V_LO/HI`, the speed
  taper in the rate-limit block, `_steer_ratehi_steps`, info `steer_ratehi_frac`.
- **`THROTTLE_SMOOTH_WEIGHT` / `throttle_smooth_penalty`** — this penalizes |Δthrottle|
  *including brake application*, i.e. it directly fights hard braking. Must go.
- **`SMOOTH_WEIGHT` / steering smoothness penalty** — recommend drop for a clean reset
  (the flat rate limit handles steering smoothness structurally). Flag for review: benign,
  Chat may prefer to keep it light.

**Keep:**
- **ONE flat steering-rate limit:** `steer_rate = 0.5` (the run13 base; the
  `if self.steer_rate > 0: clip(±steer_rate)` path stays, speed-taper removed). The single
  retained safety net, GT-Sophy style.
- Heading kill-switch, progress×alignment core, checkpoint/lap bonuses, 6-point lookahead,
  the run10 speed-scaled *reference* for heading_err (that's an obs aid, not a constraint).

---

## 4. GAMMA — hold 0.99 (reserved knob)

Hold `gamma = 0.99` for this baseline. The braking-aware dense reward credits braking
**immediately** (over-speed is penalized the moment it occurs on the approach, not only at
the eventual spin), so we do not need a long horizon, and 0.999 risks SAC value-function
instability. Raise to 0.999 only if a watch shows long-horizon braking-credit is the
bottleneck. Documented as the one reserved knob.

---

## 5. Validation plan

1. **Offline profile probe (no BeamNG):** generate `V_TARGET[i]` over the 985-pt
   centerline; print/plot v_target vs arc. Confirm: ~33 m/s on straights; drops to
   `sqrt(12·R)` at each corner (T1 25.7, hairpins ~20, T8 28.6); and the drop **begins
   before the apex over the braking distance** (T1 target falling by ~arc 315, not a cliff
   at 338). This validates the back-prop. Tune A_LAT_MAX/A_BRAKE if the profile is too
   timid or too hot.
2. **Smoke ~7k fresh:** new obs (18-dim) loads; plain SAC trains with **no NaN**; the
   dropped constraints are **actually gone** (grep the config / confirm no tc/esc/gradcaps/
   ratehi active) and **one flat rate limit remains**; new telemetry logs (v_target, over-
   speed, beta); reward finite. Calibrate W_OVER/W_SLIP here against the §1b principle.
3. **Fresh 500k** on Mike's approval. Wrapper (buffer fix) + temp logger armed.

## 6. Fresh, not warm

Fresh — obs shape changes (15->18) so warm-start is impossible, and this is a deliberate
paradigm reset (no stale critic trained on the old speed-everywhere reward). Keep run15
artifacts as the baseline-to-beat.

## 7. G14 watch question (the verdict)

After maturity, watch 3+ times:
1. **Does braking EMERGE?** Speed visibly drops on the approach to T1 (and other corners),
   tracking v_target — slow-in. This is the whole thesis.
2. **Does T1 finally CLEAR?** max_arc reliably past T1 exit (394 m) onto T2 and beyond —
   ideally the first sustained run deep onto the lap, eventually a full lap.
3. **Is it cornering, not drifting?** slip-angle β stays low (<~9°) through corners — the
   car is gripping and turning, not sliding through.

## 8. A vs B (shared build)

This build is the shared core for both paths. Run **A (pure RL done right)** first:
time-box ~2 runs / ~1M steps (~24 h). If braking does not emerge and T1 does not clear,
fall back to **B (residual RL)** — a pure-pursuit base controller tracking the speed
profile/racing line (brakes by construction) with RL learning only a correction. A is
preferred (it's the project's purpose, and B needs a robust base controller we have
already failed to build once — the run13 tracker drove off at spawn), but B is the
proven-robust fallback against the Aug-6 clock. The A/B choice does not bind until after
run16, because the first build is identical for both.

## 9. Sequence after run16 (one change each, rule resumes)

- **run17 — spawn curriculum.** Once braking works, `random_spawn` / scheduled
  `spawn_idx` (already wired: random idx + heading + start-speed, per-idx heading fix) so
  every corner gets practiced, not just the ones reachable from the start line.
- **run18 — full width-based racing line.** Replace the centerline speed profile with an
  apex-cutting racing line using the track edges (`extract_centerline.py` already computes
  left/right edges + width) — the lap-time optimization, once laps complete.

## 10. Constraints reminder for CC

- run16 is ONE coherent paradigm change (reward + obs + constraint cleanup + the offline
  speed profile). Inseparable; the one-change rule resumes at run17.
- Structural fix is reward + observation, NOT new action scripts. Keep exactly one flat
  steering-rate limit.
- Measure the speed profile offline and calibrate the reward weights in the smoke before
  the full run.
- Pattern: CC measures + refines against the real code, proposes final weights/constants,
  Mike approves, CC smokes, Mike approves the full run.
