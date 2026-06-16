# run14 plan: ESC throttle cut on slip-angle (the unguarded axis)

Date: 2026-06-15
Status: DRAFT for CC review. CC did the measurement and the core design; this
consolidates it as the run14 spec. CC pokes final holes, Mike approves, CC smokes, Mike
approves the full run. Do not implement from this draft.

SUPERSEDES `docs/mikey_run14_plan_speed_scaled_authority.md`: the speed-scaled steering
authority cap is demoted to the reserve ladder. See section 1.

Depends on: the run13 deterministic spin trace + CC's slip-angle / TC-firing analysis.

---

## 1. Why ESC, and why the authority cap is NOT the fix

The clincher: TC never fired during any of the three run13 spins. Rear longitudinal
slip stayed in -4.8..+3.9 the whole time (0 of 9 reversal steps above the DEAD=4
threshold). The spin is not wheelspin, it is the whole rear sliding sideways, a
lateral/yaw event, and TC only watches longitudinal wheelspin. So across runs 10-13 we
guarded steering magnitude, steering rate, and throttle-by-wheelspin, while the actual
spin trigger lives on a fourth axis, lateral SLIP-ANGLE, that nothing in the stack
watches.

Slip-angle beta separates the regimes cleanly: clean tracking sits at beta median 0deg,
p90 5.1deg; spin onset is beta 17-46deg. And the proven differentiator is throttle, not
steering swing size (a throttle-OFF reversal held even at +/-0.9 lock; throttle-ON snaps
it). So the fix is stability control: cut throttle on slip-angle, the real-world sibling
of the traction control we already have.

The speed-scaled authority cap (former run14) adds a third steering constraint while
leaving the lateral-throttle axis unguarded, so it would shrink the reversal swing but
leave the powered rear stepping out with nothing to catch it (TC asleep). It is demoted
to the reserve ladder; if a residual high-speed reversal persists after ESC, it becomes
the justified follow-up.

## 2. The change (one change): beta-gated ESC throttle cut

In step(), gate the applied throttle on slip-angle beta, multiplicative with the run11
traction cap:

```
beta = acos(clip(dot(vel_hat, nose_hat), -1, 1))            # slip angle (deg), from agent_state vel & dir
esc_factor = clip(1 - (beta - BETA_DEAD)/(BETA_FULL - BETA_DEAD), ESC_MIN, 1.0)
tc_factor  = clip(1 - (slip - 4.0)/(7.0 - 4.0), 0.1, 1.0)   # run11 TC, unchanged
applied_throttle = max(0, action[1]) * tc_factor * esc_factor   # brake (action[1]<0) untouched
```

- BETA_DEAD = 9 deg (above the clean-driving p90 of ~5deg, so no cut in clean cornering)
- BETA_FULL = 22 deg (deep oversteer, below the 25-46deg full-spin band)
- ESC_MIN = 0.1 (floor; cut throttle, do not brake; lifting alone restores a powered
  rear's grip)

Why beta, not |steer| or a sign-flip detector: |steer| over-cuts (it would lift in
every corner -> timid) and is speed-limited so a poor proxy; sign-flip-only is bursty
and misses sustained load; beta is the physical friction-circle quantity, a LEADING
indicator (rises 5->17deg as the rear steps out, before the spin completes, so the
one-step lag is fine), and self-calibrating (only high when the car is genuinely
sliding, which is the spin and almost never clean cornering).

Composes with TC: orthogonal signals (longitudinal wheelspin vs lateral slide),
multiplicative, the tighter constraint dominates, they never fight. A step that is both
wheelspinning and sliding should be off-throttle.

## 3. The real prize (why this is a training unlock, not just fewer spins)

ESC makes the failure self-correcting: rear steps out -> beta rises -> throttle cut ->
rear regains grip -> the car RECOVERS instead of spinning. So it finally completes the
opening straight, reaches past-T1 states for the first time, and SAC gets gradient
signal from the rest of the lap. Every prior run died before T1 with almost no
downstream experience; this is the change that should get the policy onto the track.

## 4. Keep everything else (the whole stack)

Keep run13 steering-rate limit (STEER_RATE=0.5), run12 Grad-CAPS (lambda_T=1.0), run11
TC (DEAD=4/FULL=7/MIN=0.1, wheelspin deadzone 4.0), run10 speed-scaled reference, -25,
kill-switch, 6-point lookahead, and the wrapper buffer fix (WARM_LEARNING_STARTS=5000).
Obs stays 15-dim (beta is computed from existing agent_state for the throttle transform,
it is NOT added to the observation). One new action transform (the ESC factor).

Default-OFF gate: add the ESC behind a flag / threshold defaulting to no-cut (e.g.
--esc-beta-dead very high = esc_factor 1.0 everywhere), so other runs importing the
shared env are unaffected, same protective pattern as run13's steer_rate=0 and the
authority-cap default-off.

## 5. Measure status (already done) + validation

CC has already characterized beta (clean p90 ~5deg, spin 17-46deg) and confirmed TC
never fires in the spins, so the thresholds are data-set. Remaining:
1. Probe: esc_factor cuts throttle when beta > BETA_DEAD, passes (=1.0) when beta is
   low; composes multiplicatively with TC; cut-throttle-only (brake path untouched);
   bounded.
2. Smoke ~7k fresh: ESC fires on the spin events (beta high), does NOT fire on clean
   straight/T1-approach driving (beta < BETA_DEAD), composes with TC, no NaN, all
   run10-13 mechanics intact, obs 15-dim. Log esc_cut_frac + a beta summary so we can
   confirm it is firing on slides, not clean driving.
3. Fresh 500k on Mike's approval. Wrapper (buffer fix) + temp logger armed.

Honest caveat: BETA_DEAD is set from straight + T1-approach good driving because we have
NO past-T1 cornering telemetry yet. So the smoke and the watch must confirm ESC does not
trigger on clean cornering once the car reaches corners; raise BETA_DEAD if it does.

## 6. Fresh, not warm

Fresh, consistent with the project. (Warm-from-run13 is now lower-risk with the buffer
fix and ESC would make run13's policy recoverable, but default fresh; Mike's call.)

## 7. G14 watch questions (the verdict)

After maturity, watch 3+ times:
1. THE STRAIGHT: when the rear steps out, does the car now RECOVER (throttle cut, rear
   grips, continues) instead of spinning? Does it finally complete the opening straight
   and get THROUGH T1?
2. PAST T1: does it reach the rest of the lap for the first time (the downstream states
   we have never had)?
3. OVER-CUT / timid: once it reaches corners, does ESC trigger on clean cornering and
   bog it (beta crossing BETA_DEAD in a normal corner)? Watch beta vs the corners it now
   reaches; raise BETA_DEAD if clean corners get cut.

## 8. Reserve ladder

- ESC over-cuts / timid in corners -> raise BETA_DEAD.
- Residual high-speed reversal persists (ESC fires but an unloaded-rear flick at extreme
  lock still rotates it) -> the speed-scaled steering authority cap (former run14,
  refined curve, knee at 27 m/s) becomes the justified run15.
- Cut-throttle-only insufficient -> add brake-on-beta (full ESC). Reserve.
- Do NOT add a reward penalty (outvoted, proven).

## 9. Constraints reminder for CC

- One behavioral change: the beta-gated ESC throttle cut. Keep the whole run10-13 stack
  + wrapper buffer fix. Authority cap stays in reserve, not added.
- Structural constraint, not a reward penalty.
- The thresholds are data-set (beta measured); confirm in the smoke that ESC fires on
  slides and not clean driving before the 500k.
- Pattern: CC finalizes against the real code (the beta computation from agent_state,
  the multiplicative compose with TC), proposes any threshold tweak, Mike approves, CC
  smokes, Mike approves the full run.
