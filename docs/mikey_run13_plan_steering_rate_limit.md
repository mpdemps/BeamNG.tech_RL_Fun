# run13 plan: steering-rate limit (kill the overshoot-slam)

Date: 2026-06-15
Status: DRAFT for CC review. Chat drafts, CC measures + pokes holes against the real
code, proposes the final form + constant, Mike approves, CC smokes, Mike approves the
full run. Do not implement from this draft.

Depends on: the run12 rolling_160000 spin instrumentation and `docs/track_reference.md`.

Scope note: run13 is the last piece of the stability stack, the steering analog of the
traction cap. Racing line shifts to run14+.

---

## 1. Why (the run12 instrumentation)

Big result first: run12 (Grad-CAPS + the run11 traction cap) SOLVED the throttle axis.
In every run12 spin, rear slip stays 1-3 and never leads the steering flip, the
throttle-led fishtail that drove runs 7-11 is gone.

What remains is a standalone STEERING overcorrection, measured on rolling_160000: the
smooth opening holds to arc ~192-195m at ~24-25 m/s (about two-thirds down the 294m
opening straight before T1), then the policy answers a ~2m line drift with FULL LOCK,
sweeps across, OVERSHOOTS the centerline (center_off +2.1 -> -0.66), and SLAMS the
opposite lock (Δsteer 1.31 in one step). That violent reversal at speed rotates the car
into a spin. Steer leads, slip follows. ("Points forward, goes reverse" is just the
spin-out: post-flip rotation + braked momentum sliding forward-along-track, throttle 0,
no reverse gear.)

Why not the obvious levers: throttle-rate limit is the wrong axis (slip doesn't lead).
Raising Grad-CAPS's λ_T is weak because its displacement-normalization deliberately
down-weights large one-way sweeps to preserve corner agility, so it under-penalizes
exactly this overshoot-reversal no matter how high λ_T goes. The fix is a structural
steering constraint.

## 2. The change (one change): steering slew-rate limit

In step(), rate-limit the steering command (action[0]) before it goes to BeamNG:

```
requested_steer = action[0]                                  # in [-1, 1]
delta           = requested_steer - prev_steer
delta           = clip(delta, -STEER_RATE, +STEER_RATE)      # symmetric, per 20Hz step
applied_steer   = prev_steer + delta
prev_steer      = applied_steer                              # store; reset to 0 each reset()
```

Symmetric cap (both directions): a slam either way is the failure. Forcing the wheel to
ramp does two things: it directly kills the Δ1.31 slam reversal (the violent yaw), and
it self-limits the over-application, the policy can no longer dump full lock for a 2m
drift in one step, so it ramps, the drift corrects before full lock is reached, and the
overshoot shrinks.

STEER_RATE = set from the sweep (section 4), the max |Δsteer|/step that still allows
real cornering but blocks the spin-inducing slam.

## 3. Keep everything else (it composes)

Keep run12 Grad-CAPS (λ_T=1.0, this is what solved the throttle axis, do not lose it),
run11 TC (DEAD=4/FULL=7/MIN=0.1, wheelspin deadzone 4.0), run10 speed-scaled steering
reference, -25, kill-switch, 6-point lookahead. Obs 15-dim. One new action transform
(the steering-rate limit).

Implementation note for CC: Grad-CAPS currently regularizes the policy mean μ(s). With
the env now rate-limiting steering, decide whether Grad-CAPS should keep regularizing
raw μ (a soft intent-shaper, with the rate limit as the hard backstop) or the applied
steering. Keeping it on raw μ is fine and preserves the throttle-axis smoothing; just be
aware the rate limit is the real steering enforcement now.

## 4. Measure first (sets STEER_RATE)

From the run12 traces (and/or a scripted drive of the centerline using
docs/track_reference.md), characterize two things:
1. The |Δsteer|/step REQUIRED for legitimate cornering, at each corner's actual entry
   speed, especially the demanding ones: T1 (R55, ~25-30 m/s entry), the T5/T6 hairpins
   (R35/R34, low speed), and the fast T8 (R68 at the end of the 1242m straight). This is
   the floor, STEER_RATE must not cap below it or the car cannot make the corners.
2. The |Δsteer|/step of the spin-inducing slam (measured Δ1.31; smooth-driving avg was
   0.08). This is the ceiling to stay below.
Set STEER_RATE in the gap between them. Report both ranges and the recommended value;
if the fast corners need more rate than is safe on the straight, flag it (we'd then need
the speed-scaled-authority variant instead, section 8).

## 5. Validation plan

1. Probe: |Δsteer| capped at STEER_RATE both directions; a full-lock-slam request ramps
   over several steps; composes with Grad-CAPS + TC + speed-scaled reference; bounded;
   throttle/brake path untouched.
2. Smoke ~7k fresh: the rate limit fires live (a slam request shows ramped applied
   steering), no NaN, Grad-CAPS term still active, TC still cutting, obs unchanged. Log
   a steering-rate-limit-active metric (fraction of steps clipped) alongside the
   existing fluctuation columns.
3. Fresh 500k on Mike's approval. Wrapper + temp logger armed.

NOTE on the wrapper: the run12 self-heal regressed badly (warm-restart with empty
buffer + learning_starts=0 dumped reward 160 -> -20). Before the run13 500k, the
wrapper should save/restore the replay buffer (or use learning_starts > 0 on restart)
so a self-heal does not throw away progress. This is the wrapper fix flagged at run12;
fold it in here or it will bite again.

## 6. Fresh, not warm

Fresh. We are constraining the policy's entrenched over-steering, and fresh has been
right every run. (Warm-from-run12-seg0 could save ~12h but carries the just-demonstrated
empty-buffer regression risk and entrenchment risk; only consider it if the wrapper's
buffer handling is fixed first. Default fresh.) Keep run12 artifacts.

## 7. Risks and the G14 watch questions (the verdict)

The watch decides. After maturity:
1. THE STRAIGHT AT SPEED: is the overcorrection gone, smooth held line with no
   overshoot-slam-reversal? With the throttle axis already solved, this should be the
   last piece, a car that drives the straight smooth AND fast.
2. THE CORNERS (the key risk): does it still take the corners, especially the sharp T1
   and the hairpins, and turn into the fast T8 in time? If the rate limit is too tight
   the car understeers / turns in late / misses corners. If so, raise STEER_RATE.
3. Does the overcorrection merely slow down rather than vanish (a slower oscillation)?
   If so, the magnitude, not just the rate, is the issue -> speed-scaled authority.

## 8. Reserve ladder

- Overcorrection persists as a slower oscillation -> speed-scaled steering AUTHORITY
  (cap max |steer| as a function of speed; full lock at 25 m/s is the root). run14.
- Rate limit too tight -> can't make sharp/fast corners -> raise STEER_RATE.
- Both rate and authority needed -> combine.
- Do NOT add a steering reward penalty (outvoted, proven) or lean on λ_T (exempts the
  sweep, proven this run).

## 9. Constraints reminder for CC

- One behavioral change: the steering slew-rate limit. Keep Grad-CAPS + TC + speed-
  scaled reference. The wrapper buffer fix is infra, allowed alongside.
- Measure STEER_RATE from the sweep before building (the corner-rate floor is the key
  number, so we don't cap below what the corners need). Smoke before the full run.
- Structural constraint, not a reward penalty.
- Pattern: CC measures + refines against the real code and the run12 traces, proposes
  the final form + STEER_RATE (+ fresh/warm, + the wrapper fix), Mike approves, CC
  smokes, Mike approves the full run.
