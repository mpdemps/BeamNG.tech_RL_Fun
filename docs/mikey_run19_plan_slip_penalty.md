# run19 plan: strengthen the slip penalty — teach throttle discipline at T1

Date: 2026-06-19
Status: APPROVED. CC did the measurement and the calibration; Mike approved the two
numbers (BETA_SLIP_DEAD=7.0, W_SLIP=0.15, warm-start). Next: CC makes the two-line change,
runs the probe + 7k smoke, Mike approves the 500k.

Baseline: run18 (anti-timid nudge), KEPT as-is. run19 changes ONLY the slip-angle penalty
weight and deadband. This is NOT re-adding scripted TC/ESC; it is tuning the reward term
that already exists to teach this (the GT Sophy tire-slip lesson).

---

## 0. The diagnosis (watch + trace, both confirmed, reproducible)

run18 broke the timidity (carries ~20 m/s, brakes for T1) but cannot get THROUGH T1. The
G14 watch (3x) and CC's deterministic trace (3x, identical) agree exactly: the car brakes
into T1 (~90 -> ~40 kph), then at TURN-IN floors the throttle, the rear breaks loose on the
748hp RWD car (power oversteer), and it SPINS off the outside (right side of the left
corner) at ~350 m, short of T1's 394 m exit. Spin onset arc 327-330 m, onset beta
9.6-11.1 deg, every time. Steering loads to -0.8 into the corner then snaps to +0.9
counter-steer as the rear lets go.

It is NOT braking, NOT the line, NOT over-speed. It is throttle-induced slip at the right
speed: too much throttle while the tires are also cornering, so the rear exceeds grip.

The run8 lesson applied (measure before cranking a knob): walking ep0 into the spin showed
the slip penalty fires too LATE and too WEAK, both knobs:

```
step  beta   thr   slip-pen  over-pen
 226   3.8  +0.59   -0.000   -0.442
 227   6.2  +0.62   -0.000   -0.418   rear already going, penalty STILL 0
 228   8.5  +0.65   -0.000   -0.495   clearly sliding, penalty STILL 0 (beta < 9)
 229  10.7  +0.71   -0.083   -0.641   "onset": penalty finally on, but -0.08
 231  15.1  +0.73   -0.305            still tiny vs throttle's reward
 237  29.0  +0.44   -0.998            only now big, but the car is already spinning
 245  58.2  +0.86   -2.462            useless; mid-spin
```

- Too LATE (-> BETA_DEAD): the rear starts going at beta 5-8 deg (steps 226-228, throttle
  high, steering loaded), and the penalty reads exactly 0 through that entire recoverable
  window because BETA_DEAD=9. By the time beta clears 9 the slide is committed (10 -> 60 deg
  over the next 16 steps).
- Too WEAK (-> W_SLIP): even once on, in the recoverable zone (beta 9-20) the penalty is
  -0.08 to -0.5/step, while the throttle earns ~+4/step (progress ~2 + match ~2.1). Flooring
  wins ~10x. The penalty only rivals the throttle reward at beta ~50, i.e. after the spin.
- Over-speed penalty is NOT the lever: it IS firing (-0.2 to -0.9, not nullified), but the
  car is at/near v_target (~21-24 m/s), so it stays correctly small. The spin is slip at the
  right speed, not over-speed.

## 1. The change (one change, two knobs on the same term)

| knob           | run18 | run19    | why |
|----------------|-------|----------|-----|
| BETA_SLIP_DEAD | 9.0   | 7.0      | Engages at the true onset (beta 7-8.5) instead of after. Clean tracking p90 ~5 deg, clean-steering transients peak 6-7 deg, so 7 leaves grippy cornering ~free; only the building slide pays. |
| W_SLIP         | 0.05  | 0.15 (3x)| At beta 8 -> -0.15; beta 12 -> -0.90; beta 15 -> -1.35. A steep cost gradient in the recoverable zone that rivals the match term (~2.1), so feathering beats flooring before the spin commits. |

The deadband drop is the higher-leverage half: catching the slide in the recoverable 5-8
window (where a small lift still saves it) is worth more than raw weight, because early
correction is cheap and late correction is impossible.

## 2. Timidity guard (the binding constraint)

Clean cornering <= 5-6 deg still pays ~0 (below the 7 deadband); a car at the grip limit
touching 7 pays -0.15/step (trivial vs +4). It only bites when beta is genuinely climbing
past grip, i.e. sliding, not cornering. This should not undo run18's speed win. The smoke
must confirm BOTH sides: the penalty now bites at turn-in (nonzero in the recoverable zone)
AND mean_speed holds (not timid).

## 3. Warm, not fresh

Warm-start from run18's best_model (--warm checkpoints/mikey_run18/best_model/best_model.zip,
supported by auto_restart_train.py -> SAC.load). We are locally sharpening one penalty, not
redefining the reward like run16; the braking + turn-in carry over cleanly. The Q-function
carries slight bias from the old slip weighting but re-calibrates inside the warm
learning-starts window. Not a reason to go fresh.

## 4. Keep everything from run18

The anti-timid nudge (W_MATCH=0.10, alignment-gated), the spawn curriculum, the 8m
off-track termination, the braking-aware speed-target reward, the 18-dim obs, plain SAC,
gamma 0.99, the one flat steering-rate limit (0.5), the 20 eval/* TB cards. run19 changes
only BETA_SLIP_DEAD and W_SLIP.

## 5. Validation plan

1. Probe: the slip penalty fires during a turn-in slide (beta in the recoverable zone) and
   stays ~0 in clean straight/braking driving, bounded.
2. Smoke ~7k warm: the penalty bites the turn-in oversteer (nonzero), mean_speed still up
   (not timid), over_speed_frac low, no NaN, all run18 mechanics + the 20 eval/* cards
   intact. Log the slip-penalty contribution + mean_speed so we read both sides of the
   tension.
3. Fresh-eyes 500k on Mike's approval. Wrapper (buffer fix) + temp logger armed.

## 6. G14 watch question (the verdict)

After maturity, watch 3+ times:
1. At T1 turn-in, does it stop spinning, squeezing the throttle on progressively instead of
   flooring it and breaking the rear loose?
2. Does it carry speed through the corner (not crawl it timidly)?
3. Does it finally get THROUGH T1, past the 394 m exit, the wall no run has cleared?

## 7. Reserve ladder

- Still spins after the tune -> raise W_SLIP further (5-6x) BEFORE the racing line; 3x may
  be a touch weak against the ~10x throttle reward, and the weight knob is not exhausted.
- Goes timid in corners (won't accelerate) -> lower W_SLIP.
- Slip tune still can't get it through T1 even at higher weight -> the racing line (explicit
  path + speed reference through the whole corner), the deferred Phase-1 feature.

## 8. Constraints reminder for CC

- One change: the slip-penalty deadband + weight. Everything else stays run18.
- GROUND IT IN THE TRACE: the penalty is late (beta 5-8 reads 0) AND weak (10x throttle gap
  in the recoverable zone). Both knobs, not one. This is a reward-term tune, NOT scripted
  traction control.
- The smoke must confirm both sides: penalty bites at turn-in AND mean_speed holds.
- Pattern: CC makes the two-line change, runs the probe + smoke, Mike approves the full run.
