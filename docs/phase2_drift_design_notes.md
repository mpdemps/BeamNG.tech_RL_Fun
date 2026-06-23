# Phase 2 design notes: the drift / stunt agent

Date: 2026-06-20
Status: EXPLORATORY design note, not a run plan. Parked while run22 (the controller-led
hybrid lap, Phase 1) trains, so the thinking is ready when we get here. Phase 2 per CLAUDE.md
is "a stunt agent with a custom reward function Mikey designs." This note captures the reward
sketch and the architecture options. Mikey owns the reward design.

---

## 0. The big idea (and why it fits)

Drifting is the one place where this car's defining curse becomes the goal. The 748hp
rear-drive power oversteer that wrecked our laps for 22 runs, the rear stepping out the instant
you touch the throttle, IS a drift. Phase 2 turns our biggest enemy into the entire objective.

And drift may suit RL far better than grip-racing did:
- Grip-racing is unforgiving precision (brake exactly here, never slide); pure RL plateaued on
  it at our compute.
- Drift is a dynamic balance defined by ONE clean, measurable quantity, the slip angle (beta),
  which is already in the car's observation. The reward signal is dense and obvious.
- The car's slide-happy nature is an ASSET here, not a liability.
- There is published precedent: high-speed drifting RL with slip angle in the reward (Cai et
  al., RA-L 2020). RL learns drift well.

## 1. The architecture problem (why run22's hybrid won't just drift)

The run22 controller is built to do the EXACT OPPOSITE of drifting: it tracks the
minimum-curvature GRIP line and actively corrects any slide (its beta stays under ~1.3 deg by
design). The residual is tightly bounded (delta ~0.12), so it cannot overpower the controller
to break the rear loose. Bolt a drift reward onto run22 and the controller just keeps catching
the slide; the small residual cannot win. So drift will NOT fall out of the hybrid directly.

What run22 DOES give Phase 2 is the foundation underneath: a car that knows the whole track at
speed, the racing line, slip angle in the observation, the off-track check (now fixed), and a
training pipeline that works. Phase 2 is a new setup ON TOP of that competence.

## 2. The key design choice: the controller's role in a drift

This is the open architectural question. Options, roughly from least to most RL authority:

a. SECTIONED hybrid (promising): the grip controller drives the STRAIGHTS (transit to the
   corner at speed), and the RL takes over with FULL authority in the corners to initiate and
   hold the drift. Clean division of labor: controller for transit, RL for the stunt. The
   controller gets the car to the corner consistently; the RL does the slide. Likely the best
   first structure.

b. BIG-AUTHORITY residual: keep the controller everywhere but give the RL much more authority
   (large or unbounded delta, at least in corners) so it can overpower the grip correction and
   slide. Risk: the controller fights the RL, and we drift back toward pure RL (which struggled).

c. PURE RL with a drift reward, warm-started from the project's track competence. The slide is
   exactly where we WANT the RL fully in charge, and a clean slip-angle reward may be
   RL-friendly enough to work where precise grip-racing did not. The catch is we do not yet
   have a strong standalone policy to warm from (run21's solo policy was a crawler).

d. DRIFT-LINE base controller (most work): a controller that intentionally initiates a drift
   (throttle-induced oversteer / a flick into the corner) with RL refining the slide. Bigger
   build; reserve unless a-c stall.

Leaning: start with (a) the sectioned hybrid. It reuses the lapping controller for the boring
part and hands the RL the fun part with real authority, which is where RL belongs.

## 3. The reward sketch (Mikey designs this)

Core idea: reward "sideways but controlled and still going." Concretely, a starting sketch for
Mikey to shape:

- REWARD sustained slip angle in a target DRIFT BAND, e.g. beta in ~20-45 deg. Below the band
  it is just gripping (boring); above it is a spin (out of control). Reward peaks in the band.
- GATE on staying on track (the off-track penalty we already have). A drift that leaves the
  road does not count.
- GATE on forward PROGRESS (keep covering ground along the track / racing line). Otherwise it
  learns to spin in place, which is not a drift.
- PENALIZE the full spin-out (losing it completely, beta past the band, car no longer pointing
  roughly down-track) and going off.
- Possibly later: reward HOLDING the angle through the whole corner, and LINKING drifts
  (transitioning from one corner's drift to the next), the hallmarks of a skilled drifter.

We are already instrumented for all of this: beta is in the observation and telemetry, the
on-track check is fixed, and progress is the existing reward backbone.

## 4. Open questions for Mikey

- WHERE to drift: every corner, or specific drift zones Mikey picks? (Designated drift zones
  may be easier to learn first.)
- "In control" vs "spun out": where exactly is the line? The target band (e.g. 45 deg upper
  bound) encodes this, but the watch will calibrate it.
- Whole-lap drift vs drift zones + grip elsewhere.
- Style points: do we reward bigger angles (showier) or smoother sustained drifts? Mikey's call,
  it is a stunt agent.

## 5. Sequencing and the clock

Land run22's clean, fast hybrid lap FIRST, that is essentially Phase 1's deliverable, and the
BeamNG license expires 2026-08-06, so we bank the lap before chasing the stunt. Then Phase 2:
Mikey designs the drift reward, we pick the controller's role (lean: sectioned hybrid), and we
build a drift run. If the clock gets tight, Phase 2 on a single corner / drift zone is a
satisfying, achievable target.

## 6. The narrative (for Mikey's project)

The arc writes itself: the uncontrollable power oversteer that ruined lap after lap, the thing
we spent the whole project fighting, turns out to be exactly what drifting is. Phase 1 was
teaching the car NOT to slide. Phase 2 is teaching it to slide ON PURPOSE. Same car, same curse,
opposite goal.
