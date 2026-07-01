#!/usr/bin/env bash
# run31: 500k FRESH controller-backed drift RL -- the architecture pivot away from pure RL.
# Base controller = backbone (pure-pursuit STEERING keeps the car pointed through the corner + speed
# profile); the RL is a CORNER-GATED residual (option a): tight on straights (grips, kills the
# fishtail that made runs 27-30 die before T1), BIG authority in corners (steer +/-0.5 to counter-
# steer, throttle up to +1.0 to break the rear loose), ramped in at turn-in by corner_factor. The
# controller's throttle-braking is relaxed in corners (scaled by cf) so the RL owns throttle there.
# Carries the run27 drift reward (corner-gated slip-angle match) + backward penalty (heading-gap
# closed). FRESH (residual starts near 0 = the controller laps; RL only learns the corner slide).
# --steer-rate 0.5, --random-spawn, lr 3e-4.
# The prize: eval/max_arc finally clearing the opening straight (past run29/30's ~50-123m) because
# the controller drives the straight, and a HELD corner drift (beta_err falling) without the collapse.
# HOLD until Mike says go (~half a day of CPU).
export BEAMNG_HOME=/home/mike/opt/BeamNG.tech.v0.38.5.0
cd /home/mike/projects/beamng-mikey
source venv/bin/activate
python scripts/auto_restart_train.py --base mikey_run31 --total 500000 --lr 3e-4 \
    --steer-rate 0.5 --random-spawn --drift --residual \
    2>&1 | tee logs/mikey_run31_supervisor.log
