#!/usr/bin/env bash
# run31 smoke: ~7k FRESH controller-backed drift RL. The base controller drives (steering backbone
# + speed); the RL is a CORNER-GATED residual -- tight on straights (grips, kills the fishtail),
# BIG authority in corners (steer +/-0.5 + throttle up to +1.0) to break the rear loose + hold the
# drift, ramped at turn-in by corner_factor; the controller's braking is relaxed in corners (cf).
# Validates the hybrid plumbing (--drift --residual => corner-gated), 20-dim obs, no NaN, cards.
# NOT the 500k. Fresh (no warm-start) -- the residual is small vs prior collapsed pure-RL policies.
set -uo pipefail
export BEAMNG_HOME=/home/mike/opt/BeamNG.tech.v0.38.5.0
cd /home/mike/projects/beamng-mikey
source venv/bin/activate
pkill -f BeamNG 2>/dev/null || true
sleep 3
python train_beamng.py --run-name mikey_run31_smoke --timesteps 7000 --nogpu --headless \
    --steer-rate 0.5 --random-spawn --drift --residual \
    --learning-rate 3e-4 --learning-starts 1000 --eval-freq 3000 --no-journal \
    2>&1 | tee logs/mikey_run31_smoke.console.log
