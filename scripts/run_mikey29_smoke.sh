#!/usr/bin/env bash
# run29 smoke: ~7k WARM-START drift SAC -- grip the straights, drift the corners. Warmed from run28's
# best_model (keeps the corner drift + no-reverse habit). Validates: 20-dim warm-load, the corner-
# gated TARGET (obs[19] ~0 on straights, band in corners), the straight slip penalty fires on
# straight slides + rings off into corners, drift reward still corner-gated, backward penalty holds,
# terminator not instant-terminating, straight_slip_frac telemetry live, no NaN. NOT the 500k.
# run29 changes on run28: corner-gate the target (corner_factor ramp) + straight slip penalty.
set -uo pipefail
export BEAMNG_HOME=/home/mike/opt/BeamNG.tech.v0.38.5.0
cd /home/mike/projects/beamng-mikey
source venv/bin/activate
pkill -f BeamNG 2>/dev/null || true
sleep 3
python train_beamng.py --run-name mikey_run29_drift_smoke --timesteps 7000 --nogpu --headless \
    --steer-rate 0.5 --random-spawn --drift \
    --warm-start checkpoints/mikey_run28/best_model/best_model.zip \
    --learning-rate 1e-4 --learning-starts 1000 --eval-freq 3000 --no-journal \
    2>&1 | tee logs/mikey_run29_drift_smoke.console.log
