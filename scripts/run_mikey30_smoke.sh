#!/usr/bin/env bash
# run30 smoke: ~7k WARM-START drift SAC -- STRONG straight-grip discipline. Warmed from run29's
# best_model (keeps corner drift + fixed reverse). ONE change from run29: the straight slip weight
# is raised via a dedicated W_DRIFT_STRAIGHT_SLIP (0.2, up from run29's W_SLIP 0.05) so gripping the
# opening straight clearly beats sliding. Validates: 20-dim warm-load, the stronger straight penalty
# bites straight-slides, drift still corner-gated + initiates at turn-in, backward penalty safe,
# no NaN, cards. NOT the 500k.
set -uo pipefail
export BEAMNG_HOME=/home/mike/opt/BeamNG.tech.v0.38.5.0
cd /home/mike/projects/beamng-mikey
source venv/bin/activate
pkill -f BeamNG 2>/dev/null || true
sleep 3
python train_beamng.py --run-name mikey_run30_drift_smoke --timesteps 7000 --nogpu --headless \
    --steer-rate 0.5 --random-spawn --drift \
    --warm-start checkpoints/mikey_run29/best_model/best_model.zip \
    --learning-rate 1e-4 --learning-starts 1000 --eval-freq 3000 --no-journal \
    2>&1 | tee logs/mikey_run30_drift_smoke.console.log
