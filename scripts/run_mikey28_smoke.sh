#!/usr/bin/env bash
# run28 smoke: ~7k WARM-START drift SAC on the GTS drift car -- warmed from run27's PEAK best_model
# (NOT fresh, NOT the degraded end). Validates the run28 changes: backward-motion penalty logs +
# is safe for drift, over-speed discipline active in drift corners, raised progress weight in,
# reverse telemetry (backward_frac) on the cards, 20-dim obs, no NaN. NOT the 500k -- holds for go.
# run28 changes: W_DRIFT_PROGRESS 0.3->0.6, backward penalty (W_DRIFT_BACKWARD=0.5), over-speed
# re-activated in corners. Everything else stays run27 (drift-match, small band, corner-gate,
# anti-timidity, re-tuned terminator, GTS drift config).
set -uo pipefail
export BEAMNG_HOME=/home/mike/opt/BeamNG.tech.v0.38.5.0
cd /home/mike/projects/beamng-mikey
source venv/bin/activate
pkill -f BeamNG 2>/dev/null || true
sleep 3
python train_beamng.py --run-name mikey_run28_drift_smoke --timesteps 7000 --nogpu --headless \
    --steer-rate 0.5 --random-spawn --drift \
    --warm-start checkpoints/mikey_run27/best_model/best_model.zip \
    --learning-rate 1e-4 --learning-starts 1000 --eval-freq 3000 --no-journal \
    2>&1 | tee logs/mikey_run28_drift_smoke.console.log
