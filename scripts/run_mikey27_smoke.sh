#!/usr/bin/env bash
# run27 smoke: ~7k FRESH drift SAC on the GTS drift car (drift mode + LSD diff). Validates the drift
# plumbing: 20-dim obs builds, r_drift logs and is corner-gated, the re-tuned terminator does NOT
# false-fire on controlled drifts (episodes aren't all instant-terminating), spins still terminate,
# no NaN, cards saved (incl. the beta_err drift card). NOT the 500k -- holds for Mike's go.
# Reward: corner-gated slip-angle MATCH, small target (~20 deg), W_DRIFT=1.0 / W_DRIFT_PROGRESS=0.3.
set -uo pipefail
export BEAMNG_HOME=/home/mike/opt/BeamNG.tech.v0.38.5.0
cd /home/mike/projects/beamng-mikey
source venv/bin/activate
pkill -f BeamNG 2>/dev/null || true
sleep 3
python train_beamng.py --run-name mikey_run27_drift_smoke --timesteps 7000 --nogpu --headless \
    --steer-rate 0.5 --random-spawn --drift --learning-rate 3e-4 --learning-starts 1000 \
    --eval-freq 3000 --no-journal \
    2>&1 | tee logs/mikey_run27_drift_smoke.console.log
