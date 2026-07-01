#!/usr/bin/env bash
# run29: 500k WARM-START drift SAC -- GRIP the straights, DRIFT the corners. Warmed from run28's
# best_model (keeps the corner drift + no-reverse habit), continued under the run29 reward:
#   - corner-gate the TARGET via a smooth corner_factor (0 straight -> band in corner), so the car
#     no longer chases a slide angle on the straight (the run28 straight-fishtail cause)
#   - straight SLIP PENALTY (reuse W_SLIP) that rings off at corner entry (turn-in drift still starts)
# Everything else stays run28 (corner-gated drift-match, backward penalty 0.5, corner over-speed
# discipline, progress weight 0.6, anti-timidity, re-tuned terminator, GTS drift config).
# lr 1e-4. --steer-rate 0.5, --random-spawn.
# The prize: straight_slip_frac ~0 + the car actually REACHES the corners and drifts them (beta_err
# falling in corners) instead of fishtailing off before T1.
# HOLD until Mike says go (~half a day of CPU).
export BEAMNG_HOME=/home/mike/opt/BeamNG.tech.v0.38.5.0
cd /home/mike/projects/beamng-mikey
source venv/bin/activate
python scripts/auto_restart_train.py --base mikey_run29 --total 500000 --lr 1e-4 \
    --steer-rate 0.5 --random-spawn --drift \
    --warm checkpoints/mikey_run28/best_model/best_model.zip \
    2>&1 | tee logs/mikey_run29_supervisor.log
