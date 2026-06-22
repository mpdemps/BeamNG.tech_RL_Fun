#!/usr/bin/env bash
# run24: WARM 500k residual hybrid with the THROTTLE-AUTHORITY CUT (commit 8e31645).
# Same controller-led additive residual + grip bundle as run23, but the residual bound is now
# ASYMMETRIC: steer +/-0.12, throttle [-0.12, +0.05]. The +0.05 cap limits the over-throttle-into-
# spin the policy kept doing (run22/23 ended high-variance, spinning); -0.12 keeps full lift/brake.
# Warm from run23's MAIN best_model (the ~7776 pre-outage peak) so we keep its hard-won driving and
# just retrain the throttle habit under the cap. lr 1e-4 (gentle warm fine-tune). Watch:
# residual_throttle_satfrac (is the +0.05 cap binding) + eval/term_loss_of_control falling +
# eval/max_arc clearing the 4326m lap. Reward unchanged, steer-rate 0.5, random-spawn, 19-dim.
export BEAMNG_HOME=/home/mike/opt/BeamNG.tech.v0.38.5.0
cd /home/mike/projects/beamng-mikey
source venv/bin/activate
python scripts/auto_restart_train.py --base mikey_run24 --total 500000 --lr 1e-4 \
    --warm checkpoints/mikey_run23/best_model/best_model.zip \
    --steer-rate 0.5 --random-spawn \
    --residual --residual-delta 0.12 --residual-throttle-up 0.05 \
    2>&1 | tee logs/mikey_run24_supervisor.log
