#!/usr/bin/env bash
# run26: FRESH 500k PURE-RL SAC on the GTS (do-both) car. NO controller, NO residual -- drop
# --residual entirely so the policy outputs the FULL action and learns to drive on its own.
# This is the real test: can plain RL learn to lap a car that COOPERATES (forgiving GTS limit),
# where it plateaued on the unforgiving race car (run20)?
# Everything else is current: min-curvature racing-line reward, 19-dim grip-aware obs,
# loss-of-control spin terminator, GTS grip-aware speed profile, --steer-rate 0.5, --random-spawn,
# plain SAC, lr 3e-4, fresh (not warm).
# The prize: eval/max_arc clearing 4326m (a COMPLETED lap), term_loss_of_control falling,
# mean_speed climbing toward the ~47 m/s straights the controller showed are reachable.
# HOLD until Mike says go (a 500k run is ~hours of CPU).
export BEAMNG_HOME=/home/mike/opt/BeamNG.tech.v0.38.5.0
cd /home/mike/projects/beamng-mikey
source venv/bin/activate
python scripts/auto_restart_train.py --base mikey_run26 --total 500000 --lr 3e-4 \
    --steer-rate 0.5 --random-spawn \
    2>&1 | tee logs/mikey_run26_supervisor.log
