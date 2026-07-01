#!/usr/bin/env bash
# run30: 500k WARM-START drift SAC -- STRONG straight-grip discipline. Warmed from run29's best_model
# (corner drift + fixed reverse), continued with ONE change: the straight slip penalty raised to a
# dedicated W_DRIFT_STRAIGHT_SLIP=0.2 (run29 was 0.05, far too weak -- start-line eval reached T1 in
# 0/34, dying ~50m into the opening straight fishtailing). Everything else stays run29: corner-gated
# target (corner_factor crossfade), corner drift-match, backward penalty (0.5), corner over-speed
# discipline, progress weight (0.6), GTS drift config. lr 1e-4. --steer-rate 0.5, --random-spawn.
# WATCH on the full run: (1) eval/max_arc climbing past run29's 123m toward T1 (294m) = finally
# gripping the straight; (2) backward_frac early -- a stronger straight penalty may re-trigger the
# lift-stall-reverse: SAME ABORT as run29 (if backward_frac not falling toward ~0.02 by ~50k, STOP).
# HOLD until Mike says go (~half a day of CPU).
export BEAMNG_HOME=/home/mike/opt/BeamNG.tech.v0.38.5.0
cd /home/mike/projects/beamng-mikey
source venv/bin/activate
python scripts/auto_restart_train.py --base mikey_run30 --total 500000 --lr 1e-4 \
    --steer-rate 0.5 --random-spawn --drift \
    --warm checkpoints/mikey_run29/best_model/best_model.zip \
    2>&1 | tee logs/mikey_run30_supervisor.log
