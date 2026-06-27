#!/usr/bin/env bash
# run27: FRESH 500k DRIFT SAC on the GTS drift car (Phase 2 -- teach it to slide ON PURPOSE).
# Drift mode + LSD rear diff; 20-dim obs (adds beta_target); reward = corner-gated slip-angle MATCH
# (small ~20 deg target, W_DRIFT=1.0 / W_DRIFT_PROGRESS=0.3), grip slip-penalty OFF in corners;
# terminator allows controlled drifts, fires on real spins. --steer-rate 0.5, --random-spawn, fresh.
# The prize: eval/r_drift climbing + eval/beta_err_mean falling (it's holding the target slide angle
# through corners) while it still covers ground and doesn't spin out.
# HOLD until Mike says go (a 500k run is ~half a day of CPU).
export BEAMNG_HOME=/home/mike/opt/BeamNG.tech.v0.38.5.0
cd /home/mike/projects/beamng-mikey
source venv/bin/activate
python scripts/auto_restart_train.py --base mikey_run27 --total 500000 --lr 3e-4 \
    --steer-rate 0.5 --random-spawn --drift \
    2>&1 | tee logs/mikey_run27_supervisor.log
