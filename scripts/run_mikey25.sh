#!/usr/bin/env bash
# run25: FRESH 500k residual hybrid on the GRIP-AWARE profile (commit 4f682bb+).
# = run24's controller-led additive residual (steer +/-0.12, throttle [-0.12,+0.05], loss-of-control
# terminator, grip obs[18]) but on the run25 speed profile: V_MAX 33->55 (straights open; controller
# alone hit 187 kph + lapped) and T11 cut ~18% via the off-camber A_LAT=8 cap (the spin corner gets
# grip margin). FRESH (not warm) -- sidesteps the obs[16]=v_target/V_MAX renormalization (V_MAX
# changed) and the unfamiliar-high-speed warm mismatch. Plain SAC, lr 3e-4, 19-dim, reward unchanged.
# The prize: eval/max_arc clearing 4326m (a COMPLETED lap), term_loss_of_control falling, mean_speed
# climbing toward the ~52 m/s straights.
export BEAMNG_HOME=/home/mike/opt/BeamNG.tech.v0.38.5.0
cd /home/mike/projects/beamng-mikey
source venv/bin/activate
python scripts/auto_restart_train.py --base mikey_run25 --total 500000 --lr 3e-4 \
    --steer-rate 0.5 --random-spawn \
    --residual --residual-delta 0.12 --residual-throttle-up 0.05 \
    2>&1 | tee logs/mikey_run25_supervisor.log
