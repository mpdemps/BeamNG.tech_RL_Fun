#!/usr/bin/env bash
# run23: FRESH 500k residual hybrid + grip-awareness bundle (commit aba542e).
# Same controller-led additive residual as run22 (applied = clip(base_controller + clip(policy,
# +/-0.12), -1, 1), controller at FULL, reward unchanged) PLUS the three run23 fixes for the
# T1-exit donut the watch exposed:
#   1. LOSS-OF-CONTROL termination: yaw rate >150 deg/s sustained 8 ticks -> crash penalty, so a
#      spin/donut ENDS in ~0.4s regardless of position (run22 donuted forever, corrupting the buffer).
#   2. off_track confirmed working (a contained on-road donut just couldn't trip it; #1 is the fix).
#   3. GRIP obs[18] = normalized dist-to-road (0 on road -> 1 at edge): the policy can SEE it's
#      running wide toward low-grip grass. obs 18->19 -> FRESH policy.
# Eval = full hybrid; watch eval/term_loss_of_control FALLING + eval/max_arc climbing past the
# controller's 4326m, residual_abs trending DOWN. delta=0.12, plain SAC, gamma 0.99, random-spawn.
export BEAMNG_HOME=/home/mike/opt/BeamNG.tech.v0.38.5.0
cd /home/mike/projects/beamng-mikey
source venv/bin/activate
python scripts/auto_restart_train.py --base mikey_run23 --total 500000 --lr 3e-4 \
    --steer-rate 0.5 --random-spawn \
    --residual --residual-delta 0.12 \
    2>&1 | tee logs/mikey_run23_supervisor.log
