#!/usr/bin/env bash
# run28: 500k WARM-START drift SAC -- controlled FORWARD drift. Warmed from run27's PEAK best_model
# (the ~peak drifter, NOT fresh, NOT the degraded end), continued under the run28 reward:
#   - W_DRIFT_PROGRESS 0.3 -> 0.6 (catch/control the slide, keep moving forward cleanly)
#   - backward-motion penalty (W_DRIFT_BACKWARD=0.5) -- kills the reverse flail; safe for drift
#   - over-speed brake signal RE-ACTIVATED in drift corners (slow to v_target before drifting)
# Everything else stays run27 (drift-match small-band target, corner-gate, anti-timidity, re-tuned
# terminator, GTS drift config). lr 1e-4 (gentle warm fine-tune). --steer-rate 0.5, --random-spawn.
# The prize: eval/r_drift held + eval/beta_err_mean falling + backward_frac ~0 + no over-speed-into-T1.
# HOLD until Mike says go (~half a day of CPU).
export BEAMNG_HOME=/home/mike/opt/BeamNG.tech.v0.38.5.0
cd /home/mike/projects/beamng-mikey
source venv/bin/activate
python scripts/auto_restart_train.py --base mikey_run28 --total 500000 --lr 1e-4 \
    --steer-rate 0.5 --random-spawn --drift \
    --warm checkpoints/mikey_run27/best_model/best_model.zip \
    2>&1 | tee logs/mikey_run28_supervisor.log
