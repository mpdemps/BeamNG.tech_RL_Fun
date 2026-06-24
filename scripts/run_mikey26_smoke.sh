#!/usr/bin/env bash
# run26 smoke: ~7k FRESH PURE-RL SAC on the GTS (do-both) car. NO controller, NO residual --
# the policy outputs the full action and learns to drive on its own. Validates the pure-RL
# plumbing: GTS profile + 19-dim grip obs work, loss-of-control terminator fires on spins, no
# NaN, checkpoint/milestone cards saved. NOT the 500k -- holds for Mike's go.
# Everything else is current: min-curvature racing-line reward, grip-aware GTS speed profile,
# --steer-rate 0.5, --random-spawn, plain SAC, fresh.
set -uo pipefail
export BEAMNG_HOME=/home/mike/opt/BeamNG.tech.v0.38.5.0
cd /home/mike/projects/beamng-mikey
source venv/bin/activate
pkill -f BeamNG 2>/dev/null || true
sleep 3
python train_beamng.py --run-name mikey_run26_smoke --timesteps 7000 --nogpu --headless \
    --steer-rate 0.5 --random-spawn --learning-rate 3e-4 --learning-starts 1000 \
    --eval-freq 3000 --no-journal \
    2>&1 | tee logs/mikey_run26_smoke.console.log
