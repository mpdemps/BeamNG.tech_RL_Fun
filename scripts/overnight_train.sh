#!/usr/bin/env bash
# Overnight BeamNG RL training wrapper.
#
# Run by beamng-train.service (see scripts/systemd/). Responsibilities:
#   1. Export BEAMNG_HOME. This is the canonical place the env var lives for
#      unattended runs -- non-interactive shells (systemd, cron) do NOT source
#      ~/.bashrc, so a .bashrc export would silently vanish here and trip the
#      "Set BEAMNG_HOME or pass --home" guard in train_beamng.py.
#   2. pkill any lingering BeamNG so the TechCom port (25252) is free. This is
#      the crashed-run half of the cleanup story; the clean-exit half lives in
#      train_beamng.py's finally (commit 02f3a3f).
#   3. Run training, logging stdout+stderr to a timestamped file.
#
# TIMESTEPS is overridable. The 250000 default is a PLACEHOLDER for proving the
# automation, NOT a production size: a prior 250k run produced a dead policy
# (drove into walls). Do NOT arm the 22:00 start timer with this default --
# size the run after the reward/entropy work.
set -euo pipefail

export BEAMNG_HOME=/home/mike/opt/BeamNG.tech.v0.38.5.0
REPO=/home/mike/projects/beamng-mikey
LOGDIR="$REPO/logs/overnight"
RUN="overnight_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOGDIR"
cd "$REPO"

# Clean the port in case a previous run crashed and left a zombie BeamNG.
# || true so "no process matched" (exit 1) doesn't trip set -e.
pkill -f BeamNG.tech.x64 || true
sleep 2  # let the OS release port 25252 before the fresh launch

# exec so the python process becomes the service MainPID -- required so the
# 08:00 SIGINT (KillSignal in beamng-train.service) lands on python, which
# handles it gracefully (saves final model, closes BeamNG), not on a wrapper
# shell that would not forward it.
exec ./venv/bin/python train_beamng.py \
    --run-name "$RUN" \
    --timesteps "${TIMESTEPS:-250000}" \
    --nogpu \
    >> "$LOGDIR/$RUN.log" 2>&1
