#!/usr/bin/env bash
# Pull a run's saved models (all three tiers) from the Mini PC to the G14, so
# Mikey can watch the full learning arc in order with watch_beamng.py.
#
# RUN THIS ON THE G14 -- it pulls from hermes-mini over Tailscale. The bare
# `hermes-mini` alias defaults to the wrong user, so we use mike@hermes-mini.
#
# Usage: pull_run_to_g14.sh <run-name> [dest-dir]
#   e.g. pull_run_to_g14.sh mikey_run1
set -euo pipefail

RUN="${1:?usage: pull_run_to_g14.sh <run-name> [dest-dir]}"
DEST="${2:-$HOME/beamng-checkpoints}"

mkdir -p "$DEST/$RUN"
rsync -avP "mike@hermes-mini:~/projects/beamng-mikey/checkpoints/$RUN/" "$DEST/$RUN/"

echo
echo "Pulled to $DEST/$RUN/"
echo "Watch in chronological order. NOTE: rolling_* step numbers are not zero-"
echo "padded, so use natural sort (ls -v / sort -V), NOT plain ls:"
echo "  Tier 1 (time, every 5k):  ls -v \"$DEST/$RUN\"/rolling_*.zip"
echo "  Tier 2 (furthest cp):     ls    \"$DEST/$RUN\"/milestone_cp*.zip   # cp01..cp15, zero-padded"
echo "  Tier 3 (lap trophy):      ls    \"$DEST/$RUN\"/milestone_lap_*.zip"
echo "Then on the G14: python watch_beamng.py --model <one-of-those>.zip"
