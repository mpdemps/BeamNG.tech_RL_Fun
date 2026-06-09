# Overnight training automation

Bounded nightly BeamNG training that runs in a 22:00-08:00 window and never
fights the Honcho/wger Docker stacks for CPU.

## Pieces

- `../overnight_train.sh` -- wrapper: exports `BEAMNG_HOME`, `pkill`s any
  lingering BeamNG, runs training with logging to `logs/overnight/`.
- `beamng-train.service` -- the bounded run. `Restart=no`, `KillSignal=SIGINT`,
  `CPUWeight=20`.
- `beamng-train.timer` -- starts the run at 22:00. **Stays disabled** until the
  policy work is done (the default run size is a known-insufficient placeholder).
- `beamng-train-stop.{service,timer}` -- gracefully stops the run at 08:00 via
  SIGINT.

## Why SIGINT and not SIGTERM

`train_beamng.py` catches `KeyboardInterrupt` (SIGINT) and saves the final model
in its `finally`. SIGTERM would kill python without running `finally` -> no save.
`KillSignal=SIGINT` makes `systemctl stop` hit the same graceful path Ctrl+C
uses, so the 08:00 stop is clean and free. **This must be verified once (below)
before any unattended run is trusted.**

## Install (run with sudo, in your terminal)

```bash
cd ~/projects/beamng-mikey
sudo cp scripts/systemd/beamng-train.service        /etc/systemd/system/
sudo cp scripts/systemd/beamng-train.timer          /etc/systemd/system/
sudo cp scripts/systemd/beamng-train-stop.service   /etc/systemd/system/
sudo cp scripts/systemd/beamng-train-stop.timer     /etc/systemd/system/
sudo systemctl daemon-reload
```

Do **not** `enable` either timer yet. Installing the unit files does not arm
anything; the service only runs when started manually or when a timer is
enabled.

## MANDATORY: verify the graceful SIGINT stop before arming anything

Run a short training via the service, interrupt it mid-run with `systemctl
stop`, and confirm it saved exactly like Ctrl+C does.

```bash
# 1. Start a SHORT run so the test is quick (override the placeholder size).
#    systemctl start blocks until the service is active, so background it.
sudo systemctl set-environment TIMESTEPS=8000
sudo systemctl start beamng-train.service

# 2. Watch the log until it is past the first [reset] and actually stepping
#    (BeamNG launch takes ~60s). Ctrl+C this tail once you see step activity.
tail -f ~/projects/beamng-mikey/logs/overnight/overnight_*.log

# 3. Stop it mid-run. This delivers SIGINT via KillSignal.
sudo systemctl stop beamng-train.service

# 4. Confirm the graceful path fired:
tail -n 20 ~/projects/beamng-mikey/logs/overnight/overnight_*.log
#    EXPECT: "Interrupted by user. Saving final model and journaling..."
ls -l ~/projects/beamng-mikey/checkpoints/overnight_*/final.zip   # final model saved
pgrep -f BinLinux/BeamNG.tech.x64 || echo "BeamNG terminated"     # no orphan
ss -tlnp | grep 25252 || echo "port 25252 free"                   # port released

# 5. Clean up the test environment override.
sudo systemctl unset-environment TIMESTEPS
```

Pass = step 4 shows the "Saving final model" line, `final.zip` exists, BeamNG is
gone, and 25252 is free. If instead the log just stops with no save line, the
signal path is wrong -- do not arm automation; investigate first.

## Arming (DEFERRED -- only after the reward/entropy work)

1. Set a real run size in `overnight_train.sh` (replace the 250000 placeholder)
   sized to finish inside ~10h at the measured throughput (~7.6 steps/s).
2. Enable both timers:
   ```bash
   sudo systemctl enable --now beamng-train.timer beamng-train-stop.timer
   systemctl list-timers 'beamng-*'   # confirm next-fire times
   ```

## Disarming

```bash
sudo systemctl disable --now beamng-train.timer beamng-train-stop.timer
```
