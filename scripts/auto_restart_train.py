"""Event-driven auto-warm-restart supervisor for BeamNG training.

Runs train_beamng.py in segments. Concurrently watches the active segment's
monitor CSV for the BeamNG freeze signature (max_arc<1e-3 / mean_speed~0 sustained
over N episodes). On a freeze it SIGINTs the segment (graceful save), kills the
hung sim, selects the last CLEAN rolling checkpoint (saved before the freeze
onset), and relaunches warm with learning_starts=0 -- accounting for steps already
done so the TOTAL target is honored. Normal completion stops the supervisor; a real
crash (non-freeze exit) stops it without relaunch (never loop into a crash).

This is infrastructure: it does not touch the env/reward/obs. Behavioral change is
elsewhere (run10 speed-scaled reference).

Usage:
  python scripts/auto_restart_train.py --base mikey_run10 --total 500000 \
      --lr 1e-4 [--warm <ckpt>]   # omit --warm for a fresh first segment
"""
import argparse
import csv
import glob
import os
import re
import signal
import subprocess
import sys
import time

FROZEN_ARC = 1e-3
FROZEN_SPD = 0.05
FREEZE_EPS = 5          # consecutive frozen episodes => freeze
POLL_S = 20
MAX_RESTARTS = 12
CKPT_FREQ = 5000        # must match train_beamng --checkpoint-freq default
WARM_LEARNING_STARTS = 5000   # self-heal: seed the (empty) replay buffer before training


def monitor_csv(run): return f"logs/{run}/train.monitor.csv"
def console_log(run): return f"logs/{run}.console.log"
def ckpt_dir(run): return f"checkpoints/{run}"


def read_episodes(run):
    p = monitor_csv(run)
    if not os.path.exists(p):
        return []
    try:
        with open(p) as f:
            next(f); rows = list(csv.DictReader(f))
        return rows
    except Exception:
        return []


def is_frozen(run):
    rows = read_episodes(run)
    if len(rows) < FREEZE_EPS:
        return False
    last = rows[-FREEZE_EPS:]
    return all(float(r["max_arc"]) < FROZEN_ARC or float(r["mean_speed"]) < FROZEN_SPD
               for r in last)


def steps_done(run):
    """Cumulative env steps logged this segment (sum of episode lengths)."""
    return sum(int(r["l"]) for r in read_episodes(run))


def freeze_onset_step(run):
    rows = read_episodes(run)
    cum = 0
    onsets = []
    for i, r in enumerate(rows):
        if float(r["max_arc"]) < FROZEN_ARC or float(r["mean_speed"]) < FROZEN_SPD:
            onsets.append(cum)
        cum += int(r["l"])
    # first frozen episode that is sustained to the end
    cum = 0
    for i, r in enumerate(rows):
        frozen_tail = all(float(x["max_arc"]) < FROZEN_ARC or float(x["mean_speed"]) < FROZEN_SPD
                          for x in rows[i:])
        if (float(r["max_arc"]) < FROZEN_ARC or float(r["mean_speed"]) < FROZEN_SPD) and frozen_tail:
            return cum
        cum += int(r["l"])
    return cum


def last_clean_checkpoint(run, onset):
    """Highest rolling_<N>_steps.zip with N strictly below the freeze onset."""
    best = None
    for p in glob.glob(f"{ckpt_dir(run)}/rolling_*_steps.zip"):
        m = re.search(r"rolling_(\d+)_steps", p)
        if not m:
            continue
        n = int(m.group(1))
        if n < onset and (best is None or n > best[0]):
            best = (n, p)
    return best  # (step, path) or None


def crashed(run):
    log = console_log(run)
    if not os.path.exists(log):
        return False
    try:
        with open(log, errors="ignore") as f:
            tail = f.read()[-4000:]
        return ("Traceback (most recent call last)" in tail) or ("NaN" in tail)
    except Exception:
        return False


def launch(run, timesteps, lr, warm, steer_rate):
    """Launch a training segment as a child process, tee to console log."""
    os.makedirs("logs", exist_ok=True)
    cmd = [sys.executable, "train_beamng.py", "--run-name", run,
           "--timesteps", str(timesteps), "--nogpu", "--learning-rate", str(lr),
           "--steer-rate", str(steer_rate)]
    if warm:
        # WARM_LEARNING_STARTS > batch_size (256): on a self-heal the replay buffer
        # starts EMPTY (not saved), so learning_starts=0 made SAC's first train()
        # sample 256 copies of ~1 transition -> degenerate gradient that destroyed
        # the loaded policy (run12 self-heal dumped reward 160 -> -20). Seeding the
        # buffer with WARM_LEARNING_STARTS fresh transitions before any gradient step
        # fixes it; the loaded networks are preserved through the short warmup.
        cmd += ["--warm-start", warm, "--learning-starts", str(WARM_LEARNING_STARTS)]
    log = open(console_log(run), "w")
    # new session so our signals are scoped; train_beamng manages its own BeamNG
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    return p, log


def kill_beamng():
    # bracket-trick so this supervisor's own argv never self-matches
    subprocess.run(["pkill", "-9", "-f", "[B]eamNG"], check=False)
    time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--lr", default="1e-4")
    ap.add_argument("--warm", default=None)
    ap.add_argument("--steer-rate", type=float, default=0.0)
    args = ap.parse_args()

    done = 0
    warm = args.warm
    seg = 0
    while done < args.total and seg <= MAX_RESTARTS:
        run = f"{args.base}_s{seg}" if seg > 0 else args.base
        remaining = args.total - done
        print(f"[supervisor] segment {seg}: run={run} remaining={remaining} "
              f"warm={warm or 'FRESH'}", flush=True)
        p, log = launch(run, remaining, args.lr, warm, args.steer_rate)

        outcome = None
        while True:
            ret = p.poll()
            if ret is not None:
                outcome = "complete" if ret == 0 and not crashed(run) else "crash"
                break
            if is_frozen(run):
                outcome = "freeze"
                break
            time.sleep(POLL_S)

        if outcome == "complete":
            print(f"[supervisor] segment {seg} completed cleanly. DONE.", flush=True)
            log.close()
            return
        if outcome == "crash":
            print(f"[supervisor] segment {seg} CRASHED (non-freeze). Stopping, NO relaunch. "
                  f"Inspect {console_log(run)}.", flush=True)
            log.close()
            return
        # freeze: graceful stop, find clean checkpoint, relaunch warm
        seg_steps = steps_done(run)
        onset = freeze_onset_step(run)
        print(f"[supervisor] FREEZE in segment {seg} (onset~{onset} step, "
              f"{seg_steps} steps this segment). Stopping...", flush=True)
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
        except Exception:
            pass
        try:
            p.wait(timeout=120)
        except Exception:
            try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception: pass
        log.close()
        kill_beamng()
        clean = last_clean_checkpoint(run, onset)
        if not clean:
            print(f"[supervisor] no clean checkpoint < {onset} in {ckpt_dir(run)}; STOP.", flush=True)
            return
        # count only the clean steps toward the total (discard post-freeze garbage)
        done += clean[0]
        warm = clean[1]
        seg += 1
        print(f"[supervisor] relaunching warm from {warm} (clean@{clean[0]}); "
              f"total clean done={done}", flush=True)
    print(f"[supervisor] stopped: done={done}, segments={seg}, "
          f"{'target reached' if done>=args.total else 'MAX_RESTARTS hit'}", flush=True)


if __name__ == "__main__":
    main()
