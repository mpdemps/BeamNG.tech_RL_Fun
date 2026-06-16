"""run14 measure-first: the empirical grip-limit steering-vs-speed curve.

From logs/run13_g14_spin_trace.csv (the 3 deterministic opening-straight spins).
For each speed band, the MAX |applied steer| the car SUSTAINED WITHOUT grip loss is
the grip-limit-steering proxy -- the cap target. A step is "grip-OK" if the nose is
holding now AND no spin is imminent (heading does not collapse in the next few steps)
AND slip is below runaway. The highest grip-OK |steer| per band is the most steering the
car can hold at that speed; the cap should sit at/just above it.

Also reports, per band, the |steer| that PRECEDED grip loss (the steer level on the last
good step before each spin) -- the ceiling the cap must stay below.

Measure only. No env, no BeamNG, no controller -- pure CSV analysis."""
import csv
import math
from collections import defaultdict

CSV = "logs/run13_g14_spin_trace.csv"
HEAD_OK = 0.85          # nose-vs-tangent: holding
HEAD_COLLAPSE = 0.70    # below this within the lookahead = spin imminent
LOOKAHEAD = 5           # steps ahead to check for an imminent collapse
SLIP_RUNAWAY = 6.0      # |slip| above this = breaking loose
# speed bands in m/s
BANDS = [(0, 15), (15, 18), (18, 21), (21, 24), (24, 27), (27, 30), (30, 99)]


def band_of(v):
    for lo, hi in BANDS:
        if lo <= v < hi:
            return (lo, hi)
    return BANDS[-1]


def main():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k not in ("term",) else v) for k, v in r.items()})

    # index rows by episode so lookahead doesn't cross episode boundaries
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[int(r["ep"])].append(r)

    grip_ok = defaultdict(list)     # band -> [ |steer| ] on grip-OK steps
    pre_loss = defaultdict(list)    # band -> [ |steer| ] on last good step before a collapse
    for ep, ers in by_ep.items():
        n = len(ers)
        for i, r in enumerate(ers):
            v = r["kph"] / 3.6
            ab = abs(r["app_steer"])
            future = ers[i:i + LOOKAHEAD + 1]
            imminent = any(fr["head_align"] < HEAD_COLLAPSE for fr in future)
            slip_ok = all(abs(fr["slip"]) < SLIP_RUNAWAY for fr in future)
            if r["head_align"] > HEAD_OK and not imminent and slip_ok:
                grip_ok[band_of(v)].append(ab)
            # last good step before a collapse: this step good, a collapse begins next few
            nxt = ers[i + 1:i + LOOKAHEAD + 1]
            if (r["head_align"] > HEAD_OK and abs(r["slip"]) < SLIP_RUNAWAY
                    and any(fr["head_align"] < HEAD_COLLAPSE for fr in nxt)):
                pre_loss[band_of(v)].append(ab)

    def pct(xs, p):
        if not xs:
            return float("nan")
        s = sorted(xs)
        return s[min(len(s) - 1, int(len(s) * p))]

    print(f"=== grip-limit steering vs speed (from {CSV}) ===")
    print(f"  grip-OK = head_align>{HEAD_OK}, no head<{HEAD_COLLAPSE} within {LOOKAHEAD} steps, |slip|<{SLIP_RUNAWAY}\n")
    print(f"{'band m/s':>10} {'kph':>9} {'n_ok':>5} {'maxOK|s|':>9} {'p90OK':>7} {'medOK':>7} | "
          f"{'n_preloss':>9} {'med preloss|s|':>14}")
    for b in BANDS:
        ok = grip_ok[b]; pl = pre_loss[b]
        kph = f"{b[0]*3.6:.0f}-{b[1]*3.6:.0f}"
        mx = max(ok) if ok else float('nan')
        print(f"{b[0]:>4}-{b[1]:<5} {kph:>9} {len(ok):>5} {mx:>9.3f} {pct(ok,0.9):>7.3f} "
              f"{pct(ok,0.5):>7.3f} | {len(pl):>9} {pct(pl,0.5):>14.3f}")

    # the spin band: what |steer| at what speed actually preceded the collapses
    print("\n=== the collapses (per episode: speed + |steer| at the last good step) ===")
    for ep, ers in by_ep.items():
        for i, r in enumerate(ers):
            nxt = ers[i + 1:i + 4]
            if (r["head_align"] > HEAD_OK and any(fr["head_align"] < HEAD_COLLAPSE for fr in nxt)):
                print(f"  EP{ep} step{int(r['step'])} arc{r['arc']:.0f}m  "
                      f"{r['kph']/3.6:.1f} m/s ({r['kph']:.0f} kph)  "
                      f"|app_steer|={abs(r['app_steer']):.3f}  slip={r['slip']:.1f}")
                break  # first collapse in the episode is the opening-straight spin


if __name__ == "__main__":
    main()
