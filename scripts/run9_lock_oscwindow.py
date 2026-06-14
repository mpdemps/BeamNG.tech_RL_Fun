"""run9 lock-in (robust): zigzag-debounced reversal detection.

A reversal counts only when center_off retraces from its current swing extreme by
more than REV_AMP_DEAD (jitter-immune). osc-active = >=2 such reversals within the
last N steps; charge -WEAVE_WEIGHT*max(0,|lat_vel|-LAT_DEAD)*straightness per active
step. Measure the real weave reversal spacing from this detector, set N to span one
full cycle (2 reversals), validate against the trace + control cases INCLUDING a
NOISY single correction (must read ~0 by construction). Offline only."""
import bisect
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beamng_env import CENTERLINE, WEAVE_BEND_DEAD, WEAVE_BEND_FULL

N = len(CENTERLINE)
def dist(a, b): return math.hypot(b[0] - a[0], b[1] - a[1])
CUM = [0.0] * N
for i in range(1, N):
    CUM[i] = CUM[i - 1] + dist(CENTERLINE[i - 1], CENTERLINE[i])
TRACK = CUM[N - 1] + dist(CENTERLINE[N - 1], CENTERLINE[0])
def yaw(idx):
    p = CENTERLINE[(idx - 1) % N]; q = CENTERLINE[(idx + 1) % N]
    return math.atan2(q[1] - p[1], q[0] - p[0])
def idx_at_arc(arc):
    arc %= TRACK
    return min(max(bisect.bisect_right(CUM, arc) - 1, 0), N - 1)
def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi
def straightness_at(arc, H):
    b = abs(wrap(yaw(idx_at_arc(arc + H)) - yaw(idx_at_arc(arc))))
    return max(0.0, min(1.0, (WEAVE_BEND_FULL - b) / (WEAVE_BEND_FULL - WEAVE_BEND_DEAD)))

LAT_DEAD = 0.05      # charge deadband, m/step (1.0 m/s lateral)
REV_AMP_DEAD = 0.10  # reversal must retrace >10cm from the swing extreme to count
H = 40
W = 3.0
import statistics as st_


def reversals(offs):
    """Zigzag reversal step-indices: a reversal when center_off retraces from the
    current swing extreme by > REV_AMP_DEAD. Returns list of step indices."""
    revs = []
    direction = 0            # +1 rising, -1 falling, 0 unknown
    extreme = offs[0]
    for i, o in enumerate(offs):
        if direction >= 0 and o > extreme:
            extreme = o
        elif direction <= 0 and o < extreme:
            extreme = o
        if direction >= 0 and extreme - o > REV_AMP_DEAD:
            if direction == 1: revs.append(i)
            direction = -1; extreme = o
        elif direction <= 0 and o - extreme > REV_AMP_DEAD:
            if direction == -1: revs.append(i)
            direction = 1; extreme = o
        elif direction == 0:
            direction = 1 if o >= extreme else -1
    return revs


def locked_penalty(offs, arcs, Nwin, W, LAT_DEAD, H):
    revs = set(reversals(offs))
    out = []
    prev = None
    rev_hist = []
    for i, o in enumerate(offs):
        if i in revs: rev_hist.append(i)
        rev_hist = [s for s in rev_hist if s > i - Nwin]
        osc = 1 if len(rev_hist) >= 2 else 0
        lv = 0.0 if prev is None else o - prev
        swing = max(0.0, abs(lv) - LAT_DEAD)
        out.append(-W * swing * osc * straightness_at(arcs[i], H))
        prev = o
    return out


# ---------- load trace ----------
rows = []
with open("logs/run8_weave_trace.csv") as f:
    for r in csv.DictReader(f):
        rows.append({k: (v if k == "reason" else float(v)) for k, v in r.items()})
pre = [r for r in rows if r["killswitch"] < 0.5]
offs = [r["center_off"] for r in pre]
arcs = [r["arc"] for r in pre]

# ---------- 1) real weave reversal spacing (zigzag, jitter-immune) ----------
rv = reversals(offs)
gaps = [rv[i] - rv[i - 1] for i in range(1, len(rv))]
print(f"=== 1) zigzag reversal spacing (retrace>{REV_AMP_DEAD}m), pre-killswitch n={len(pre)} ===")
if gaps:
    g = sorted(gaps)
    print(f"  {len(rv)} real reversals; gaps(steps) min={min(g)} median={g[len(g)//2]} "
          f"mean={st_.mean(gaps):.1f} p90={g[int(len(g)*0.9)]} max={max(g)}")
    print(f"  all gaps: {gaps}")
med = sorted(gaps)[len(gaps)//2]
N_CHOICE = 2 * med
print(f"\n=== 2) WEAVE_OSC_WINDOW = 2 x median-gap({med}) = {N_CHOICE} steps "
      f"(one full cycle; >=2 reversals fall inside during a weave) ===")

# ---------- 3+4) locked form over trace ----------
print(f"\n=== 3) locked form (2+ zigzag-reversals in last N={N_CHOICE}), W={W}, LAT_DEAD={LAT_DEAD}, H={H} ===")
wi = [i for i, r in enumerate(pre) if 120 <= r["arc"] < 280]
pall = locked_penalty(offs, arcs, N_CHOICE, W, LAT_DEAD, H)
pw = [pall[i] for i in wi]
print(f"  TRACE weave 120-280m: mean={st_.mean(pw):+.4f} sum={sum(pw):+.2f} min={min(pw):+.3f} "
      f"nonzero={sum(1 for x in pw if x<0)}/{len(pw)}   (V4 -0.085; 1-rev/N=50 -0.112; old -0.003)")
print(f"  TRACE full episode  : mean={st_.mean(pall):+.4f} sum={sum(pall):+.2f} min={min(pall):+.3f}")

# ---------- control cases ----------
print(f"\n=== 4) control cases (must read ~0 except the weave) ===")
def run_case(name, offs_c, arc=200.0):
    arcs_c = [arc] * len(offs_c)
    p = locked_penalty(offs_c, arcs_c, N_CHOICE, W, LAT_DEAD, H)
    print(f"  {name:34s} zigzag_revs={len(reversals(offs_c)):2d}  sum={sum(p):+.3f} "
          f"mean={st_.mean(p):+.4f} active={sum(1 for x in p if x<0)}")

run_case("held line @2.2m (+noise)", [2.2 + 0.03 * math.sin(k) for k in range(40)])
run_case("monotonic drift-back 3->0m", [3.0 - 0.12 * k for k in range(26)])
# clean single correction
run_case("single correction (clean)", [0,.3,.6,.9,1.2,1.4,1.5,1.4,1.1,.8,.5,.2,0]+[0.0]*20)
# NOISY single correction: same arc + realistic +-3cm jitter on every step
run_case("single correction (NOISY +-3cm)",
         [v + (0.03 if k % 2 else -0.03) for k, v in
          enumerate([0,.3,.6,.9,1.2,1.4,1.5,1.4,1.1,.8,.5,.2,0]+[0.0]*20)])
run_case("oscillating weave +-0.4m", [0.4 * math.sin(k * 0.9) for k in range(40)])
# oscillation inside a corner (straightness gate)
ca = 335.0
pc = locked_penalty([0.4 * math.sin(k * 0.9) for k in range(40)], [ca] * 40, N_CHOICE, W, LAT_DEAD, H)
print(f"  {'oscillation INSIDE corner@335':34s} straightness={straightness_at(ca,H):.2f}  sum={sum(pc):+.3f}")
