"""run9 weave-redesign OFFLINE validation. No sim (training untouched).

Three parts:
 A) Corner geometry + straightness-horizon sweep: replicate _cum_arc /
    _smoothed_forward_yaw from CENTERLINE, detect every corner, and for each
    candidate horizon H report where the straightness gate releases relative to
    each corner (does it stay live through the 180-280m back straight but release
    before legit turn-in, for ALL corners not just R40?).
 B) New weave term over the REAL trace (logs/run8_weave_trace.csv): compute the
    proposed lateral-velocity-reversal penalty (and a steering-reversal alternate,
    and an amplitude-charged variant) with straightness recomputed at the chosen
    horizon. Show it bites the 120-280m weave (strongly negative, not -0.003).
 C) Synthetic control cases through the IDENTICAL penalty fn: held line at a
    steady offset, monotonic drift-back recovery, real corner -> must read ~0.
"""
import bisect
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beamng_env import CENTERLINE, WEAVE_BEND_DEAD, WEAVE_BEND_FULL

N = len(CENTERLINE)


def dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


CUM = [0.0] * N
for i in range(1, N):
    CUM[i] = CUM[i - 1] + dist(CENTERLINE[i - 1], CENTERLINE[i])
TRACK = CUM[N - 1] + dist(CENTERLINE[N - 1], CENTERLINE[0])


def yaw(idx):
    p = CENTERLINE[(idx - 1) % N]
    q = CENTERLINE[(idx + 1) % N]
    return math.atan2(q[1] - p[1], q[0] - p[0])


def idx_at_arc(arc):
    arc %= TRACK
    return min(max(bisect.bisect_right(CUM, arc) - 1, 0), N - 1)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def bend_over(arc, H):
    """|heading(arc+H) - heading(arc)|, exactly as the reward's straightness uses."""
    near = idx_at_arc(arc)
    far = idx_at_arc(arc + H)
    return abs(wrap(yaw(far) - yaw(near)))


def straightness_at(arc, H):
    b = bend_over(arc, H)
    return max(0.0, min(1.0, (WEAVE_BEND_FULL - b) / (WEAVE_BEND_FULL - WEAVE_BEND_DEAD)))


# ---------- A) corners + horizon sweep ----------
print(f"track_length={TRACK:.0f}m, {N} centerline pts\n")
# Detect corners: local turn rate over a 25m window; group contiguous high-turn arcs.
STEP = 5.0
turn = {}
s = 0.0
while s < TRACK:
    turn[round(s)] = math.degrees(abs(wrap(yaw(idx_at_arc(s + 25)) - yaw(idx_at_arc(s)))))
    s += STEP
arcs = sorted(turn)
# corner = contiguous run where 25m-turn > 8deg; apex = max turn in the run
corners = []
i = 0
while i < len(arcs):
    if turn[arcs[i]] > 8:
        j = i
        while j < len(arcs) and turn[arcs[j]] > 8:
            j += 1
        seg = arcs[i:j]
        apex = max(seg, key=lambda a: turn[a])
        # total signed turn through the corner (sharpness): use 25m-window peak
        corners.append((apex, turn[apex], seg[0], seg[-1]))
        i = j
    else:
        i += 1
print(f"=== A) detected {len(corners)} corners (apex_arc, 25m-turn_deg, entry..exit arc) ===")
for apex, t, a0, a1 in corners:
    print(f"  apex@{apex:4d}m  turn25={t:4.0f}deg  span {a0}-{a1}m")

HS = [30, 40, 50, 60, 80]
print(f"\n=== turn-in lead: metres BEFORE apex the straightness gate releases, per horizon ===")
print(f"   (approaching the corner, the largest distance-before-apex at which straightness")
print(f"    has dropped below 1.0 contiguously to the apex; bigger = releases earlier/further out)")
print(f"{'apex':>5} {'turn':>4} " + " ".join(f"H{h:>3}" for h in HS))
for apex, t, a0, a1 in corners:
    cells = []
    for H in HS:
        lead = 0
        # walk outward from the apex; lead = furthest contiguous back still released
        for back in range(0, 220):
            if straightness_at(apex - back, H) < 1.0:
                lead = back
            else:
                break
        cells.append(f"{lead:>3}")
    print(f"{apex:>5} {t:>4.0f} " + "  ".join(cells))
print("  (metres before apex the gate is released; want ~30-50m for turn-in, not 100m+)")
print("  back-straight liveness check: straightness at arc 260m for each H:")
print("   " + "  ".join(f"H{H}={straightness_at(260,H):.2f}" for H in HS))
print("   straightness at arc 270m:")
print("   " + "  ".join(f"H{H}={straightness_at(270,H):.2f}" for H in HS))


# ---------- load trace ----------
rows = []
with open("logs/run8_weave_trace.csv") as f:
    for r in csv.DictReader(f):
        rows.append({k: float(v) if k not in ("reason",) else v for k, v in r.items()})


def penalties(rows, H, W, LAT_DEAD, STEER_DEAD=0.1):
    """Compute three candidate per-step penalties over a row sequence.
    V1 proposed: -W*max(0,|lat_vel|-DEAD)*reversed*straightness
    V2 amplitude: at each lateral reversal, charge the swing amplitude since the
       previous reversal: -W*max(0,amp-AMP_DEAD)*straightness  (AMP_DEAD=2*LAT_DEAD)
    V3 steer-reversal: -W*max(0,|steer|-STEER_DEAD)*steer_reversed*straightness
    Uses straightness recomputed at horizon H (NOT the trace's H=80)."""
    out = []
    prev_off = None
    prev_lv = 0.0
    prev_steer_sign = 0
    swing_accum = 0.0
    steps_since_rev = 999
    OSC_WINDOW = 4
    for r in rows:
        off = r["center_off"]
        st = straightness_at(r["arc"], H)
        lv = 0.0 if prev_off is None else (off - prev_off)
        rev = 1 if (prev_lv != 0 and lv != 0 and (lv > 0) != (prev_lv > 0)) else 0
        swing = max(0.0, abs(lv) - LAT_DEAD)
        v1 = -W * swing * rev * st
        # V2 amplitude since last reversal (single dump at the reversal)
        swing_accum += abs(lv)
        v2 = 0.0
        if rev:
            v2 = -W * max(0.0, swing_accum - 2 * LAT_DEAD) * st
            swing_accum = 0.0
        # V4 smooth: charge |lateral_vel| every step while oscillation is active
        # (a reversal happened within the last OSC_WINDOW steps). Monotonic motion
        # never reverses -> osc stays off -> unpenalized. Spreads the charge over
        # the swing instead of one spike.
        if rev:
            steps_since_rev = 0
        else:
            steps_since_rev += 1
        osc = 1 if steps_since_rev < OSC_WINDOW else 0
        v4 = -W * swing * osc * st
        # V3 steering reversal
        ssign = 1 if r["steer"] > 0 else (-1 if r["steer"] < 0 else 0)
        srev = 1 if (prev_steer_sign != 0 and ssign != 0 and ssign != prev_steer_sign) else 0
        v3 = -W * max(0.0, abs(r["steer"]) - STEER_DEAD) * srev * st
        out.append((v1, v2, v3, v4, lv, rev, st))
        prev_off = off
        prev_lv = lv if lv != 0 else prev_lv
        if ssign != 0:
            prev_steer_sign = ssign
    return out


# ---------- B) new term over real trace ----------
H_CHOICE = 40
W = 3.0
LAT_DEAD = 0.05  # m/step at 20Hz == 1.0 m/s lateral
print(f"\n=== B) new term over real trace  (H={H_CHOICE}m, WEAVE_WEIGHT={W}, LAT_DEAD={LAT_DEAD}m/step) ===")
import statistics as st_

for label, lo, hi in [("WEAVE 120-280m", 120, 280), ("full episode", 0, 99999)]:
    seg = [r for r in rows if lo <= r["arc"] < hi and r["killswitch"] < 0.5]
    if not seg:
        continue
    p = penalties(seg, H_CHOICE, W, LAT_DEAD)
    v1 = [x[0] for x in p]; v2 = [x[1] for x in p]; v3 = [x[2] for x in p]; v4 = [x[3] for x in p]
    print(f"  [{label}] n={len(seg)} (pre-killswitch)")
    print(f"     V1 lat-vel-reversal(instant): mean={st_.mean(v1):+.4f}  sum={sum(v1):+.2f}  min={min(v1):+.3f}  nonzero={sum(1 for x in v1 if x<0)}")
    print(f"     V2 swing-amplitude  (spike) : mean={st_.mean(v2):+.4f}  sum={sum(v2):+.2f}  min={min(v2):+.3f}  nonzero={sum(1 for x in v2 if x<0)}")
    print(f"     V4 osc-window |latvel|(smooth):mean={st_.mean(v4):+.4f}  sum={sum(v4):+.2f}  min={min(v4):+.3f}  nonzero={sum(1 for x in v4 if x<0)}")
    print(f"     V3 steer-reversal           : mean={st_.mean(v3):+.4f}  sum={sum(v3):+.2f}  min={min(v3):+.3f}  nonzero={sum(1 for x in v3 if x<0)}")
print(f"  (run8's old position-gated term over the same weave: mean -0.003)")


# ---------- C) synthetic control cases ----------
def synth(name, offs, arc=200.0, steer=None):
    seq = []
    for k, o in enumerate(offs):
        seq.append({"arc": arc, "center_off": o,
                    "steer": (steer[k] if steer else 0.0), "killswitch": 0.0})
    p = penalties(seq, H_CHOICE, W, LAT_DEAD)
    v1 = [x[0] for x in p]; v2 = [x[1] for x in p]; v3 = [x[2] for x in p]; v4 = [x[3] for x in p]
    print(f"  {name:34s} V2 sum={sum(v2):+.3f} | V4 sum={sum(v4):+.3f} | V3 sum={sum(v3):+.3f}")


print(f"\n=== C) synthetic control cases (must read ~0; H={H_CHOICE}m straight @arc200) ===")
# held line at steady 2.2m offset with tiny sensor noise
held = [2.2 + 0.01 * math.sin(k) for k in range(30)]
synth("held line @2.2m (+noise)", held, steer=[0.02 * math.sin(k) for k in range(30)])
# monotonic drift-back recovery 3.0m -> 0
drift = [3.0 - 0.12 * k for k in range(26)]
synth("monotonic drift-back 3->0m", drift, steer=[-0.4] * 26)
# weave: oscillation +-0.4m around line (the disease)
weave = [0.4 * math.sin(k * 0.9) for k in range(30)]
synth("oscillating weave +-0.4m", weave, steer=[0.6 * math.sin(k * 0.9) for k in range(30)])
# real corner: straightness 0 there, so use an arc in a corner
corner_arc = corners[0][0] if corners else 300.0
seqc = [{"arc": corner_arc, "center_off": 1.0 - 0.05 * k, "steer": 0.5, "killswitch": 0.0} for k in range(20)]
pc = penalties(seqc, H_CHOICE, W, LAT_DEAD)
print(f"  real corner @apex{corner_arc}m (straightness={straightness_at(corner_arc,H_CHOICE):.2f})  "
      f"V2 sum={sum(x[1] for x in pc):+.3f}  V4 sum={sum(x[3] for x in pc):+.3f}  V3 sum={sum(x[2] for x in pc):+.3f}")
