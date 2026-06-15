"""Generate docs/track_reference.md + docs/track_reference.png from the centerline
geometry. Read-only on the data; no sim."""
import bisect
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data.centerline_racetrack_builtin import CENTERLINE

N = len(CENTERLINE)
XY = np.array([(p[0], p[1]) for p in CENTERLINE])


def dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


CUM = [0.0] * N
for i in range(1, N):
    CUM[i] = CUM[i - 1] + dist(CENTERLINE[i - 1], CENTERLINE[i])
TRACK = CUM[N - 1] + dist(CENTERLINE[N - 1], CENTERLINE[0])


def yaw(i):
    p = CENTERLINE[(i - 1) % N]; q = CENTERLINE[(i + 1) % N]
    return math.atan2(q[1] - p[1], q[0] - p[0])


def iat(a):
    a %= TRACK
    return min(max(bisect.bisect_right(CUM, a) - 1, 0), N - 1)


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


# ---- detect corners: per-step signed turn rate; group contiguous SAME-SIGN
#      above-threshold runs, MERGE runs separated by a short gap, then DROP
#      gentle wiggles below a minimum total turn. ----
STEP = 2.0
WIN = 20.0                       # turn-rate window (m)
THRESH = math.radians(6)         # per-WIN turn to be "in a corner"
MERGE_GAP = 35.0                 # merge same-direction corners closer than this (m)
MIN_TURN = math.radians(15)      # drop corners that turn less than this overall

arcs = list(np.arange(0.0, TRACK, STEP))
rate = [wrap(yaw(iat(a + WIN)) - yaw(iat(a))) for a in arcs]   # signed turn over next WIN
sign = [(1 if r > THRESH else (-1 if r < -THRESH else 0)) for r in rate]

# contiguous same-sign runs
raw = []
i = 0
while i < len(arcs):
    if sign[i] != 0:
        j = i
        while j < len(arcs) and sign[j] == sign[i]:
            j += 1
        raw.append([arcs[i], arcs[j - 1], sign[i]])
        i = j
    else:
        i += 1
# merge same-direction runs separated by < MERGE_GAP
merged = []
for r in raw:
    if merged and merged[-1][2] == r[2] and (r[0] - merged[-1][1]) < MERGE_GAP:
        merged[-1][1] = r[1]
    else:
        merged.append(r[:])

corners = []
for a0, a1, sg in merged:
    entry, exit_ = a0, a1 + WIN     # exit ~ where the turn-window ends
    total = wrap(yaw(iat(exit_)) - yaw(iat(entry)))
    if abs(total) < MIN_TURN:
        continue
    span = (exit_ - entry)
    R = span / abs(total) if abs(total) > 1e-6 else float("inf")   # mean radius through the corner
    # apex = tightest point (max local turn rate)
    apex = max(np.arange(a0, a1 + 0.1, STEP), key=lambda a: abs(wrap(yaw(iat(a + WIN)) - yaw(iat(a)))))
    corners.append(dict(entry=entry, apex=float(apex), exit=exit_, R=R,
                        dir="L" if total > 0 else "R", turn_deg=math.degrees(abs(total))))
corners.sort(key=lambda c: c["apex"])
for k, c in enumerate(corners):
    c["T"] = f"T{k+1}"

# ---- straights = ring-complement of the corner intervals (robust to overlaps
#      and the start/finish seam) ----
iv = sorted([(c["entry"], c["exit"]) for c in corners])
# union-merge overlapping/adjacent corner intervals
uni = []
for a, b in iv:
    if uni and a <= uni[-1][1]:
        uni[-1][1] = max(uni[-1][1], b)
    else:
        uni.append([a, b])
straights = []
for k in range(len(uni)):
    s_start = uni[k][1]
    s_end = uni[(k + 1) % len(uni)][0]
    length = (s_end - s_start) % TRACK
    if length < 5:
        continue
    nxt = next((c["T"] for c in corners if abs(c["entry"] - (s_end % TRACK)) < 1e-3), "T1")
    straights.append(dict(start=s_start % TRACK, end=s_end % TRACK, length=length, before=nxt))

# ---- known landmarks (run8 geometry report) + measured speeds ----
# map a landmark name to the nearest corner apex
def nearest_T(apex_arc):
    c = min(corners, key=lambda c: abs(c["apex"] - apex_arc))
    return c["T"], c["apex"]

print(f"track_length={TRACK:.0f} m, {len(corners)} corners")
for c in corners:
    print(f"  {c['T']:>3} apex@{c['apex']:>5.0f}m R~{c['R']:>4.0f}m {c['dir']} turn={c['turn_deg']:.0f}deg")
longest = max(straights, key=lambda s: s["length"])
print(f"  longest straight: {longest['length']:.0f}m -> {longest['before']}")

# =============== write docs/track_reference.md ===============
lines = []
lines.append("# West Coast USA — track reference\n")
lines.append("Generated from `data/centerline_racetrack_builtin.py` (BeamNG DecalRoad "
             "geometry, road ID 59564). Lap walked by arc distance.\n")
lines.append(f"- **Total lap length:** {TRACK:.0f} m ({N} centerline points)")
lines.append(f"- **Corners:** {len(corners)} (sustained turn-rate > 6°/20 m, same-direction "
             "runs merged, total turn > 15°). Radius ≈ span / total-turn (mean radius "
             "through the corner); direction L/R = left/right for a forward lap.")
lines.append(f"- **Longest straight:** {longest['length']:.0f} m into {longest['before']}.")
lines.append("- **Speeds:** measured where the policy has driven it; the opening "
             "straight reaches ~115 kph (~32 m/s) before T1 (run8/run10 watches). "
             "Deeper straights not yet driven at clean pace → projected only.\n")

lines.append("## Lap sequence (by arc distance)\n")
lines.append("| arc start→end | feature | length / radius | dir | turn | notes |")
lines.append("|---|---|---|---|---|---|")
# interleave straights and corners in arc order
items = []
for s_ in straights:
    items.append((s_["start"], "straight", s_))
for c in corners:
    items.append((c["entry"], "corner", c))
items.sort(key=lambda x: x[0])
LANDMARKS = {  # apex_arc (approx, run8 report) -> (name, note)
}
for start, kind, obj in items:
    if kind == "straight":
        note = "longest straight" if obj is longest else ""
        lines.append(f"| {obj['start']:.0f}→{obj['end']:.0f} m | STRAIGHT (before {obj['before']}) "
                     f"| {obj['length']:.0f} m | — | — | {note} |")
    else:
        lines.append(f"| {obj['entry']:.0f}→{obj['exit']:.0f} m | **{obj['T']}** corner (apex {obj['apex']:.0f} m) "
                     f"| R≈{obj['R']:.0f} m | {obj['dir']} | {obj['turn_deg']:.0f}° | |")

def near(arc):  # detected turn nearest an arc
    return min(corners, key=lambda c: abs(c["apex"] - arc))
hp = sorted([near(1274)["T"], near(1336)["T"]], key=lambda t: int(t[1:]))
lines.append("\n## Known landmarks (run8 geometry report)\n")
lines.append("Radii in the table are geometric estimates (span / total turn) and can "
             "differ from the original landmark names; positions are the alignment.\n")
lines.append(f"- **First corner — \"R40\" (T1)** at apex ~{near(300)['apex']:.0f} m: the corner "
             "every policy reaches off the opening straight (entry ~295 m).")
lines.append("- **694 m longest straight** — centerline idx 470–531 = **arc 1717–2405 m** — "
             "into the **\"R252\" kink** (~R252 is gentler than this detector's ~190 m "
             "threshold, so it sits inside the straight stretch, not flagged as a turn).")
lines.append(f"- **~508 m straight into \"R109\"** — the next straight (~arc 2405–2940 m) into "
             f"the corner near {near(2940)['T']} (apex ~{near(2940)['apex']:.0f} m).")
lines.append(f"- **\"R15–R17\" hairpin pair** at arc ~1300 m = **{hp[0]}/{hp[1]}** "
             f"(apex ~{near(1274)['apex']:.0f} / {near(1336)['apex']:.0f} m) — the tightest sequence.")
lines.append(f"- **\"R143\" median corner** ≈ {near(596)['T']}/{near(3280)['T']} (mid-radius corners).")
lines.append(f"- **Total length ~{TRACK:.0f} m.**\n")
lines.append("Note: very gentle bends (radius above ~190 m, e.g. the R252 kink) are below "
             "the turn-detector threshold and are folded into the adjacent straight; the "
             "longest detected straight therefore spans the 694 m straight + R252 kink + "
             "508 m straight as one ~1270 m run between T7 and T8.\n")
lines.append("See `docs/track_reference.png` for the labeled layout.\n")

os.makedirs("docs", exist_ok=True)
with open("docs/track_reference.md", "w") as f:
    f.write("\n".join(lines))
print("wrote docs/track_reference.md")

# =============== write docs/track_reference.png ===============
fig, ax = plt.subplots(figsize=(13, 11))
ax.plot(XY[:, 0], XY[:, 1], "-", color="0.55", lw=2, zorder=1)
# start/finish
ax.plot(XY[0, 0], XY[0, 1], "ks", ms=10, zorder=5)
ax.annotate("START/FINISH (0 m)", (XY[0, 0], XY[0, 1]), textcoords="offset points",
            xytext=(8, 8), fontsize=9, weight="bold")
# corners
for c in corners:
    p = CENTERLINE[iat(c["apex"])]
    ax.plot(p[0], p[1], "o", color="crimson", ms=9, zorder=6)
    ax.annotate(f"{c['T']}\n{c['apex']:.0f}m R{c['R']:.0f}{c['dir']}", (p[0], p[1]),
                textcoords="offset points", xytext=(6, 6), fontsize=8,
                color="crimson", weight="bold")
# longest straight midpoint label
mid = (longest["start"] + longest["length"] / 2) % TRACK
pm = CENTERLINE[iat(mid)]
ax.annotate(f"694 m main straight\n→ {longest['before']}", (pm[0], pm[1]),
            textcoords="offset points", xytext=(10, -18), fontsize=9,
            color="navy", weight="bold")
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_title(f"West Coast USA racetrack — {TRACK:.0f} m, {len(corners)} corners "
             f"(T1 = first corner R40)")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
fig.tight_layout()
fig.savefig("docs/track_reference.png", dpi=110)
print("wrote docs/track_reference.png")
